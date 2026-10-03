"""Command line entry point: birdy run | roi | classify | purge."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
import tomllib
from pathlib import Path

from birdy.classifier import CropLoader, ImageNetBirdClassifier, TwoStageClassifier
from birdy.config import Config, ConfigError, load_config
from birdy.live import LiveConfig
from birdy.species import InatBirdClassifier
from birdy.store import Store

log = logging.getLogger("birdy")


def build_parser() -> argparse.ArgumentParser:
    # `--config` is a global option and must precede the subcommand:
    # the systemd unit runs `birdy --config /path/birdy.toml run`.
    p = argparse.ArgumentParser(prog="birdy")
    p.add_argument("--config", default="birdy.toml", help="path to birdy.toml")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="run the capture service")
    roi = sub.add_parser("roi", help="save one lores frame (with ROI drawn) for picking coordinates")
    roi.add_argument("--out", default="roi.png")
    cls = sub.add_parser("classify", help="classify a single image file")
    cls.add_argument("image")
    sub.add_parser("purge", help="run the retention purge once")
    return p


def _classifier(config: Config, live: LiveConfig) -> TwoStageClassifier:
    gate = ImageNetBirdClassifier(
        config.classifier.model,
        config.classifier.labels,
        config.classifier.bird_classes,
        threshold=config.classifier.threshold,
        lores_size=config.camera.lores_size,
    )
    species = InatBirdClassifier(config.species.model, config.species.labels)
    return TwoStageClassifier(gate, species, live, lores_size=config.camera.lores_size)


def cmd_run(config: Config, config_path: Path) -> int:
    import uvicorn

    from birdy.capture import run_capture
    from birdy.pipeline import Pipeline
    from birdy.retention import RetentionJob
    from birdy.state import RuntimeState
    from birdy.watcher import ConfigWatcher
    from birdy.web.app import create_app

    live = LiveConfig(config)
    store = Store(config.storage.data_dir)
    # Startup config only: site name isn't live-editable and lat/lon in the DB are informational.
    store.set_site(config.site.name, config.site.lat, config.site.lon)
    state = RuntimeState()

    # One classifier for the pipeline worker and the web upload box: it serialises inference
    # under its own lock (TFLite interpreters are not thread-safe).
    clf = _classifier(config, live)
    pipeline = Pipeline(store, clf, live)
    pipeline.start()
    retention = RetentionJob(store, live)
    retention.start()
    watcher = ConfigWatcher(config_path, live)
    watcher.start()

    app = create_app(store, state, live, config_path, queue_depth=pipeline.queue_depth, classifier=clf)
    server = uvicorn.Server(uvicorn.Config(app, host=config.web.host, port=config.web.port, log_level="warning"))
    web_thread = threading.Thread(target=server.run, name="web", daemon=True)
    web_thread.start()

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())

    log.info(
        "birdy starting: site=%s daylight_gate=%s web=%s:%d",
        config.site.name, config.site.lat is not None, config.web.host, config.web.port,
    )
    try:
        # If the port is taken, uvicorn sys.exit(1)s inside its thread; without this
        # check the service would run headless forever. Fail fast so systemd restarts us.
        for _ in range(50):
            if server.started or not web_thread.is_alive():
                break
            time.sleep(0.1)
        if not server.started:
            log.error("web server failed to start on %s:%d", config.web.host, config.web.port)
            return 1
        run_capture(live, store, pipeline, state, stop)
        return 0
    except Exception:
        # capture.py already logged the traceback via log.exception.
        return 1
    finally:
        server.should_exit = True
        watcher.stop()
        pipeline.stop()
        retention.stop()
        store.close()


def cmd_roi(config: Config, out: str) -> int:
    import cv2
    import numpy as np

    from birdy.capture import grab_lores_frame

    try:
        frame = grab_lores_frame(config)
    except RuntimeError as e:
        print(
            f"camera unavailable ({e}); if the service is running use "
            "http://birdy.local:8000/roi.png or 'sudo systemctl stop birdy' first",
            file=sys.stderr,
        )
        return 1
    img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    pts = np.array(config.motion.roi, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
    cv2.imwrite(out, img)
    print(f"saved {out} ({img.shape[1]}x{img.shape[0]}); current roi={list(map(list, config.motion.roi))}")
    return 0


def cmd_classify(config: Config, image: str) -> int:
    live = LiveConfig(config)
    clf = _classifier(config, live)
    img = CropLoader(config.camera.lores_size).load(Path(image), None)
    p_gate = clf.gate.predict(img)
    pred, top = clf.classify_image(img)
    threshold = config.classifier.threshold
    rescued = ", rescued by species" if pred.is_bird and p_gate < threshold else ""
    print(
        f"bird={pred.is_bird} p_bird={pred.confidence:.3f} (gate {p_gate:.3f}{rescued}) "
        f"species={pred.species or '-'}"
    )
    print("gate (ImageNet):")
    for label, prob in clf.gate.top_k(Path(image), None, k=5):
        print(f"  {prob:.3f}  {label}")
    print("species (iNat):")
    for label, prob in top:
        print(f"  {prob:.3f}  {label.common} ({label.latin})")
    return 0


def cmd_purge(config: Config) -> int:
    from birdy.retention import run_once

    store = Store(config.storage.data_dir)
    try:
        deleted = run_once(store, config.retention.non_bird_days)
    finally:
        store.close()
    print(f"deleted {deleted}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        config = load_config(Path(args.config))
    except FileNotFoundError:
        print(f"config error: {args.config}: file not found", file=sys.stderr)
        return 2
    except tomllib.TOMLDecodeError as e:
        print(f"config error: {args.config}: invalid TOML: {e}", file=sys.stderr)
        return 2
    except ConfigError as e:
        print(f"config error: {args.config}: {e}", file=sys.stderr)
        return 2
    try:
        if args.command == "run":
            return cmd_run(config, Path(args.config))
        if args.command == "classify":
            return cmd_classify(config, args.image)
    except (ValueError, OSError) as e:
        # A missing or corrupt model file fails while the classifier is built. Exit 2 like
        # a config error so systemd does not restart-loop on something only a human can fix.
        print(f"model error: {e}", file=sys.stderr)
        return 2
    if args.command == "roi":
        return cmd_roi(config, args.out)
    if args.command == "purge":
        return cmd_purge(config)
    return 2


if __name__ == "__main__":
    sys.exit(main())
