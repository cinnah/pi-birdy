"""Command line entry point: birdy run | roi | classify | purge | telegram-chat-id | publish."""

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
from birdy.notify import HttpTelegramTransport, TelegramError, TelegramNotifier, fetch_chat_ids
from birdy.publish import Publisher
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
    tg = sub.add_parser("telegram-chat-id", help="print the chat ids that have messaged the bot (bot setup)")
    tg.add_argument("--token", default=None, help="override [telegram] bot_token")
    p_publish = sub.add_parser("publish", help="build the gallery mirror and rsync it to the server once")
    p_publish.add_argument(
        "--rerank",
        action="store_true",
        help="re-rank every species/day with the current thresholds before pushing"
        " (also runs automatically the first time publishing is enabled)",
    )
    return p


class ModelLoadError(Exception):
    """A model or labels file is missing/corrupt: only a human can fix it (exit 2, no restart)."""


def _classifier(config: Config, live: LiveConfig) -> TwoStageClassifier:
    try:
        gate = ImageNetBirdClassifier(
            config.classifier.model,
            config.classifier.labels,
            config.classifier.bird_classes,
            threshold=config.classifier.threshold,
            lores_size=config.camera.lores_size,
        )
        species = InatBirdClassifier(config.species.model, config.species.labels)
    except (ValueError, OSError, RuntimeError) as e:  # LiteRT raises RuntimeError on corrupt models
        raise ModelLoadError(str(e)) from e
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
    notifier = None
    if config.telegram.enabled:
        notifier = TelegramNotifier(live, store, HttpTelegramTransport(config.telegram.bot_token))
        notifier.start()
    publisher = None
    if config.publish.enabled:
        publisher = Publisher(live, store, config.storage.data_dir / "publish")
        publisher.start()
    pipeline = Pipeline(store, clf, live, notifier=notifier, publisher=publisher)
    pipeline.start()
    retention = RetentionJob(store, live)
    retention.start()
    watcher = ConfigWatcher(config_path, live)
    watcher.start()

    app = create_app(
        store, state, live, config_path,
        queue_depth=pipeline.queue_depth, classifier=clf, notifier=notifier, publisher=publisher,
    )
    server = uvicorn.Server(uvicorn.Config(app, host=config.web.host, port=config.web.port, log_level="warning"))
    web_thread = threading.Thread(target=server.run, name="web", daemon=True)
    web_thread.start()

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())

    log.info(
        "birdy starting: site=%s daylight_gate=%s alerts=%s publish=%s web=%s:%d",
        config.site.name, config.site.lat is not None, config.telegram.enabled, config.publish.enabled,
        config.web.host, config.web.port,
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
        if publisher is not None:
            publisher.stop()
        if notifier is not None:
            notifier.stop()
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
    """Classify one image and print both models' views.

    `clf.gate.predict` / `top_k` bypass the classifier's lock and decode again: fine for this
    one-shot CLI, never do that in the service.
    """
    live = LiveConfig(config)
    clf = _classifier(config, live)
    try:
        img = CropLoader(config.camera.lores_size).load(Path(image), None)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
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


def cmd_telegram_chat_id(config: Config, token_override: str | None) -> int:
    token = token_override or config.telegram.bot_token
    if not token:
        print("no bot token: set [telegram] bot_token in birdy.toml or pass --token", file=sys.stderr)
        return 2
    try:
        chats = fetch_chat_ids(token)
    except (TelegramError, OSError) as e:
        print(f"telegram error: {e}", file=sys.stderr)
        return 1
    if not chats:
        print("no chats yet: open your bot in Telegram, send it any message, then run this again", file=sys.stderr)
        return 1
    for chat_id, name, text in chats:
        print(f"{chat_id}\t{name}\t{text}")
    return 0


def cmd_publish(config: Config, rerank: bool = False) -> int:
    """Build the gallery mirror and rsync it to the server once; meant for setup and debugging.

    Safe while the service runs: both processes take an flock on the mirror folder's `.lock`,
    so if they collide one of them skips its run with a log line (here: exit 1).
    """
    if not config.publish.enabled:
        print("publishing is disabled: set [publish] enabled = true (with host) in birdy.toml", file=sys.stderr)
        return 2
    store = Store(config.storage.data_dir)
    try:
        publisher = Publisher(LiveConfig(config), store, config.storage.data_dir / "publish")
        if rerank:
            print(f"re-ranked {publisher.evaluate_all()} days")
        ok = publisher.run_once()
    finally:
        store.close()
    if ok:
        print(f"published to {config.publish.user}@{config.publish.host}:{config.publish.remote_dir}")
        return 0
    print(f"rsync failed (see log above); key: {config.publish.ssh_key}", file=sys.stderr)
    return 1


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
    except ModelLoadError as e:
        # Exit 2 like a config error so systemd does not restart-loop on something only a
        # human can fix. Other startup errors (e.g. an unmounted data dir) keep restarting.
        print(f"model error: {e}", file=sys.stderr)
        return 2
    if args.command == "roi":
        return cmd_roi(config, args.out)
    if args.command == "purge":
        return cmd_purge(config)
    if args.command == "telegram-chat-id":
        return cmd_telegram_chat_id(config, args.token)
    if args.command == "publish":
        return cmd_publish(config, args.rerank)
    return 2


if __name__ == "__main__":
    sys.exit(main())
