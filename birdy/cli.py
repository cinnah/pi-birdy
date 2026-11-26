"""Command line entry point: birdy run | roi | classify | purge | telegram-chat-id | publish | species-list | dslr."""

from __future__ import annotations

import argparse
import logging
import signal
import socket
import sys
import threading
import time
import tomllib
from pathlib import Path

import cv2

from birdy.classifier import CropLoader, ImageNetBirdClassifier, TwoStageClassifier
from birdy.config import Config, ConfigError, load_config
from birdy.detector import BirdDetector
from birdy.dslr import Dslr, DslrError, region_crop, species_crop
from birdy.live import LiveConfig
from birdy.motion import crop_box
from birdy.notify import HttpTelegramTransport, TelegramError, TelegramNotifier, fetch_chat_ids
from birdy.publish import Publisher
from birdy.species import BioClipSpecies, load_bank
from birdy.species_list import SpeciesList
from birdy.store import Store
from birdy.visits import Visits

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
    cls = sub.add_parser("classify", help="classify an image file, or a stored capture with its crop")
    cls.add_argument("image", nargs="?", help="image file (classified whole, no crop)")
    cls.add_argument("--capture", type=int, default=None, help="capture id: classify its photo with the crop the service uses")
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
    sl = sub.add_parser("species-list", help="print the local species list from iNaturalist; --refresh fetches it now")
    sl.add_argument("--refresh", action="store_true")
    ds = sub.add_parser("dslr", help="the second camera: print its settings, or take a test frame")
    ds.add_argument("action", choices=("info", "shot"))
    ds.add_argument("--out", default=None, help="shot: where to save the frame (default data/dslr-test.jpg)")
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
        species = BioClipSpecies(config.species.model, config.species.bank)
        detector = BirdDetector(config.detector.model, config.detector.labels)
    except (ValueError, OSError, RuntimeError, ImportError) as e:  # LiteRT raises RuntimeError on corrupt models; ImportError: no onnxruntime
        raise ModelLoadError(str(e)) from e
    species_list = SpeciesList(config.storage.data_dir, species.bank)
    return TwoStageClassifier(gate, species, live, lores_size=config.camera.lores_size, detector=detector, species_list=species_list)


def _listen_socket(host: str, port: int) -> socket.socket | None:
    """A dual-stack socket for host "::", else None (uvicorn binds the host itself).

    birdy.local resolves to the Pi's IPv6 address too, and asyncio makes "::" IPv6-only,
    so a browser that tries IPv6 first would hang. Clearing IPV6_V6ONLY accepts both.
    """
    if host != "::":
        return None
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        sock.bind((host, port))
    except OSError:
        sock.close()
        raise
    return sock


def cmd_run(config: Config, config_path: Path) -> int:
    import uvicorn

    from birdy.capture import run_capture
    from birdy.pipeline import Pipeline
    from birdy.retention import RetentionJob
    from birdy.state import RuntimeState
    from birdy.watcher import ConfigWatcher
    from birdy.web.app import create_app

    # Bound before any thread starts, so a taken port exits cleanly for systemd to retry.
    try:
        sock = _listen_socket(config.web.host, config.web.port)
    except OSError as e:
        log.error("web server failed to start on %s:%d: %s", config.web.host, config.web.port, e)
        return 1

    live = LiveConfig(config)
    store = Store(config.storage.data_dir)
    # Startup config only: site name isn't live-editable and lat/lon in the DB are informational.
    store.set_site(config.site.name, config.site.lat, config.site.lon)
    state = RuntimeState()

    # One classifier for the pipeline worker and the web upload box: it serialises inference
    # under its own lock (TFLite interpreters and the onnxruntime session are not thread-safe).
    clf = _classifier(config, live)
    model_id = getattr(getattr(clf, "species", None), "model_id", None)
    if model_id is not None:
        n = store.reset_species_for_model(model_id)
        if n:
            log.info("species model %s is new: %d birds will be renamed by the backfill", model_id, n)
    species_list = getattr(clf, "species_list", None)
    if species_list is not None:
        species_list.start_refresh(config.site.lat, config.site.lon, config.species.region_radius_km)
        if config.site.lat is not None and species_list.current() is None:
            # No stored list: wait for the first fetch so the history rename uses the local list.
            log.info("species list: waiting for the first fetch")
            species_list.stop(timeout=60)
            cur = species_list.current()
            if cur is not None:
                log.info("species list: %d species within %d km", len(cur.species), cur.radius_km)
            else:
                log.warning("species list: not fetched yet; using the whole bank for now")
        apply_candidates = getattr(clf, "apply_candidates", None)
        if apply_candidates is not None:
            try:
                apply_candidates()   # so /species shows the right candidate count before the first bird
            except Exception:
                log.exception("applying the species candidates failed; retried on the next bird")
    built = store.migrate_visits(config.visits.gap_seconds)
    if built:
        queued = store.count_classification_backlog()
        # The backfill takes 50 rows per 5-min sweep at ~3.5 s each: whichever bound is longer.
        eta_min = max(queued * 3.5 / 60, -(-queued // 50) * 5)
        log.info("visits: built %d visits over history; %d photos queued for embeddings (~%d min)",
                 built, queued, round(eta_min))
    # Always built: the whole [telegram] section is live on /settings. Idle while messages are off.
    notifier = TelegramNotifier(live, HttpTelegramTransport(lambda: live.get().telegram.bot_token))
    notifier.start()
    # Always built: `publish.enabled` and the gallery connection are live on /settings. Idle while off.
    publisher = Publisher(live, store, config.storage.data_dir / "publish")
    publisher.start()
    # Always built: `dslr.enabled` is a live switch on /settings. Nothing runs gphoto2 while it is off.
    dslr = Dslr(store, live)
    if config.dslr.enabled:
        try:
            info = dslr.info()
            log.info("%s connected: %s", getattr(dslr, "model", None) or "extra camera", ", ".join(f"{k}={v}" for k, v in info.items()))
        except DslrError:
            pass   # logged by Dslr; captures fall back to the Pi photo until it answers
    # Getattr: test doubles have no species model; Visits then votes by the stored model species.
    bank = getattr(getattr(clf, "species", None), "bank", None)
    visits = Visits(store, live, bank=bank, species_list=species_list, notifier=notifier, publisher=publisher)
    pipeline = Pipeline(store, clf, live, lores_size=config.camera.lores_size, visits=visits, dslr=dslr)
    pipeline.start()
    if store.migrated_visits and publisher.enabled:
        publisher.evaluate_all()   # the migration re-grouped history; re-rank every day once
    retention = RetentionJob(store, live, visits=visits)
    retention.start()
    watcher = ConfigWatcher(config_path, live)
    watcher.start()

    app = create_app(
        store, state, live, config_path,
        queue_depth=pipeline.queue_depth, classifier=clf, notifier=notifier, publisher=publisher,
        visits=visits, dslr=dslr,
    )
    server = uvicorn.Server(uvicorn.Config(app, host=config.web.host, port=config.web.port, log_level="warning"))
    web_thread = threading.Thread(
        target=server.run, kwargs={"sockets": [sock] if sock else None}, name="web", daemon=True,
    )
    web_thread.start()

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())

    log.info(
        "birdy starting: site=%s daylight_gate=%s alerts=%s publish=%s dslr=%s web=%s:%d",
        config.site.name, config.site.lat is not None, config.telegram.enabled, config.publish.enabled,
        config.dslr.enabled, config.web.host, config.web.port,
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
        if species_list is not None:
            species_list.stop()
        store.close()


def cmd_roi(config: Config, out: str) -> int:
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


def cmd_classify(config: Config, image: str | None, capture_id: int | None = None) -> int:
    """Classify one image and print both models' views.

    With `--capture`, the stored photo is cropped exactly as the pipeline crops it (motion
    crop, else the ROI crop; for a capture promoted to its DSLR frame, the species crop of that frame). `clf.gate.predict` / `top_k_image` / `clf.detector.detect` bypass
    the classifier's lock: fine for this one-shot CLI, never do that in the service.
    """
    if (image is None) == (capture_id is None):
        print("classify: give an image file or --capture <id>", file=sys.stderr)
        return 2
    bbox, pad = None, None
    dslr_img = None
    if capture_id is not None:
        store = Store(config.storage.data_dir)
        c = store.get(capture_id)
        if c is None:
            print(f"no capture {capture_id}", file=sys.stderr)
            return 1
        path = store.photos_dir / c.path
        if c.dslr_box is not None:
            frame = cv2.imread(str(store.photos_dir / c.dslr_path))
            if frame is None:
                print(f"cannot read {c.dslr_path}", file=sys.stderr)
                return 1
            dslr_img = species_crop(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), c.dslr_box)
            print(f"capture {c.id} {c.dslr_path}: dslr crop {dslr_img.shape[1]}x{dslr_img.shape[0]} px around the bird")
        else:
            crop = crop_box(c.motion_box, config.motion.roi, config.camera.lores_size)
            bbox, pad = crop.bbox, crop.pad
            b = crop.bbox
            print(f"capture {c.id} {c.path}: {crop.kind} crop x={b.x} y={b.y} w={b.w} h={b.h} (lores px)")
    else:
        path = Path(image)
    live = LiveConfig(config)
    clf = _classifier(config, live)
    try:
        img = dslr_img if dslr_img is not None else CropLoader(config.camera.lores_size).load(path, bbox, pad)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    p_gate = clf.gate.predict(img)
    pred, top = clf.classify_image(img)
    p_det = pred.detector_confidence
    print(
        f"bird={pred.is_bird} p_bird={pred.confidence:.3f} (gate {p_gate:.3f}, detector {p_det:.3f}) "
        f"species={pred.species or '-'}"
    )
    print("gate (ImageNet):")
    for label, prob in clf.gate.top_k_image(img, k=5):
        print(f"  {prob:.3f}  {label}")
    if pred.detector_box is not None:
        x, y, w, h = pred.detector_box
        print(f"detector (COCO): bird {p_det:.3f} box x={x:.2f} y={y:.2f} w={w:.2f} h={h:.2f}")
    else:
        try:
            dets = clf.detector.detect(img)
        except Exception as e:
            print(f"detector (COCO): failed ({e})")
        else:
            if dets:
                print(f"detector (COCO): no bird (best: {dets[0].label} {dets[0].score:.3f})")
            else:
                print("detector (COCO): no detections")
    if pred.is_bird:
        print(f"species (BioCLIP 2, {clf.species.candidate_count} candidates):")
        for label, prob in top:
            print(f"  {prob:.3f}  {label.common} ({label.latin})")
    else:
        print("species: skipped (not a bird)")
    return 0


_DSLR_EXPECTED = {"capturetarget": "Internal RAM", "autopoweroff": "0", "focusmode": "Manual"}


def cmd_dslr(config: Config, action: str, out: str | None = None) -> int:
    """`info`: the camera's settings, with a warning for each one birdy needs different.
    `shot`: one test frame, a copy with `dslr.region` drawn, and what the detector finds in the
    region. Works while the service runs (when both reach for the camera at once, one fails)."""
    store = Store(config.storage.data_dir)
    try:
        dslr = Dslr(store, LiveConfig(config))
        try:
            if action == "info":
                values = dslr.info()
                print(f"model: {getattr(dslr, 'model', None) or 'unknown'}")
                for k, v in values.items():
                    print(f"{k}: {v}")
                for k, want in _DSLR_EXPECTED.items():
                    if k in values and values[k] != want:
                        print(f"warning: {k} is '{values[k]}', birdy expects '{want}'")
                return 0
            dest = Path(out) if out else config.storage.data_dir / "dslr-test.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            took = dslr.capture(dest)
        except DslrError as e:
            print(f"{getattr(dslr, 'name', 'extra camera')}: {e}", file=sys.stderr)
            return 1
        frame = cv2.imread(str(dest))
        if frame is None:
            print(f"{getattr(dslr, 'name', 'extra camera')}: {dest} could not be decoded", file=sys.stderr)
            return 1
        h, w = frame.shape[:2]
        x, y, rw, rh = config.dslr.region
        marked = frame.copy()
        cv2.rectangle(marked, (int(x * w), int(y * h)), (int((x + rw) * w), int((y + rh) * h)), (0, 176, 255), max(2, w // 400))
        region_file = dest.with_name(f"{dest.stem}-region{dest.suffix}")
        cv2.imwrite(str(region_file), marked)
        print(f"saved {dest} ({w}x{h}) in {took:.2f} s; region drawn in {region_file}")
        region_rgb = cv2.cvtColor(region_crop(frame, config.dslr.region), cv2.COLOR_BGR2RGB)
        det = BirdDetector(config.detector.model, config.detector.labels).best_bird(region_rgb)
        if det is None:
            print(f"detector in region {list(config.dslr.region)}: no bird")
        else:
            bx, by, bw, bh = det.box
            print(f"detector in region {list(config.dslr.region)}: bird {det.score:.3f} "
                  f"box x={bx:.2f} y={by:.2f} w={bw:.2f} h={bh:.2f} ({bw * 100:.0f} % of the region's width)")
        return 0
    finally:
        store.close()


def cmd_purge(config: Config) -> int:
    """Run the retention job once, without `Visits` (no classifier is built here). Acceptable: only
    non-birds are purged, which never vote in their visit; a visit they leave without a bird is never
    revisited, but it no longer counts for the first-visit check (`count_visits_with_species`)."""
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
        publisher = Publisher(LiveConfig(config), store, config.storage.data_dir / "publish", rsync_timeout=3600.0)
        if rerank:
            print(f"re-ranked {publisher.evaluate_all()} days")
        ok = publisher.run_once()
    finally:
        store.close()
    if ok:
        print(f"published to {config.publish.user}@{config.publish.host}:{config.publish.remote_dir}")
        return 0
    print(
        "publish run skipped (another run in progress) or rsync failed; see the log above."
        f" key: {config.publish.ssh_key}",
        file=sys.stderr,
    )
    return 1


def cmd_species_list(config: Config, refresh: bool = False) -> int:
    from datetime import datetime, timezone

    if config.site.lat is None or config.site.lon is None:
        print("species-list: set site.lat and site.lon in birdy.toml to enable the local species list", file=sys.stderr)
        return 1
    try:
        bank = load_bank(config.species.bank)
    except ValueError as e:
        print(f"model error: {e}", file=sys.stderr)
        return 2
    sl = SpeciesList(config.storage.data_dir, bank)
    fetched = False
    if refresh or sl.current() is None:
        fetched = sl.refresh_if_needed(config.site.lat, config.site.lon, config.species.region_radius_km, force=refresh)
        if not fetched and sl.current() is None:
            print("species-list: fetch failed and no list is stored", file=sys.stderr)
            return 1
    cur = sl.current()
    age = max(0, (datetime.now(timezone.utc) - cur.fetched_at).days)
    print(f"{len(cur.species)} species within {cur.radius_km} km of {cur.lat:.3f},{cur.lon:.3f} (fetched {age} days ago)")
    for s in sorted(cur.species, key=lambda s: -s["count"]):
        print(f"  {s['count']:6d}  {s['common']} ({s['latin']})")
    unknown = sl.unknown_names(config.species.include + config.species.exclude)
    if unknown:
        print("unknown names in species.include/exclude: " + ", ".join(unknown))
    if refresh and not fetched:
        print(f"species-list: refresh failed (or too few species); showing the list fetched {age} days ago", file=sys.stderr)
        return 1
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
            return cmd_classify(config, args.image, args.capture)
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
    if args.command == "species-list":
        return cmd_species_list(config, args.refresh)
    if args.command == "dslr":
        return cmd_dslr(config, args.action, args.out)
    return 2


if __name__ == "__main__":
    sys.exit(main())
