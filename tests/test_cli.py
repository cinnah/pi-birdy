import dataclasses
import signal
import socket
import subprocess
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from birdy.cli import build_parser, cmd_run

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
DETECTOR_TOML = (
    f'[detector]\nmodel = "{ROOT}/models/efficientdet_lite0_320_ptq.tflite"\n'
    f'labels = "{ROOT}/models/coco_labels.txt"\n'
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "birdy.cli", *args],
        capture_output=True,
        text=True,
    )


def _bioclip_available():
    return (ROOT / "models" / "bioclip2-vision-int8.onnx").exists() and (ROOT / "models" / "bioclip2-birds.npz").exists()


def test_parser_subcommands():
    p = build_parser()
    assert p.parse_args(["run"]).command == "run"
    assert p.parse_args(["roi", "--out", "x.png"]).out == "x.png"
    assert p.parse_args(["classify", "a.jpg"]).image == "a.jpg"
    assert p.parse_args(["purge"]).command == "purge"


def test_default_config_path():
    assert build_parser().parse_args(["run"]).config == "birdy.toml"


def test_classify_command_prints_prediction(tmp_path):
    if not _bioclip_available():
        pytest.skip("BioCLIP model not built")
    bird = FIXTURES / "bird.jpg"
    if not bird.exists() or not (ROOT / "models" / "mobilenet_v2_1.0_224_quant.tflite").exists():
        pytest.skip("fixtures/model missing")
    cfg = tmp_path / "birdy.toml"
    cfg.write_text(
        '[site]\nname = "t"\n[motion]\nroi = [[0,0],[10,0],[10,10],[0,10]]\n'
        f'[classifier]\nmodel = "{ROOT}/models/mobilenet_v2_1.0_224_quant.tflite"\n'
        f'labels = "{ROOT}/models/imagenet_labels.txt"\nbird_classes = "{ROOT}/models/imagenet_bird_classes.txt"\n'
        f'[species]\nmodel = "{ROOT}/models/bioclip2-vision-int8.onnx"\n'
        f'bank = "{ROOT}/models/bioclip2-birds.npz"\n'
        + DETECTOR_TOML
    )
    proc = _run_cli("--config", str(cfg), "classify", str(bird))
    assert proc.returncode == 0, proc.stderr
    assert "bird=True" in proc.stdout
    assert "gate (ImageNet):" in proc.stdout
    assert "species (BioCLIP 2" in proc.stdout
    assert "detector (COCO):" in proc.stdout
    assert "(gate " in proc.stdout and ", detector " in proc.stdout


def test_classify_prints_no_bird_for_cat(tmp_path):
    if not _bioclip_available():
        pytest.skip("BioCLIP model not built")
    cat = FIXTURES / "cat.jpg"
    if not cat.exists() or not (ROOT / "models" / "mobilenet_v2_1.0_224_quant.tflite").exists():
        pytest.skip("fixtures/model missing")
    cfg = tmp_path / "birdy.toml"
    cfg.write_text(
        '[site]\nname = "t"\n[motion]\nroi = [[0,0],[10,0],[10,10],[0,10]]\n'
        f'[classifier]\nmodel = "{ROOT}/models/mobilenet_v2_1.0_224_quant.tflite"\n'
        f'labels = "{ROOT}/models/imagenet_labels.txt"\nbird_classes = "{ROOT}/models/imagenet_bird_classes.txt"\n'
        f'[species]\nmodel = "{ROOT}/models/bioclip2-vision-int8.onnx"\n'
        f'bank = "{ROOT}/models/bioclip2-birds.npz"\n'
        + DETECTOR_TOML
    )
    proc = _run_cli("--config", str(cfg), "classify", str(cat))
    assert proc.returncode == 0, proc.stderr
    assert "bird=False" in proc.stdout
    assert "detector (COCO): no bird (best:" in proc.stdout


def _classify_config(tmp_path, species_model: str) -> Path:
    cfg = tmp_path / "birdy.toml"
    cfg.write_text(
        '[site]\nname = "t"\n[motion]\nroi = [[0,0],[10,0],[10,10],[0,10]]\n'
        f'[classifier]\nmodel = "{ROOT}/models/mobilenet_v2_1.0_224_quant.tflite"\n'
        f'labels = "{ROOT}/models/imagenet_labels.txt"\nbird_classes = "{ROOT}/models/imagenet_bird_classes.txt"\n'
        f'[species]\nmodel = "{species_model}"\n'
        f'bank = "{ROOT}/models/bioclip2-birds.npz"\n'
        + DETECTOR_TOML
    )
    return cfg


def test_missing_species_model_exits_2(tmp_path):
    cfg = _classify_config(tmp_path, str(tmp_path / "nope.onnx"))
    proc = _run_cli("--config", str(cfg), "classify", str(FIXTURES / "bird.jpg"))
    assert proc.returncode == 2
    assert "model error" in proc.stderr


def test_unreadable_image_exits_1(tmp_path):
    if not _bioclip_available():
        pytest.skip("BioCLIP model not built")
    cfg = _classify_config(tmp_path, f"{ROOT}/models/bioclip2-vision-int8.onnx")
    proc = _run_cli("--config", str(cfg), "classify", str(tmp_path / "missing.jpg"))
    assert proc.returncode == 1
    assert "cannot read image" in proc.stderr
    assert "model error" not in proc.stderr

def test_bad_config_exits_2(tmp_path):
    cfg = tmp_path / "birdy.toml"
    cfg.write_text('[site]\nname = "t"\n[motion\nroi = [[0,0],[10,0],[10,10],[0,10]]\n')
    proc = _run_cli("--config", str(cfg), "purge")
    assert proc.returncode == 2
    assert "config error" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_invalid_config_value_exits_2(tmp_path):
    cfg = tmp_path / "birdy.toml"
    cfg.write_text('[site]\nname = "t"\n[motion]\nroi = [[0,0],[10,0]]\n')
    proc = _run_cli("--config", str(cfg), "purge")
    assert proc.returncode == 2
    assert "config error" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_purge_command_end_to_end(tmp_path):
    import cv2
    import numpy as np

    from birdy.store import Store

    data_dir = tmp_path / "data"
    cfg = tmp_path / "birdy.toml"
    cfg.write_text(
        '[site]\nname = "t"\n[motion]\nroi = [[0,0],[10,0],[10,10],[0,10]]\n'
        f'[storage]\ndata_dir = "{data_dir}"\n[retention]\nnon_bird_days = 7\n'
    )
    store = Store(data_dir)
    old = datetime.now().astimezone() - timedelta(days=10)
    abs_path, rel_path = store.photo_path_for(old)
    assert cv2.imwrite(str(abs_path), np.zeros((8, 8, 3), dtype=np.uint8))
    capture_id = store.add_capture(old, rel_path, 42)
    store.update_classification(capture_id, False, 0.1, None, None)
    store.close()
    assert abs_path.exists()

    proc = _run_cli("--config", str(cfg), "purge")
    assert proc.returncode == 0, proc.stderr
    assert "deleted 1" in proc.stdout
    assert not abs_path.exists()


def test_missing_config_exits_2(tmp_path):
    proc = _run_cli("--config", str(tmp_path / "nope.toml"), "purge")
    assert proc.returncode == 2
    assert "config error" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_run_wires_live_config_and_exits_without_camera(tmp_path):
    """On the Mac picamera2 is missing: `run` must start everything, then exit 1 cleanly."""
    if not _bioclip_available():
        pytest.skip("BioCLIP model not built")
    if not (ROOT / "models" / "mobilenet_v2_1.0_224_quant.tflite").exists():
        pytest.skip("model missing")
    cfg = tmp_path / "birdy.toml"
    cfg.write_text(
        '[site]\nname = "t"\n[motion]\nroi = [[0,0],[10,0],[10,10],[0,10]]\n[storage]\ndata_dir = "data"\n'
        f'[web]\nport = {_free_port()}\n'
        f'[classifier]\nmodel = "{ROOT}/models/mobilenet_v2_1.0_224_quant.tflite"\n'
        f'labels = "{ROOT}/models/imagenet_labels.txt"\nbird_classes = "{ROOT}/models/imagenet_bird_classes.txt"\n'
        f'[species]\nmodel = "{ROOT}/models/bioclip2-vision-int8.onnx"\n'
        f'bank = "{ROOT}/models/bioclip2-birds.npz"\n'
        + DETECTOR_TOML
    )
    proc = subprocess.run(
        [sys.executable, "-m", "birdy.cli", "--config", str(cfg), "run"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 1, proc.stderr
    assert "birdy starting" in proc.stderr
    assert "picamera2" in proc.stderr
    assert "Traceback" not in proc.stderr.split("camera session failed")[0]  # no errors before the expected one


def test_cmd_run_passes_live_everywhere(config, config_path, monkeypatch):
    """In-process: every live-aware component gets the same LiveConfig and the web app is up."""
    from birdy import capture, pipeline, retention, watcher
    from birdy.config import WebConfig
    from birdy.live import LiveConfig

    port = _free_port()
    config = dataclasses.replace(
        config, web=WebConfig(host="127.0.0.1", port=port),
        site=dataclasses.replace(config.site, lat=38.8, lon=-9.39),
    )
    seen: dict[str, object] = {}
    order: list[str] = []

    def record(name, init, live_index):
        def wrapper(self, *a, **kw):
            seen[name] = a[live_index]
            return init(self, *a, **kw)
        return wrapper

    # Pipeline(store, classifier, live); RetentionJob(store, live); ConfigWatcher(path, live)
    monkeypatch.setattr(retention.RetentionJob, "__init__", record("retention", retention.RetentionJob.__init__, 1))
    monkeypatch.setattr(watcher.ConfigWatcher, "__init__", record("watcher", watcher.ConfigWatcher.__init__, 1))
    class FakeSpeciesModel:
        model_id = "test/1"

    class FakeList:
        def __init__(self):
            self.started = []
            self.stopped = False

        def start_refresh(self, lat, lon, radius):
            self.started.append((lat, lon, radius))

        def current(self):
            return None              # no stored list: cmd_run waits for the first fetch

        def stop(self, timeout=5.0):
            order.append("list-stop")
            self.stopped = True

    class Clf:
        species = FakeSpeciesModel()
        species_list = FakeList()

        def apply_candidates(self):
            order.append("apply-candidates")

    clf = Clf()
    monkeypatch.setattr("birdy.cli._classifier", lambda cfg, live: clf)
    calls = []
    monkeypatch.setattr("birdy.store.Store.reset_species_for_model", lambda self, mid: calls.append(mid) or 0)
    real_init = pipeline.Pipeline.__init__

    migrations = []

    def migrate_visits(self, gap_seconds):
        order.append("migrate")
        migrations.append(gap_seconds)
        return 0

    monkeypatch.setattr("birdy.store.Store.migrate_visits", migrate_visits)

    def pipeline_init(self, store, classifier, live, notifier=None, publisher=None, lores_size=(320, 240), visits=None,
                      dslr=None):
        order.append("pipeline-init")
        seen["pipeline_classifier"] = classifier
        seen["pipeline_obj"] = self
        seen["pipeline_visits"] = visits
        seen["pipeline_lores_size"] = lores_size
        seen["pipeline_dslr"] = dslr
        return real_init(self, store, classifier, live, notifier=notifier, publisher=publisher,
                         lores_size=lores_size, visits=visits, dslr=dslr)

    monkeypatch.setattr(pipeline.Pipeline, "__init__", record("pipeline", pipeline_init, 2))
    real_start = pipeline.Pipeline.start

    def pipeline_start(self):
        order.append("pipeline-start")
        return real_start(self)

    monkeypatch.setattr(pipeline.Pipeline, "start", pipeline_start)

    from birdy.web import app as web_app
    real_create_app = web_app.create_app

    def create_app(*a, **kw):
        seen["web_classifier"] = kw.get("classifier")
        seen["web_notifier"] = kw.get("notifier", "missing")
        seen["web_visits"] = kw.get("visits")
        return real_create_app(*a, **kw)

    monkeypatch.setattr(web_app, "create_app", create_app)

    def fake_run_capture(live, store, pipeline_, state, stop):
        seen["capture"] = live
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=5) as resp:
            seen["status_http"] = resp.status
            seen["status_body"] = resp.read()

    monkeypatch.setattr(capture, "run_capture", fake_run_capture)

    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        rc = cmd_run(config, config_path)
    finally:
        for s, h in handlers.items():
            signal.signal(s, h)

    assert rc == 0
    assert clf.species_list.started == [(config.site.lat, config.site.lon, config.species.region_radius_km)]
    assert clf.species_list.stopped is True
    # The first list fetch is joined (and the candidates applied) before the first sweep.
    # The history migration runs after that and before the pipeline exists.
    assert order[:5] == ["list-stop", "apply-candidates", "migrate", "pipeline-init", "pipeline-start"], order
    assert migrations == [config.visits.gap_seconds]
    assert calls == ["test/1"]
    assert seen["status_http"] == 200
    assert b"motion_score" in seen["status_body"]
    lives = [seen[k] for k in ("pipeline", "retention", "watcher", "capture")]
    assert isinstance(lives[0], LiveConfig)
    assert all(v is lives[0] for v in lives), seen
    assert lives[0].get() is config
    # The pipeline and the upload box share one classifier (and therefore its lock).
    assert seen["pipeline_classifier"] is clf
    assert seen["web_classifier"] is clf
    # The pipeline crops with the camera's lores size, not a hard-coded default.
    assert seen["pipeline_lores_size"] == config.camera.lores_size
    assert seen["pipeline_dslr"] is not None      # always built: dslr.enabled is a live switch
    # Alerts are off in the default config: no notifier anywhere.
    assert seen["pipeline_obj"].visits._notifier is None
    assert seen["web_notifier"] is None
    # One Visits for the pipeline, the web app (and retention).
    assert seen["pipeline_visits"] is not None and seen["pipeline_obj"].visits is seen["pipeline_visits"]
    assert seen["web_visits"] is seen["pipeline_visits"]


def test_telegram_chat_id_prints_ids(capsys, config, monkeypatch):
    import dataclasses

    from birdy import cli

    monkeypatch.setattr(cli, "fetch_chat_ids", lambda token: [(11, "Yazz", "test")])
    cfg = dataclasses.replace(config, telegram=dataclasses.replace(config.telegram, bot_token="T"))
    assert cli.cmd_telegram_chat_id(cfg, None) == 0
    assert "11\tYazz\ttest" in capsys.readouterr().out


def test_telegram_chat_id_without_token_exits_2(capsys, config):
    from birdy import cli

    assert cli.cmd_telegram_chat_id(config, None) == 2
    assert "bot_token" in capsys.readouterr().err


def test_telegram_chat_id_no_chats_exits_1(capsys, config, monkeypatch):
    from birdy import cli

    monkeypatch.setattr(cli, "fetch_chat_ids", lambda token: [])
    assert cli.cmd_telegram_chat_id(config, "T") == 1
    assert "send" in capsys.readouterr().err.lower()


def test_parser_telegram_chat_id():
    a = build_parser().parse_args(["telegram-chat-id", "--token", "x"])
    assert a.command == "telegram-chat-id" and a.token == "x"


def test_cmd_run_builds_notifier_when_enabled(config_path, monkeypatch):
    import dataclasses

    from birdy import cli
    from birdy.config import load_config

    cfg = load_config(config_path)
    cfg = dataclasses.replace(cfg, telegram=dataclasses.replace(cfg.telegram, enabled=True, bot_token="T", chat_id="42"))
    built = {}

    class FakeNotifier:
        def __init__(self, live, transport):
            built["transport"] = transport
            built["self"] = self

        def start(self):
            built["started"] = True

        def stop(self, timeout=5.0):
            built["stopped"] = True

    monkeypatch.setattr(cli, "TelegramNotifier", FakeNotifier)
    monkeypatch.setattr(cli, "HttpTelegramTransport", lambda token: ("transport", token))
    monkeypatch.setattr(cli, "_classifier", lambda c, l: object())
    seen = {}
    monkeypatch.setattr("birdy.capture.run_capture", lambda live, store, pipeline, state, stop: seen.update(pipeline=pipeline))
    import birdy.web.app as webapp

    real_create_app = webapp.create_app
    monkeypatch.setattr(webapp, "create_app", lambda *a, **k: seen.update(notifier=k.get("notifier")) or real_create_app(*a, **k))
    cfg = dataclasses.replace(cfg, web=dataclasses.replace(cfg.web, port=_free_port()))
    assert cli.cmd_run(cfg, config_path) == 0
    assert built["transport"] == ("transport", "T") and built["started"] and built["stopped"]
    assert seen["notifier"] is built["self"] and seen["pipeline"].visits._notifier is built["self"]


def test_cmd_run_builds_publisher_when_enabled(config_path, monkeypatch):
    import dataclasses

    from birdy import cli
    from birdy.config import load_config

    cfg = load_config(config_path)
    cfg = dataclasses.replace(cfg, publish=dataclasses.replace(cfg.publish, enabled=True, host="h"))
    built = {}

    order = []

    class FakePublisher:
        def __init__(self, live, store, mirror_dir):
            built["mirror_dir"] = mirror_dir
            built["self"] = self

        def start(self):
            built["started"] = True

        def stop(self, timeout=10.0):
            built["stopped"] = True
            order.append("publisher.stop")

        def evaluate_all(self):   # a fresh DB is migrated too (0 visits): history is re-ranked once
            return 0

    monkeypatch.setattr(cli, "Publisher", FakePublisher)
    import birdy.pipeline as pipeline_mod
    from birdy.store import Store

    real_pipeline_stop, real_close = pipeline_mod.Pipeline.stop, Store.close
    monkeypatch.setattr(pipeline_mod.Pipeline, "stop", lambda self, *a, **k: order.append("pipeline.stop") or real_pipeline_stop(self, *a, **k))
    monkeypatch.setattr(Store, "close", lambda self: order.append("store.close") or real_close(self))
    monkeypatch.setattr(cli, "_classifier", lambda c, l: object())
    seen = {}
    monkeypatch.setattr("birdy.capture.run_capture", lambda live, store, pipeline, state, stop: seen.update(pipeline=pipeline))
    import birdy.web.app as webapp

    real_create_app = webapp.create_app
    monkeypatch.setattr(webapp, "create_app", lambda *a, **k: seen.update(publisher=k.get("publisher")) or real_create_app(*a, **k))
    cfg = dataclasses.replace(cfg, web=dataclasses.replace(cfg.web, port=_free_port()))
    assert cli.cmd_run(cfg, config_path) == 0
    assert built["mirror_dir"] == cfg.storage.data_dir / "publish" and built["started"] and built["stopped"]
    assert seen["publisher"] is built["self"] and seen["pipeline"].visits._publisher is built["self"]
    assert order == ["pipeline.stop", "publisher.stop", "store.close"]


def test_cmd_run_reranks_after_the_visits_migration(config_path, monkeypatch):
    import dataclasses

    from birdy import cli
    from birdy.config import load_config

    cfg = load_config(config_path)
    cfg = dataclasses.replace(cfg, publish=dataclasses.replace(cfg.publish, enabled=True, host="h"))
    calls = []

    class FakePublisher:
        def __init__(self, live, store, mirror_dir):
            pass

        def start(self):
            calls.append("start")

        def stop(self, timeout=10.0):
            calls.append("stop")

        def evaluate_all(self):
            calls.append("evaluate_all")
            return 0

        def evaluate_day(self, species, when):
            return False

        def notify(self):
            pass

    def migrate_visits(self, gap_seconds):
        self.migrated_visits = True
        return 2

    monkeypatch.setattr(cli, "Publisher", FakePublisher)
    monkeypatch.setattr("birdy.store.Store.migrate_visits", migrate_visits)
    monkeypatch.setattr(cli, "_classifier", lambda c, l: object())
    monkeypatch.setattr("birdy.capture.run_capture", lambda live, store, pipeline, state, stop: None)
    cfg = dataclasses.replace(cfg, web=dataclasses.replace(cfg.web, port=_free_port()))
    assert cli.cmd_run(cfg, config_path) == 0
    assert calls.count("evaluate_all") == 1 and calls.index("start") < calls.index("evaluate_all")


def test_cmd_run_no_publisher_when_disabled(config_path, monkeypatch):
    import dataclasses

    from birdy import cli
    from birdy.config import load_config

    cfg = load_config(config_path)
    monkeypatch.setattr(cli, "Publisher", lambda *a, **k: pytest.fail("Publisher built while disabled"))
    monkeypatch.setattr(cli, "_classifier", lambda c, l: object())
    seen = {}
    monkeypatch.setattr("birdy.capture.run_capture", lambda live, store, pipeline, state, stop: seen.update(pipeline=pipeline))
    import birdy.web.app as webapp

    real_create_app = webapp.create_app
    monkeypatch.setattr(webapp, "create_app", lambda *a, **k: seen.update(publisher=k.get("publisher")) or real_create_app(*a, **k))
    cfg = dataclasses.replace(cfg, web=dataclasses.replace(cfg.web, port=_free_port()))
    assert cli.cmd_run(cfg, config_path) == 0
    assert seen["publisher"] is None and seen["pipeline"].visits._publisher is None


def test_publish_command_runs_once(tmp_path, monkeypatch, capsys):
    import birdy.cli as cli

    class FakePublisher:
        instances = []

        def __init__(self, live, store, mirror_dir, **kw):
            self.mirror_dir = mirror_dir
            self.kw = kw
            FakePublisher.instances.append(self)

        def run_once(self):
            return True

    monkeypatch.setattr("birdy.cli.Publisher", FakePublisher)
    cfg = tmp_path / "birdy.toml"
    cfg.write_text('[site]\nname="t"\n[motion]\nroi=[[0,0],[9,0],[9,9]]\n[storage]\ndata_dir="data"\n[publish]\nenabled=true\nhost="h"\n')
    assert cli.main(["--config", str(cfg), "publish"]) == 0
    assert FakePublisher.instances[0].mirror_dir == tmp_path / "data" / "publish"
    assert FakePublisher.instances[0].kw["rsync_timeout"] == 3600.0  # manual one-shot: longer cap
    assert "published" in capsys.readouterr().out


def test_publish_command_rerank(tmp_path, monkeypatch, capsys):
    import birdy.cli as cli
    calls = []

    class FakePublisher:
        def __init__(self, live, store, mirror_dir, **kw):
            pass

        def evaluate_all(self):
            calls.append("evaluate_all")
            return 7

        def run_once(self):
            calls.append("run_once")
            return True

    monkeypatch.setattr("birdy.cli.Publisher", FakePublisher)
    cfg = tmp_path / "birdy.toml"
    cfg.write_text('[site]\nname="t"\n[motion]\nroi=[[0,0],[9,0],[9,9]]\n[storage]\ndata_dir="data"\n[publish]\nenabled=true\nhost="h"\n')
    assert cli.main(["--config", str(cfg), "publish", "--rerank"]) == 0
    assert calls == ["evaluate_all", "run_once"]
    assert "re-ranked 7 days" in capsys.readouterr().out
    calls.clear()
    assert cli.main(["--config", str(cfg), "publish"]) == 0
    assert calls == ["run_once"]


def test_publish_command_exit_2_when_disabled(tmp_path, capsys):
    import birdy.cli as cli
    cfg = tmp_path / "birdy.toml"
    cfg.write_text('[site]\nname="t"\n[motion]\nroi=[[0,0],[9,0],[9,9]]\n[storage]\ndata_dir="data"\n')
    assert cli.main(["--config", str(cfg), "publish"]) == 2
    assert "publish" in capsys.readouterr().err


def test_publish_command_exit_1_when_rsync_fails(tmp_path, monkeypatch, capsys):
    import birdy.cli as cli

    class FailingPublisher:
        def __init__(self, live, store, mirror_dir, **kw):
            pass

        def run_once(self):
            return False

    monkeypatch.setattr("birdy.cli.Publisher", FailingPublisher)
    cfg = tmp_path / "birdy.toml"
    cfg.write_text('[site]\nname="t"\n[motion]\nroi=[[0,0],[9,0],[9,9]]\n[storage]\ndata_dir="data"\n[publish]\nenabled=true\nhost="h"\n')
    assert cli.main(["--config", str(cfg), "publish"]) == 1
    err = capsys.readouterr().err
    assert "another run in progress" in err and "rsync failed" in err


def test_parser_classify_capture():
    args = build_parser().parse_args(["classify", "--capture", "7"])
    assert args.capture == 7 and args.image is None


def test_classify_needs_image_or_capture(config_path):
    from birdy import cli

    assert cli.main(["--config", str(config_path), "classify"]) == 2


def test_classify_capture_uses_stored_crop(tmp_path):
    if not _bioclip_available():
        pytest.skip("BioCLIP model not built")
    import cv2

    from birdy.store import Store

    bird = FIXTURES / "bird.jpg"
    if not bird.exists() or not (ROOT / "models" / "mobilenet_v2_1.0_224_quant.tflite").exists():
        pytest.skip("fixtures/model missing")
    data = tmp_path / "data"
    store = Store(data)
    when = datetime(2026, 10, 3, 10, 0).astimezone()
    abs_path, rel = store.photo_path_for(when)
    cv2.imwrite(str(abs_path), cv2.imread(str(bird)))
    with_box = store.add_capture(when, rel, 100, motion_box=(2 / 320, 2 / 240, 5 / 320, 5 / 240))
    without = store.add_capture(when, rel, 100)
    cfg = _classify_config(tmp_path, f"{ROOT}/models/bioclip2-vision-int8.onnx")
    cfg.write_text(cfg.read_text() + f'[storage]\ndata_dir = "{data}"\n')

    proc = _run_cli("--config", str(cfg), "classify", "--capture", str(with_box))
    assert proc.returncode == 0, proc.stderr
    assert "motion crop" in proc.stdout and "gate (ImageNet):" in proc.stdout

    proc = _run_cli("--config", str(cfg), "classify", "--capture", str(without))
    assert proc.returncode == 0, proc.stderr
    assert "roi crop" in proc.stdout

    eos = str(rel).replace(".jpg", "_eos.jpg")
    cv2.imwrite(str(store.photos_dir / eos), cv2.imread(str(bird)))
    store.store_verdict(without, is_bird=True, confidence=0.9, species=None, species_confidence=None,
                        dslr_path=eos, dslr_box=(0.25, 0.25, 0.5, 0.5))
    proc = _run_cli("--config", str(cfg), "classify", "--capture", str(without))
    assert proc.returncode == 0, proc.stderr
    assert "dslr crop" in proc.stdout

    proc = _run_cli("--config", str(cfg), "classify", "--capture", "999")
    assert proc.returncode == 1


def test_listen_socket_dual_stack_accepts_ipv4_and_ipv6():
    from birdy.cli import _listen_socket

    sock = _listen_socket("::", 0)
    try:
        sock.listen()
        port = sock.getsockname()[1]
        for family, addr in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
            with socket.socket(family, socket.SOCK_STREAM) as c:
                c.settimeout(2)
                c.connect((addr, port))
    finally:
        sock.close()


def test_listen_socket_none_for_plain_hosts():
    from birdy.cli import _listen_socket

    assert _listen_socket("0.0.0.0", 0) is None


def test_missing_detector_model_exits_2(tmp_path):
    if not _bioclip_available():
        pytest.skip("BioCLIP model not built")
    cfg = _classify_config(tmp_path, f"{ROOT}/models/bioclip2-vision-int8.onnx")
    cfg.write_text(cfg.read_text().replace("efficientdet_lite0_320_ptq.tflite", "nope.tflite"))
    assert "nope.tflite" in cfg.read_text()
    proc = _run_cli("--config", str(cfg), "classify", str(FIXTURES / "bird.jpg"))
    assert proc.returncode == 2 and "model error" in proc.stderr


def test_detector_labels_without_bird_exit_2(tmp_path):
    if not _bioclip_available():
        pytest.skip("BioCLIP model not built")
    labels = tmp_path / "labels.txt"
    labels.write_text("person\ncat\n")
    cfg = _classify_config(tmp_path, f"{ROOT}/models/bioclip2-vision-int8.onnx")
    cfg.write_text(cfg.read_text().replace(f"{ROOT}/models/coco_labels.txt", str(labels)))
    assert str(labels) in cfg.read_text()
    proc = _run_cli("--config", str(cfg), "classify", str(FIXTURES / "bird.jpg"))
    assert proc.returncode == 2 and "model error" in proc.stderr


def test_classify_cat_skips_species(tmp_path):
    if not _bioclip_available():
        pytest.skip("BioCLIP model not built")
    cfg = _classify_config(tmp_path, f"{ROOT}/models/bioclip2-vision-int8.onnx")
    proc = _run_cli("--config", str(cfg), "classify", str(FIXTURES / "cat.jpg"))
    assert proc.returncode == 0 and "species: skipped (not a bird)" in proc.stdout


def test_parser_species_list():
    assert build_parser().parse_args(["species-list"]).refresh is False
    assert build_parser().parse_args(["species-list", "--refresh"]).refresh is True


def _bank(path):
    np.savez(path, ids=np.array([13094], np.int32), latin=np.array(["Phoenicurus ochruros"]), common=np.array(["Black Redstart"]),
             emb=np.eye(1, 4, dtype=np.float16), logit_scale=np.float32(100.0), model=np.str_("x"), prompt=np.str_("p"))
    return path


def test_cmd_species_list_prints_list_and_refreshes(capsys, config, monkeypatch, tmp_path):
    from birdy import cli
    from birdy.config import SiteConfig, StorageConfig
    from birdy.species_list import LocalList

    config = dataclasses.replace(config, site=SiteConfig("t", 38.8, -9.39), storage=StorageConfig(tmp_path / "data"))
    config = dataclasses.replace(config, species=dataclasses.replace(config.species, bank=_bank(tmp_path / "bank.npz")))
    LocalList(datetime.now(timezone.utc), 38.8, -9.39, 50, [{"id": 13094, "latin": "Phoenicurus ochruros", "common": "Black Redstart", "count": 3}]).save(tmp_path / "data" / "species" / "local.json")
    assert cli.cmd_species_list(config, refresh=False) == 0
    out = capsys.readouterr().out
    assert "1 species within 50 km" in out and "Black Redstart" in out
    called = {}

    def fake_fetch(lat, lon, r, **kw):
        called["r"] = r
        return [{"id": 13094, "latin": "Phoenicurus ochruros", "common": "Black Redstart", "count": 5}] * 25, r
    monkeypatch.setattr("birdy.species_list.fetch_species_counts", fake_fetch)
    assert cli.cmd_species_list(config, refresh=True) == 0
    assert called["r"] == 50


def test_cmd_species_list_refresh_failure_exits_1(capsys, config, monkeypatch, tmp_path):
    from birdy import cli
    from birdy.config import SiteConfig, StorageConfig
    from birdy.species_list import LocalList

    config = dataclasses.replace(config, site=SiteConfig("t", 38.8, -9.39), storage=StorageConfig(tmp_path / "data"))
    config = dataclasses.replace(config, species=dataclasses.replace(config.species, bank=_bank(tmp_path / "bank.npz")))
    LocalList(datetime.now(timezone.utc), 38.8, -9.39, 50, [{"id": 13094, "latin": "Phoenicurus ochruros", "common": "Black Redstart", "count": 3}]).save(tmp_path / "data" / "species" / "local.json")

    def failing_fetch(*a, **kw):
        raise OSError("offline")
    monkeypatch.setattr("birdy.species_list.fetch_species_counts", failing_fetch)
    assert cli.cmd_species_list(config, refresh=True) == 1
    captured = capsys.readouterr()
    assert "Black Redstart" in captured.out
    assert "species-list: refresh failed (or too few species); showing the list fetched 0 days ago" in captured.err


def test_cmd_run_tolerates_a_classifier_without_species_attrs(config_path, monkeypatch):
    from birdy import cli
    from birdy.config import load_config

    cfg = load_config(config_path)
    cfg = dataclasses.replace(cfg, web=dataclasses.replace(cfg.web, port=_free_port()))
    monkeypatch.setattr(cli, "_classifier", lambda c, l: object())
    monkeypatch.setattr("birdy.capture.run_capture", lambda live, store, pipeline, state, stop: None)
    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        assert cli.cmd_run(cfg, config_path) == 0
    finally:
        for s, h in handlers.items():
            signal.signal(s, h)


def test_cmd_species_list_without_location_exits_1(capsys, config):
    from birdy import cli
    assert cli.cmd_species_list(config, refresh=False) == 1
    assert "site.lat" in capsys.readouterr().err


class _FakeDslr:
    def __init__(self, *a, **k): pass

    def info(self, keys=None):
        return {"capturetarget": "Memory card", "imageformat": "Large Fine JPEG", "autopoweroff": "30",
                "focusmode": "Manual", "autoexposuremode": "TV", "iso": "Auto", "shuttercounter": "22764"}

    def capture(self, dest):
        cv2.imwrite(str(dest), np.zeros((800, 1200, 3), np.uint8))
        return 1.25


class _FakeDetector:
    def __init__(self, *a, **k): pass

    def best_bird(self, img):
        return None


def test_dslr_info_prints_values_and_warnings(config_path, monkeypatch, capsys):
    from birdy import cli
    monkeypatch.setattr("birdy.cli.Dslr", _FakeDslr)
    assert cli.main(["--config", str(config_path), "dslr", "info"]) == 0
    out = capsys.readouterr().out
    assert "shuttercounter: 22764" in out
    assert "warning: capturetarget is 'Memory card'" in out and "warning: autopoweroff is '30'" in out
    assert "warning: focusmode" not in out


def test_dslr_shot_saves_the_frame_and_the_region_copy(config_path, tmp_path, monkeypatch, capsys):
    from birdy import cli
    monkeypatch.setattr("birdy.cli.Dslr", _FakeDslr)
    monkeypatch.setattr("birdy.cli.BirdDetector", _FakeDetector)
    out_file = tmp_path / "shot.jpg"
    assert cli.main(["--config", str(config_path), "dslr", "shot", "--out", str(out_file)]) == 0
    assert out_file.exists() and (tmp_path / "shot-region.jpg").exists()
    out = capsys.readouterr().out
    assert "1200x800" in out and "1.25 s" in out and "no bird" in out


def test_dslr_error_is_reported(config_path, monkeypatch, capsys):
    from birdy import cli
    from birdy.dslr import DslrError

    class Broken(_FakeDslr):
        def info(self, keys=None):
            raise DslrError("No camera found.")

    monkeypatch.setattr("birdy.cli.Dslr", Broken)
    assert cli.main(["--config", str(config_path), "dslr", "info"]) == 1
    assert "No camera found." in capsys.readouterr().err
