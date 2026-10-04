import dataclasses
import signal
import socket
import subprocess
import sys
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from birdy.cli import build_parser, cmd_run

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


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


def test_parser_subcommands():
    p = build_parser()
    assert p.parse_args(["run"]).command == "run"
    assert p.parse_args(["roi", "--out", "x.png"]).out == "x.png"
    assert p.parse_args(["classify", "a.jpg"]).image == "a.jpg"
    assert p.parse_args(["purge"]).command == "purge"


def test_default_config_path():
    assert build_parser().parse_args(["run"]).config == "birdy.toml"


def test_classify_command_prints_prediction(tmp_path):
    bird = FIXTURES / "bird.jpg"
    if not bird.exists() or not (ROOT / "models" / "mobilenet_v2_1.0_224_quant.tflite").exists():
        pytest.skip("fixtures/model missing")
    cfg = tmp_path / "birdy.toml"
    cfg.write_text(
        '[site]\nname = "t"\n[motion]\nroi = [[0,0],[10,0],[10,10],[0,10]]\n'
        f'[classifier]\nmodel = "{ROOT}/models/mobilenet_v2_1.0_224_quant.tflite"\n'
        f'labels = "{ROOT}/models/imagenet_labels.txt"\nbird_classes = "{ROOT}/models/imagenet_bird_classes.txt"\n'
        f'[species]\nmodel = "{ROOT}/models/mobilenet_v2_1.0_224_inat_bird_quant.tflite"\n'
        f'labels = "{ROOT}/models/inat_bird_labels.txt"\n'
    )
    proc = _run_cli("--config", str(cfg), "classify", str(bird))
    assert proc.returncode == 0, proc.stderr
    assert "bird=True" in proc.stdout
    assert "gate (ImageNet):" in proc.stdout
    assert "species (iNat):" in proc.stdout



def _classify_config(tmp_path, species_model: str) -> Path:
    cfg = tmp_path / "birdy.toml"
    cfg.write_text(
        '[site]\nname = "t"\n[motion]\nroi = [[0,0],[10,0],[10,10],[0,10]]\n'
        f'[classifier]\nmodel = "{ROOT}/models/mobilenet_v2_1.0_224_quant.tflite"\n'
        f'labels = "{ROOT}/models/imagenet_labels.txt"\nbird_classes = "{ROOT}/models/imagenet_bird_classes.txt"\n'
        f'[species]\nmodel = "{species_model}"\n'
        f'labels = "{ROOT}/models/inat_bird_labels.txt"\n'
    )
    return cfg


def test_missing_species_model_exits_2(tmp_path):
    cfg = _classify_config(tmp_path, str(tmp_path / "nope.tflite"))
    proc = _run_cli("--config", str(cfg), "classify", str(FIXTURES / "bird.jpg"))
    assert proc.returncode == 2
    assert "model error" in proc.stderr


def test_unreadable_image_exits_1(tmp_path):
    if not (ROOT / "models" / "mobilenet_v2_1.0_224_inat_bird_quant.tflite").exists():
        pytest.skip("model missing")
    cfg = _classify_config(tmp_path, f"{ROOT}/models/mobilenet_v2_1.0_224_inat_bird_quant.tflite")
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
    if not (ROOT / "models" / "mobilenet_v2_1.0_224_quant.tflite").exists():
        pytest.skip("model missing")
    cfg = tmp_path / "birdy.toml"
    cfg.write_text(
        '[site]\nname = "t"\n[motion]\nroi = [[0,0],[10,0],[10,10],[0,10]]\n[storage]\ndata_dir = "data"\n'
        f'[web]\nport = {_free_port()}\n'
        f'[classifier]\nmodel = "{ROOT}/models/mobilenet_v2_1.0_224_quant.tflite"\n'
        f'labels = "{ROOT}/models/imagenet_labels.txt"\nbird_classes = "{ROOT}/models/imagenet_bird_classes.txt"\n'
        f'[species]\nmodel = "{ROOT}/models/mobilenet_v2_1.0_224_inat_bird_quant.tflite"\n'
        f'labels = "{ROOT}/models/inat_bird_labels.txt"\n'
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
    config = dataclasses.replace(config, web=WebConfig(host="127.0.0.1", port=port))
    seen: dict[str, object] = {}

    def record(name, init, live_index):
        def wrapper(self, *a, **kw):
            seen[name] = a[live_index]
            return init(self, *a, **kw)
        return wrapper

    # Pipeline(store, classifier, live); RetentionJob(store, live); ConfigWatcher(path, live)
    monkeypatch.setattr(retention.RetentionJob, "__init__", record("retention", retention.RetentionJob.__init__, 1))
    monkeypatch.setattr(watcher.ConfigWatcher, "__init__", record("watcher", watcher.ConfigWatcher.__init__, 1))
    clf = object()
    monkeypatch.setattr("birdy.cli._classifier", lambda cfg, live: clf)
    real_init = pipeline.Pipeline.__init__

    def pipeline_init(self, store, classifier, live, notifier=None, publisher=None):
        seen["pipeline_classifier"] = classifier
        seen["pipeline_notifier"] = notifier
        return real_init(self, store, classifier, live, notifier=notifier, publisher=publisher)

    monkeypatch.setattr(pipeline.Pipeline, "__init__", record("pipeline", pipeline_init, 2))

    from birdy.web import app as web_app
    real_create_app = web_app.create_app

    def create_app(*a, **kw):
        seen["web_classifier"] = kw.get("classifier")
        seen["web_notifier"] = kw.get("notifier", "missing")
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
    assert seen["status_http"] == 200
    assert b"motion_score" in seen["status_body"]
    lives = [seen[k] for k in ("pipeline", "retention", "watcher", "capture")]
    assert isinstance(lives[0], LiveConfig)
    assert all(v is lives[0] for v in lives), seen
    assert lives[0].get() is config
    # The pipeline and the upload box share one classifier (and therefore its lock).
    assert seen["pipeline_classifier"] is clf
    assert seen["web_classifier"] is clf
    # Alerts are off in the default config: no notifier anywhere.
    assert seen["pipeline_notifier"] is None
    assert seen["web_notifier"] is None


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
        def __init__(self, live, store, transport):
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
    assert seen["notifier"] is built["self"] and seen["pipeline"]._notifier is built["self"]


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
    assert seen["publisher"] is built["self"] and seen["pipeline"]._publisher is built["self"]
    assert order == ["pipeline.stop", "publisher.stop", "store.close"]


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
    assert seen["publisher"] is None and seen["pipeline"]._publisher is None


def test_publish_command_runs_once(tmp_path, monkeypatch, capsys):
    import birdy.cli as cli

    class FakePublisher:
        instances = []

        def __init__(self, live, store, mirror_dir, **kw):
            self.mirror_dir = mirror_dir
            FakePublisher.instances.append(self)

        def run_once(self):
            return True

    monkeypatch.setattr("birdy.cli.Publisher", FakePublisher)
    cfg = tmp_path / "birdy.toml"
    cfg.write_text('[site]\nname="t"\n[motion]\nroi=[[0,0],[9,0],[9,9]]\n[storage]\ndata_dir="data"\n[publish]\nenabled=true\nhost="h"\n')
    assert cli.main(["--config", str(cfg), "publish"]) == 0
    assert FakePublisher.instances[0].mirror_dir == tmp_path / "data" / "publish"
    assert "published" in capsys.readouterr().out


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
    assert "rsync failed" in capsys.readouterr().err
