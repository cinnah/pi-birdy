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

    def pipeline_init(self, store, classifier, live):
        seen["pipeline_classifier"] = classifier
        return real_init(self, store, classifier, live)

    monkeypatch.setattr(pipeline.Pipeline, "__init__", record("pipeline", pipeline_init, 2))

    from birdy.web import app as web_app
    real_create_app = web_app.create_app

    def create_app(*a, **kw):
        seen["web_classifier"] = kw.get("classifier")
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
