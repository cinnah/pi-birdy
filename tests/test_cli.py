import subprocess
import sys
from pathlib import Path

import pytest

from birdy.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


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
    )
    proc = _run_cli("--config", str(cfg), "classify", str(bird))
    assert proc.returncode == 0, proc.stderr
    assert "bird=True" in proc.stdout


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


def test_missing_config_exits_2(tmp_path):
    proc = _run_cli("--config", str(tmp_path / "nope.toml"), "purge")
    assert proc.returncode == 2
    assert "config error" in proc.stderr
    assert "Traceback" not in proc.stderr
