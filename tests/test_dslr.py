import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

from birdy.dslr import Dslr, DslrError, box_in_frame, dslr_rel, region_crop, species_crop
from birdy.store import Store

NOW = datetime(2026, 11, 24, 12, 0).astimezone()


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "data")


class Runner:
    """Stands in for subprocess.run: records calls and plays a scripted gphoto2."""

    def __init__(self, returncode=0, stdout="", stderr="", write=b"jpegbytes", raises=None):
        self.returncode, self.stdout, self.stderr, self.write, self.raises = returncode, stdout, stderr, write, raises
        self.calls: list[list[str]] = []

    def __call__(self, cmd, capture_output=True, text=True, timeout=None):
        self.calls.append(cmd)
        if self.raises is not None:
            raise self.raises
        if "--filename" in cmd and self.write is not None:
            Path(cmd[cmd.index("--filename") + 1]).write_bytes(self.write)
        return subprocess.CompletedProcess(cmd, self.returncode, self.stdout, self.stderr)


def make(store, live, runner):
    return Dslr(store, live, runner=runner, now=lambda: NOW)


def tmp_files(d: Path) -> list[Path]:
    return [p for p in d.iterdir() if p.name.endswith(".tmp")]


def test_capture_writes_the_file_atomically(store, live, tmp_path):
    r = Runner()
    dest = tmp_path / "a_eos.jpg"
    took = make(store, live, r).capture(dest)
    assert dest.read_bytes() == b"jpegbytes" and took >= 0 and not tmp_files(tmp_path)
    assert r.calls[0][:2] == ["gphoto2", "--capture-image-and-download"] and "--force-overwrite" in r.calls[0]


@pytest.mark.parametrize("runner, state", [
    (Runner(returncode=1, write=None, stderr="*** Error: No camera found. ***\n"), "not_connected"),
    (Runner(returncode=1, stderr="*** Error ***\nCould not claim the USB device\n"), "error"),
    (Runner(returncode=0, write=None), "error"),                       # exit 0 but no file
    (Runner(returncode=0, write=b""), "error"),                        # empty file
    (Runner(raises=subprocess.TimeoutExpired("gphoto2", 15)), "error"),
    (Runner(raises=FileNotFoundError("gphoto2")), "error"),
])
def test_capture_failures_raise_and_leave_nothing(store, live, tmp_path, runner, state):
    d = make(store, live, runner)
    dest = tmp_path / "a_eos.jpg"
    with pytest.raises(DslrError):
        d.capture(dest)
    assert not dest.exists() and not tmp_files(tmp_path)
    assert d.status().state == state


def test_daily_cap_blocks_before_gphoto2_runs(store, live, tmp_path):
    for i in range(live.get().dslr.daily_cap):
        when = NOW - timedelta(minutes=1 + i)
        abs_path, rel = store.photo_path_for(when)
        cv2.imwrite(str(abs_path), np.zeros((6, 8, 3), np.uint8))
        cid = store.add_capture(when, rel, 1)
        store.store_verdict(cid, is_bird=True, confidence=0.9, species=None, species_confidence=None,
                            dslr_path=str(rel), dslr_box=None)
    r = Runner()
    d = make(store, live, r)
    with pytest.raises(DslrError, match="daily cap"):
        d.capture(tmp_path / "a_eos.jpg")
    assert r.calls == [] and d.status().state == "cap_reached"
    assert d.status().shots_today == live.get().dslr.daily_cap


def test_one_warning_per_distinct_error_and_recovery_is_logged(store, live, tmp_path, caplog):
    bad = Runner(returncode=1, write=None, stderr="*** Error: No camera found. ***\n")
    d = make(store, live, bad)
    with caplog.at_level(logging.INFO, logger="birdy.dslr"):
        for _ in range(3):
            with pytest.raises(DslrError):
                d.capture(tmp_path / "a_eos.jpg")
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1
        d._runner = Runner()
        d.capture(tmp_path / "a_eos.jpg")
        assert any("ready" in r.getMessage() for r in caplog.records if r.levelno == logging.INFO)
    assert d.status().state == "ready"


INFO_OUT = """Label: Capture Target
Readonly: 0
Type: RADIO
Current: Internal RAM
Choice: 0 Internal RAM
END
Label: Auto Power Off
Readonly: 0
Type: TEXT
Current: 0
END
"""


def test_info_parses_current_values(store, live):
    r = Runner(stdout=INFO_OUT)
    d = make(store, live, r)
    assert d.info(("capturetarget", "autopoweroff")) == {"capturetarget": "Internal RAM", "autopoweroff": "0"}
    assert r.calls[0] == ["gphoto2", "--get-config", "capturetarget", "--get-config", "autopoweroff"]
    assert d.status().state == "ready"


def test_info_failure(store, live):
    d = make(store, live, Runner(returncode=1, stderr="*** Error: No camera found. ***\n"))
    with pytest.raises(DslrError):
        d.info()
    assert d.status().state == "not_connected"


def test_dslr_rel():
    assert dslr_rel(Path("2026/11/24/101502_123.jpg")) == Path("2026/11/24/101502_123_eos.jpg")


def test_region_crop_and_box_mapping():
    frame = np.zeros((200, 400, 3), np.uint8)
    region = (0.25, 0.5, 0.5, 0.5)
    assert region_crop(frame, region).shape[:2] == (100, 200)
    # A box in the middle of the region, as fractions of the region -> fractions of the frame.
    assert box_in_frame((0.5, 0.5, 0.2, 0.2), region) == pytest.approx((0.5, 0.75, 0.1, 0.1))


def test_species_crop_is_a_square_twice_the_box():
    frame = np.zeros((2000, 3000, 3), np.uint8)
    crop = species_crop(frame, (0.5, 0.5, 0.1, 0.05))        # box 300 x 100 px -> side 600
    assert crop.shape[:2] == (600, 600)


def test_species_crop_has_a_minimum_and_stays_in_the_frame():
    frame = np.zeros((2000, 3000, 3), np.uint8)
    assert species_crop(frame, (0.0, 0.0, 0.01, 0.01)).shape[:2] == (224, 224)   # tiny box in the corner
    assert species_crop(frame, (0.1, 0.1, 0.8, 0.8)).shape[:2] == (2000, 2000)   # capped by the frame height


def test_a_failed_save_raises_dslr_error_and_leaves_nothing(store, live, tmp_path, monkeypatch):
    def boom(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("birdy.dslr.os.replace", boom)
    d = make(store, live, Runner())
    dest = tmp_path / "a_eos.jpg"
    with pytest.raises(DslrError, match="could not save the frame"):
        d.capture(dest)
    assert not dest.exists() and not tmp_files(tmp_path) and d.status().state == "error"
