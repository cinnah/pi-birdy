import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from birdy.classifier import Prediction
from birdy.config import load_config
from birdy.species import SpeciesLabel
from birdy.state import RuntimeState
from birdy.store import Store
from birdy.web.app import create_app

T0 = datetime(2026, 10, 3, 10, 0, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "data")


@pytest.fixture
def state():
    s = RuntimeState()
    s.update(state="capturing", motion_score=42)
    return s


BLUE_TIT = SpeciesLabel("Cyanistes caeruleus", "Eurasian Blue Tit")


class FakeTwoStage:
    def __init__(self):
        self.images = []

    def classify_image(self, img, threshold=None):
        self.images.append(img.shape)
        pred = Prediction(is_bird=True, confidence=0.93, species="Eurasian Blue Tit", species_confidence=0.81)
        return pred, [(BLUE_TIT, 0.81), (SpeciesLabel("Parus major", "Great Tit"), 0.05)]


@pytest.fixture
def classifier():
    return FakeTwoStage()


@pytest.fixture
def client(store, state, live, config_path, classifier):
    return TestClient(create_app(store, state, live, config_path, classifier=classifier))


def add(store, i, is_bird):
    when = T0 + timedelta(minutes=i)
    abs_path, rel = store.photo_path_for(when)
    cv2.imwrite(str(abs_path), np.full((60, 80, 3), 90, np.uint8))
    cid = store.add_capture(when, rel, 100 + i)
    store.update_classification(cid, is_bird, 0.9 if is_bird else 0.1, "Robin" if is_bird else None, 0.8 if is_bird else None)
    return store.get(cid)


def test_index_renders_captures(client, store):
    bird = add(store, 0, True)
    junk = add(store, 1, False)
    r = client.get("/")
    assert r.status_code == 200
    assert f"/thumbs/{bird.thumb_path}" in r.text
    assert f"/photos/{bird.path}" in r.text
    assert "Robin" in r.text
    assert f"/thumbs/{junk.thumb_path}" in r.text


def test_index_filter_birds(client, store):
    bird = add(store, 0, True)
    junk = add(store, 1, False)
    r = client.get("/?filter=birds")
    assert bird.thumb_path in r.text and junk.thumb_path not in r.text


def test_index_bad_filter_is_400(client):
    assert client.get("/?filter=cats").status_code == 400


def test_index_pagination(client, store):
    for i in range(60):
        add(store, i, True)
    r = client.get("/?page=2")
    assert r.status_code == 200
    assert "page=1" in r.text  # link back


def test_status_json(client, store):
    add(store, 0, True)
    r = client.get("/status")
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "capturing"
    assert body["motion_score"] == 42
    assert body["queue_depth"] == 0
    assert body["pending_count"] == 0
    assert body["disk_free_bytes"] > 0
    assert body["last_capture_at"] is not None
    assert body["uptime_seconds"] >= 0


def test_photos_and_thumbs_are_served(client, store):
    c = add(store, 0, True)
    assert client.get(f"/photos/{c.path}").status_code == 200
    assert client.get(f"/thumbs/{c.thumb_path}").status_code == 200


def test_roi_png_404_without_frame(client):
    assert client.get("/roi.png").status_code == 404


def test_roi_png_renders_frame(client, state):
    state.update(latest_lores=np.full((240, 320), 100, np.uint8))
    r = client.get("/roi.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_page_clamps_to_last_page(client, store):
    for i in range(60):
        add(store, i, True)
    r = client.get("/?page=999")
    assert r.status_code == 200
    assert "page 2 / 2" in r.text
    assert r.text.count('class="tile"') == 10


def test_page_zero_on_empty_store(client):
    r = client.get("/?page=0")
    assert r.status_code == 200
    assert "page 1 / 1" in r.text
    assert "No captures yet." in r.text


def test_status_line_shows_local_time(client, store):
    add(store, 0, True)
    r = client.get("/")
    assert T0.astimezone().strftime("%Y-%m-%d %H:%M:%S") in r.text


def test_pending_tile_renders(client, store):
    abs_path, rel = store.photo_path_for(T0)
    cv2.imwrite(str(abs_path), np.full((60, 80, 3), 90, np.uint8))
    store.add_capture(T0, rel, 100)
    r = client.get("/")
    assert r.status_code == 200
    assert ">pending<" in r.text


def post_settings(client, **overrides):
    form = {
        "roi": "[[40, 60], [280, 60], [280, 220], [40, 220]]",
        "pixel_threshold": "25",
        "min_area": "150",
        "background_alpha": "0.05",
        "cooldown_seconds": "3",
        "classifier_threshold": "0.5",
        "species_threshold": "0.5",
        "species_rescue_threshold": "0.7",
        "lat": "",
        "lon": "",
        "margin_minutes": "30",
        "non_bird_days": "7",
    }
    form.update(overrides)
    return client.post("/settings", data=form, follow_redirects=False)


def test_frame_jpg_404_without_frame(client):
    assert client.get("/frame.jpg").status_code == 404


def test_frame_jpg_returns_jpeg(client, state):
    state.update(latest_lores=np.full((240, 320), 100, np.uint8))
    r = client.get("/frame.jpg")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.headers["cache-control"] == "no-store"
    assert r.content[:2] == b"\xff\xd8"


def test_index_links_to_settings(client):
    assert 'href="/settings"' in client.get("/").text


def test_settings_page_shows_current_values(client, config):
    r = client.get("/settings")
    assert r.status_code == 200
    assert 'name="min_area"' in r.text and 'value="150"' in r.text
    assert json.dumps([list(p) for p in config.motion.roi]) in r.text
    assert "edit birdy.toml" in r.text  # read-only block
    assert str(config.camera.main_size[0]) in r.text


def test_settings_post_saves_applies_and_redirects(client, config_path, live):
    r = post_settings(client, min_area="400", roi="[[10, 10], [100, 10], [100, 100]]", lat="38.72", lon="-9.14")
    assert r.status_code == 303
    assert r.headers["location"] == "/settings?saved=1"
    assert live.version == 1
    cfg = live.get()
    assert cfg.motion.min_area == 400 and cfg.motion.roi == ((10, 10), (100, 10), (100, 100))
    assert cfg.site.lat == 38.72
    assert load_config(config_path) == cfg
    assert "applied live" in client.get("/settings?saved=1").text


@pytest.mark.parametrize(
    "overrides, key, field",
    [
        ({"roi": "[[0, 0], [1, 1]]"}, "motion.roi", "roi"),
        ({"min_area": "-1"}, "motion.min_area", "min_area"),
        ({"lat": "38.7"}, "site.lat", "lat"),
        ({"background_alpha": "abc"}, "motion.background_alpha", "background_alpha"),
    ],
)
def test_settings_post_invalid_renders_error_and_writes_nothing(client, config_path, live, overrides, key, field):
    before = config_path.read_bytes()
    r = post_settings(client, **overrides)
    assert r.status_code == 422
    assert key in r.text
    if field == "roi":
        assert 'id="roi-hint" class="hint error"' in r.text
    else:
        assert re.search(rf'<div class="row error">\s*<label for="{field}"', r.text)
    assert config_path.read_bytes() == before
    assert live.version == 0


def test_settings_post_keeps_submitted_values_on_error(client):
    r = post_settings(client, min_area="-1", pixel_threshold="77")
    assert 'value="77"' in r.text


def test_settings_post_oserror_is_500(client, config_path, live, monkeypatch):
    def boom(*a, **k):
        raise PermissionError("ro")

    monkeypatch.setattr("birdy.web.app.save_settings", boom)
    before = config_path.read_bytes()
    r = post_settings(client, min_area="400")
    assert r.status_code == 500
    assert "could not write" in r.text
    assert live.version == 0
    assert config_path.read_bytes() == before


def test_settings_post_missing_field_422(client):
    form = {
        "roi": "[[40, 60], [280, 60], [280, 220], [40, 220]]",
        "pixel_threshold": "25",
        "background_alpha": "0.05",
        "cooldown_seconds": "3",
        "classifier_threshold": "0.5",
        "lat": "",
        "lon": "",
        "margin_minutes": "30",
        "non_bird_days": "7",
    }
    r = client.post("/settings", data=form, follow_redirects=False)
    assert r.status_code == 422
    assert re.search(r'<div class="row error">\s*<label for="min_area"', r.text)


def test_settings_post_missing_roi_422(client, live):
    form = {
        "pixel_threshold": "25",
        "min_area": "150",
        "background_alpha": "0.05",
        "cooldown_seconds": "3",
        "classifier_threshold": "0.5",
        "lat": "",
        "lon": "",
        "margin_minutes": "30",
        "non_bird_days": "7",
    }
    r = client.post("/settings", data=form, follow_redirects=False)
    assert r.status_code == 422
    assert "motion.roi" in r.text
    assert live.version == 0


def test_settings_post_ignores_unknown_fields(client):
    assert post_settings(client, foo="bar").status_code == 303


def test_settings_page_has_novalidate_and_saved_roi(client):
    text = client.get("/settings").text
    assert "novalidate" in text
    assert "data-saved='[[40, 60], [280, 60], [280, 220], [40, 220]]'" in text


def test_settings_page_has_species_rows(client):
    r = client.get("/settings").text
    assert 'name="species_threshold"' in r and 'name="species_rescue_threshold"' in r


def test_index_species_dropdown_and_filter(client, store):
    robin = add(store, 0, True)
    junk = add(store, 1, False)
    r = client.get("/")
    assert '<select name="species"' in r.text and "Robin (1)" in r.text
    r2 = client.get("/?species=Robin")
    assert robin.thumb_path in r2.text and junk.thumb_path not in r2.text
    assert client.get("/?species=Dodo").status_code == 400


def test_index_links_to_species_page(client):
    assert 'href="/species"' in client.get("/").text


def test_species_page_empty(client):
    r = client.get("/species")
    assert r.status_code == 200 and "No species yet" in r.text


def test_species_page_table_and_best_photo(client, store):
    a = add(store, 0, True)
    b = add(store, 1, True)
    r = client.get("/species").text
    assert "Robin" in r and ">2<" in r
    assert f"/photos/{b.path}" in r or f"/photos/{a.path}" in r  # best photo link present


def test_classify_upload_renders_result(client, classifier, store):
    data = (Path(__file__).parent / "fixtures" / "bird.jpg").read_bytes()
    before = store.count("all")
    r = client.post("/classify", files={"image": ("phone.jpg", data, "image/jpeg")})
    assert r.status_code == 200
    assert "Eurasian Blue Tit" in r.text and "81%" in r.text and "bird" in r.text
    assert classifier.images and classifier.images[0][2] == 3
    assert store.count("all") == before
    assert not any(store.photos_dir.rglob("*.jpg"))


def test_classify_upload_garbage_422(client):
    r = client.post("/classify", files={"image": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 422 and "could not decode" in r.text


def test_classify_upload_missing_422(client):
    assert client.post("/classify", data={}).status_code == 422


def test_classify_upload_too_large_413(client):
    big = b"\xff\xd8" + b"\0" * (10 * 1024 * 1024 + 1)
    assert client.post("/classify", files={"image": ("big.jpg", big, "image/jpeg")}).status_code == 413


def test_classify_without_classifier_503(store, state, live, config_path):
    c = TestClient(create_app(store, state, live, config_path))
    data = (Path(__file__).parent / "fixtures" / "bird.jpg").read_bytes()
    assert c.post("/classify", files={"image": ("p.jpg", data, "image/jpeg")}).status_code == 503


def test_status_last_species(client, store):
    assert client.get("/status").json()["last_species"] is None
    add(store, 0, True)
    assert client.get("/status").json()["last_species"] == "Robin"
