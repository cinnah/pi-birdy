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
from birdy.notify import TelegramError
from birdy.species import SpeciesLabel, load_bank
from birdy.state import RuntimeState
from birdy.store import Store
from birdy.visits import Visits
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


class FakeBank:
    reported_names = frozenset({"black redstart", "eurasian hoopoe"})   # common names; latin is not what a capture says

    def resolve(self, name):
        return {"black redstart": 1, "phoenicurus ochruros": 1, "eurasian hoopoe": 2}.get(name.strip().casefold())


class FakeSpeciesModel:
    bank = FakeBank()
    candidate_count = 354


class FakeSpeciesList:
    version = 1

    def current(self):
        from birdy.species_list import LocalList
        return LocalList(datetime.now(timezone.utc), 38.8, -9.39, 50, [{"id": 1, "latin": "Phoenicurus ochruros", "common": "Black Redstart", "count": 3}])

    def unknown_names(self, names, reported_only=False):
        if reported_only:
            return [n for n in names if n.strip().casefold() not in FakeBank.reported_names]
        return [n for n in names if FakeBank().resolve(n) is None]


class FakeTwoStage:
    def __init__(self):
        self.images = []
        self.species = FakeSpeciesModel()
        self.species_list = FakeSpeciesList()

    def classify_image(self, img, threshold=None):
        self.images.append(img.shape)
        pred = Prediction(
            is_bird=True, confidence=0.93, species="Eurasian Blue Tit", species_confidence=0.81,
            detector_confidence=0.42, detector_box=(0.1, 0.1, 0.5, 0.5),
        )
        return pred, [(BLUE_TIT, 0.81), (SpeciesLabel("Parus major", "Great Tit"), 0.05)]


@pytest.fixture
def classifier():
    return FakeTwoStage()


class FakeNotifier:
    def __init__(self, raise_exc=None):
        self.raise_exc = raise_exc
        self.tests = 0

    def send_test(self):
        self.tests += 1
        if self.raise_exc:
            raise self.raise_exc

    def consider(self, *a, **k):
        return None


@pytest.fixture
def notifier():
    return FakeNotifier()


def _bank(tmp_path):
    # unit vectors: row 0 = Black Redstart, row 1 = European Robin, row 2 = Eurasian Hoopoe
    p = tmp_path / "bank.npz"
    np.savez(p, ids=np.array([1, 2, 3], np.int32), latin=np.array(["Phoenicurus ochruros", "Erithacus rubecula", "Upupa epops"]),
             common=np.array(["Black Redstart", "European Robin", "Eurasian Hoopoe"]), emb=np.eye(3, 4, dtype=np.float16),
             logit_scale=np.float32(100.0), model=np.str_("x"), prompt=np.str_("p"))
    return load_bank(p)


class FakeVisitsList:
    """The species list `Visits` holds: whole bank as candidates, a 2-species local list."""
    version = 1

    def candidates(self, cfg):
        return None

    def current(self):
        from birdy.species_list import LocalList
        return LocalList(datetime.now(timezone.utc), 38.8, -9.39, 50, [
            {"id": 2, "latin": "Erithacus rubecula", "common": "European Robin", "count": 4},
            {"id": 1, "latin": "Phoenicurus ochruros", "common": "Black Redstart", "count": 9},
        ])


@pytest.fixture
def visits(store, live, tmp_path, publisher):
    return Visits(store, live, bank=_bank(tmp_path), species_list=FakeVisitsList(), notifier=None, publisher=publisher)


@pytest.fixture
def client(store, state, live, config_path, classifier, notifier, visits):
    return TestClient(create_app(store, state, live, config_path, classifier=classifier, notifier=notifier, visits=visits))


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
    r = client.get("/?filter=all")
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
    assert T0.astimezone().strftime("%d/%m/%Y %H:%M:%S") in r.text


def test_pending_tile_renders(client, store):
    abs_path, rel = store.photo_path_for(T0)
    cv2.imwrite(str(abs_path), np.full((60, 80, 3), 90, np.uint8))
    store.add_capture(T0, rel, 100)
    r = client.get("/")
    assert r.status_code == 200
    assert ">pending<" in r.text


def _base_form():
    return {
        "roi": "[[40, 60], [280, 60], [280, 220], [40, 220]]",
        "pixel_threshold": "25",
        "min_area": "150",
        "background_alpha": "0.05",
        "cooldown_seconds": "3",
        "classifier_threshold": "0.5",
        "species_threshold": "0.5",
        "species_include": "",
        "species_exclude": "",
        "detector_threshold": "0.2",
        "visits_gap_seconds": "60",
        "lat": "",
        "lon": "",
        "margin_minutes": "30",
        "non_bird_days": "7",
        "watchlist": "",
        "alert_new_species": "on",
        "cooldown_minutes": "30",
        "publish_min_confidence": "0.6",
        "per_species_per_day": "3",
        "dslr_region": "0, 0, 1, 1",
        "dslr_daily_cap": "200",
    }


def post_settings(client, **overrides):
    form = _base_form()
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
    assert 'name="species_threshold"' in r and 'name="species_include"' in r and 'species_rescue_threshold' not in r


def test_index_species_dropdown_and_filter(client, store):
    robin = add(store, 0, True)
    junk = add(store, 1, False)
    r = client.get("/")
    assert '<select name="species"' in r.text and "Robin (1)" in r.text
    r2 = client.get("/?species=Robin")
    assert robin.thumb_path in r2.text and junk.thumb_path not in r2.text
    r3 = client.get("/?species=Dodo")
    assert r3.status_code == 200 and "No Dodo photos" in r3.text
    assert '<option value="Dodo" selected>Dodo (0)</option>' in r3.text


def test_alert_link_after_not_a_bird_renders_empty_grid(client, store):
    # A Telegram alert links to /?species=Robin; the photo is corrected afterwards.
    robin = add(store, 0, True)
    client.post(f"/captures/{robin.id}/label", data={"label": "not_bird"})
    r = client.get("/?species=Robin")
    assert r.status_code == 200 and "No Robin photos" in r.text
    assert robin.thumb_path not in r.text and "marked <i>Not a bird</i> or deleted" in r.text


def test_species_with_birds_empty_in_other_view(client, store):
    add(store, 0, True)
    r = client.get("/?filter=non-birds&species=Robin")
    assert "No Robin photos" in r.text and "None in this view" in r.text
    assert "or deleted" not in r.text and "Robin (0)" not in r.text


def test_index_links_to_species_page(client):
    assert 'href="/species"' in client.get("/").text


def test_species_page_empty(client):
    r = client.get("/species")
    assert r.status_code == 200 and "No species yet" in r.text


def test_species_page_table_and_best_photo(client, store):
    a = add(store, 0, True)
    b = add(store, 1, True)
    store.update_classification(a.id, True, 0.9, "Robin", 0.95)  # a is the more confident Robin
    r = client.get("/species").text
    assert "Robin" in r and ">2<" in r
    assert f"/photos/{a.path}" in r and f"/photos/{b.path}" not in r


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


def test_classify_upload_empty_422(client):
    r = client.post("/classify", files={"image": ("a.jpg", b"", "image/jpeg")})
    assert r.status_code == 422 and "choose a photo first" in r.text


def test_index_empty_species_param_shows_all(client, store):
    bird = add(store, 0, True)
    junk = add(store, 1, False)
    r = client.get("/?filter=all&species=")
    assert r.status_code == 200
    assert bird.thumb_path in r.text and junk.thumb_path in r.text


def test_index_defaults_to_birds(client, store):
    bird = add(store, 0, True)
    junk = add(store, 1, False)
    r = client.get("/")
    assert bird.thumb_path in r.text and junk.thumb_path not in r.text
    assert 'class="active">birds<' in r.text


class FakeNotABird:
    def classify_image(self, img, threshold=None):
        return Prediction(is_bird=False, confidence=0.12, species=None, species_confidence=None), []


def test_classify_upload_non_bird(store, state, live, config_path):
    c = TestClient(create_app(store, state, live, config_path, classifier=FakeNotABird()))
    data = (Path(__file__).parent / "fixtures" / "cat.jpg").read_bytes()
    r = c.post("/classify", files={"image": ("cat.jpg", data, "image/jpeg")})
    assert r.status_code == 200
    assert "not a bird (12% bird)" in r.text
    assert "species guesses" not in r.text


def test_species_labelled_non_bird_is_hidden(client, store):
    when = T0
    abs_path, rel = store.photo_path_for(when)
    cv2.imwrite(str(abs_path), np.full((60, 80, 3), 90, np.uint8))
    cid = store.add_capture(when, rel, 100)
    store.update_classification(cid, False, 0.3, "Sparrow", 0.9)  # species label on a non-bird row
    index = client.get("/").text
    assert 'value="Sparrow"' not in index and "Sparrow (1)" not in index  # tile shows the label, dropdown doesn't
    assert "Sparrow" not in client.get("/species").text
    assert "No Sparrow photos" in client.get("/?species=Sparrow").text  # only bird rows count
    assert client.get("/status").json()["last_species"] is None


def test_status_last_species_returns_newest(client, store):
    add(store, 0, True)  # Robin at T0
    when = T0 + timedelta(minutes=5)
    abs_path, rel = store.photo_path_for(when)
    cv2.imwrite(str(abs_path), np.full((60, 80, 3), 90, np.uint8))
    cid = store.add_capture(when, rel, 100)
    store.update_classification(cid, True, 0.9, "Blue Tit", 0.7)
    assert client.get("/status").json()["last_species"] == "Blue Tit"


def test_species_filter_excludes_non_bird_rows(client, store):
    junk = add(store, 0, False)
    store.update_classification(junk.id, False, 0.1, "Robin", 0.8)  # non-bird row labelled Robin
    bird = add(store, 1, True)
    r = client.get("/?species=Robin").text
    assert bird.thumb_path in r and junk.thumb_path not in r
    assert "(1)" in r


def test_settings_alerts_group_renders(client, store, live, config):
    import dataclasses

    add(store, 0, True)  # a "Robin" bird → species hint
    live.replace(dataclasses.replace(config, telegram=dataclasses.replace(config.telegram, watchlist=("Hoopoe",), bot_token="SECRET123", chat_id="42")))
    r = client.get("/settings").text
    assert '<textarea id="watchlist" name="watchlist"' in r and ">Hoopoe</textarea>" in r
    assert 'name="alert_new_species" value="on" checked' in r
    assert 'name="cooldown_minutes"' in r
    assert "Robin" in r  # species hint
    assert "telegram.bot_token" in r and "SECRET123" not in r
    assert "Send test alert" in r


def test_settings_post_saves_alert_fields(client, live, config_path):
    r = post_settings(client, watchlist="Eurasian Hoopoe\nBlack Redstart", cooldown_minutes="10")
    assert r.status_code == 303
    t = live.get().telegram
    assert t.watchlist == ("Eurasian Hoopoe", "Black Redstart") and t.cooldown_minutes == 10 and t.alert_new_species is True
    data = _base_form()
    data.pop("alert_new_species")  # an unchecked checkbox is simply absent from the POST
    r2 = client.post("/settings", data=data, follow_redirects=False)
    assert r2.status_code == 303 and live.get().telegram.alert_new_species is False


def test_settings_post_invalid_cooldown_highlights_row(client):
    r = post_settings(client, cooldown_minutes="-5")
    assert r.status_code == 422 and "telegram.cooldown_minutes" in r.text
    assert re.search(r'<div class="row error">\s*<label for="cooldown_minutes"', r.text)


def test_settings_failed_post_never_leaks_token(client, live, config):
    import dataclasses

    live.replace(dataclasses.replace(config, telegram=dataclasses.replace(config.telegram, bot_token="SECRET123", chat_id="42")))
    r = post_settings(client, cooldown_minutes="-5")
    assert r.status_code == 422 and "telegram.bot_token" in r.text and "SECRET123" not in r.text


def test_settings_button_disabled_without_notifier(store, state, live, config, config_path, classifier):
    import dataclasses

    tg = dataclasses.replace(config.telegram, enabled=True, bot_token="SECRET123", chat_id="42")
    live.replace(dataclasses.replace(config, telegram=tg))
    c = TestClient(create_app(store, state, live, config_path, classifier=classifier))
    r = c.get("/settings").text
    assert re.search(r'<button class="btn" type="submit" form="test-alert"\s+disabled>Send test alert</button>', r)
    assert "Alerts are off. Set [telegram] in birdy.toml and restart." in r
    assert "SECRET123" not in r


def test_test_alert_success(client, notifier):
    r = client.post("/settings/test-alert")
    assert r.status_code == 200 and "test alert sent" in r.text and notifier.tests == 1
    # result banner sits at the top, before the main form
    assert r.text.index("test alert sent") < r.text.index('<form method="post" action="/settings"')


def test_test_alert_telegram_error(store, state, live, config_path, classifier):
    bad = FakeNotifier(raise_exc=TelegramError("Unauthorized", 401))
    c = TestClient(create_app(store, state, live, config_path, classifier=classifier, notifier=bad))
    r = c.post("/settings/test-alert")
    assert r.status_code == 502 and "Unauthorized" in r.text


def test_test_alert_disabled_503(store, state, live, config_path, classifier):
    c = TestClient(create_app(store, state, live, config_path, classifier=classifier))
    r = c.post("/settings/test-alert")
    assert r.status_code == 503 and "disabled" in r.text
    assert "disabled" in c.get("/settings").text  # button state explained


def test_settings_page_shows_publishing_group(client):
    html = client.get("/settings").text
    assert "Publishing" in html
    assert 'name="publish_min_confidence"' in html and 'name="per_species_per_day"' in html
    assert "publish.site_id" in html and "publish.enabled" in html


def test_settings_post_saves_publish_fields(client, live):
    r = post_settings(client, publish_min_confidence="0.9", per_species_per_day="1")
    assert r.status_code == 303
    assert live.get().publish.min_confidence == 0.9 and live.get().publish.per_species_per_day == 1


class FakePublisher:
    def __init__(self):
        self.calls = []
        self.forgotten = []
        self.evaluated = []
        self.days = []
        self.notified = 0

    def forget(self, capture):
        self.forgotten.append(capture.id)

    def evaluate(self, capture):
        self.evaluated.append(capture.id)

    def evaluate_day(self, species, when):
        self.days.append((species, when))

    def notify(self):
        self.notified += 1

    def set_manual(self, capture_id, state):
        self.calls.append((capture_id, state))
        if capture_id == 999:
            raise KeyError(capture_id)


@pytest.fixture
def publisher():
    return FakePublisher()


@pytest.fixture
def pclient(store, state, live, config_path, classifier, notifier, publisher):
    return TestClient(create_app(store, state, live, config_path, classifier=classifier, notifier=notifier, publisher=publisher))


def test_index_without_publisher_has_no_buttons(client, store):
    add(store, 0, True)
    html = client.get("/").text
    assert "/publish" not in html and "filter=published" not in html


def test_index_renders_badges_and_buttons(pclient, store):
    a = add(store, 0, True)
    b = add(store, 1, True)
    c = add(store, 2, True)
    store.set_published(a.id, "auto")
    store.set_published(b.id, "forced")
    store.set_published(c.id, "hidden")
    html = pclient.get("/").text
    assert html.count('class="badge') == 3
    assert ">published<" in html and ">forced<" in html and ">hidden<" in html
    assert f'action="/captures/{a.id}/publish"' in html
    assert "filter=published" in html


def test_publish_post_sets_state_and_redirects_back(pclient, store, publisher):
    a = add(store, 0, True)
    r = pclient.post(f"/captures/{a.id}/publish", data={"state": "hidden", "next": "/?filter=birds&page=2"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/?filter=birds&page=2"
    assert publisher.calls == [(a.id, "hidden")]
    r = pclient.post(f"/captures/{a.id}/publish", data={"state": "auto"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    assert publisher.calls[-1] == (a.id, None)


def test_publish_post_rejects_bad_input(pclient, store, publisher):
    a = add(store, 0, True)
    assert pclient.post(f"/captures/{a.id}/publish", data={"state": "maybe"}).status_code == 400
    assert pclient.post("/captures/999/publish", data={"state": "forced"}).status_code == 404
    for bad in ("//evil.example/x", "/\\evil.example", "http://evil", "/settings"):
        r = pclient.post(f"/captures/{a.id}/publish", data={"state": "forced", "next": bad}, follow_redirects=False)
        assert r.headers["location"] == "/", bad


def test_publish_post_503_without_publisher(client, store):
    a = add(store, 0, True)
    assert client.post(f"/captures/{a.id}/publish", data={"state": "forced"}).status_code == 503


def test_index_filter_published(pclient, store):
    a = add(store, 0, True)
    add(store, 1, True)
    store.set_published(a.id, "auto")
    html = pclient.get("/?filter=published").text
    assert html.count('class="tile"') == 1


def test_index_buttons_per_state(pclient, store):
    caps = [add(store, i, True) for i in range(4)]
    states = [None, "auto", "forced", "hidden"]
    for c, st in zip(caps, states):
        if st:
            store.set_published(c.id, st)
    html = pclient.get("/").text
    forms = {}
    for chunk in html.split("<form")[1:]:
        chunk = chunk.split("</form>")[0]
        m = re.search(r'action="/captures/(\d+)/publish"', chunk)
        if m is None:
            continue  # the header's camera switch
        cid = int(m.group(1))
        forms[cid] = set(re.findall(r'<button name="state" value="([a-z]+)"', chunk))
    expected = {None: {"forced", "hidden"}, "auto": {"forced", "hidden"}, "forced": {"hidden", "auto"}, "hidden": {"forced", "auto"}}
    assert forms == {c.id: expected[st] for c, st in zip(caps, states)}


def test_camera_switch_sets_override_until_next_daylight_change(client, state, live):
    r = client.post("/camera", data={"mode": "off", "next": "/settings"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/settings"
    assert state.override == "off"
    # no lat/lon in the test config: no daylight schedule, so it holds until switched back
    assert state.override_until is None
    assert client.get("/status").json()["override"] == "off"
    page = client.get("/settings").text
    assert "Back to schedule" in page
    r = client.post("/camera", data={"mode": "auto", "next": "//evil"}, follow_redirects=False)
    assert r.headers["location"] == "/"
    assert state.override is None
    assert client.post("/camera", data={"mode": "maybe"}).status_code == 400


def test_camera_switch_with_site_holds_until_next_change(client, state, live):
    import dataclasses

    cfg = live.get()
    live.replace(dataclasses.replace(cfg, site=dataclasses.replace(cfg.site, lat=38.72, lon=-9.14)))
    client.post("/camera", data={"mode": "on"}, follow_redirects=False)
    assert state.override == "on"
    assert state.override_until is not None and state.override_until > datetime.now().astimezone()
    assert "Turn" in client.get("/").text


def test_delete_capture_removes_it_and_redirects_back(pclient, store, publisher):
    a = add(store, 0, True)
    photo = store.photos_dir / a.path
    r = pclient.post(f"/captures/{a.id}/delete", data={"next": "/?filter=birds&page=2"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/?filter=birds&page=2"
    assert store.get(a.id) is None and not photo.exists()
    assert publisher.forgotten == [a.id]


def test_delete_capture_without_publisher_and_bad_input(client, store):
    a = add(store, 0, False)
    r = client.post(f"/captures/{a.id}/delete", data={"next": "//evil"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    assert store.get(a.id) is None
    assert client.post("/captures/999/delete").status_code == 404


def test_delete_non_birds_bulk_only_deletes_non_birds(pclient, store, publisher):
    bird = add(store, 0, True)
    junk1, junk2 = add(store, 1, False), add(store, 2, False)
    r = pclient.post("/captures/delete-non-birds", data={"ids": [str(bird.id), str(junk1.id), str(junk2.id), "x"], "next": "/?filter=non-birds"},
                     follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/?filter=non-birds"
    assert store.get(bird.id) is not None
    assert store.get(junk1.id) is None and store.get(junk2.id) is None


def test_index_has_delete_buttons_and_bulk_only_on_non_birds(client, store):
    a = add(store, 0, True)
    j = add(store, 1, False)
    html = client.get("/?filter=all").text
    assert f'action="/captures/{a.id}/delete"' in html and f'action="/captures/{j.id}/delete"' in html
    assert "/captures/delete-non-birds" not in html
    html = client.get("/?filter=non-birds").text
    assert "/captures/delete-non-birds" in html and f'name="ids" value="{j.id}"' in html


def _add_photo(store, box=None):
    abs_path, rel = store.photo_path_for(T0)
    cv2.imwrite(str(abs_path), np.full((480, 640, 3), 90, np.uint8))   # 2x the 320x240 lores frame
    return store.add_capture(T0, rel, 100, motion_box=box)


def test_crop_jpg_is_the_motion_crop(client, store):
    cid = _add_photo(store, (100 / 320, 100 / 240, 20 / 320, 20 / 240))
    r = client.get(f"/captures/{cid}/crop.jpg")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (80, 80)   # 40 lores px square, x2


def test_crop_jpg_falls_back_to_roi(client, store, config):
    from birdy.motion import roi_bbox

    cid = _add_photo(store)
    img = cv2.imdecode(np.frombuffer(client.get(f"/captures/{cid}/crop.jpg").content, np.uint8), cv2.IMREAD_COLOR)
    rb = roi_bbox(config.motion.roi)   # 240 x 160 lores, padded 10 % each side, x2
    assert img.shape[1] == round((rb.w * 1.2) * 2) and img.shape[0] == round((rb.h * 1.2) * 2)


def test_crop_jpg_draws_the_detector_box(client, store):
    cid = _add_photo(store, (100 / 320, 100 / 240, 20 / 320, 20 / 240))   # 80x80 crop
    plain = cv2.imdecode(np.frombuffer(client.get(f"/captures/{cid}/crop.jpg").content, np.uint8), cv2.IMREAD_COLOR)
    store.update_classification(cid, True, 0.6, None, None, det_confidence=0.6, det_box=(0.25, 0.25, 0.5, 0.5))
    boxed = cv2.imdecode(np.frombuffer(client.get(f"/captures/{cid}/crop.jpg").content, np.uint8), cv2.IMREAD_COLOR)
    assert boxed.shape == plain.shape
    b, g, r = boxed[20, 40]   # top edge of the box, BGR; #ffb000 = (0, 176, 255) ± JPEG
    assert r > 200 and g > 150 and b < 80
    assert not np.array_equal(plain[20, 40], boxed[20, 40])


def test_classify_upload_shows_detector(client, classifier, store):
    data = (Path(__file__).parent / "fixtures" / "bird.jpg").read_bytes()
    r = client.post("/classify", files={"image": ("b.jpg", data, "image/jpeg")})
    assert r.status_code == 200 and "detector 42%" in r.text


def test_crop_jpg_unknown_capture(client):
    assert client.get("/captures/999/crop.jpg").status_code == 404


def test_grid_tile_links_model_view(client, store):
    cid = _add_photo(store)
    store.update_classification(cid, True, 0.9, None, None)
    assert f'href="/captures/{cid}/crop.jpg"' in client.get("/?filter=all").text


def test_crop_jpg_missing_photo(client, store):
    cid = _add_photo(store)
    (store.photos_dir / store.get(cid).path).unlink()
    assert client.get(f"/captures/{cid}/crop.jpg").status_code == 404


def test_not_a_bird_button_on_bird_tiles_only(client, store):
    bird = add(store, 0, True)
    junk = add(store, 1, False)
    html = client.get("/?filter=all").text
    assert ">Not a bird<" in _tile(html, bird.id)
    junk_tile = _tile(html, junk.id)
    assert f'action="/captures/{junk.id}/label"' in junk_tile and ">Not a bird<" not in junk_tile


def test_mark_not_bird_and_back(pclient, store, publisher):
    a = add(store, 0, True)
    store.set_published(a.id, "auto")
    r = pclient.post(f"/captures/{a.id}/label", data={"label": "not_bird", "next": "/?filter=birds"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/?filter=birds"
    c = store.get(a.id)
    assert c.is_bird is False and c.species is None and c.published is None
    assert publisher.forgotten == [a.id]            # its species/day is re-ranked, the gallery syncs
    html = pclient.get("/?filter=non-birds").text
    assert 'class="tag you">you<' in html and ">no bird<" in html and "Back to model" in html and "model: Robin" in html
    r = pclient.post(f"/captures/{a.id}/label", data={"label": "clear"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    c = store.get(a.id)
    assert c.is_bird is True and c.species == "Robin"
    assert publisher.evaluated == [a.id]


def test_label_rejects_bad_input(client, store):
    a = add(store, 0, True)
    assert client.post(f"/captures/{a.id}/label", data={"label": "robin"}).status_code == 400
    assert client.post("/captures/999/label", data={"label": "not_bird"}).status_code == 404
    abs_path, rel = store.photo_path_for(T0)
    pending = store.add_capture(T0, rel, 100)
    assert client.post(f"/captures/{pending}/label", data={"label": "not_bird"}).status_code == 409


def test_take_photo_button_asks_the_capture_loop(client, state):
    page = client.get("/").text
    assert 'formaction="/camera/capture"' in page
    r = client.post("/camera/capture", data={"next": "/?filter=all"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/?filter=all"
    assert state.take_capture_request() is True
    assert state.take_capture_request() is False
    r = client.post("/camera/capture", data={"next": "//evil"}, follow_redirects=False)
    assert r.headers["location"] == "/"


def test_take_photo_refused_while_camera_is_off(client, state):
    state.update(state="daylight_paused")
    assert 'formaction="/camera/capture"' not in client.get("/").text
    r = client.post("/camera/capture", data={"next": "/"})
    assert r.status_code == 409
    assert state.take_capture_request() is False


def test_settings_page_has_detector_threshold(client):
    r = client.get("/settings").text
    assert 'name="detector_threshold"' in r and 'value="0.2"' in r


def test_settings_rejects_unknown_species_names(client):
    r = client.post("/settings", data={**_base_form(), "species_include": "Dodo"}, follow_redirects=False)
    assert r.status_code == 422 and "species.include" in r.text and "Dodo" in r.text
    r = client.post("/settings", data={**_base_form(), "species_exclude": "Unicorn"}, follow_redirects=False)
    assert r.status_code == 422 and "species.exclude" in r.text
    r = client.post("/settings", data={**_base_form(), "watchlist": "Black Redstart\nUnicorn"}, follow_redirects=False)
    assert r.status_code == 422 and "telegram.watchlist" in r.text and "Unicorn" in r.text
    r = client.post("/settings", data={**_base_form(), "watchlist": "Phoenicurus ochruros"}, follow_redirects=False)
    assert r.status_code == 422 and "telegram.watchlist" in r.text and "use the English name shown on /species" in r.text
    r = client.post("/settings", data={**_base_form(), "species_include": "Phoenicurus ochruros"}, follow_redirects=False)
    assert r.status_code == 303                                 # include/exclude take latin names
    r = client.post("/settings", data={**_base_form(), "species_include": "eurasian hoopoe", "watchlist": "black redstart"}, follow_redirects=False)
    assert r.status_code == 303


def test_settings_shows_species_list_state(client):
    assert "1 species within 50 km" in client.get("/settings").text


def test_settings_without_classifier_skips_name_validation(store, state, live, config_path):
    app = create_app(store, state, live, config_path)
    c = TestClient(app)
    r = c.post("/settings", data={**_base_form(), "species_include": "Dodo"}, follow_redirects=False)
    assert r.status_code == 303


def test_species_page_shows_candidate_count(client):
    assert "354 candidates" in client.get("/species").text


def test_classify_upload_non_bird_shows_species_skipped(store, state, live, config_path):
    app = create_app(store, state, live, config_path, classifier=FakeNotABird())
    c = TestClient(app)
    data = (Path(__file__).parent / "fixtures" / "cat.jpg").read_bytes()
    r = c.post("/classify", files={"image": ("c.jpg", data, "image/jpeg")})
    assert r.status_code == 200 and "species: skipped (not a bird)" in r.text and "species guesses" not in r.text


def test_settings_page_has_visits_gap(client):
    r = client.get("/settings").text
    assert 'name="visits_gap_seconds"' in r and 'value="60"' in r


# --- corrections, Fix panel, visits -------------------------------------------------

def _e(i):
    return np.eye(4, dtype=np.float32)[i]


def vadd(store, visits, seconds, *, bird=True, row=1, species="European Robin", p=0.7):
    """A classified photo through Visits (row 1 = European Robin in the test bank)."""
    when = T0 + timedelta(seconds=seconds)
    abs_path, rel = store.photo_path_for(when)
    cv2.imwrite(str(abs_path), np.full((60, 80, 3), 90, np.uint8))
    cid = store.add_capture(when, rel, 100)
    store.store_verdict(cid, is_bird=bird, confidence=0.9 if bird else 0.1, species=species if bird else None,
                        species_confidence=p if bird else None, embedding=_e(row) if bird else None)
    visits.photo_classified(store.get(cid), live=False)
    return cid


def test_label_species_corrects_the_whole_visit(client, store, visits):
    a, b = vadd(store, visits, 0), vadd(store, visits, 20)
    vid = store.get(a).visit_id
    assert vid is not None and store.get(b).visit_id == vid and store.get(a).species == "European Robin"
    r = client.post(f"/captures/{a}/label",
                    data={"label": "species", "species_name": "black redstart", "next": "/?filter=birds&page=2"},
                    follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/?filter=birds&page=2"
    for cid in (a, b):
        c = store.get(cid)
        assert c.species == "Black Redstart" and c.manual_species == "Black Redstart" and c.manual_at is not None
    # the spec's own form works too
    r = client.post(f"/captures/{b}/label", data={"label": "species:Eurasian Hoopoe"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    assert store.get(a).species == "Eurasian Hoopoe"


def test_label_bird_unknown_and_clear(client, store, visits):
    a, b = vadd(store, visits, 0), vadd(store, visits, 20)
    r = client.post(f"/captures/{a}/label", data={"label": "bird_unknown"}, follow_redirects=False)
    assert r.status_code == 303
    for cid in (a, b):
        c = store.get(cid)
        assert c.is_bird is True and c.species is None and c.manual_is_bird is True
    r = client.post(f"/captures/{b}/label", data={"label": "clear", "next": "//evil"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    for cid in (a, b):
        c = store.get(cid)
        assert c.manual_at is None and c.manual_is_bird is None and c.species == "European Robin"


def test_label_not_bird_through_visits(client, store, visits, publisher):
    a, b = vadd(store, visits, 0), vadd(store, visits, 20)
    store.set_published(a, "forced")
    r = client.post(f"/captures/{a}/label", data={"label": "not_bird"}, follow_redirects=False)
    assert r.status_code == 303
    for cid in (a, b):
        c = store.get(cid)
        assert c.is_bird is False and c.species is None and c.manual_is_bird is False and c.published is None
    assert publisher.notified >= 1          # the gallery drops the forced photo on the next sync


def test_label_species_rejects_unknown_or_empty_name(client, store, visits):
    a = vadd(store, visits, 0)
    r = client.post(f"/captures/{a}/label", data={"label": "species", "species_name": "Dodo"})
    assert r.status_code == 400 and "Dodo" in r.json()["detail"]
    assert client.post(f"/captures/{a}/label", data={"label": "species", "species_name": "  "}).status_code == 400
    assert client.post(f"/captures/{a}/label", data={"label": "species:"}).status_code == 400
    assert store.get(a).manual_at is None
    assert client.post("/captures/999/label", data={"label": "species", "species_name": "Black Redstart"}).status_code == 404


def test_label_without_visits_falls_back(store, state, live, config_path):
    c = TestClient(create_app(store, state, live, config_path))
    a = add(store, 0, True)
    assert c.post(f"/captures/{a.id}/label", data={"label": "species", "species_name": "Black Redstart"}).status_code == 503
    assert c.post(f"/captures/{a.id}/label", data={"label": "bird_unknown"}).status_code == 503
    assert store.get(a.id).manual_at is None
    assert c.post(f"/captures/{a.id}/label", data={"label": "not_bird"}, follow_redirects=False).status_code == 303
    assert store.get(a.id).is_bird is False
    assert c.post(f"/captures/{a.id}/label", data={"label": "clear"}, follow_redirects=False).status_code == 303
    assert store.get(a.id).is_bird is True


def _tile(html, cid):
    """The markup of one grid tile (from its photo link to the next tile)."""
    start = html.index(f'action="/captures/{cid}/delete"')
    start = html.rindex('<div class="tile', 0, start)
    end = html.find('<div class="tile', start + 1)
    return html[start:end if end != -1 else len(html)]


def test_fix_panel_per_tile_state(client, store, visits):
    bird = vadd(store, visits, 0)
    junk = vadd(store, visits, 600, bird=False)
    fixed = vadd(store, visits, 1200)
    client.post(f"/captures/{fixed}/label", data={"label": "species", "species_name": "Black Redstart"})
    html = client.get("/?filter=all").text
    t = _tile(html, bird)
    assert '<details class="fix"><summary class="pb">Fix</summary>' in t
    assert ">Not a bird<" in t and ">Bird, species unknown<" in t and ">Set<" in t
    assert 'name="species_name" list="species-names"' in t and "Back to model" not in t
    assert 'class="tag you"' not in t and "model:" not in t
    t = _tile(html, junk)          # a bird the model missed can be fixed too
    assert 'class="fix"' in t and ">Bird, species unknown<" in t and 'name="species_name"' in t and ">Set<" in t
    assert ">Not a bird<" not in t and "Back to model" not in t
    t = _tile(html, fixed)
    assert "Back to model" in t and 'class="tag you">you<' in t and "model: European Robin 70%" in t
    assert ">Black Redstart<" in t


def test_visit_link_and_filter(client, store, visits):
    a, b = vadd(store, visits, 0), vadd(store, visits, 20)
    lone = vadd(store, visits, 600)
    vid = store.get(a).visit_id
    html = client.get("/").text
    assert f'<a class="visit" href="/?visit={vid}">visit · 2</a>' in _tile(html, a)
    assert 'class="visit"' not in _tile(html, lone)
    client.post(f"/captures/{b}/label", data={"label": "not_bird"})   # still in the visit, now a non-bird
    html = client.get(f"/?visit={vid}").text
    assert f'action="/captures/{a}/delete"' in html and f'action="/captures/{b}/delete"' in html
    assert f'action="/captures/{lone}/delete"' not in html
    assert "<b>2</b> photos in visit" in html
    assert client.get("/?visit=abc").status_code == 422


def test_visit_pager_keeps_the_visit(client, store, visits, monkeypatch):
    monkeypatch.setattr("birdy.web.app.PAGE_SIZE", 1)
    a, b = vadd(store, visits, 0), vadd(store, visits, 20)
    vadd(store, visits, 600)
    vid = store.get(a).visit_id
    html = client.get(f"/?visit={vid}").text
    assert f'href="/?filter=all&page=2&visit={vid}"' in html and f'action="/captures/{b}/delete"' in html
    html = client.get(f"/?visit={vid}&page=2").text
    assert f'action="/captures/{a}/delete"' in html and f'href="/?filter=all&page=1&visit={vid}"' in html


def test_non_bird_corrected_to_a_bird(client, store, visits):
    junk = vadd(store, visits, 0, bird=False)
    assert store.get(junk).visit_id is None
    r = client.post(f"/captures/{junk}/label", data={"label": "bird_unknown", "next": "/?filter=all"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/?filter=all"
    c = store.get(junk)
    assert c.is_bird is True and c.species is None and c.visit_id is not None
    other = vadd(store, visits, 600, bird=False)
    client.post(f"/captures/{other}/label", data={"label": "species", "species_name": "Eurasian Hoopoe"})
    c = store.get(other)
    assert c.is_bird is True and c.species == "Eurasian Hoopoe" and c.visit_id is not None
    assert "Eurasian Hoopoe" in _tile(client.get("/").text, other)


def test_species_datalist(client, store, visits):
    add(store, 0, True)          # seen: "Robin"
    vadd(store, visits, 600)     # seen: "European Robin" (also on the local list)
    html = client.get("/").text
    m = re.search(r'<datalist id="species-names">(.*?)</datalist>', html)
    assert m is not None and html.count('id="species-names"') == 1
    assert re.findall(r'<option value="([^"]+)">', m.group(1)) == ["Black Redstart", "European Robin", "Robin"]


def test_delete_routes_re_resolve_the_visit(client, store, visits):
    calls = []
    orig = visits.photo_removed
    visits.photo_removed = lambda vid: (calls.append(vid), orig(vid))
    a, b = vadd(store, visits, 0), vadd(store, visits, 20)
    vid = store.get(a).visit_id
    lone = add(store, 30, True)      # no visit: no hook
    r = client.post(f"/captures/{a}/delete", follow_redirects=False)
    assert r.status_code == 303 and calls == [vid] and store.get_visit(vid) is not None
    client.post(f"/captures/{lone.id}/delete")
    assert calls == [vid]
    c, d = vadd(store, visits, 3600), vadd(store, visits, 3620)
    vid2 = store.get(c).visit_id
    client.post(f"/captures/{c}/label", data={"label": "not_bird"})
    r = client.post("/captures/delete-non-birds", data={"ids": [str(c), str(d), str(b)]}, follow_redirects=False)
    assert r.status_code == 303 and calls == [vid, vid2]            # one call per visit, birds untouched
    assert store.get(b) is not None and store.get_visit(vid2) is None


def test_settings_page_has_dslr_fields(client):
    html = client.get("/settings").text
    assert 'name="dslr_region"' in html and 'name="dslr_daily_cap"' in html


def test_settings_post_saves_dslr(client, live):
    r = post_settings(client, dslr_region="0.1, 0.2, 0.3, 0.4", dslr_daily_cap="25")
    assert r.status_code in (200, 303)
    assert live.get().dslr.region == (0.1, 0.2, 0.3, 0.4) and live.get().dslr.daily_cap == 25
