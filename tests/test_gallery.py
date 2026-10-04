"""The static gallery has no build step; check its files reference each other consistently."""

import json
import re
from pathlib import Path

GALLERY = Path(__file__).resolve().parents[1] / "gallery"
PAGES = ["index.html", "site.html", "species.html"]


def test_pages_exist():
    for name in PAGES + ["app.js", "style.css", "sites.example.json"]:
        assert (GALLERY / name).exists(), name


def test_pages_reference_existing_local_assets():
    for name in PAGES:
        html = (GALLERY / name).read_text()
        for ref in re.findall(r'(?:src|href)="([^"#?]+)', html):
            if "://" in ref:
                continue
            assert (GALLERY / ref).exists(), f"{name} -> {ref}"


def test_pages_name_a_renderer_that_exists():
    js = (GALLERY / "app.js").read_text()
    for name in PAGES:
        m = re.search(r'<body data-page="(\w+)"', (GALLERY / name).read_text())
        assert m, f"{name} has no data-page"
        assert f"async function {m.group(1)}(" in js, f"{name}: no renderer {m.group(1)}"


def test_app_only_fetches_gallery_data():
    js = (GALLERY / "app.js").read_text()
    args = re.findall(r"fetch\(([^,)]+)", js)
    assert args, "app.js never fetches"
    for a in args:
        assert "sites.json" in a or "sites/" in a, a


def test_pages_are_responsive_and_titled():
    for name in PAGES:
        html = (GALLERY / name).read_text()
        assert '<meta name="viewport"' in html and "<title>" in html


def test_sites_example_is_valid():
    sites = json.loads((GALLERY / "sites.example.json").read_text())
    assert sites and all({"id", "name", "place"} <= set(s) for s in sites)
    assert all(re.fullmatch(r"[a-z0-9-]{1,32}", s["id"]) for s in sites)
