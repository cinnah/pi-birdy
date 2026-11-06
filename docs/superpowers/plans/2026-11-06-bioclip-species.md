# BioCLIP 2 Species Recognition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 2017 iNat species model with BioCLIP 2 (int8 ONNX, zero-shot over a per-site species list fetched from iNaturalist), run it only on photos already judged birds, drop species rescue, and rename stored history once with gallery slots moving correctly.

**Architecture:** A Mac-side build script produces `models/bioclip2-vision-int8.onnx` and a global bank `models/bioclip2-birds.npz` (name embeddings for ~10,900 iNat bird species); both are gitignored and rsynced by `deploy.sh`. `birdy/species.py` becomes `BioClipSpecies` (onnxruntime, softmax over the current candidate ids). `birdy/species_list.py` fetches/refreshes `data/species/local.json` and computes candidates = local ids + `species.include` − `species.exclude`. `TwoStageClassifier` runs species only when `is_bird` and refreshes candidates when the live config or the list changes. A `meta` table records the species model id; a change resets `species_checked` on uncorrected birds so the existing backfill renames them, calling `publisher.forget(old)` before `evaluate(new)`.

**Tech Stack:** Python 3.11/3.13, onnxruntime (new dep), numpy, OpenCV, FastAPI/Jinja, SQLite, pytest; Mac-only build: torch + open_clip_torch + onnx. Spec: `docs/superpowers/specs/2026-11-06-bioclip-species-design.md`.

**Conventions:** tests with `.venv/bin/pytest`; config keys in `config.py` + `birdy.toml.example`; commit after each task ending with
```
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Agk9eUVj9dbThFrVMdZie7
```
Work on a branch `bioclip-species` off master.

---

## File map

| File | Change |
|---|---|
| `birdy/config.py`, `birdy.toml.example`, `birdy/watcher.py` | `[species]`: `bank`, `region_radius_km`, `include`, `exclude`; `rescue_threshold` removed (`labels` removed in Task 7) |
| `birdy/settings.py`, `birdy/web/templates/settings.html`, `birdy/web/app.py` | rescue row gone; include/exclude textareas; radius read-only; name validation against the bank |
| `birdy/species.py` | `SpeciesLabel` (kept), `Bank`, `load_bank`, `BioClipSpecies`; `InatBirdClassifier`/`parse_label` removed in Task 7 |
| `birdy/species_list.py` (new) | iNat fetch, `LocalList`, `SpeciesList` (load/refresh thread/candidates) |
| `birdy/classifier.py` | species only for birds; rescue removed; candidate refresh hook |
| `birdy/store.py` | `meta` table, `get_meta/set_meta`, `reset_species_for_model` |
| `birdy/pipeline.py` | rename → `publisher.forget(old)` then `evaluate(new)` |
| `birdy/cli.py` | `_classifier` builds BioCLIP + list; `cmd_run` reset + refresh thread; `classify` output; `species-list` command |
| `deploy/build_species_model.py`, `deploy/build-species-model.sh`, `models/README.md`, `.gitignore`, `pyproject.toml` | artefacts + deps |
| `CLAUDE.md` | docs |
| tests: `test_config.py`, `test_settings.py`, `test_watcher.py`, `test_web.py`, `test_species.py` (rewritten), `test_species_list.py` (new), `test_two_stage.py`, `test_store.py`, `test_pipeline.py`, `test_cli.py`, `tests/fixtures/species_counts.json` (new) | |

---

### Task 1: Config — new `[species]` keys, rescue removed

**Files:**
- Modify: `birdy/config.py`, `birdy.toml.example`, `birdy/watcher.py`, `birdy/settings.py`, `birdy/web/templates/settings.html`, `birdy/web/app.py`, `birdy/classifier.py` (rescue only), `birdy/cli.py` (rescue print only)
- Test: `tests/test_config.py`, `tests/test_settings.py`, `tests/test_watcher.py`, `tests/test_web.py`, `tests/test_two_stage.py`, `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_config.py`: replace `test_species_defaults`, `test_species_rescue_must_not_be_below_threshold` and `test_species_thresholds_range` with:

```python
def test_species_defaults(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    assert cfg.species.model == tmp_path / "models" / "bioclip2-vision-int8.onnx"
    assert cfg.species.bank == tmp_path / "models" / "bioclip2-birds.npz"
    assert cfg.species.labels == tmp_path / "models" / "inat_bird_labels.txt"   # removed in Task 7
    assert cfg.species.threshold == 0.5
    assert cfg.species.region_radius_km == 50
    assert cfg.species.include == () and cfg.species.exclude == ()
    assert not hasattr(cfg.species, "rescue_threshold")


def test_species_threshold_range(tmp_path):
    with pytest.raises(ConfigError, match="species.threshold"):
        load_config(write(tmp_path, MINIMAL + "\n[species]\nthreshold = 1.5\n"))


def test_species_unknown_key_ignored(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL + "\n[species]\nrescue_threshold = 1.0\n"))
    assert cfg.species.threshold == 0.5


@pytest.mark.parametrize("text", ["region_radius_km = 0", "region_radius_km = 501", 'region_radius_km = "x"'])
def test_species_radius_range(tmp_path, text):
    with pytest.raises(ConfigError, match="species.region_radius_km"):
        load_config(write(tmp_path, MINIMAL + f"\n[species]\n{text}\n"))


def test_species_include_exclude_lists(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL + '\n[species]\ninclude = [" Hoopoe ", ""]\nexclude = ["Rock Pigeon"]\n'))
    assert cfg.species.include == ("Hoopoe",) and cfg.species.exclude == ("Rock Pigeon",)
    with pytest.raises(ConfigError, match="species.include"):
        load_config(write(tmp_path, MINIMAL + "\n[species]\ninclude = [1]\n"))
```

`tests/test_settings.py`: in `default_patch` remove `species_rescue_threshold=...` and add `species_include=config.species.include, species_exclude=config.species.exclude,`; in `FORM` remove `"species_rescue_threshold": "0.7",` and add `"species_include": "", "species_exclude": "",`; in `FULL_TOML` replace the `rescue_threshold = 0.9` line with `bank = "models/other-birds.npz"\nregion_radius_km = 80\ninclude = ["Hoopoe"]\nexclude = ["Rock Pigeon"]`; delete `test_species_rescue_below_threshold_rejected`; change the two rescue-using tests (`test_species_thresholds_roundtrip` at ~309 and `test_parse_form_species_thresholds` at ~319) to use only `species_threshold`; append:

```python
def test_species_include_exclude_roundtrip(config_path, config):
    new = save_settings(config_path, config, default_patch(config, species_include=("Hoopoe", "Blue Tit"), species_exclude=("Rock Pigeon",)))
    assert new.species.include == ("Hoopoe", "Blue Tit") and new.species.exclude == ("Rock Pigeon",)
    assert load_config(config_path).species.exclude == ("Rock Pigeon",)


def test_parse_form_species_lists():
    p = parse_settings_form({**FORM, "species_include": "Hoopoe\r\n\r\n Blue Tit \n", "species_exclude": "Rock Pigeon"})
    assert p.species_include == ("Hoopoe", "Blue Tit") and p.species_exclude == ("Rock Pigeon",)
```

`tests/test_watcher.py`: in `test_save_settings_then_watch_is_noop` replace `species_rescue_threshold=cur.species.rescue_threshold,` with `species_include=cur.species.include, species_exclude=cur.species.exclude,`; append:

```python
def test_species_bank_change_warns(config_path, live, caplog):
    w = ConfigWatcher(config_path, live)
    config_path.write_text(config_path.read_text() + '\n[species]\nbank = "models/other.npz"\n')
    bump_mtime(config_path)
    with caplog.at_level(logging.WARNING):
        assert w.check_once() is True
    assert "restart birdy to apply them" in caplog.text
```

`tests/test_web.py`: in `_base_form` remove `"species_rescue_threshold": "0.7",` and add `"species_include": "", "species_exclude": "",`; at ~334 change the assertion to `assert 'name="species_threshold"' in r and 'name="species_include"' in r and 'species_rescue_threshold' not in r`.

`tests/test_two_stage.py`: delete `test_rescue_above_rescue_threshold`, `test_no_rescue_between_thresholds`, `test_rescue_still_works_with_detector_below`; change `test_thresholds_follow_live_config` to:

```python
def test_species_threshold_follows_live_config(live, config):
    clf = two_stage(live, 0.9, 0.6)
    assert clf.classify_image(IMG)[0].species == "Eurasian Blue Tit"
    live.replace(dataclasses.replace(config, species=dataclasses.replace(config.species, threshold=0.7)))
    assert clf.classify_image(IMG)[0].species is None
```

and append:

```python
def test_confident_species_never_rescues(live):
    pred, _ = two_stage(live, 0.2, 0.99, p_det=0.1).classify_image(IMG)
    assert pred.is_bird is False and pred.confidence == 0.2
```

`tests/test_cli.py`: nothing yet (the "rescued by species" marker just disappears).

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_config.py tests/test_settings.py tests/test_two_stage.py -q`
Expected: failures on `cfg.species.bank`, `region_radius_km`, `SettingsPatch` kwargs, `test_confident_species_never_rescues`.

- [ ] **Step 3: Implement**

`birdy/config.py` — `SpeciesConfig`:

```python
@dataclass(frozen=True)
class SpeciesConfig:
    model: Path                     # restart-only; BioCLIP 2 image tower (int8 ONNX)
    bank: Path                      # restart-only; name embeddings for every iNat bird species
    labels: Path                    # legacy iNat label file; removed with InatBirdClassifier
    threshold: float                # name a bird at or above this (live)
    region_radius_km: int           # iNaturalist radius for the local species list; refetch at next start
    include: tuple[str, ...]        # names added to the local list, latin or common (live)
    exclude: tuple[str, ...]        # names dropped from it (live)
```

Parser (replace the species block):

```python
    sp = raw.get("species", {})
    species_threshold = _num(sp, "species", "threshold", 0.5, float)
    _check(0.0 <= species_threshold <= 1.0, "species.threshold must be between 0 and 1")
    region_radius_km = _num(sp, "species", "region_radius_km", 50, int)
    _check(1 <= region_radius_km <= 500, "species.region_radius_km must be between 1 and 500")
    sp_include = _str_list(sp, "species", "include")
    sp_exclude = _str_list(sp, "species", "exclude")
```

Constructor:

```python
        species=SpeciesConfig(
            model=_path(base, sp, "species", "model", "models/bioclip2-vision-int8.onnx"),
            bank=_path(base, sp, "species", "bank", "models/bioclip2-birds.npz"),
            labels=_path(base, sp, "species", "labels", "models/inat_bird_labels.txt"),
            threshold=species_threshold,
            region_radius_km=region_radius_km,
            include=sp_include,
            exclude=sp_exclude,
        ),
```

`birdy.toml.example` `[species]` block:

```toml
[species]
model = "models/bioclip2-vision-int8.onnx"   # BioCLIP 2 image tower; built by deploy/build-species-model.sh; restart-only
bank = "models/bioclip2-birds.npz"           # name embeddings for every iNaturalist bird species; same script; restart-only
threshold = 0.5            # name a bird at or above this confidence; applies without a restart
region_radius_km = 50      # the local species list = birds seen on iNaturalist within this radius of [site] lat/lon; a change refetches at the next start
include = []               # names to add to the local list (latin or common), e.g. ["Eurasian Hoopoe"]; live
exclude = []               # names to drop from it; live
```

`birdy/watcher.py` `_restart_only`: replace `cfg.species.model, cfg.species.labels,` with `cfg.species.model, cfg.species.bank, cfg.species.region_radius_km,`.

`birdy/settings.py`: `FIELD_KEYS`: remove the rescue line, add `"species_include": "species.include", "species_exclude": "species.exclude",`; `SettingsPatch`: remove `species_rescue_threshold`, add `species_include: tuple[str, ...]` and `species_exclude: tuple[str, ...]`; `config_to_dict["species"]` = `{"model", "bank", "labels", "threshold", "region_radius_km", "include": list(...), "exclude": list(...)}`; `apply_patch`: remove the rescue line, add `data["species"]["include"] = list(patch.species_include)` and `exclude` likewise; `parse_settings_form`: remove rescue, add `species_include=_lines(form, "species_include"), species_exclude=_lines(form, "species_exclude"),` where

```python
def _lines(form: Mapping[str, str], field: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in form.get(field, "").splitlines() if s.strip())
```

and `_watchlist` becomes `return _lines(form, "watchlist")`.

`birdy/web/templates/settings.html`: delete the `species_rescue_threshold` row. After the `species_threshold` row add:

```jinja
    <div class="row {{ 'error' if error_field == 'species_include' else '' }}">
      <label for="species_include">Also consider</label>
      {{ explain("Species to add to the local list (latin or common name), one per line.", "species.include") }}
      <textarea id="species_include" name="species_include" rows="3">{{ v['species_include'] }}</textarea>
    </div>
    {% if error_field == 'species_include' %}<div class="hint error">{{ error_message }}</div>{% endif %}
    <div class="row {{ 'error' if error_field == 'species_exclude' else '' }}">
      <label for="species_exclude">Never name</label>
      {{ explain("Species to drop from the local list, one per line.", "species.exclude") }}
      <textarea id="species_exclude" name="species_exclude" rows="3">{{ v['species_exclude'] }}</textarea>
    </div>
    {% if error_field == 'species_exclude' %}<div class="hint error">{{ error_message }}</div>{% endif %}
```

`birdy/web/app.py` settings values: remove `"species_rescue_threshold"`, add `"species_include": "\n".join(cfg.species.include), "species_exclude": "\n".join(cfg.species.exclude),`; in `readonly` add `"species.bank": str(cfg.species.bank), "species.region_radius_km": cfg.species.region_radius_km,`.

`birdy/classifier.py` `classify_image`: `t_sp = cfg.species.threshold` (no rescue), and

```python
        species = label.common if label is not None and p_sp >= t_sp else None
        is_bird = p_bird >= t_gate or p_det >= t_det
        confidence = max(p_bird, p_det)
```

Update the class docstring and module docstring (no rescue). `birdy/cli.py` `cmd_classify`: drop the `rescued` variable and its `{rescued}` in the verdict line.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest -q`
Expected: all pass (`test_config_to_dict_covers_every_field` included).

- [ ] **Step 5: Commit**

```bash
git add -A birdy birdy.toml.example tests
git commit -m "Config: [species] bank/region_radius_km/include/exclude; species rescue removed"
```

---

### Task 2: `Bank` and `BioClipSpecies`

**Files:**
- Modify: `birdy/species.py` (add; keep `InatBirdClassifier` until Task 7), `pyproject.toml`
- Test: `tests/test_species.py` (append; the iNat tests stay until Task 7)

- [ ] **Step 1: Dependencies**

`pyproject.toml`: add `"onnxruntime>=1.20",` to `dependencies`; add `"onnx>=1.16",` to the `dev` extra. Run `.venv/bin/pip install -e ".[dev]" -q`.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_species.py`:

```python
import json

from birdy.species import Bank, BioClipSpecies, load_bank

BIOCLIP = ROOT / "models" / "bioclip2-vision-int8.onnx"
BANK = ROOT / "models" / "bioclip2-birds.npz"
D = 8  # embedding size of the stand-in


def write_standin_onnx(path: Path, d: int = D, seed: int = 0) -> np.ndarray:
    """pixels[1,3,224,224] -> mean over pixels per channel (3) -> MatMul W[3,d] -> embedding[1,d]."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(seed)
    w = rng.standard_normal((3, d)).astype(np.float32)
    nodes = [
        helper.make_node("ReduceMean", ["pixels"], ["m"], axes=[2, 3], keepdims=0),  # [1,3]
        helper.make_node("MatMul", ["m", "W"], ["embedding"]),
    ]
    graph = helper.make_graph(
        nodes, "standin",
        [helper.make_tensor_value_info("pixels", TensorProto.FLOAT, [1, 3, 224, 224])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, d])],
        initializer=[numpy_helper.from_array(w, "W")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, str(path))
    return w


def write_bank(path: Path, emb: np.ndarray, model_id: str = "standin/1") -> None:
    np.savez(
        path,
        ids=np.array([11, 22, 33], np.int32),
        latin=np.array(["Phoenicurus ochruros", "Erithacus rubecula", "Passer domesticus"]),
        common=np.array(["Black Redstart", "European Robin", "House Sparrow"]),
        emb=emb.astype(np.float16),
        logit_scale=np.float32(100.0),
        model=np.str_(model_id),
        prompt=np.str_("a photo of {latin} ({common})."),
    )


@pytest.fixture
def standin(tmp_path):
    w = write_standin_onnx(tmp_path / "m.onnx")
    # three unit text vectors; an image of a given flat colour maps to colour @ W, so pick
    # bank rows that are exactly the embeddings of pure red / green / blue images.
    MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
    STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
    def emb_of(rgb):
        px = (np.array(rgb, np.float32) / 255.0 - MEAN) / STD
        e = px @ w
        return e / np.linalg.norm(e)
    bank = np.vstack([emb_of((255, 0, 0)), emb_of((0, 255, 0)), emb_of((0, 0, 255))])
    write_bank(tmp_path / "b.npz", bank)
    return tmp_path / "m.onnx", tmp_path / "b.npz"


def flat(rgb, h=100, w=120):
    return np.full((h, w, 3), rgb, np.uint8)


def test_load_bank_shapes(standin):
    _, bank_path = standin
    b = load_bank(bank_path)
    assert isinstance(b, Bank) and b.ids.tolist() == [11, 22, 33] and b.emb.shape == (3, D) and b.emb.dtype == np.float32
    assert b.model == "standin/1" and b.logit_scale == 100.0
    assert b.resolve("black redstart") == 11 and b.resolve("ERITHACUS RUBECULA") == 22 and b.resolve("Dodo") is None


def test_top_over_whole_bank(standin):
    sp = BioClipSpecies(*standin)
    top = sp.top(flat((255, 0, 0)), k=3)
    assert len(top) == 3 and top[0][0].common == "Black Redstart"
    assert top[0][1] > 0.9 and abs(sum(p for _, p in top) - 1.0) < 1e-4
    assert top[0][0] == SpeciesLabel("Phoenicurus ochruros", "Black Redstart")


def test_candidates_restrict_the_softmax(standin):
    sp = BioClipSpecies(*standin)
    assert sp.set_candidates([22, 33]) == 2
    top = sp.top(flat((255, 0, 0)), k=5)
    assert len(top) == 2 and all(l.common != "Black Redstart" for l, _ in top)
    assert abs(sum(p for _, p in top) - 1.0) < 1e-4
    assert sp.set_candidates(None) == 3


def test_unknown_candidate_ids_are_ignored_and_empty_means_no_species(standin):
    sp = BioClipSpecies(*standin)
    assert sp.set_candidates([99, 11]) == 1
    assert sp.top(flat((0, 255, 0)))[0][0].common == "Black Redstart"
    assert sp.set_candidates([]) == 0
    assert sp.top(flat((0, 255, 0))) == []


def test_model_id_mismatch_raises(standin, tmp_path):
    model, _ = standin
    write_bank(tmp_path / "other.npz", np.eye(3, D, dtype=np.float32), model_id="bioclip-2/int8/deadbeef")
    with pytest.raises(ValueError, match="bank"):
        BioClipSpecies(model, tmp_path / "other.npz")


def test_model_id_is_derived_from_the_onnx_file(standin):
    sp = BioClipSpecies(*standin, check_model=False)
    assert sp.model_id.startswith("bioclip-2/int8/") and len(sp.model_id.split("/")[-1]) == 12


@pytest.mark.skipif(not (BIOCLIP.exists() and BANK.exists()), reason="BioCLIP model/bank not built")
class TestRealBioClip:
    def test_bird_fixture_gets_a_bird_name(self):
        sp = BioClipSpecies(BIOCLIP, BANK)
        label, p = sp.top(rgb(FIXTURES / "bird.jpg"), k=1)[0]
        assert p > 0.3 and label.latin and label.common
```

Note on `test_model_id_mismatch_raises`: the stand-in bank's `model` is `"standin/1"`, which must be accepted by the constructor only because the test writes it that way — define the rule as: the bank's `model` must equal `BioClipSpecies.model_id`, where `model_id = "bioclip-2/int8/" + sha256(onnx)[:12]`, **unless** `check_model=False`; and the `standin` fixture must therefore write the real derived id. Implement the fixture accordingly: compute `hashlib.sha256(path.read_bytes()).hexdigest()[:12]` after `write_standin_onnx` and pass `model_id=f"bioclip-2/int8/{h}"` to `write_bank`. Adjust `test_load_bank_shapes` to assert `b.model.startswith("bioclip-2/int8/")`.

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_species.py -q`
Expected: `ImportError: cannot import name 'Bank'`.

- [ ] **Step 4: Implement** — add to `birdy/species.py` (keep the existing classes for now; change the module docstring to "Bird species recognition: BioCLIP 2 zero-shot over a bank of iNaturalist bird names (the Coral iNat model below is legacy)."):

```python
import hashlib
from collections.abc import Iterable

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
INPUT = 224


@dataclass(frozen=True)
class Bank:
    """Name embeddings for every iNaturalist bird species (built on the Mac, see deploy/build-species-model.sh)."""

    ids: np.ndarray        # int32 [N]  iNaturalist taxon ids
    latin: list[str]
    common: list[str]      # "" when iNaturalist has no English name
    emb: np.ndarray        # float32 [N, D], L2-normalised
    logit_scale: float
    model: str             # "bioclip-2/int8/<12 hex>": must match the image tower
    prompt: str

    def resolve(self, name: str) -> int | None:
        """iNat id for a latin or common name, case-insensitive; None if unknown."""
        return self._index().get(name.strip().casefold())

    def _index(self) -> dict[str, int]:
        idx = getattr(self, "_idx", None)
        if idx is None:
            idx = {}
            for i, (lat, com) in enumerate(zip(self.latin, self.common)):
                idx.setdefault(lat.casefold(), int(self.ids[i]))
                if com:
                    idx.setdefault(com.casefold(), int(self.ids[i]))
            object.__setattr__(self, "_idx", idx)
        return idx


def load_bank(path: Path) -> Bank:
    try:
        z = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as e:
        raise ValueError(f"{path}: cannot read species bank: {e}") from e
    try:
        return Bank(
            ids=z["ids"].astype(np.int32),
            latin=[str(s) for s in z["latin"]],
            common=[str(s) for s in z["common"]],
            emb=z["emb"].astype(np.float32),
            logit_scale=float(z["logit_scale"]),
            model=str(z["model"]),
            prompt=str(z["prompt"]),
        )
    except KeyError as e:
        raise ValueError(f"{path}: species bank is missing {e}") from e


def model_id_of(onnx_path: Path) -> str:
    return "bioclip-2/int8/" + hashlib.sha256(Path(onnx_path).read_bytes()).hexdigest()[:12]


def preprocess(img_rgb: np.ndarray) -> np.ndarray:
    """CLIP preprocessing: short side to 224 (bicubic), centre crop, normalise, NCHW float32."""
    h, w = img_rgb.shape[:2]
    s = INPUT / max(1, min(h, w))
    nh, nw = max(INPUT, round(h * s)), max(INPUT, round(w * s))
    img = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_CUBIC)
    t, l = (nh - INPUT) // 2, (nw - INPUT) // 2
    img = img[t:t + INPUT, l:l + INPUT]
    x = (img.astype(np.float32) / 255.0 - CLIP_MEAN) / CLIP_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[np.newaxis])


class BioClipSpecies:
    """Zero-shot species over the current candidate ids. Not thread-safe; called under
    `TwoStageClassifier._lock`. ~3.5 s per image on a Pi 4 (int8 ViT-L/14)."""

    def __init__(self, model_path: Path, bank_path: Path, threads: int = 3, check_model: bool = True) -> None:
        import onnxruntime as ort

        self.model_id = model_id_of(model_path)
        self.bank = load_bank(bank_path)
        if check_model and self.bank.model != self.model_id:
            raise ValueError(f"{bank_path}: bank built for {self.bank.model}, model is {self.model_id}; rebuild both")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self._sess = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self._input = self._sess.get_inputs()[0].name
        self._cand = np.arange(len(self.bank.ids))   # row indices into the bank
        self.set_candidates(None)

    def set_candidates(self, ids: Iterable[int] | None) -> int:
        """Restrict `top` to these iNat ids (None = whole bank). Unknown ids are ignored. Returns the count."""
        if ids is None:
            self._cand = np.arange(len(self.bank.ids))
        else:
            wanted = set(int(i) for i in ids)
            self._cand = np.array([i for i, tid in enumerate(self.bank.ids) if int(tid) in wanted], dtype=np.int64)
        return int(len(self._cand))

    @property
    def candidate_count(self) -> int:
        return int(len(self._cand))

    def embed(self, img_rgb: np.ndarray) -> np.ndarray:
        e = self._sess.run(None, {self._input: preprocess(img_rgb)})[0][0].astype(np.float32)
        return e / max(1e-12, float(np.linalg.norm(e)))

    def top(self, img_rgb: np.ndarray, k: int = 5) -> list[tuple[SpeciesLabel, float]]:
        if len(self._cand) == 0:
            return []
        e = self.embed(img_rgb)
        logits = self.bank.logit_scale * (self.bank.emb[self._cand] @ e)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        order = np.argsort(p)[::-1][:k]
        return [
            (SpeciesLabel(self.bank.latin[self._cand[i]], self.bank.common[self._cand[i]] or self.bank.latin[self._cand[i]]), float(p[i]))
            for i in order
        ]
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/pytest tests/test_species.py -q`
Expected: all pass (real-model class skipped until Task 9).

- [ ] **Step 6: Commit**

```bash
git add birdy/species.py pyproject.toml tests/test_species.py
git commit -m "Species: BioClipSpecies — zero-shot over an iNaturalist name bank (onnxruntime)"
```

---

### Task 3: Classifier — species only for birds, candidate refresh hook

**Files:**
- Modify: `birdy/classifier.py`
- Test: `tests/test_two_stage.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_two_stage.py`, give `FakeSpecies` a `set_candidates` recorder:

```python
class FakeSpecies:
    def __init__(self, p, raise_exc=None):
        self.p = p
        self.raise_exc = raise_exc
        self.calls = 0
        self.candidate_sets: list = []

    def set_candidates(self, ids):
        self.candidate_sets.append(None if ids is None else sorted(ids))
        return 0 if ids is None else len(list(ids))

    def top(self, img, k=5, skip_background=True):
        ...  # unchanged
```

Append:

```python
def test_species_not_run_for_non_birds(live):
    clf = two_stage(live, 0.1, 0.9, p_det=0.1)
    pred, top = clf.classify_image(IMG)
    assert pred.is_bird is False and pred.species is None and pred.species_confidence is None
    assert top == [] and clf.species.calls == 0


def test_species_runs_for_detector_birds(live):
    clf = two_stage(live, 0.1, 0.9, p_det=0.6)
    pred, top = clf.classify_image(IMG)
    assert pred.is_bird and pred.species == "Eurasian Blue Tit" and clf.species.calls == 1


class FakeList:
    def __init__(self, ids):
        self.ids, self.version = ids, 1

    def candidates(self, species_cfg):
        return self.ids


def test_candidates_follow_the_list_and_live_config(live, config):
    lst = FakeList(frozenset({11, 22}))
    clf = TwoStageClassifier(FakeGate(0.9), FakeSpecies(0.8), live, lores_size=(320, 240), detector=FakeDetector(0.0), species_list=lst)
    clf.classify_image(IMG)
    assert clf.species.candidate_sets == [[11, 22]]
    clf.classify_image(IMG)
    assert len(clf.species.candidate_sets) == 1          # unchanged -> not recomputed
    lst.ids, lst.version = frozenset({33}), 2
    clf.classify_image(IMG)
    assert clf.species.candidate_sets[-1] == [33]
    live.replace(dataclasses.replace(config, species=dataclasses.replace(config.species, exclude=("x",))))
    clf.classify_image(IMG)
    assert len(clf.species.candidate_sets) == 3          # live version bump recomputes too


def test_no_list_means_whole_bank(live):
    clf = two_stage(live, 0.9, 0.8)
    clf.classify_image(IMG)
    assert clf.species.candidate_sets == []               # no species_list given: never touched
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_two_stage.py -q` → `TypeError ... unexpected keyword argument 'species_list'` and `test_species_not_run_for_non_birds` fails (species called).

- [ ] **Step 3: Implement** in `birdy/classifier.py`

`__init__`: add `species_list: "SpeciesList | None" = None` (import under `TYPE_CHECKING` from `birdy.species_list`), store it as the public attribute `self.species_list = species_list` (the CLI and the web read it) and `self._cand_key: tuple | None = None`. In `_refresh_candidates` use `self.species_list`.

`classify_image` body:

```python
        cfg = self._live.get()
        t_gate = cfg.classifier.threshold if threshold is None else threshold
        t_det = cfg.detector.threshold
        t_sp = cfg.species.threshold

        with self._lock:
            p_bird = self.gate.predict(img_rgb)
            det = None
            if self.detector is not None:
                try:
                    det = self.detector.best_bird(img_rgb)
                except Exception as e:
                    key = f"{type(e).__name__}: {e}"
                    if key not in self._detector_errors:
                        log.exception("detector failed (%s); continuing without it", key)
                        self._detector_errors.add(key)
            p_det = det.score if det is not None else 0.0
            is_bird = p_bird >= t_gate or p_det >= t_det
            top = []
            if is_bird:
                # The species model has no "nothing here" answer (14 % of empty frames get a confident
                # name), so it only ever sees photos the gate or detector already called a bird.
                self._refresh_candidates(cfg)
                try:
                    top = self.species.top(img_rgb, k=5)
                except Exception as e:
                    key = f"{type(e).__name__}: {e}"
                    if key not in self._species_errors:
                        log.exception("species model failed (%s); continuing without a species", key)
                        self._species_errors.add(key)
                    top = []
        label, p_sp = top[0] if top else (None, 0.0)
        species = label.common if label is not None and p_sp >= t_sp else None
        pred = Prediction(
            is_bird=is_bird,
            confidence=max(p_bird, p_det),
            species=species,
            species_confidence=p_sp if species is not None else None,
            detector_confidence=p_det,
            detector_box=det.box if det is not None else None,
        )
        return pred, top

    def _refresh_candidates(self, cfg) -> None:
        """Re-apply the candidate set when the live config or the local list changed. Caller holds the lock."""
        if self.species_list is None:
            return
        key = (self._live.version, self.species_list.version)
        if key == self._cand_key:
            return
        n = self.species.set_candidates(self.species_list.candidates(cfg.species))
        log.info("species candidates: %d", n)
        self._cand_key = key
```

Update the docstrings (species only for birds; rescue gone).

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_two_stage.py tests/test_web.py tests/test_pipeline.py -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add birdy/classifier.py tests/test_two_stage.py
git commit -m "Classifier: species only for birds; candidate set follows the live config and the local list"
```

---

### Task 4: `birdy/species_list.py` — fetch, store, refresh, candidates

**Files:**
- Create: `birdy/species_list.py`, `tests/fixtures/species_counts.json`
- Test: `tests/test_species_list.py`

- [ ] **Step 1: Fixture** — write `tests/fixtures/species_counts.json` (a trimmed real response shape):

```json
{"total_results": 4, "page": 1, "per_page": 500, "results": [
 {"count": 120, "taxon": {"id": 13094, "name": "Phoenicurus ochruros", "rank": "species", "preferred_common_name": "Black Redstart"}},
 {"count": 2,   "taxon": {"id": 13095, "name": "Phoenicurus phoenicurus", "rank": "species", "preferred_common_name": "Common Redstart"}},
 {"count": 40,  "taxon": {"id": 559248, "name": "Anas platyrhynchos × Cairina moschata", "rank": "hybrid", "preferred_common_name": "Mallard × Muscovy Duck"}},
 {"count": 9,   "taxon": {"id": 12942, "name": "Erithacus rubecula", "rank": "species"}}
]}
```

- [ ] **Step 2: Write the failing tests** — `tests/test_species_list.py`:

```python
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from birdy.species import load_bank
from birdy.species_list import LocalList, SpeciesList, fetch_species_counts, parse_species_counts, needs_refresh

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_keeps_species_with_three_observations():
    data = json.loads((FIXTURES / "species_counts.json").read_text())
    rows = parse_species_counts([data])
    assert [r["id"] for r in rows] == [13094, 12942]
    assert rows[0] == {"id": 13094, "latin": "Phoenicurus ochruros", "common": "Black Redstart", "count": 120}
    assert rows[1]["common"] == "Erithacus rubecula"   # no English name -> latin


class FakeHttp:
    def __init__(self, pages_by_radius):
        self.pages_by_radius, self.calls = pages_by_radius, []

    def __call__(self, lat, lon, radius_km, page):
        self.calls.append((lat, lon, radius_km, page))
        return self.pages_by_radius[radius_km][page - 1]


def many(n, start=1000):
    return {"results": [{"count": 5, "taxon": {"id": start + i, "name": f"Sp {i}", "rank": "species", "preferred_common_name": f"Bird {i}"}} for i in range(n)]}


def test_fetch_pages_until_short_page():
    http = FakeHttp({50: [many(500), many(20, 2000)]})
    rows = fetch_species_counts(38.8, -9.39, 50, http=http, sleep=lambda s: None)
    assert len(rows) == 520 and [c[3] for c in http.calls] == [1, 2]


def test_fetch_retries_once_at_four_times_radius_when_too_few():
    http = FakeHttp({50: [many(5)], 200: [many(30)]})
    rows = fetch_species_counts(38.8, -9.39, 50, http=http, sleep=lambda s: None)
    assert len(rows) == 30 and [c[2] for c in http.calls] == [50, 200]


def test_local_list_roundtrip_atomic(tmp_path):
    lst = LocalList(fetched_at=datetime(2026, 11, 6, tzinfo=timezone.utc), lat=38.8, lon=-9.39, radius_km=50,
                    species=[{"id": 13094, "latin": "Phoenicurus ochruros", "common": "Black Redstart", "count": 120}])
    lst.save(tmp_path / "species" / "local.json")
    back = LocalList.load(tmp_path / "species" / "local.json")
    assert back == lst and not list((tmp_path / "species").glob("*.tmp*"))
    assert LocalList.load(tmp_path / "missing.json") is None


@pytest.mark.parametrize("change,expected", [
    ({}, False),
    ({"lat": 39.0}, True),
    ({"radius_km": 80}, True),
    ({"age_days": 91}, True),
])
def test_needs_refresh(change, expected):
    now = datetime(2026, 11, 6, tzinfo=timezone.utc)
    lst = LocalList(fetched_at=now - timedelta(days=change.get("age_days", 1)), lat=38.8, lon=-9.39, radius_km=50, species=[])
    assert needs_refresh(lst, change.get("lat", 38.8), -9.39, change.get("radius_km", 50), now) is expected
    assert needs_refresh(None, 38.8, -9.39, 50, now) is True


def bank_file(tmp_path):
    p = tmp_path / "bank.npz"
    np.savez(p, ids=np.array([13094, 12942, 1], np.int32),
             latin=np.array(["Phoenicurus ochruros", "Erithacus rubecula", "Upupa epops"]),
             common=np.array(["Black Redstart", "European Robin", "Eurasian Hoopoe"]),
             emb=np.eye(3, 4, dtype=np.float16), logit_scale=np.float32(100.0), model=np.str_("x"), prompt=np.str_("p"))
    return load_bank(p)


class Cfg:
    def __init__(self, include=(), exclude=()):
        self.include, self.exclude = include, exclude


def test_candidates_from_list_plus_include_minus_exclude(tmp_path, caplog):
    sl = SpeciesList(tmp_path / "data", bank_file(tmp_path))
    assert sl.candidates(Cfg()) is None                      # no list, nothing configured -> whole bank
    LocalList(fetched_at=datetime.now(timezone.utc), lat=38.8, lon=-9.39, radius_km=50,
              species=[{"id": 13094, "latin": "Phoenicurus ochruros", "common": "Black Redstart", "count": 1}]).save(tmp_path / "data" / "species" / "local.json")
    sl.reload()
    assert sl.candidates(Cfg()) == frozenset({13094})
    assert sl.candidates(Cfg(include=("eurasian hoopoe",), exclude=("Black Redstart",))) == frozenset({1})
    with caplog.at_level("WARNING"):
        assert sl.candidates(Cfg(include=("Dodo",))) == frozenset({13094})
    assert "Dodo" in caplog.text
    assert sl.unknown_names(("Dodo", "European Robin")) == ["Dodo"]


def test_refresh_fetches_saves_and_bumps_version(tmp_path, monkeypatch):
    sl = SpeciesList(tmp_path / "data", bank_file(tmp_path))
    data = json.loads((FIXTURES / "species_counts.json").read_text())
    monkeypatch.setattr("birdy.species_list.fetch_species_counts", lambda lat, lon, r, **kw: parse_species_counts([data]))
    v0 = sl.version
    assert sl.refresh_if_needed(38.8, -9.39, 50) is True
    assert sl.version == v0 + 1 and sl.current().radius_km == 50 and (tmp_path / "data" / "species" / "local.json").exists()
    assert sl.refresh_if_needed(38.8, -9.39, 50) is False     # fresh -> no fetch
    assert sl.refresh_if_needed(None, None, 50) is False      # no location -> no list


def test_refresh_failure_keeps_old_file(tmp_path, monkeypatch, caplog):
    sl = SpeciesList(tmp_path / "data", bank_file(tmp_path))
    old = LocalList(fetched_at=datetime.now(timezone.utc) - timedelta(days=100), lat=38.8, lon=-9.39, radius_km=50, species=[])
    old.save(tmp_path / "data" / "species" / "local.json")
    sl.reload()
    monkeypatch.setattr("birdy.species_list.fetch_species_counts", lambda *a, **kw: (_ for _ in ()).throw(OSError("down")))
    with caplog.at_level("WARNING"):
        assert sl.refresh_if_needed(38.8, -9.39, 50) is False
    assert "species list" in caplog.text and sl.current() == old


def test_refresh_thread_runs_once_and_stops(tmp_path, monkeypatch):
    sl = SpeciesList(tmp_path / "data", bank_file(tmp_path))
    ran = threading.Event()
    monkeypatch.setattr(sl, "refresh_if_needed", lambda *a: ran.set() or True)
    sl.start_refresh(38.8, -9.39, 50)
    assert ran.wait(2)
    sl.stop()
```

- [ ] **Step 3: Run to verify failure** → `ModuleNotFoundError: birdy.species_list`.

- [ ] **Step 4: Implement** `birdy/species_list.py`:

```python
"""The site's local species list: birds seen on iNaturalist near the Pi, used to restrict the
BioCLIP candidates. Stored at data/species/local.json; refreshed at service start when stale."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from birdy.species import Bank

log = logging.getLogger(__name__)

API = "https://api.inaturalist.org/v1/observations/species_counts"
USER_AGENT = "birdy (+https://birdy.akibeo.com)"
MIN_OBSERVATIONS = 3
MIN_SPECIES = 20          # fewer -> retry once at 4x radius
MAX_AGE = timedelta(days=90)
PER_PAGE = 500


@dataclass(frozen=True)
class LocalList:
    fetched_at: datetime
    lat: float
    lon: float
    radius_km: int
    species: list[dict]    # {id, latin, common, count}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".json.tmp{os.getpid()}")
        payload = {**asdict(self), "fetched_at": self.fetched_at.isoformat()}
        tmp.write_text(json.dumps(payload, indent=1))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "LocalList | None":
        try:
            d = json.loads(Path(path).read_text())
            return cls(datetime.fromisoformat(d["fetched_at"]), float(d["lat"]), float(d["lon"]), int(d["radius_km"]), list(d["species"]))
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def ids(self) -> frozenset[int]:
        return frozenset(int(s["id"]) for s in self.species)


def _http_get(lat: float, lon: float, radius_km: int, page: int) -> dict:
    q = urllib.parse.urlencode({"lat": lat, "lng": lon, "radius": radius_km, "taxon_id": 3,
                                "quality_grade": "research", "locale": "en", "per_page": PER_PAGE, "page": page})
    req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.load(resp)


def parse_species_counts(pages: Iterable[dict]) -> list[dict]:
    out = []
    for page in pages:
        for r in page.get("results", []):
            t = r.get("taxon") or {}
            if t.get("rank") != "species" or int(r.get("count", 0)) < MIN_OBSERVATIONS:
                continue
            out.append({"id": int(t["id"]), "latin": t["name"], "common": t.get("preferred_common_name") or t["name"], "count": int(r["count"])})
    return out


def fetch_species_counts(lat: float, lon: float, radius_km: int, http: Callable = _http_get, sleep: Callable = time.sleep) -> list[dict]:
    """All pages for the radius; retried once at 4x radius when fewer than MIN_SPECIES come back."""
    def pages(radius):
        got, page = [], 1
        while True:
            data = http(lat, lon, radius, page)
            got.append(data)
            if len(data.get("results", [])) < PER_PAGE:
                return got
            page += 1
            sleep(1.0)
    rows = parse_species_counts(pages(radius_km))
    if len(rows) < MIN_SPECIES:
        sleep(1.0)
        rows = parse_species_counts(pages(radius_km * 4))
    return rows


def needs_refresh(current: LocalList | None, lat: float, lon: float, radius_km: int, now: datetime) -> bool:
    if current is None:
        return True
    moved = abs(current.lat - lat) > 1e-6 or abs(current.lon - lon) > 1e-6 or current.radius_km != radius_km
    return moved or now - current.fetched_at > MAX_AGE


class SpeciesList:
    """Holds the current list and turns it + include/exclude into candidate ids for the bank."""

    def __init__(self, data_dir: Path, bank: Bank) -> None:
        self.path = Path(data_dir) / "species" / "local.json"
        self.bank = bank
        self.version = 0
        self._lock = threading.Lock()
        self._current: LocalList | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._warned: set[str] = set()
        self.reload()

    def reload(self) -> None:
        with self._lock:
            self._current = LocalList.load(self.path)
            self.version += 1

    def current(self) -> LocalList | None:
        with self._lock:
            return self._current

    def unknown_names(self, names: Iterable[str]) -> list[str]:
        return [n for n in names if self.bank.resolve(n) is None]

    def candidates(self, species_cfg) -> frozenset[int] | None:
        """Local ids + include − exclude; None when there is no list and nothing configured (whole bank)."""
        cur = self.current()
        if cur is None and not species_cfg.include and not species_cfg.exclude:
            return None
        ids = set(cur.ids()) if cur is not None else set(int(i) for i in self.bank.ids)
        for name in species_cfg.include:
            tid = self.bank.resolve(name)
            if tid is None:
                self._warn(name)
            else:
                ids.add(tid)
        for name in species_cfg.exclude:
            tid = self.bank.resolve(name)
            if tid is None:
                self._warn(name)
            else:
                ids.discard(tid)
        return frozenset(ids)

    def _warn(self, name: str) -> None:
        if name not in self._warned:
            log.warning("species.include/exclude: unknown name %r (not in the bank); ignored", name)
            self._warned.add(name)

    def refresh_if_needed(self, lat: float | None, lon: float | None, radius_km: int, force: bool = False) -> bool:
        """Fetch when the file is missing, stale or for another place (or `force`); True if a new list was saved."""
        if lat is None or lon is None:
            return False
        if not force and not needs_refresh(self.current(), lat, lon, radius_km, datetime.now(timezone.utc)):
            return False
        try:
            rows = fetch_species_counts(lat, lon, radius_km)
        except Exception as e:
            log.warning("species list fetch failed (%s); keeping the current list", e)
            return False
        LocalList(datetime.now(timezone.utc), lat, lon, radius_km, rows).save(self.path)
        self.reload()
        log.info("species list: %d species within %d km", len(rows), radius_km)
        return True

    def start_refresh(self, lat: float | None, lon: float | None, radius_km: int) -> None:
        self._thread = threading.Thread(target=self.refresh_if_needed, args=(lat, lon, radius_km), name="species-list", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout)
```

- [ ] **Step 5: Run to verify pass** → `.venv/bin/pytest tests/test_species_list.py -q` all pass.

- [ ] **Step 6: Commit**

```bash
git add birdy/species_list.py tests/test_species_list.py tests/fixtures/species_counts.json
git commit -m "Species list: iNaturalist local species within region_radius_km, include/exclude, 90-day refresh"
```

---

### Task 5: Store — `meta` table and species-model reset

**Files:** `birdy/store.py`, `tests/test_store.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_store.py`):

```python
def test_meta_roundtrip(store):
    assert store.get_meta("species_model") is None
    store.set_meta("species_model", "bioclip-2/int8/abc")
    assert store.get_meta("species_model") == "bioclip-2/int8/abc"
    store.set_meta("species_model", "x")
    assert store.get_meta("species_model") == "x"


def test_reset_species_for_model_marks_uncorrected_done_birds(store):
    bird = add_species(store, T0, "Black Phoebe", 0.8)
    nonbird = add(store, T0 + timedelta(minutes=1)); store.update_classification(nonbird, False, 0.1, None, None)
    corrected = add_species(store, T0 + timedelta(minutes=2), "Robin", 0.9); store.mark_not_bird(corrected)
    pending = add(store, T0 + timedelta(minutes=3))
    assert store.reset_species_for_model("bioclip-2/int8/abc") == 1
    assert store.get(bird).species_checked is False and store.get(bird).species == "Black Phoebe"
    assert store.get(nonbird).species_checked is True and store.get(corrected).species_checked is True
    assert store.get(pending).status == "pending"
    assert store.reset_species_for_model("bioclip-2/int8/abc") == 0        # same model: idempotent
    store.update_classification(bird, True, 0.8, "Black Redstart", 0.9)
    assert store.reset_species_for_model("bioclip-2/int8/def") == 1        # new model: again
```

(`add_species` exists in the file; check its signature and adapt.)

- [ ] **Step 2: Run to verify failure** → `AttributeError: 'Store' object has no attribute 'get_meta'`.

- [ ] **Step 3: Implement** in `birdy/store.py`: add to `SCHEMA`:

```sql
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
```

and methods after `site()`:

```python
    # --- meta -------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else row["value"]

    def set_meta(self, key: str, value: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def reset_species_for_model(self, model_id: str) -> int:
        """A new species model: every uncorrected, classified bird gets its species re-run by the
        backfill. Returns how many rows were marked; 0 when the model is unchanged."""
        if self.get_meta("species_model") == model_id:
            return 0
        with self._lock, self._conn:
            n = self._conn.execute(
                "UPDATE captures SET species_checked = 0 WHERE status = 'done' AND is_bird = 1 AND manual_at IS NULL"
            ).rowcount
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES ('species_model', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (model_id,),
            )
        return int(n)
```

- [ ] **Step 4: Run to verify pass** → `.venv/bin/pytest tests/test_store.py -q`.

- [ ] **Step 5: Commit**

```bash
git add birdy/store.py tests/test_store.py
git commit -m "Store: meta table; a new species model resets species_checked on uncorrected birds"
```

---

### Task 6: Pipeline — renames move publish slots

**Files:** `birdy/pipeline.py`, `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests** (append; `FakePublisher` in the file records `evaluated`; add a `forgotten` list to it if missing: `def forget(self, c): self.forgotten.append(c.id)`):

```python
def test_backfill_rename_forgets_old_species_then_evaluates_new(store, live):
    ev = add(store, 0)
    store.update_classification(ev.capture_id, True, 0.9, "Black Phoebe", 0.8)
    unchecked(store, [ev.capture_id])                      # species pass pending (new model)
    pub = FakePublisher()
    Pipeline(store, ReturnsFake(Prediction(is_bird=True, confidence=0.9, species="Black Redstart", species_confidence=0.95)), live, publisher=pub).sweep()
    c = store.get(ev.capture_id)
    assert c.species == "Black Redstart" and c.species_confidence == 0.95
    assert pub.forgotten == [ev.capture_id] and pub.evaluated == [ev.capture_id]
    assert pub.order == ["forget", "evaluate"]


def test_backfill_rename_to_nothing_forgets_old_species(store, live):
    ev = add(store, 0)
    store.update_classification(ev.capture_id, True, 0.9, "Black Phoebe", 0.8)
    unchecked(store, [ev.capture_id])
    pub = FakePublisher()
    Pipeline(store, ReturnsFake(Prediction(is_bird=True, confidence=0.9)), live, publisher=pub).sweep()
    c = store.get(ev.capture_id)
    assert c.species is None and pub.forgotten == [ev.capture_id] and pub.evaluated == []


def test_backfill_same_species_does_not_forget(store, live):
    ev = add(store, 0)
    store.update_classification(ev.capture_id, True, 0.9, "Robin", 0.8)
    unchecked(store, [ev.capture_id])
    pub = FakePublisher()
    Pipeline(store, ReturnsFake(Prediction(is_bird=True, confidence=0.9, species="Robin", species_confidence=0.85)), live, publisher=pub).sweep()
    assert pub.forgotten == [] and pub.evaluated == [ev.capture_id]
```

Give `FakePublisher` an `order` list appended with `"forget"`/`"evaluate"` in each method.

- [ ] **Step 2: Run to verify failure** → `AttributeError: 'FakePublisher' object has no attribute 'forgotten'` / assertions.

- [ ] **Step 3: Implement** in `_backfill` (`birdy/pipeline.py`): replace the species-write rule with

```python
            # Species: a row the species model hasn't seen with the current model (species_checked = 0)
            # takes the new verdict, even "no species"; a seen row keeps its label (the crop may have
            # changed with the ROI; rewriting would move publish slots for nothing). A promoted
            # non-bird gets one like a new row.
            write_species = (not c.species_checked) or promoted
            if write_species:
                species, species_confidence = pred.species, pred.species_confidence
```

(drop the earlier `and bool(pred.species or not c.is_bird)` clause: with a new model, a bird whose re-run has no species must lose its old wrong name). After the store write:

```python
        renamed = write_species and bool(c.species) and c.species != species
        if self._publisher is not None and renamed:
            try:
                self._publisher.forget(c)          # the old species/day loses this row's slot
            except Exception:
                log.exception("publish forget failed for capture %d", c.id)
        got_species = write_species and bool(pred.species)
        if self._publisher is not None and got_species:
            try:
                self._publisher.evaluate(self._store.get(c.id))
            except Exception:
                log.exception("publish decision failed for capture %d", c.id)
        return got_species, promoted
```

Check `test_backfill_keeps_a_birds_species_when_the_rerun_has_none` from the detector plan: it used `detector_unchecked` (species_checked = 1) so it still holds; the earlier `test_backfill_writes_species_for_unseen_rows` holds too.

- [ ] **Step 4: Run to verify pass** → `.venv/bin/pytest tests/test_pipeline.py -q`.

- [ ] **Step 5: Commit**

```bash
git add birdy/pipeline.py tests/test_pipeline.py
git commit -m "Pipeline: a renamed bird forgets its old species' slot before the new one is ranked"
```

---

### Task 7: CLI — build BioCLIP, species list thread, model reset, `species-list`, legacy removal

**Files:** `birdy/cli.py`, `birdy/species.py` (remove `InatBirdClassifier`, `parse_label`, the TFLite import), `birdy/config.py` (remove `labels`), `birdy/settings.py` (`config_to_dict` labels line), `birdy/watcher.py` (no change), `tests/test_cli.py`, `tests/test_species.py` (remove iNat tests), `tests/test_config.py` (labels assertion), `tests/test_settings.py` (FULL_TOML `labels` line if any), `models/download.sh` + `models/README.md` (drop the iNat model lines), delete `models/mobilenet_v2_1.0_224_inat_bird_quant.tflite` and `models/inat_bird_labels.txt`

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py`: in `DETECTOR_TOML`-using configs the `[species]` lines currently point at the iNat model; change `_classify_config(tmp_path, species_model)` to write `[species]\nmodel = "{species_model}"\nbank = "{ROOT}/models/bioclip2-birds.npz"\n` and make the two classify-output tests skip unless `ROOT/models/bioclip2-vision-int8.onnx` and the bank exist (`pytest.skip("BioCLIP model not built")`). Update `test_missing_species_model_exits_2` to expect `"model error"` still. Change `assert "species (iNat):" in proc.stdout` to `assert "species (BioCLIP 2" in proc.stdout`. Add:

```python
def test_classify_cat_skips_species(tmp_path):
    cfg = _classify_config(tmp_path, f"{ROOT}/models/bioclip2-vision-int8.onnx")
    if not (ROOT / "models" / "bioclip2-vision-int8.onnx").exists():
        pytest.skip("BioCLIP model not built")
    proc = _run_cli("--config", str(cfg), "classify", str(FIXTURES / "cat.jpg"))
    assert proc.returncode == 0 and "species: skipped (not a bird)" in proc.stdout


def test_parser_species_list():
    assert build_parser().parse_args(["species-list"]).refresh is False
    assert build_parser().parse_args(["species-list", "--refresh"]).refresh is True


def test_cmd_species_list_prints_list_and_refreshes(capsys, config, monkeypatch, tmp_path):
    from birdy import cli
    from birdy.species_list import LocalList
    from datetime import datetime, timezone
    import dataclasses
    from birdy.config import SiteConfig, StorageConfig
    config = dataclasses.replace(config, site=SiteConfig("t", 38.8, -9.39), storage=StorageConfig(tmp_path / "data"))
    bank = tmp_path / "bank.npz"
    np.savez(bank, ids=np.array([13094], np.int32), latin=np.array(["Phoenicurus ochruros"]), common=np.array(["Black Redstart"]),
             emb=np.eye(1, 4, dtype=np.float16), logit_scale=np.float32(100.0), model=np.str_("x"), prompt=np.str_("p"))
    config = dataclasses.replace(config, species=dataclasses.replace(config.species, bank=bank))
    LocalList(datetime.now(timezone.utc), 38.8, -9.39, 50, [{"id": 13094, "latin": "Phoenicurus ochruros", "common": "Black Redstart", "count": 3}]).save(tmp_path / "data" / "species" / "local.json")
    assert cli.cmd_species_list(config, refresh=False) == 0
    out = capsys.readouterr().out
    assert "1 species within 50 km" in out and "Black Redstart" in out
    called = {}
    monkeypatch.setattr("birdy.species_list.fetch_species_counts", lambda lat, lon, r, **kw: called.setdefault("r", r) and [])
    assert cli.cmd_species_list(config, refresh=True) == 0
    assert called["r"] == 50


def test_cmd_species_list_without_location_exits_1(capsys, config):
    from birdy import cli
    assert cli.cmd_species_list(config, refresh=False) == 1
    assert "site.lat" in capsys.readouterr().err
```

(`np` import at the top of the test file if missing.) In `test_cmd_run_passes_live_everywhere`, the fake classifier `clf = object()` must now also satisfy `cmd_run`'s calls: use

```python
    class Clf:
        class species:
            model_id = "test/1"
        species_list = None
    clf = Clf()
```

and assert afterwards that `store.get_meta("species_model") == "test/1"` is not required (the store is internal to cmd_run) — instead assert the log contains `"species model"` via caplog. Keep it minimal: assert `rc == 0` as before.

`tests/test_species.py`: delete the `parse_label` tests, the `clf` fixture and `TestRealModel`; drop the `InatBirdClassifier, parse_label` import.
`tests/test_config.py`: drop the `labels` assertion. `tests/test_settings.py`: FULL_TOML has no `labels` line (check).

- [ ] **Step 2: Run to verify failure** → `AttributeError: module 'birdy.cli' has no attribute 'cmd_species_list'` etc.

- [ ] **Step 3: Implement**

`birdy/species.py`: remove `InatBirdClassifier`, `parse_label`, `_LABEL_RE`, `re`, and the `load_interpreter` import; module docstring "Bird species recognition: BioCLIP 2 zero-shot over a bank of iNaturalist bird names."

`birdy/config.py`: remove `labels` from `SpeciesConfig` and the constructor. `birdy/settings.py` `config_to_dict["species"]`: remove `"labels"`. `tests/test_config.py::test_species_defaults`: remove the labels line.

`birdy/cli.py`:

```python
from birdy.species import BioClipSpecies
from birdy.species_list import SpeciesList


def _classifier(config: Config, live: LiveConfig) -> TwoStageClassifier:
    try:
        gate = ImageNetBirdClassifier(...)                      # unchanged
        species = BioClipSpecies(config.species.model, config.species.bank)
        detector = BirdDetector(config.detector.model, config.detector.labels)
    except (ValueError, OSError, RuntimeError) as e:
        raise ModelLoadError(str(e)) from e
    species_list = SpeciesList(config.storage.data_dir, species.bank)
    return TwoStageClassifier(gate, species, live, lores_size=config.camera.lores_size, detector=detector, species_list=species_list)
```

(`TwoStageClassifier` exposes it as `clf.species_list`, set in Task 3.)

`cmd_run`, after `clf = _classifier(config, live)`:

```python
    n = store.reset_species_for_model(clf.species.model_id)
    if n:
        log.info("species model %s is new: %d birds will be renamed by the backfill", clf.species.model_id, n)
    clf.species_list.start_refresh(config.site.lat, config.site.lon, config.species.region_radius_km)
```

and in the `finally`: `clf.species_list.stop()` before `store.close()`. Guard both with `getattr(clf, "species_list", None)` so test doubles work, and `getattr(getattr(clf, "species", None), "model_id", None)` likewise (skip the reset when absent).

`cmd_classify`: species block becomes

```python
    if pred.is_bird:
        print(f"species (BioCLIP 2, {clf.species.candidate_count} candidates):")
        for label, prob in top:
            print(f"  {prob:.3f}  {label.common} ({label.latin})")
    else:
        print("species: skipped (not a bird)")
```

`species-list` command — parser: `sl = sub.add_parser("species-list", help="print the local species list (from iNaturalist); --refresh fetches it now"); sl.add_argument("--refresh", action="store_true")`; dispatch `if args.command == "species-list": return cmd_species_list(config, args.refresh)`; implementation:

```python
def cmd_species_list(config: Config, refresh: bool = False) -> int:
    from datetime import datetime, timezone

    from birdy.species import load_bank

    if config.site.lat is None or config.site.lon is None:
        print("species-list: set site.lat and site.lon in birdy.toml to enable the local species list", file=sys.stderr)
        return 1
    try:
        bank = load_bank(config.species.bank)
    except ValueError as e:
        print(f"model error: {e}", file=sys.stderr)
        return 2
    sl = SpeciesList(config.storage.data_dir, bank)
    if refresh or sl.current() is None:
        fetched = sl.refresh_if_needed(config.site.lat, config.site.lon, config.species.region_radius_km, force=refresh)
        if not fetched and sl.current() is None:
            print("species-list: fetch failed and no list is stored", file=sys.stderr)
            return 1
    cur = sl.current()
    age = (datetime.now(timezone.utc) - cur.fetched_at).days
    print(f"{len(cur.species)} species within {cur.radius_km} km of {cur.lat:.3f},{cur.lon:.3f} (fetched {age} days ago)")
    for s in sorted(cur.species, key=lambda s: -s["count"]):
        print(f"  {s['count']:6d}  {s['common']} ({s['latin']})")
    unknown = sl.unknown_names(config.species.include + config.species.exclude)
    if unknown:
        print("unknown names in species.include/exclude: " + ", ".join(unknown))
    return 0
```

(`force=refresh` makes `--refresh` fetch even when the file is fresh; the test monkeypatches `fetch_species_counts` and expects a call with radius 50.)

`models/download.sh`: remove the two iNat curl lines and its echo; `models/README.md`: replace the iNat paragraph with a pointer to `deploy/build-species-model.sh` (written in Task 9); `git rm` the two iNat files.

- [ ] **Step 4: Run to verify pass** → `.venv/bin/pytest -q` all pass (BioCLIP-dependent tests skipped).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "CLI: BioCLIP 2 species with the local list; species-list command; iNat model removed"
```

---

### Task 8: Web — name validation, species page

**Files:** `birdy/web/app.py`, `birdy/web/templates/settings.html`, `birdy/web/templates/species.html`, `tests/test_web.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_web.py`; `FakeTwoStage` gets a `species_list` attribute and a `species` attribute with `bank.resolve`):

```python
class FakeBank:
    def resolve(self, name):
        return {"black redstart": 1, "eurasian hoopoe": 2}.get(name.strip().casefold())


class FakeSpeciesModel:
    bank = FakeBank()
    candidate_count = 354


class FakeSpeciesList:
    version = 1

    def current(self):
        from birdy.species_list import LocalList
        from datetime import datetime, timezone
        return LocalList(datetime.now(timezone.utc), 38.8, -9.39, 50, [{"id": 1, "latin": "Phoenicurus ochruros", "common": "Black Redstart", "count": 3}])

    def unknown_names(self, names):
        return [n for n in names if FakeBank().resolve(n) is None]


# in FakeTwoStage.__init__: self.species = FakeSpeciesModel(); self.species_list = FakeSpeciesList()


def test_settings_rejects_unknown_species_names(client):
    r = client.post("/settings", data={**_base_form(), "species_include": "Dodo"}, follow_redirects=False)
    assert r.status_code == 422 and "species.include" in r.text and "Dodo" in r.text
    r = client.post("/settings", data={**_base_form(), "watchlist": "Black Redstart\nUnicorn"}, follow_redirects=False)
    assert r.status_code == 422 and "telegram.watchlist" in r.text and "Unicorn" in r.text
    r = client.post("/settings", data={**_base_form(), "species_include": "eurasian hoopoe"}, follow_redirects=False)
    assert r.status_code == 303


def test_settings_shows_species_list_state(client):
    r = client.get("/settings").text
    assert "1 species within 50 km" in r


def test_species_page_shows_list_and_skipped_result(client, store):
    assert "354 candidates" in client.get("/species").text
    data = (Path(__file__).parent / "fixtures" / "cat.jpg").read_bytes()
    # FakeTwoStage returns a bird; add a not-a-bird fake for this one
```

Replace the last test's tail with a dedicated fake: use the existing `FakeNotABird` class (returns `is_bird=False`) and assert `"species: skipped" in r.text` on the `/classify` response (the result dict's `top` is `[]`, so the page shows the skipped note instead of guesses).

- [ ] **Step 2: Run to verify failure** → assertions fail (no validation, no state text).

- [ ] **Step 3: Implement**

`birdy/web/app.py` `settings_post`: after `patch = parse_settings_form(form)` add

```python
            sl = getattr(classifier, "species_list", None)
            if sl is not None:
                for key, names in (("species.include", patch.species_include), ("species.exclude", patch.species_exclude), ("telegram.watchlist", patch.watchlist)):
                    unknown = sl.unknown_names(names)
                    if unknown:
                        raise ConfigError(f"{key}: unknown species {', '.join(unknown)} (not in the name bank)")
```

(inside the existing `try`, so the field highlighting works via `FIELD_KEYS`).

`settings_context`: add `"species_list": _species_list_state()` where

```python
    def _species_list_state() -> str:
        sl = getattr(classifier, "species_list", None)
        cur = sl.current() if sl is not None else None
        if cur is None:
            return "no local list — set a location to enable it" if live.get().site.lat is None else "no local list yet (fetching at start)"
        age = (datetime.now(timezone.utc) - cur.fetched_at).days
        return f"{len(cur.species)} species within {cur.radius_km} km (fetched {age} days ago)"
```

`settings.html`: under the species threshold row's section header, add `<p class="hint">Local species list: {{ species_list }}</p>` (before the include textarea).

`species_context`: add `"candidates": getattr(getattr(classifier, "species", None), "candidate_count", None)`; `species.html` header subtitle: `Runs gate, detector and species{% if candidates %} ({{ candidates }} candidates){% endif %}. Nothing is stored.`; in the readout, replace the `{% if result.top %}` block's inner `key` text with: when `not result.is_bird` show `<span class="key">species: skipped (not a bird)</span>` (and no guesses), else the guesses as today.

- [ ] **Step 4: Run to verify pass** → `.venv/bin/pytest tests/test_web.py -q`.

- [ ] **Step 5: Commit**

```bash
git add birdy/web tests/test_web.py
git commit -m "Web: unknown species names rejected on save; species-list state; skipped species shown"
```

---

### Task 9: Mac build script, artefacts, real-model tests

**Files:**
- Create: `deploy/build_species_model.py`, `deploy/build-species-model.sh`
- Modify: `models/README.md`, `.gitignore`

- [ ] **Step 1: `.gitignore`** — append:

```
models/bioclip2-vision-int8.onnx
models/bioclip2-birds.npz
.build-venv/
```

- [ ] **Step 2: `deploy/build-species-model.sh`**

```bash
#!/usr/bin/env bash
# Build the BioCLIP 2 species artefacts on the Mac (needs ~3 GB of downloads, ~15 min):
#   models/bioclip2-vision-int8.onnx  (image tower, int8)
#   models/bioclip2-birds.npz         (name embeddings for every iNaturalist bird species)
# Both are gitignored; deploy/deploy.sh rsyncs models/ to the Pi.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .build-venv ] || /opt/homebrew/bin/python3.11 -m venv .build-venv
.build-venv/bin/pip install -q --upgrade pip
.build-venv/bin/pip install -q torch torchvision open_clip_torch onnx onnxruntime numpy
.build-venv/bin/python -I deploy/build_species_model.py "$@"
```

- [ ] **Step 3: `deploy/build_species_model.py`**

```python
"""Export BioCLIP 2's image tower to int8 ONNX and embed every iNaturalist bird name.

Usage: python -I deploy/build_species_model.py [--skip-export] [--skip-bank] [--out models]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import torch

HF = "hf-hub:imageomics/bioclip-2"
TAXA = "https://api.inaturalist.org/v1/taxa"
UA = "birdy (+https://birdy.akibeo.com)"
PROMPT = "a photo of {latin} ({common})."
PROMPT_NO_COMMON = "a photo of {latin}."


def export_onnx(model, out: Path) -> None:
    class Vision(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m.encode_image(x)

    fp32 = out.with_name("bioclip2-vision-fp32.onnx")
    torch.onnx.export(Vision(model).eval(), torch.zeros(1, 3, 224, 224), str(fp32), input_names=["pixels"],
                      output_names=["embedding"], opset_version=17, dynamo=False)
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(fp32), str(out), weight_type=QuantType.QInt8)
    fp32.unlink()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")


def fetch_bird_taxa() -> list[dict]:
    """Every active species under Aves (taxon 3) via id_above paging (the page*per_page cap is 10,000)."""
    out, last = [], 0
    while True:
        q = urllib.parse.urlencode({"taxon_id": 3, "rank": "species", "is_active": "true", "per_page": 200,
                                    "order_by": "id", "order": "asc", "id_above": last})
        req = urllib.request.Request(f"{TAXA}?{q}", headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
        results = data.get("results", [])
        if not results:
            break
        for t in results:
            out.append({"id": int(t["id"]), "latin": t["name"], "common": t.get("preferred_common_name") or ""})
        last = max(int(t["id"]) for t in results)
        print(f"  taxa: {len(out)} (last id {last})", file=sys.stderr)
        time.sleep(1.0)
    return out


def build_bank(model, tokenizer, taxa: list[dict], model_id: str, out: Path) -> None:
    prompts = [PROMPT.format(latin=t["latin"], common=t["common"]) if t["common"] else PROMPT_NO_COMMON.format(latin=t["latin"]) for t in taxa]
    embs = []
    with torch.no_grad():
        for i in range(0, len(prompts), 64):
            e = model.encode_text(tokenizer(prompts[i:i + 64]))
            embs.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float16))
            if i % 1024 == 0:
                print(f"  embeddings: {i}/{len(prompts)}", file=sys.stderr)
    np.savez(
        out,
        ids=np.array([t["id"] for t in taxa], np.int32),
        latin=np.array([t["latin"] for t in taxa]),
        common=np.array([t["common"] for t in taxa]),
        emb=np.vstack(embs),
        logit_scale=np.float32(float(model.logit_scale.exp())),
        model=np.str_(model_id),
        prompt=np.str_(PROMPT),
    )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB, {len(taxa)} species)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--skip-bank", action="store_true")
    ap.add_argument("--taxa-cache", default=".build-venv/bird_taxa.json")
    args = ap.parse_args()
    import open_clip

    out = Path(args.out)
    onnx_path = out / "bioclip2-vision-int8.onnx"
    model, _, _ = open_clip.create_model_and_transforms(HF)
    model.eval()
    if not args.skip_export:
        export_onnx(model, onnx_path)
    model_id = "bioclip-2/int8/" + hashlib.sha256(onnx_path.read_bytes()).hexdigest()[:12]
    print("model id", model_id)
    if not args.skip_bank:
        cache = Path(args.taxa_cache)
        if cache.exists():
            taxa = json.loads(cache.read_text())
        else:
            taxa = fetch_bird_taxa()
            cache.write_text(json.dumps(taxa))
        build_bank(model, open_clip.get_tokenizer(HF), taxa, model_id, out / "bioclip2-birds.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Verify the taxa endpoint before the long run**

Run: `curl -s -A "birdy (+https://birdy.akibeo.com)" "https://api.inaturalist.org/v1/taxa?taxon_id=3&rank=species&is_active=true&per_page=3&order_by=id&order=asc&id_above=0" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['total_results'], [(t['id'], t['name'], t.get('preferred_common_name')) for t in d['results']])"`
Expected: a total around 11,000 and three species with ids ascending. If `id_above` is not honoured (same ids on a second call with a larger `id_above`), stop and report — the fetch must then use the iNaturalist open-data `taxa.csv` export instead.

- [ ] **Step 5: Build** — `chmod +x deploy/build-species-model.sh && deploy/build-species-model.sh` (15–30 min). Expected: both files in `models/`, `model id bioclip-2/int8/…` printed, bank ~17 MB with ~10,900 species.

- [ ] **Step 6: Real-model tests** — Run: `.venv/bin/pytest tests/test_species.py tests/test_cli.py -q` → the BioCLIP-dependent tests now run and pass (bird.jpg gets a bird name; cat.jpg "skipped").

- [ ] **Step 7: `models/README.md`** — replace the iNat paragraph with:

```markdown
`bioclip2-vision-int8.onnx` + `bioclip2-birds.npz` are built on the Mac by
`deploy/build-species-model.sh` (BioCLIP 2, imageomics/bioclip-2, MIT; image tower
exported to ONNX and dynamically quantised to int8, ~292 MB; name embeddings for every
active iNaturalist bird species, ~17 MB). They are gitignored (GitHub size limit) and
reach the Pi through `deploy/deploy.sh`. The bank's `model` field must match the onnx's
sha256 prefix; rebuild both together. Species are scored zero-shot against the site's
local list (`data/species/local.json`, see CLAUDE.md).
```

- [ ] **Step 8: Commit**

```bash
git add deploy/build_species_model.py deploy/build-species-model.sh models/README.md .gitignore
git commit -m "Build: BioCLIP 2 int8 ONNX export and iNaturalist bird-name bank (Mac-side)"
```

---

### Task 10: Docs, full suite, deploy notes

**Files:** `CLAUDE.md`

- [ ] **Step 1: CLAUDE.md**
  - Commands: `.venv/bin/birdy species-list [--refresh]   # the site's iNaturalist species list (candidates for naming)`; a Mac line `deploy/build-species-model.sh   # one-off: BioCLIP 2 ONNX + name bank into models/ (gitignored, ~310 MB)`.
  - Architecture classifier bullet: species = `BioClipSpecies` (`birdy/species.py`, BioCLIP 2 int8 ONNX via onnxruntime, zero-shot over the bank restricted to the local list + `species.include` − `exclude`), **runs only when the photo is a bird** (measured 2026-11-06: 14 % of empty frames get a confident name otherwise); no rescue; ~3.5 s per bird photo on the Pi; `confidence = max(gate, detector)`.
  - New bullet for `birdy/species_list.py` (fetch rules, file, refresh thread, `include/exclude` live, unknown names rejected by the settings page / warned from the file, no location = whole bank).
  - Backfill bullet: "a row with `species_checked = 0` always takes the new verdict; `store.reset_species_for_model` (meta table) marks every uncorrected bird when the species model id changes, so a model swap renames history (~40 min for 640 rows); a rename calls `publisher.forget(old)` then `evaluate(new)`."
  - Gotchas: `species.model/bank/region_radius_km` restart-only; `threshold/include/exclude` live; the bank and onnx must be rebuilt together; watchlist names must be iNaturalist English names; the Pi's `rescue_threshold` key is ignored; Chrome + `birdy.local`: set `use-ipv6=no` in `/etc/avahi/avahi-daemon.conf` on the Pi.
  - Remove the two iNat-model gotchas ("never re-softmax", "iNat … uint8 1/256") and update "All three models must get the same RGB array" (still true).
  - Models list: `models/` now holds the ImageNet gate, the COCO detector, and the gitignored BioCLIP files.

- [ ] **Step 2: Full suite** — `.venv/bin/pytest -q` → all pass.

- [ ] **Step 3: Commit** — `git add CLAUDE.md && git commit -m "Docs: BioCLIP 2 species and the local species list in CLAUDE.md"`.

- [ ] **Step 4: Deploy (user's terminal, since the Mac shell may not reach the LAN)** — after merge to master: `deploy/deploy.sh` (first run pushes ~310 MB), then
  ```bash
  ssh yassine@birdy.local 'journalctl -u birdy --since "-2 min" --no-pager | grep -E "species|backfill|error|Traceback"'
  ```
  Expected: `species list: N species within 50 km`, `species model bioclip-2/int8/… is new: 1xx birds will be renamed by the backfill`, then `backfill: 50 rows, … with species …` lines every 5 min, no tracebacks; `free -m` shows ~1 GB used by the service; `/species` lists Black Redstart within the hour.
