"""Bird species recognition: BioCLIP 2 zero-shot over a bank of iNaturalist bird names.
(`InatBirdClassifier` below is the legacy Coral iNat model, removed with the CLI swap.)"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from birdy.tflite import load_interpreter

_LABEL_RE = re.compile(r"^(.*?)\s*\((.*)\)\s*$")

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
INPUT = 224


@dataclass(frozen=True)
class SpeciesLabel:
    latin: str
    common: str


def parse_label(line: str) -> SpeciesLabel:
    """`Cyanistes caeruleus (Eurasian Blue Tit)` -> latin/common; a bare name is used for both."""
    line = line.strip()
    m = _LABEL_RE.match(line)
    if m:
        return SpeciesLabel(m.group(1).strip(), m.group(2).strip())
    return SpeciesLabel(line, line)


class InatBirdClassifier:
    def __init__(self, model_path: Path, labels_path: Path) -> None:
        self._interp = load_interpreter(model_path)
        self._interp.allocate_tensors()
        self._input = self._interp.get_input_details()[0]
        self._output = self._interp.get_output_details()[0]
        _, self._in_h, self._in_w, _ = self._input["shape"]
        self.labels = [parse_label(l) for l in Path(labels_path).read_text().splitlines() if l.strip()]
        n_out = int(self._output["shape"][-1])
        if len(self.labels) != n_out:
            raise ValueError(f"{labels_path}: {len(self.labels)} labels but the model has {n_out} outputs")

    def predict(self, img_rgb: np.ndarray) -> list[tuple[SpeciesLabel, float]]:
        """All classes with their probability, best first. The model's output is already
        softmax-scaled (uint8, scale 1/256), so it is only dequantised — never re-softmaxed."""
        img = cv2.resize(img_rgb, (self._in_w, self._in_h), interpolation=cv2.INTER_AREA)
        if self._input["dtype"] == np.uint8:
            tensor = img[np.newaxis].astype(np.uint8)
        else:
            tensor = (img[np.newaxis].astype(np.float32) / 127.5) - 1.0
        self._interp.set_tensor(self._input["index"], tensor)
        self._interp.invoke()
        out = self._interp.get_tensor(self._output["index"])[0]
        scale, zero_point = self._output["quantization"]
        probs = (out.astype(np.float32) - zero_point) * scale if scale else out.astype(np.float32)
        order = np.argsort(probs)[::-1]
        return [(self.labels[i], float(probs[i])) for i in order]

    def top(self, img_rgb: np.ndarray, k: int = 5, skip_background: bool = True) -> list[tuple[SpeciesLabel, float]]:
        ranked = self.predict(img_rgb)
        if skip_background:
            ranked = [(l, p) for l, p in ranked if l.latin != "background"]
        return ranked[:k]


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
    index: dict[str, int] = field(default_factory=dict, repr=False, compare=False)  # casefolded name -> id

    def resolve(self, name: str) -> int | None:
        """iNat id for a latin or common name, case-insensitive; None if unknown."""
        return self.index.get(name.strip().casefold())


def _name_index(ids: np.ndarray, latin: list[str], common: list[str]) -> dict[str, int]:
    idx: dict[str, int] = {}
    for tid, lat, com in zip(ids, latin, common):
        idx.setdefault(lat.casefold(), int(tid))
        if com:
            idx.setdefault(com.casefold(), int(tid))
    return idx


def load_bank(path: Path) -> Bank:
    try:
        z = np.load(path, allow_pickle=False)
        ids = z["ids"].astype(np.int32)
        latin = [str(s) for s in z["latin"]]
        common = [str(s) for s in z["common"]]
        return Bank(
            ids=ids,
            latin=latin,
            common=common,
            emb=z["emb"].astype(np.float32),
            logit_scale=float(z["logit_scale"]),
            model=str(z["model"]),
            prompt=str(z["prompt"]),
            index=_name_index(ids, latin, common),
        )
    except (OSError, ValueError, KeyError) as e:
        raise ValueError(f"{path}: cannot read species bank: {e}") from e


def model_id_of(onnx_path: Path) -> str:
    with open(onnx_path, "rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    return "bioclip-2/int8/" + digest[:12]


def preprocess(img_rgb: np.ndarray) -> np.ndarray:
    """CLIP preprocessing: short side to 224 (area when shrinking, bicubic when enlarging), centre crop, normalise, NCHW float32."""
    h, w = img_rgb.shape[:2]
    s = INPUT / max(1, min(h, w))
    nh, nw = max(INPUT, round(h * s)), max(INPUT, round(w * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    img = cv2.resize(img_rgb, (nw, nh), interpolation=interp)
    t, l = (nh - INPUT) // 2, (nw - INPUT) // 2
    img = img[t:t + INPUT, l:l + INPUT]
    x = (img.astype(np.float32) / 255.0 - CLIP_MEAN) / CLIP_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[np.newaxis])


class BioClipSpecies:
    """Zero-shot species over the current candidate ids. Not thread-safe; called under
    `TwoStageClassifier._lock`. ~3.5 s per image on a Pi 4 (int8 ViT-L/14).
    `set_candidates` and `top` are both called under `TwoStageClassifier._lock`."""

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
        d, bank_d = self._sess.get_outputs()[0].shape[-1], self.bank.emb.shape[1]
        if isinstance(d, int) and d != bank_d:
            raise ValueError(f"{model_path}: embedding size {d} but the bank has {bank_d}")
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
        cand = self._cand
        if len(cand) == 0:
            return []
        e = self.embed(img_rgb)
        logits = self.bank.logit_scale * (self.bank.emb[cand] @ e)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        order = np.argsort(p)[::-1][:k]
        out = []
        for i in order:
            row = int(cand[i])
            latin = self.bank.latin[row]
            out.append((SpeciesLabel(latin, self.bank.common[row] or latin), float(p[i])))
        return out
