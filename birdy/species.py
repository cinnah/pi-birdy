"""Bird species recognition: BioCLIP 2 zero-shot over a bank of iNaturalist bird names."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
INPUT = 224


@dataclass(frozen=True)
class SpeciesLabel:
    latin: str
    common: str


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
    reported_names: frozenset[str] = field(default_factory=frozenset, repr=False, compare=False)  # casefolded `common or latin`: what a capture's species says

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
            reported_names=frozenset((com or lat).casefold() for lat, com in zip(latin, common)),
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


def score_candidates(bank: Bank, cand_rows: np.ndarray, emb: np.ndarray) -> list[tuple[int, float]]:
    """Softmax of logit_scale · E[cand] · emb over the candidate rows; [(bank row, p)] best first."""
    if len(cand_rows) == 0:
        return []
    logits = bank.logit_scale * (bank.emb[cand_rows] @ emb)
    p = np.exp(logits - logits.max())
    p /= p.sum()
    order = np.argsort(p)[::-1]
    return [(int(cand_rows[i]), float(p[i])) for i in order]


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
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
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

    def top_with_embedding(
        self, img_rgb: np.ndarray, k: int = 5
    ) -> tuple[list[tuple[SpeciesLabel, float]], np.ndarray | None]:
        """`top` plus the image's L2-normalised embedding (None when there are no candidates)."""
        cand = self._cand
        if len(cand) == 0:
            return [], None
        e = self.embed(img_rgb)
        out = []
        for row, p in score_candidates(self.bank, cand, e)[:k]:
            latin = self.bank.latin[row]
            out.append((SpeciesLabel(latin, self.bank.common[row] or latin), p))
        return out, e

    def top(self, img_rgb: np.ndarray, k: int = 5) -> list[tuple[SpeciesLabel, float]]:
        return self.top_with_embedding(img_rgb, k)[0]
