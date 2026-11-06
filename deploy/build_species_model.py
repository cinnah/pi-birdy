"""Export BioCLIP 2's image tower to int8 ONNX and embed every iNaturalist bird name.

Run on the Mac through deploy/build-species-model.sh (needs torch + open_clip; the Pi never does):
    python -I deploy/build_species_model.py [--skip-export] [--skip-bank] [--out models] [--taxa-cache FILE]

Outputs (gitignored, rsynced to the Pi by deploy/deploy.sh):
    models/bioclip2-vision-int8.onnx   image tower, dynamically quantised to int8 (~292 MB)
    models/bioclip2-birds.npz          name embeddings for every active iNat bird species (~20 MB)
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
    """Every active species under Aves (taxon 3), paged with id_above (page*per_page is capped at 10,000)."""
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


def model_id_of(onnx_path: Path) -> str:
    with open(onnx_path, "rb") as f:
        return "bioclip-2/int8/" + hashlib.file_digest(f, "sha256").hexdigest()[:12]


def build_bank(model, tokenizer, taxa: list[dict], model_id: str, out: Path) -> None:
    prompts = [
        PROMPT.format(latin=t["latin"], common=t["common"]) if t["common"] else PROMPT_NO_COMMON.format(latin=t["latin"])
        for t in taxa
    ]
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="models")
    ap.add_argument("--skip-export", action="store_true", help="reuse an existing models/bioclip2-vision-int8.onnx")
    ap.add_argument("--skip-bank", action="store_true")
    ap.add_argument("--taxa-cache", default=".build-venv/bird_taxa.json", help="cached iNat taxa list (fetched when missing)")
    args = ap.parse_args()
    import open_clip

    out = Path(args.out)
    onnx_path = out / "bioclip2-vision-int8.onnx"
    model, _, _ = open_clip.create_model_and_transforms(HF)
    model.eval()
    if not args.skip_export:
        export_onnx(model, onnx_path)
    model_id = model_id_of(onnx_path)
    print("model id", model_id)
    if not args.skip_bank:
        cache = Path(args.taxa_cache)
        if cache.exists():
            taxa = json.loads(cache.read_text())
        else:
            taxa = fetch_bird_taxa()
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(taxa))
        build_bank(model, open_clip.get_tokenizer(HF), taxa, model_id, out / "bioclip2-birds.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
