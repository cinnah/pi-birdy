#!/usr/bin/env bash
# Build the BioCLIP 2 species artefacts on the Mac (needs ~3 GB of downloads, ~15 min):
#   models/bioclip2-vision-int8.onnx  (image tower, int8, ~300 MB)
#   models/bioclip2-birds.npz         (name embeddings for every iNaturalist bird species, ~20 MB)
# Both are gitignored; deploy/deploy.sh rsyncs models/ to the Pi. Rebuild both together: the
# bank records the onnx's sha256 prefix and the service refuses a mismatched pair.
# Usage: deploy/build-species-model.sh [--skip-export] [--skip-bank]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .build-venv ] || /opt/homebrew/bin/python3.11 -m venv .build-venv
.build-venv/bin/pip install -q --upgrade pip
.build-venv/bin/pip install -q torch torchvision open_clip_torch onnx onnxruntime numpy
.build-venv/bin/python -I deploy/build_species_model.py "$@"
