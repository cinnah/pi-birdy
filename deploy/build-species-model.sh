#!/usr/bin/env bash
# Build the BioCLIP 2 species artefacts on the Mac (needs ~3 GB of downloads, ~15 min):
#   models/bioclip2-vision-int8.onnx  (image tower, int8, ~300 MB)
#   models/bioclip2-birds.npz         (name embeddings for every iNaturalist bird species, ~20 MB)
# Both are gitignored; deploy/deploy.sh rsyncs models/ to the Pi. Rebuild both together: the
# bank records the onnx's sha256 prefix and the service refuses a mismatched pair.
# Afterwards models/SPECIES_RELEASE holds the new files' checksums under a new release name;
# deploy/release-models.sh uploads them, then commit and push SPECIES_RELEASE (see models/README.md).
# Usage: deploy/build-species-model.sh [--skip-export] [--skip-bank]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .build-venv ] || /opt/homebrew/bin/python3.11 -m venv .build-venv
.build-venv/bin/pip install -q --upgrade pip
.build-venv/bin/pip install -q torch torchvision open_clip_torch onnx onnxruntime numpy
.build-venv/bin/python -I deploy/build_species_model.py "$@"

# Record the build: next release name (models-N+1 when the files changed) + checksums.
old=$(head -1 models/SPECIES_RELEASE); sums=$(cd models && shasum -a 256 bioclip2-vision-int8.onnx bioclip2-birds.npz)
if [ "$sums" != "$(tail -n +2 models/SPECIES_RELEASE)" ]; then
  new="models-$(( ${old#models-} + 1 ))"
  printf '%s\n%s\n' "$new" "$sums" > models/SPECIES_RELEASE
  echo "models/SPECIES_RELEASE: $old -> $new. Test it (deploy/deploy.sh), then deploy/release-models.sh, then commit and push."
fi
