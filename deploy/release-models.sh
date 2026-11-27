#!/usr/bin/env bash
# Publish the species model in models/ as the GitHub release named in models/SPECIES_RELEASE
# (deploy/install.sh downloads it from there). Run on the Mac after deploy/build-species-model.sh;
# needs the `gh` CLI, logged in. Then commit and push models/SPECIES_RELEASE: Pis update on their
# next `git pull && deploy/install.sh`. Usage: deploy/release-models.sh
set -euo pipefail
cd "$(dirname "$0")/.."
tag=$(head -1 models/SPECIES_RELEASE)
(cd models && tail -n +2 SPECIES_RELEASE | shasum -a 256 -c) || { echo "models/ does not match models/SPECIES_RELEASE" >&2; exit 1; }
gh release create "$tag" models/bioclip2-vision-int8.onnx models/bioclip2-birds.npz models/LICENSE \
  --title "Species model ($tag)" --notes "The species model deploy/install.sh downloads: BioCLIP 2 (Imageomics Institute, MIT) with its image tower exported to ONNX and quantised to int8, and its text embeddings of iNaturalist's bird species names. These are modified forms of BioCLIP 2, not the original files; LICENSE holds the notices. Not needed by hand: the installer fetches and checks them."
