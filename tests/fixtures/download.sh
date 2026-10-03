#!/usr/bin/env bash
# Two public-domain/CC photos used only by the classifier integration test.
set -euo pipefail
cd "$(dirname "$0")"
UA="birdy-tests/0.1 (github.com; contact via repo)"
curl -sSL -A "$UA" -o bird.jpg \
  "https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Eopsaltria_australis_-_Mogo_Campground.jpg/960px-Eopsaltria_australis_-_Mogo_Campground.jpg"
curl -sSL -A "$UA" -o cat.jpg \
  "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/960px-Cat03.jpg"
ls -la bird.jpg cat.jpg
