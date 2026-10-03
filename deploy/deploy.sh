#!/usr/bin/env bash
# Sync the repo to the Pi and restart the service. Usage: deploy/deploy.sh
set -euo pipefail
HOST="${BIRDY_HOST:-yassine@birdy.local}"
cd "$(dirname "$0")/.."

rsync -az --delete \
  --exclude data/ --exclude .venv/ --exclude birdy.toml --exclude .git/ \
  --exclude '__pycache__/' --exclude .pytest_cache/ --exclude roi.png \
  ./ "$HOST:~/pi-birdy/"

ssh "$HOST" 'cd ~/pi-birdy && .venv/bin/pip install -q -e . && sudo systemctl restart birdy && sleep 3 && systemctl is-active birdy && journalctl -u birdy -n 5 --no-pager'
