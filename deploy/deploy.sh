#!/usr/bin/env bash
# Sync the repo to the Pi and restart the service. Usage: deploy/deploy.sh
set -euo pipefail
HOST="${BIRDY_HOST:-yassine@birdy.local}"
cd "$(dirname "$0")/.."

rsync -az --delete \
  --exclude data/ --exclude .venv/ --exclude .build-venv/ --exclude birdy.toml --exclude .git/ \
  --exclude '__pycache__/' --exclude .pytest_cache/ --exclude roi.png \
  ./ "$HOST:~/pi-birdy/"

# Update the unit file if it changed. The Pi's sudoers rule allows only the literal
# `/usr/bin/cp /home/yassine/pi-birdy/deploy/birdy.service /etc/systemd/system/birdy.service`
# and `/usr/bin/systemctl daemon-reload`, so keep that exact absolute source path and
# no extra flags. The command is single-quoted so $HOME expands on the Pi (/home/yassine).
ssh "$HOST" 'cd ~/pi-birdy && .venv/bin/pip install -q -e . && { cmp -s "$HOME/pi-birdy/deploy/birdy.service" /etc/systemd/system/birdy.service || { sudo cp "$HOME/pi-birdy/deploy/birdy.service" /etc/systemd/system/birdy.service && sudo systemctl daemon-reload; }; } && sudo systemctl restart birdy && sleep 3 && systemctl is-active birdy && journalctl -u birdy -n 5 --no-pager'
