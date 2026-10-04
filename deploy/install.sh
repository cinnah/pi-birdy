#!/usr/bin/env bash
# One-time setup on the Pi. Run from the repo root: deploy/install.sh
# First time: deploy/deploy.sh (syncs, then fails at pip - expected), then
#   ssh -t yassine@birdy.local 'cd ~/pi-birdy && deploy/install.sh'
# -t is required: sudo prompts for a password (no blanket NOPASSWD on this Pi).
set -euo pipefail
cd "$(dirname "$0")/.."

sudo apt-get update
sudo apt-get install -y python3-picamera2 python3-opencv python3-numpy python3-venv rsync

# --system-site-packages: picamera2 and cv2 come from apt, not pip.
[ -d .venv ] || python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .

[ -f birdy.toml ] || { cp birdy.toml.example birdy.toml; echo "created birdy.toml - edit site/motion settings"; }
chmod 600 birdy.toml   # holds the Telegram bot token; enforced on every run, not only on creation
mkdir -p data

# SSH key the publisher uses to rsync the gallery mirror to the server (see CLAUDE.md "Gallery setup").
if [ ! -f ~/.ssh/birdy_publish ]; then
  mkdir -p ~/.ssh && chmod 700 ~/.ssh
  ssh-keygen -t ed25519 -N "" -f ~/.ssh/birdy_publish -C "birdy-publish@$(hostname)" >/dev/null
  echo "created ~/.ssh/birdy_publish; add this line to the gallery server's authorized_keys (see CLAUDE.md):"
  echo "command=\"/usr/bin/rrsync /srv/birdy/sites/home\",restrict $(cat ~/.ssh/birdy_publish.pub)"
fi

sudo cp deploy/birdy.service /etc/systemd/system/birdy.service
sudo systemctl daemon-reload
sudo systemctl enable birdy
echo "installed. start with: sudo systemctl start birdy"
