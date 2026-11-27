#!/usr/bin/env bash
# Setup on the Pi, for any user and any folder. Run it from the repo: deploy/install.sh
# It is safe to run again (after an update, or when a step failed).
# Over ssh use `ssh -t <user>@<pi> 'cd ~/pi-birdy && deploy/install.sh'`: sudo asks for a password.
# Everything after this script is done on the settings page it prints at the end.
set -euo pipefail
cd "$(dirname "$0")/.."

sudo apt-get update
sudo apt-get install -y python3-picamera2 python3-opencv python3-numpy python3-venv rsync gphoto2

# --system-site-packages: picamera2 and cv2 come from apt, not pip.
[ -d .venv ] || python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .

# The small models (bird check, detector) download from their public sources.
[ -s models/mobilenet_v2_1.0_224_quant.tflite ] && [ -s models/efficientdet_lite0_320_ptq.tflite ] || models/download.sh

# The species model (~310 MB, too large for git) comes from this project's release page.
# models/SPECIES_RELEASE names the release this code wants (line 1) and the sha256 of its two
# files; a file that is missing or differs is downloaded again, so `git pull` + this script
# also updates the model. Both files are checked before either is replaced: they only work as a pair.
# BIRDY_MODELS_URL = your own source: then only missing files are fetched and nothing is checked.
MODELS_TAG=$(head -1 models/SPECIES_RELEASE)
species_ok() { [ "$(cd models && sha256sum "bioclip2-vision-int8.onnx$1" "bioclip2-birds.npz$1" 2>/dev/null | awk '{print $1}')" = "$(tail -n +2 models/SPECIES_RELEASE | awk '{print $1}')" ]; }
if [ -n "${BIRDY_MODELS_URL:-}" ]; then
  for f in bioclip2-vision-int8.onnx bioclip2-birds.npz; do
    [ -s "models/$f" ] || { echo "downloading models/$f ..."; curl -fL --retry 3 -o "models/$f.part" "$BIRDY_MODELS_URL/$f" && mv "models/$f.part" "models/$f"; } || {
      rm -f "models/$f.part"; echo "could not download $f from $BIRDY_MODELS_URL" >&2; exit 1; }
  done
elif ! species_ok ""; then
  MODELS_URL="https://github.com/cinnah/pi-birdy/releases/download/$MODELS_TAG"
  echo "downloading the species model ($MODELS_TAG, ~310 MB) ..."
  for f in bioclip2-vision-int8.onnx bioclip2-birds.npz; do
    curl -fL --retry 3 -o "models/$f.part" "$MODELS_URL/$f" || {
      rm -f models/*.part
      echo "could not download $f from $MODELS_URL" >&2
      [ -s "models/$f" ] && echo "keeping the species model already here; run this script again later." >&2 && break
      echo "birdy cannot start without it. Copy it into models/ from another birdy, or set BIRDY_MODELS_URL, then run this script again." >&2
      exit 1
    }
  done
  if [ -f models/bioclip2-birds.npz.part ]; then
    species_ok ".part" || { rm -f models/*.part; echo "the downloaded species model does not match models/SPECIES_RELEASE; nothing was replaced." >&2; exit 1; }
    for f in bioclip2-vision-int8.onnx bioclip2-birds.npz; do mv "models/$f.part" "models/$f"; done
  fi
fi

[ -f birdy.toml ] || { cp birdy.toml.example birdy.toml; echo "created birdy.toml"; }
chmod 600 birdy.toml   # holds the Telegram bot token; enforced on every run, not only on creation
mkdir -p data

# SSH key the publisher uses to rsync the gallery mirror to the server (see CLAUDE.md "Gallery setup").
if [ ! -f ~/.ssh/birdy_publish ]; then
  mkdir -p ~/.ssh && chmod 700 ~/.ssh
  ssh-keygen -t ed25519 -N "" -f ~/.ssh/birdy_publish -C "birdy-publish@$(hostname)" >/dev/null
  echo "created ~/.ssh/birdy_publish"
fi
if [ -f ~/.ssh/birdy_publish.pub ]; then
  echo "gallery server authorized_keys line for this Pi (see CLAUDE.md):"
  # The restricted folder follows [publish] site_id in birdy.toml (default "home").
  SITE_ID=$(sed -n 's/^site_id *= *"\([a-z0-9-]*\)".*/\1/p' birdy.toml 2>/dev/null | head -1); SITE_ID="${SITE_ID:-home}"
  echo "command=\"/usr/bin/rrsync /srv/birdy/sites/$SITE_ID\",restrict $(cat ~/.ssh/birdy_publish.pub)"
fi

# The unit file is written for this user and this folder (deploy/birdy.service is the template).
sed -e "s|^User=.*|User=$USER|" -e "s|/home/yassine/pi-birdy|$PWD|g" deploy/birdy.service | sudo tee /etc/systemd/system/birdy.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now birdy

PORT=$(sed -n '/^\[web\]/,/^\[/s/^port *= *\([0-9]*\).*/\1/p' birdy.toml | head -1); PORT="${PORT:-8000}"
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo
echo "birdy is installed and running. Finish the setup in a browser on the same network:"
echo "  http://$(hostname).local:$PORT/settings"
[ -n "$IP" ] && echo "  http://$IP:$PORT/settings   (if the first address does not open)"
echo "Each part of that page has a 'How to set this up' guide."
