#!/usr/bin/env bash
# Push the fixed gallery pages (gallery/) to the server's /srv/birdy/. Never touches sites/ or sites.json.
# Usage: deploy/gallery.sh            (override with GALLERY_HOST=user@host GALLERY_ROOT=/srv/birdy)
set -euo pipefail
HOST="${GALLERY_HOST:-root@91.98.229.209}"
ROOT="${GALLERY_ROOT:-/srv/birdy}"
cd "$(dirname "$0")/.."
rsync -rlt --no-owner --no-group --exclude sites/ --exclude sites.json gallery/ "$HOST:$ROOT/"
echo "gallery pages synced to $HOST:$ROOT/"
