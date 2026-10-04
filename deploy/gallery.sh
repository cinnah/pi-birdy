#!/usr/bin/env bash
# Push the fixed gallery pages (gallery/) to the server's /srv/birdy/. Never touches sites/ or sites.json,
# and leaves the design record (gallery/DESIGN.md, gallery/.impeccable/) at home.
# The HTML is pushed with "style.css?v=<hash>" / "app.js?v=<hash>" links (hash of both files), so a
# browser that cached the old assets fetches the new ones on the next page load. The server ignores the query.
# Usage: deploy/gallery.sh            (override with GALLERY_HOST=user@host GALLERY_ROOT=/srv/birdy)
set -euo pipefail
HOST="${GALLERY_HOST:-root@91.98.229.209}"
ROOT="${GALLERY_ROOT:-/srv/birdy}"
cd "$(dirname "$0")/.."
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp -R gallery/. "$STAGE/"
V="$(cat gallery/style.css gallery/app.js | shasum | cut -c1-8)"
perl -pi -e 's/"(style\.css|app\.js)"/"$1?v='"$V"'"/g' "$STAGE"/*.html
rsync -rlt --no-owner --no-group --exclude sites/ --exclude sites.json --exclude DESIGN.md --exclude .impeccable/ "$STAGE/" "$HOST:$ROOT/"
echo "gallery pages synced to $HOST:$ROOT/ (assets v=$V)"
