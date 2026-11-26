# Birdy

A Raspberry Pi camera behind a window photographs the birds at a feeder. It keeps the photos that
show a bird, names the species, and can send a Telegram message for each visit and share the best
photos on a public gallery. Everything runs on the Pi; the photos stay at home unless you switch
the gallery on.

## What you need

- A Raspberry Pi 4 with Raspberry Pi OS (Trixie, 64-bit) on your home network.
- A Raspberry Pi Camera Module (v2 or newer), pointed at the feeder through a window.
- Optional: a DSLR that `gphoto2` supports, on USB, for sharper photos.

## Install

On the Pi (or over `ssh -t`):

```bash
git clone https://github.com/cinnah/pi-birdy.git ~/pi-birdy
cd ~/pi-birdy
deploy/install.sh
```

The script installs what birdy needs, downloads the models, starts the service, and prints the
address of the settings page. It asks for your password once (`sudo`). Running it again is safe.

## Set it up

Open the address the script printed (`http://<your-pi>.local:8000/settings`) on a phone or
computer on the same network. The page has five parts, each with a **How to set this up** guide:

1. **Camera**: draw the watch area on the live picture, do the walk test, name the feeder and
   give its location.
2. **Messages**: connect a Telegram bot so each bird visit sends you a photo.
3. **Extra camera**: add a DSLR on USB.
4. **Gallery**: share the best photos on a public page.
5. **System**: the few settings that are read once at start-up.

Only the first part is needed. The photos are at `http://<your-pi>.local:8000/`.

The page has no password: anyone on your home network can open it. Do not expose it to the internet.

## Update

```bash
cd ~/pi-birdy && git pull && deploy/install.sh && sudo systemctl restart birdy
```

## For developers

`CLAUDE.md` describes the architecture, the commands and the conventions; `docs/superpowers/specs/`
holds the designs. Tests run on any machine: `pip install -e ".[dev]" && pytest`.
