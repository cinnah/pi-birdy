# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

- **The builder/operator** (the owner): develops Birdy on a Mac, deploys to the Pi, tunes detection, reads logs.
- **The household**: people in the house browse the captured bird photos and may also open `/settings` to tune things. They are not technical and don't know what `pixel_threshold` or `background_alpha` mean.

All of them open the pages on phones and laptops/desktops about equally. On a phone they are often standing at the window, looking at the feeder.

## Product Purpose

A Raspberry Pi camera behind a window photographs birds at a feeder. It filters out non-bird motion with a local model, names the species, keeps the photos, and sends Telegram alerts for chosen species. The web UI is how the household sees what visited and how they adjust what counts as a visit. It works when the household enjoys the birds and the captures are mostly real birds, without anyone having to SSH in.

## Positioning

It's a private, local, single-feeder instrument: one window, one camera, one household. Nothing goes to a cloud service and there are no accounts. It isn't a product sold to anyone.

## Operating Context

- Served by FastAPI + Jinja templates from the Pi at `http://birdy.local:8000`. LAN only, no auth, no CSRF token by design.
- Pages: `/` (capture grid with filters and live status), `/species` (per-species summary plus an upload box to classify a photo), `/settings` (live camera frame with a click-to-draw ROI polygon, motion/classifier/species/site/retention/Telegram tunables, test alert).
- Settings apply live without a restart, except for the restart-only keys, which are shown read-only.
- Runs on a Raspberry Pi 4: pages must stay light, with no build step and no heavy client frameworks. The settings page polls `/frame.jpg` and `/status` once a second.

## Capabilities and Constraints

- The templates are plain HTML/CSS/vanilla JS served by the Pi. There's no bundler, and no internet access can be assumed at the feeder, so fonts and assets must be local or system.
- ROI coordinates are in 320×240 lores space.
- Every tunable lives in `birdy.toml`. The settings form must cover every live key (see CLAUDE.md "Adding a tunable key").
- The Telegram bot token is secret and is shown only as "set"/"not set".

## Evidence on Hand

- Real bird photos and thumbnails exist on the Pi under `data/`. Species names come from the 965-species iNat label set in `models/`.
- No logo or brand assets exist. Don't invent any.

## Product Principles

1. **The birds are the content.** The UI exists to show photos and help get better ones.
2. **Tuning must be understandable by a non-technical household member**: plain-language labels and live feedback over raw config names, while still exposing the real key for the operator.
3. **Live and honest**: show what the camera sees and what the detector thinks right now. Distinguish live-applied changes from restart-only ones.
4. **Light on the Pi**: no frameworks, no remote assets, minimal JS.

## Accessibility & Inclusion

Must work at phone width with touch (ROI drawing uses pointer events) and on desktop. Both light and dark modes are already supported and must stay.
