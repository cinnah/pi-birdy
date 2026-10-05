# Gallery "About" page — design

## Goal

One public page at `https://birdy.akibeo.com/about.html` that tells a visitor what birdy is, why it exists, how it works, what the rig is, how to join the shared record, and where the credits go. It is a pamphlet tucked into the record book: readable in five minutes on a phone, a reference afterwards.

Decisions taken with the user on 2026-10-05:

- The GitHub repo (`github.com/cinnah/pi-birdy`) stays **private** for now. The page says the code is in a private repository while the project settles and that collaborators get access with their folder; the URL is not printed.
- No contact details, no personal names and no repo URL on the page (user's instruction, 2026-10-05, replacing the earlier email decision): collaborators are told to ask the owner directly.
- **Less tutorial**: no step-by-step install. The page explains the parts and the pipeline, and points to the repo (via the email) for the install.
- A **photo of the rig** will be supplied by the user later. Until then a placeholder plate (ruled 4:3 frame in the paper tint, typed "Photo of the rig to follow", with the real caption) holds the slot so the layout can be judged now (user's request, 2026-10-05).

## Mode and place in the world

Read mode, inside the recording-sheet world recorded in `gallery/DESIGN.md`: paper, ink, 1 px rules, Libre Franklin printed labels, Courier Prime typed entries, red only for focus. One measured text column (max 62ch) under the same masthead. No hero, cards, icons, or illustration beyond the rig photo and one typed flow diagram.

Masthead on every gallery page gains an "About" link:

- `index.html`: `<p class="tagline">Daily record</p>` becomes `<nav><a href="about.html">About</a></nav>`. The "Daily record" wording is dropped from the masthead; the sheet's own heading already names the page.
- `site.html`, `species.html`: the existing nav gets a second link, `Daily record · About`.
- `about.html`: nav is `Daily record` (back to `index.html`).

Links in the masthead keep the 44 px hit area (`.hit` or equivalent padding).

## Page structure (`gallery/about.html`)

Static HTML, no `app.js`, no JavaScript at all. `<main>` is a single `<article>` with six `<section>`s, each headed by a printed label (`h2`, same style as the sheet's `.head` labels: 11 px / 600 / .14em uppercase, `--rule-ink`) and separated by a 1 px rule. A short in-page index right under the masthead lists the six sections as anchor links on one typed line.

1. **Why** (`#why`) — the manifesto, first person, about 250 words. Three ideas in order: it is fun to build (a Pi, a camera, a shower basket, a weekend); it is a shared record (several windows, one sheet, compare feeders side by side); it is data (every visit a row with species, time and confidence, kept honestly, including what the model gets wrong). Ends with the invitation to join (anchor to `#join`).
2. **How it works** (`#how`) — six numbered plain-language steps, then a typed flow diagram in a `<pre>` (Courier, `overflow-x: auto` on phones):
   1. A low-resolution stream is compared frame to frame inside a region you draw around the feeder. Movement there is a trigger.
   2. The camera takes a full still.
   3. A small on-device model (MobileNetV2, ImageNet) decides bird or not from the summed bird-class probability.
   4. A second model (Coral's iNaturalist bird classifier, 965 species) names the species; a confident species guess can rescue a frame the first model rejected.
   5. Photo, thumbnail and a row (time, species, confidence) go to SQLite on the Pi. A LAN page lets the household review, tune, and delete.
   6. The best photos per species per day are shrunk to 1600 px and rsynced to a static folder on a web server. The gallery pages read that folder's `manifest.json` in the browser; the server runs no code.
3. **The rig** (`#rig`) — the rig photo (placeholder plate until supplied) as a `<figure>` with a typed caption, then a parts list as a `<dl>` with typed values: Raspberry Pi 4 · Camera Module v2.1 (8 MP, fixed focus ~1 m to infinity) · official USB-C power supply · microSD card (16 GB or more) · a shower basket with suction cups, which holds the Pi and camera against the window glass · a feeder about a metre outside the window. One sentence each on placement (lens flush against the glass, indoors, daylight only) and rough cost (about €100 all in). Values marked as the user's own setup, not a shopping list.
4. **Join the record** (`#join`) — what a collaborator needs (a Pi 4 with a camera, a window with a feeder, a home network); what they get (a column on the daily record, their own monthly sheet, a species card entry); how it works technically in two sentences (their Pi pushes its own folder to the shared server over an SSH key restricted to that folder; nothing else is shared); "if you know me, ask" with no address.
5. **Code** (`#code`) — the code is in a private repository while the project settles; collaborators get access with their folder. No URL. One typed line listing the stack: Python 3.13, picamera2, OpenCV, LiteRT (TFLite), SQLite, FastAPI, rsync; gallery pages are vanilla HTML/CSS/JS.
6. **References and credits** (`#refs`) — a `<dl>` or list of typed entries with licences:
   - thjkral/BirdDetection — the project that inspired this one (link).
   - MobileNetV2 1.0 224 quantised, TensorFlow hosted models — Apache-2.0.
   - iNaturalist bird classifier (MobileNetV2 uint8), google-coral/test_data — Apache-2.0.
   - picamera2 / libcamera (Raspberry Pi), ai-edge-litert (Google), astral, FastAPI, Caddy.
   - Libre Franklin (Impallari Type) and Courier Prime (Quote-Unquote Apps) — SIL Open Font License.
   - Photos © each feeder's owner.

Footer: the same 3 px double red foot rule the sheets use, with one typed line: a link "Back to the daily record" (`index.html`).

## Rig photo

`gallery/about/rig.jpg` (≤ 1600 px wide, JPEG 85, same treatment as published photos) plus `gallery/about/rig-thumb.jpg` (400 px) for `srcset`. The `<figure>` ships now with a `div.plate` placeholder (`role="img"`, descriptive `aria-label`); when the file exists the `div` is replaced by the `<img>` with `srcset`. `alt` describes the scene (the Pi in a shower basket on the window, feeder outside).

## Styles

New rules in `gallery/style.css`, scoped under `.about`:

- `.about article { max-width: 62ch; }`; body text `font: 15px/1.6 var(--type)` (Courier) in `--ink`; `h2` reuse the printed-label style; `p + p` spacing 1em; `section` separated by `border-top: 1px solid var(--rule)` with 28 px padding.
- `.about .index` — one typed line of anchor links separated by ` · `, wraps on phones; links underlined with `--rule-ink`, 44 px hit area via padding.
- `.about dl` — two-column grid (`minmax(140px, 200px) 1fr`) with printed `dt` and typed `dd`; one column under 480 px.
- `.about pre` — Courier 13px/1.4, 1 px rule box, `overflow-x: auto`.
- `.about figure` — full column width, `img` `width: 100%; height: auto; border: 1px solid var(--ink)`; caption typed, `--rule-ink`.
- Dark mode via the existing tokens; nothing new.

## Deploy

`deploy/gallery.sh` copies all of `gallery/` and rewrites the `style.css`/`app.js` links in every `*.html`, so `about.html` and `about/` are pushed and hashed without changes. Nothing to add. Confirm after the first push that `https://birdy.akibeo.com/about.html` is 200 and the stylesheet link carries `?v=`.

## Accessibility and quality bar

- Landmarks: `header`, `nav`, `main`, `footer`; `h1` (masthead) then `h2` per section; skip-free reading order.
- Contrast ≥ 4.5:1 for all text in light and dark (tokens already pass).
- All links ≥ 44 px hit area; focus visible in `--red` as elsewhere.
- No horizontal overflow at 360 px; the `<pre>` scrolls internally.
- No JavaScript, no remote assets; fonts preloaded as on the other pages.
- `impeccable detect --json gallery` reports nothing new beyond the known intentional findings.

## Testing

Playwright pass (same harness as the redesign): render `about.html` at 1440 and 390 px, light and dark; assert no overflow, fonts loaded, no console errors, every `href` resolves (local files, anchors, `mailto:`), contrast and hit-area probes; screenshot both widths for inspection. Verify the masthead "About" link renders on `index.html`, `site.html`, `species.html` without wrapping the baseline row at 390 px. After deploy: `curl -I` on `about.html` and the rig photo (if present).

## Out of scope

Install walkthrough, multiple pages, Markdown rendering, a public repo, a licence file (to be chosen when the repo opens), analytics, comments.
