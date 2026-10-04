# Gallery About Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `gallery/about.html`, a static page (manifesto, how it works, the rig, join, code, references) in the recording-sheet look, linked from every gallery masthead and pushed by the existing deploy script.

**Architecture:** One hand-written HTML file with no JavaScript, styled by new `.about` rules appended to the shared `gallery/style.css`. The three existing pages only gain a masthead link. Verification is a Playwright script run against a local `http.server`, then `deploy/gallery.sh`.

**Tech Stack:** Static HTML/CSS; Playwright (from `/Users/yassine/.npm/_npx/705bc6b22212b352/node_modules/playwright`); `python3 -m http.server`; `deploy/gallery.sh` (rsync).

Spec: `docs/superpowers/specs/2026-10-05-about-page-design.md`. Direction contract: `.impeccable/surfaces/gallery-index-html.md`. Read `/Users/yassine/.claude/skills/impeccable/reference/craft-floor.md` before editing any gallery file.

Repo rules: commit on `master` (the user works there); never touch the user's cartaveo repo or the server's Caddy; never push `gallery/DESIGN.md` or `.impeccable/` to the server (gallery.sh excludes them). Commit messages end with
```
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QCPDonjwfAsrfGJWTbFtyB
```

---

## File map

- Create `gallery/about.html` — the page (static markup, all copy).
- Modify `gallery/style.css` — `.mast nav` gap for two links; new `/* about */` block appended before the phone `@media (max-width: 719px)` block, plus phone rules inside it.
- Modify `gallery/index.html`, `gallery/site.html`, `gallery/species.html` — masthead "About" link.
- Create `.impeccable/surfaces/gallery-about-html.md` — surface brief (Read mode) for the new page.
- Modify `gallery/DESIGN.md`, `CLAUDE.md` — one short note each.
- Scratch: `/private/tmp/claude-501/-Users-yassine-www-pi-birdy/a0942696-7d59-46f4-bfae-4ac8e8affaac/scratchpad/audit/about.js` — Playwright check.

---

### Task 1: Masthead "About" link on the three sheets

**Files:**
- Modify: `gallery/index.html:17-20`, `gallery/site.html:17-20`, `gallery/species.html:17-20`
- Modify: `gallery/style.css:48-50`

- [ ] **Step 1: Replace the index masthead tagline with a nav**

In `gallery/index.html` replace
```html
  <p class="tagline">Daily record</p>
```
with
```html
  <nav><a class="hit" href="about.html">About</a></nav>
```

- [ ] **Step 2: Add the About link to the feeder and species sheets**

In both `gallery/site.html` and `gallery/species.html` replace
```html
  <nav><a class="hit" href="index.html">Daily record</a></nav>
```
with
```html
  <nav><a class="hit" href="index.html">Daily record</a><a class="hit" href="about.html">About</a></nav>
```

- [ ] **Step 3: Space the two nav links**

In `gallery/style.css`, after the line starting `.mast nav, .mast .tagline {`, add
```css
.mast nav { display: flex; gap: 20px; }
```
(`.hit` keeps each link 44px tall; the 20px gap keeps the two targets apart on a phone.)

- [ ] **Step 4: Check the row at phone width**

```bash
cd /Users/yassine/www/pi-birdy && (python3 -m http.server 8765 --directory gallery >/dev/null 2>&1 &) && sleep 1
cat > /private/tmp/claude-501/-Users-yassine-www-pi-birdy/a0942696-7d59-46f4-bfae-4ac8e8affaac/scratchpad/audit/mast.js <<'EOF'
const { chromium } = require('/Users/yassine/.npm/_npx/705bc6b22212b352/node_modules/playwright');
(async () => {
  const b = await chromium.launch(); const ctx = await b.newContext({ viewport: { width: 360, height: 740 } });
  for (const p of ['index.html', 'site.html?site=home', 'species.html?site=home&species=Eurasian%20Blue%20Tit']) {
    const pg = await ctx.newPage(); await pg.goto('http://localhost:8765/' + p);
    const r = await pg.evaluate(() => { const m = document.querySelector('.mast'); const h = m.querySelector('h1').getBoundingClientRect(); const links = [...m.querySelectorAll('nav a')].map(a => { const r = a.getBoundingClientRect(); return { t: a.textContent, h: Math.round(r.height), top: Math.round(r.top) }; }); return { mastH: Math.round(m.getBoundingClientRect().height), h1top: Math.round(h.top), links }; });
    console.log(p, JSON.stringify(r));
  }
  await b.close();
})();
EOF
node /private/tmp/claude-501/-Users-yassine-www-pi-birdy/a0942696-7d59-46f4-bfae-4ac8e8affaac/scratchpad/audit/mast.js
```
Expected: every link `h` ≥ 44; `mastH` the same on all three pages (one baseline row, no wrap); link `top` values equal within a page.

- [ ] **Step 5: Commit**

```bash
git add gallery/index.html gallery/site.html gallery/species.html gallery/style.css
git commit -m "Gallery: About link in every masthead"
```

---

### Task 2: The About page and its styles

**Files:**
- Create: `gallery/about.html`
- Modify: `gallery/style.css` (append an `/* about */` block right before `@media (max-width: 719px) {`, and phone rules inside that block)

- [ ] **Step 1: Write `gallery/about.html`**

Use the exact head of `gallery/index.html` (charset, viewport, theme-color metas, the inline SVG favicon, the two font preloads, `style.css`), with this title/description and no `app.js`:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>birdy — about</title>
<meta name="description" content="What birdy is: a Raspberry Pi camera at a window, a daily record of the feeder's visitors, a sheet shared between friends' windows. How it works, the rig, how to join, and credits.">
<meta name="theme-color" media="(prefers-color-scheme: light)" content="#eef2e6">
<meta name="theme-color" media="(prefers-color-scheme: dark)" content="#1b1e22">
<link rel="icon" href="COPY THE data:image/svg+xml FAVICON FROM index.html VERBATIM">
<link rel="preload" as="font" type="font/woff2" crossorigin href="fonts/CourierPrime-Regular.woff2">
<link rel="preload" as="font" type="font/woff2" crossorigin href="fonts/LibreFranklin.woff2">
<link rel="stylesheet" href="style.css">
</head>
<body data-page="about">
<div class="sheet about">
<header class="mast">
  <h1><a class="hit" href="index.html">birdy</a></h1>
  <nav><a class="hit" href="index.html">Daily record</a></nav>
</header>
<main id="main">
<div class="head">
  <h2>About this record<span class="sub">A camera at the window, one sheet a day, a column per feeder</span></h2>
  <div class="meta"><p class="label">Notes</p>Sintra, October 2026</div>
</div>
<nav class="index typed" aria-label="On this page"><a href="#why">Why</a> · <a href="#how">How it works</a> · <a href="#rig">The rig</a> · <a href="#join">Join the record</a> · <a href="#code">Code</a> · <a href="#refs">References</a></nav>
<article>

<section id="why">
<h3 class="label">Why</h3>
<p>There is a feeder outside my window in Sintra and I wanted to know who comes to it. Not roughly, not "the blue tits are back", but which species, how often, at what hour, with a photo for every visit.</p>
<p>The first part is that this is a weekend's build. A Raspberry Pi 4, the standard camera module, a shower basket with suction cups to hang it on the glass, and a feeder a metre away. The code watches a low-resolution stream for movement, takes a still, asks two small models running on the Pi whether it is a bird and which one, and files the answer. Nothing leaves the house except the photos I choose to publish.</p>
<p>The second part is the one I care about most. A record of one window is a diary; a record of several windows is a dataset. Every feeder that joins gets a column on the daily sheet, so the same species can be compared across gardens on the same day: who had the goldfinches, when the robin starts, what arrives first after rain. It is a collaboration between friends who happen to have windows.</p>
<p>The third part is honesty. The species model is three megabytes and it is wrong sometimes. The record keeps the confidence of every guess, the review page lets us correct or delete, and nothing is averaged away. A sheet with a visible mistake is worth more than a chart that hides one.</p>
<p>If you have a window, a feeder and a free weekend, <a href="#join">there is a column waiting</a>.</p>
</section>

<section id="how">
<h3 class="label">How it works</h3>
<ol>
<li>The camera streams a small 320×240 picture. Inside a region you draw around the feeder, each frame is compared with a slowly updating background. Enough change there is a trigger; everything outside the region (curtains, branches, reflections) is ignored.</li>
<li>The camera takes a full-size still.</li>
<li>A small model on the Pi (MobileNetV2 trained on ImageNet) looks at the crop around the movement and gives the probability that it is a bird, summed over every bird class it knows.</li>
<li>A second model (the iNaturalist bird classifier, 965 species) names the species. A confident species guess can rescue a frame the first model turned down.</li>
<li>Photo, thumbnail and a row with time, species and both confidences go into SQLite on the Pi. A page on the home network shows the day's captures, lets the household tune the region and thresholds live, and delete what is wrong. Optionally a Telegram message goes out for chosen species.</li>
<li>The best photos of each species, per day, are shrunk to 1600 px and copied with rsync into that feeder's folder on a plain web server. These gallery pages read each folder's manifest in the browser; the server runs no code.</li>
</ol>
<pre aria-label="Flow of one visit">camera ──▶ motion inside the drawn region ──▶ still photo
                                                 │
                              bird or not? ◀─────┘   MobileNetV2 · ImageNet
                                   │
                              which species?          iNaturalist birds · 965
                                   │
                              row + photo + thumb ──▶ SQLite ──▶ review page (LAN)
                                   │
                              best of the day, per species ──rsync──▶ web server ──▶ this record</pre>
</section>

<section id="rig">
<h3 class="label">The rig</h3>
<figure>
<div class="plate" role="img" aria-label="Placeholder for a photo of the rig: the Pi and camera in a shower basket on the window, the feeder outside"><span class="typed">Photo of the rig to follow</span></div>
<figcaption>The Pi and its camera in a shower basket hung on the window glass; the feeder a metre outside.</figcaption>
</figure>
<dl>
<dt>Computer</dt><dd>Raspberry Pi 4, Raspberry Pi OS Lite (64-bit)</dd>
<dt>Camera</dt><dd>Camera Module v2.1, 8 MP, fixed focus from about a metre to infinity, no night vision</dd>
<dt>Power</dt><dd>The official USB-C supply</dd>
<dt>Storage</dt><dd>A microSD card, 16 GB or more; photos live on it</dd>
<dt>Mount</dt><dd>A shower basket with suction cups, found by chance. It hangs on the window glass and holds the Pi and the camera at exactly the right height</dd>
<dt>Outside</dt><dd>A feeder about a metre from the window</dd>
<dt>Cost</dt><dd>About €100 all in, less if a Pi is already in a drawer</dd>
</dl>
<p>The lens sits flush against the glass, indoors, so there is no weatherproofing to do. The camera only runs between sunrise and sunset; at night the Pi sleeps. This is one working setup, not the only one: anything that holds a Pi camera still at a window will do.</p>
</section>

<section id="join">
<h3 class="label">Join the record</h3>
<p>You need a Raspberry Pi 4 with a camera module, a window with a feeder outside it, and a home network. You get a column on the daily record next to the other feeders, your own monthly sheet, and your birds on the species cards.</p>
<p>Your Pi keeps its own photos and database. Once a quarter of an hour it copies its published photos and a manifest to its own folder on the shared server over an SSH key that can reach only that folder. Nothing else is shared, and you choose what is published from your review page.</p>
<p>Write to <a href="mailto:yassine@e-xperience.pt">yassine@e-xperience.pt</a> and I will set up your folder and send you the install notes.</p>
</section>

<section id="code">
<h3 class="label">Code</h3>
<p>The code is at <span class="typed">github.com/cinnah/pi-birdy</span>. The repository is private while the project settles; email for access and it is yours.</p>
<p>Python 3.13 on the Pi: picamera2, OpenCV, LiteRT (TensorFlow Lite), SQLite, FastAPI, rsync. The gallery pages are plain HTML, CSS and JavaScript with no build step and no dependencies.</p>
</section>

<section id="refs">
<h3 class="label">References and credits</h3>
<dl>
<dt>Inspiration</dt><dd><a href="https://github.com/thjkral/BirdDetection">thjkral/BirdDetection</a>, the project that showed a Pi and a camera could do this</dd>
<dt>Bird gate</dt><dd>MobileNetV2 1.0 224, quantised, from the TensorFlow hosted models (<a href="https://arxiv.org/abs/1801.04381">paper</a>). Apache-2.0</dd>
<dt>Species</dt><dd>iNaturalist bird classifier (MobileNetV2, 965 classes) from <a href="https://github.com/google-coral/test_data">google-coral/test_data</a>. Apache-2.0</dd>
<dt>Camera</dt><dd><a href="https://github.com/raspberrypi/picamera2">picamera2</a> and libcamera, Raspberry Pi</dd>
<dt>Inference</dt><dd><a href="https://github.com/google-ai-edge/LiteRT">LiteRT</a> (ai-edge-litert), Google</dd>
<dt>Daylight</dt><dd><a href="https://github.com/sffjunkie/astral">astral</a></dd>
<dt>Review page</dt><dd><a href="https://fastapi.tiangolo.com">FastAPI</a></dd>
<dt>Server</dt><dd><a href="https://caddyserver.com">Caddy</a>, serving static files</dd>
<dt>Type</dt><dd><a href="https://github.com/impallari/Libre-Franklin">Libre Franklin</a> by Impallari Type and <a href="https://quoteunquoteapps.com/courierprime/">Courier Prime</a> by Quote-Unquote Apps, both under the <a href="https://openfontlicense.org">SIL Open Font License</a></dd>
<dt>Photos</dt><dd>© each feeder's owner</dd>
</dl>
</section>

</article>
<footer class="foot"><a class="hit" href="index.html">Back to the daily record</a></footer>
</main>
</div>
</body>
</html>
```

Replace the favicon placeholder with the exact `href` value from `gallery/index.html` line 10 (`grep -n 'rel="icon"' gallery/index.html`).

- [ ] **Step 2: Append the about styles to `gallery/style.css`**

Insert this block immediately before the line `@media (max-width: 719px) {`:

```css
/* about: a typed note tucked into the record. One measured column, printed labels, typed text. */
.about article { max-width: 62ch; }
.about .head .meta { font: 15px/1.5 var(--type); }
.about .index { margin: 0; padding: 10px 0; border-bottom: 1px solid var(--rule); font-size: 14px; line-height: 1.6; color: var(--rule-ink); }
.about .index a { display: inline-block; padding: 10px 0; margin: -10px 0; color: var(--ink); text-decoration: underline; text-decoration-color: var(--rule-ink); text-underline-offset: 3px; text-decoration-thickness: 1px; }
.about section { padding: 28px 0 8px; border-bottom: 1px solid var(--rule); }
.about section:last-child { border-bottom: 0; }
.about section .label { margin-bottom: 14px; }
.about p, .about li, .about dd { font: 15px/1.65 var(--type); }
.about p { margin: 0 0 1em; }
.about article a { text-decoration: underline; text-decoration-color: var(--rule-ink); text-underline-offset: 3px; text-decoration-thickness: 1px; }
.about ol { margin: 0 0 1.2em; padding-left: 1.6em; }
.about li { margin: 0 0 .6em; }
.about li::marker { font-family: var(--type); color: var(--rule-ink); }
.about pre { margin: 0 0 1.2em; padding: 14px 16px; border: 1px solid var(--rule); font: 13px/1.45 var(--type); color: var(--ink); overflow-x: auto; }
.about dl { display: grid; grid-template-columns: minmax(120px, 160px) 1fr; gap: 0 24px; margin: 0 0 1.2em; }
.about dt { font: 600 11px/1.3 var(--print); letter-spacing: .14em; text-transform: uppercase; color: var(--rule-ink); padding: 8px 0 0; border-top: 1px solid var(--rule); }
.about dd { margin: 0; padding: 6px 0 8px; border-top: 1px solid var(--rule); }
.about dt:first-of-type, .about dd:first-of-type { border-top: 0; }
.about figure { margin: 0 0 1.4em; }
.about figure img, .about .plate { display: block; width: 100%; height: auto; border: 1px solid var(--ink); }
.about .plate { aspect-ratio: 4 / 3; display: grid; place-items: center; background: var(--tint); font: 13px/1.5 var(--type); color: var(--rule-ink); letter-spacing: .02em; }
.about figcaption { margin-top: 6px; font: 13px/1.5 var(--type); color: var(--rule-ink); }
.about .foot a { color: var(--ink); }
```

Then inside the existing `@media (max-width: 719px) {` block, add before its closing brace:

```css
  .about dl { grid-template-columns: 1fr; }
  .about dt { border-top: 1px solid var(--rule); padding-top: 10px; }
  .about dd { border-top: 0; padding-top: 2px; }
  .about dt:first-of-type { border-top: 0; }
```

Paragraph links need a 44px target only where they are the sole way to an action; the in-text anchors are supplementary (the index and the footer carry the padded targets). The `.about .index a` and `.foot .hit` links are ≥ 44px tall via padding.

- [ ] **Step 3: Render and inspect**

```bash
cd /Users/yassine/www/pi-birdy && (python3 -m http.server 8765 --directory gallery >/dev/null 2>&1 &) && sleep 1
cat > /private/tmp/claude-501/-Users-yassine-www-pi-birdy/a0942696-7d59-46f4-bfae-4ac8e8affaac/scratchpad/audit/about.js <<'EOF'
const { chromium } = require('/Users/yassine/.npm/_npx/705bc6b22212b352/node_modules/playwright');
const fs = require('fs');
const OUT = '/private/tmp/claude-501/-Users-yassine-www-pi-birdy/a0942696-7d59-46f4-bfae-4ac8e8affaac/scratchpad/audit/';
(async () => {
  const b = await chromium.launch();
  for (const [name, vp, scheme] of [['about-desk', { width: 1440, height: 900 }, 'light'], ['about-phone', { width: 360, height: 740 }, 'light'], ['about-dark', { width: 1440, height: 900 }, 'dark']]) {
    const ctx = await b.newContext({ viewport: vp, colorScheme: scheme });
    const pg = await ctx.newPage(); const errors = [];
    pg.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
    pg.on('pageerror', e => errors.push(String(e)));
    await pg.goto('http://localhost:8765/about.html'); await pg.evaluate(() => document.fonts.ready);
    const r = await pg.evaluate(async () => {
      const overflow = document.documentElement.scrollWidth > document.documentElement.clientWidth;
      const fonts = [...document.fonts].filter(f => f.status === 'loaded').map(f => f.family);
      const small = [...document.querySelectorAll('.index a, .mast a, .foot a')].map(a => ({ t: a.textContent.trim(), h: Math.round(a.getBoundingClientRect().height) })).filter(x => x.h < 44);
      const col = Math.round(document.querySelector('.about article').getBoundingClientRect().width);
      const pre = document.querySelector('pre'); const preScroll = pre.scrollWidth > pre.clientWidth;
      const hrefs = [...document.querySelectorAll('a[href]')].map(a => a.getAttribute('href'));
      const anchors = hrefs.filter(h => h.startsWith('#')).filter(h => !document.querySelector(h));
      const local = hrefs.filter(h => !/^(#|https?:|mailto:)/.test(h));
      const missing = [];
      for (const h of local) { const res = await fetch(h, { method: 'HEAD' }); if (!res.ok) missing.push(h); }
      return { overflow, fonts: [...new Set(fonts)], small, col, preScroll, badAnchors: anchors, missing, headings: [...document.querySelectorAll('h1,h2,h3')].map(h => h.tagName + ' ' + h.textContent.trim().slice(0, 30)) };
    });
    console.log(name, JSON.stringify({ ...r, errors }, null, 0));
    await pg.screenshot({ path: OUT + name + '.png', fullPage: true });
    await ctx.close();
  }
  await b.close();
})();
EOF
node /private/tmp/claude-501/-Users-yassine-www-pi-birdy/a0942696-7d59-46f4-bfae-4ac8e8affaac/scratchpad/audit/about.js
```
Expected for all three: `overflow: false`, `fonts` includes both "Libre Franklin" and "Courier Prime", `small: []`, `badAnchors: []`, `missing: []`, `errors: []`; `col` ≤ 62ch (about 560px desktop, 328px phone); phone `preScroll: true` is acceptable. Then open the three PNGs (Read tool) and check: one measured column, printed labels over typed text, rules only 1px blue, red only on the masthead/foot double rules, dark mode is the paper sheet on a dark desk.

- [ ] **Step 4: Check all external links resolve**

```bash
for u in https://github.com/thjkral/BirdDetection https://arxiv.org/abs/1801.04381 https://github.com/google-coral/test_data https://github.com/raspberrypi/picamera2 https://github.com/google-ai-edge/LiteRT https://github.com/sffjunkie/astral https://fastapi.tiangolo.com https://caddyserver.com https://github.com/impallari/Libre-Franklin https://quoteunquoteapps.com/courierprime/ https://openfontlicense.org; do printf '%s ' "$u"; curl -sIL -o /dev/null -w '%{http_code}\n' "$u"; done
```
Expected: every line ends in 200 (a 403 from a bot-blocking site is acceptable if the page opens in a browser; a 404 means fix the URL).

- [ ] **Step 5: Run the design detector**

```bash
cd /Users/yassine/www/pi-birdy && /Users/yassine/.claude/skills/impeccable/scripts/impeccable detect --json gallery 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); [print(f["rule"] if isinstance(f,dict) and "rule" in f else f, "::", f.get("file","") if isinstance(f,dict) else "") for f in (d.get("findings") or d if isinstance(d,list) else [])]' | grep about
```
Expected: nothing new for `about.html`/the `.about` block beyond the known intentional findings (fonts/colours differ from the root DESIGN.md, which documents the Pi UI). Fix anything else.

- [ ] **Step 6: Commit**

```bash
git add gallery/about.html gallery/style.css
git commit -m "Gallery: About page — why, how it works, the rig, join, code, references"
```

---

### Task 3: Surface brief, design record, CLAUDE.md

**Files:**
- Create: `.impeccable/surfaces/gallery-about-html.md`
- Modify: `gallery/DESIGN.md` (append a section), `CLAUDE.md` (architecture bullet on the gallery pages)

- [ ] **Step 1: Write the surface brief**

```markdown
---
version: 1
slug: "gallery-about-html"
primary_target: "gallery/about.html"
related_targets: ["gallery/style.css","gallery/index.html","gallery/site.html","gallery/species.html"]
---

# Gallery About page (birdy.akibeo.com/about.html)

Scope: one static page, no JavaScript. Visitor mode: Read. Audience: a friend sent the link, or a collaborator who wants to understand the thing they are about to put in their window. Job: understand what birdy is and why, how a visit becomes a row on the sheet, what the rig is, how to join, who to credit. Constraints: same world as the sheets (`gallery-index-html.md`); static files; self-hosted fonts; light and dark; never loud; no tutorial (install notes travel by email while the repo is private).

## Direction

A typed note tucked into the record book: the sheet's masthead and head, then one measured column (62ch) of Courier Prime under printed Libre Franklin labels, sections separated by the same 1px blue rule, parts and credits as ruled definition lists, the pipeline as a typed flow in a ruled box. Red only on the double rules and focus. Links underlined in rule ink. The rig photo is a framed print at column width with a typed caption; until it is supplied a ruled 4:3 plate in paper tint holds the slot with a typed line. Nothing moves.
```

- [ ] **Step 2: Append to `gallery/DESIGN.md`**

Add a section at the end:

```markdown
## About page (`about.html`)

A typed note tucked into the record: masthead and sheet head as on the other pages, then an `article` capped at 62ch. Body copy, list items and `dd`s are Courier Prime 15/1.65; section heads are `.label` (printed, 11px, tracked); sections are divided by the 1px rule with 28px above. Definition lists (`.about dl`) are two ruled columns (`minmax(120px,160px) 1fr`), one column under 720px. The flow diagram is a `pre` in a 1px rule box that scrolls sideways on phones. In-text links underline in rule ink; the index line and the footer carry the 44px targets. No JavaScript, no greys, red only on the double rules and focus. Every masthead carries an About link (`.mast nav` is a flex row with a 20px gap).
```

- [ ] **Step 3: Note it in `CLAUDE.md`**

In the architecture bullet that starts `The gallery pages in \`gallery/\` (vanilla JS) render in the browser`, after the sentence ending `and the look is recorded in \`gallery/DESIGN.md\`, not the Pi UI's \`DESIGN.md\`.` add:

```
`gallery/about.html` is the one static page (no JS): manifesto, how it works, the rig, how to join, code, credits; its copy lives in the HTML itself, and the rig photo slot is `gallery/about/rig.jpg` (a ruled `.plate` placeholder stands in the `figure` until the photo is supplied; then replace the `div.plate` with `<img src="about/rig.jpg" srcset="about/rig-thumb.jpg 400w, about/rig.jpg 1600w" sizes="(max-width: 719px) calc(100vw - 32px), 560px" alt="…">`).
```

- [ ] **Step 4: Commit**

```bash
git add .impeccable/surfaces/gallery-about-html.md gallery/DESIGN.md CLAUDE.md
git commit -m "Gallery: About page recorded (surface brief, design record, CLAUDE.md)"
```

---

### Task 4: Deploy and verify live

**Files:** none (runs `deploy/gallery.sh`)

- [ ] **Step 1: Push**

```bash
cd /Users/yassine/www/pi-birdy && deploy/gallery.sh
```
Expected: `gallery pages synced to root@91.98.229.209:/srv/birdy/ (assets v=<hash>)`.

- [ ] **Step 2: Verify**

```bash
curl -sI 'https://birdy.akibeo.com/about.html' | head -1
curl -s 'https://birdy.akibeo.com/about.html' | grep -o 'style.css?v=[0-9a-f]*'
curl -s 'https://birdy.akibeo.com/index.html' | grep -c 'href="about.html"'
curl -sI 'https://birdy.akibeo.com/DESIGN.md' | head -1
```
Expected: `HTTP/2 200`; a `style.css?v=…` link; `1`; `HTTP/2 404` (the design record stays home).

- [ ] **Step 3: Live render**

Run the Task 2 Step 3 script with `http://localhost:8765/about.html` replaced by `https://birdy.akibeo.com/about.html` (the local `fetch` HEAD checks then run against the live server). Expected: same results as locally.
