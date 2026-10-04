---
name: Birdy
description: The feeder camera's own on-screen display; a trail-cam OSD and setup menu served from the Pi.
colors:
  osd-black: "#0c0d0c"
  night-ground-raised: "#131513"
  night-field: "#101210"
  osd-white: "#f2f2ee"
  night-dim: "#9aa39d"
  night-rule: "#2b2f2c"
  night-rule-strong: "#4a504b"
  night-seg-off: "#232724"
  signal-amber: "#ffb000"
  rec-red-night: "#ff4d3d"
  osd-dim: "#7d8580"
  lcd-ground: "#c9cebd"
  lcd-ground-raised: "#bec4b0"
  lcd-field: "#d3d8c8"
  lcd-ink: "#141812"
  lcd-dim: "#434a3e"
  lcd-rule: "#99a08f"
  lcd-rule-strong: "#6f7766"
  lcd-seg-off: "#b1b8a3"
  rec-red-day: "#a3201a"
typography:
  readout:
    fontFamily: "Martian Mono, ui-monospace, SF Mono, Menlo, monospace"
    fontSize: "20px"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "0"
    fontVariation: "'wdth' 87.5"
  verdict:
    fontFamily: "Martian Mono, ui-monospace, SF Mono, Menlo, monospace"
    fontSize: "22px"
    fontWeight: 800
    lineHeight: 1.1
    letterSpacing: "0.02em"
    fontVariation: "'wdth' 87.5"
  menu-head:
    fontFamily: "Martian Mono, ui-monospace, SF Mono, Menlo, monospace"
    fontSize: "12px"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "0.14em"
    fontVariation: "'wdth' 87.5"
  row-label:
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
    fontSize: "15px"
    fontWeight: 600
    lineHeight: 1.45
  body:
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.45
    fontFeature: "'tnum'"
  explain:
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.4
  value:
    fontFamily: "Martian Mono, ui-monospace, SF Mono, Menlo, monospace"
    fontSize: "14px"
    fontWeight: 500
    lineHeight: 1.3
    fontVariation: "'wdth' 87.5"
  label-button:
    fontFamily: "Martian Mono, ui-monospace, SF Mono, Menlo, monospace"
    fontSize: "12px"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "0.1em"
    fontVariation: "'wdth' 87.5"
  osd-strip:
    fontFamily: "Martian Mono, ui-monospace, SF Mono, Menlo, monospace"
    fontSize: "12px"
    fontWeight: 500
    lineHeight: 1
    letterSpacing: "0.06em"
    fontVariation: "'wdth' 87.5"
  key:
    fontFamily: "Martian Mono, ui-monospace, SF Mono, Menlo, monospace"
    fontSize: "11px"
    fontWeight: 400
    lineHeight: 1.3
    letterSpacing: "0.02em"
    fontVariation: "'wdth' 87.5"
  stamp:
    fontFamily: "Martian Mono, ui-monospace, SF Mono, Menlo, monospace"
    fontSize: "11px"
    fontWeight: 500
    lineHeight: 1.3
    letterSpacing: "0.03em"
    fontVariation: "'wdth' 81"
rounded:
  none: "0"
  lamp: "50%"
spacing:
  hairline-gap: "2px"
  xs: "6px"
  sm: "8px"
  md: "12px"
  gutter: "16px"
  lg: "24px"
  group: "28px"
  strip-height: "44px"
components:
  button:
    backgroundColor: "transparent"
    textColor: "{colors.osd-white}"
    typography: "{typography.label-button}"
    rounded: "{rounded.none}"
    padding: "0 16px"
    height: "40px"
  button-active:
    backgroundColor: "{colors.osd-white}"
    textColor: "{colors.osd-black}"
  button-primary:
    backgroundColor: "{colors.signal-amber}"
    textColor: "{colors.osd-black}"
    typography: "{typography.label-button}"
    rounded: "{rounded.none}"
    padding: "0 16px"
    height: "40px"
  button-primary-hover:
    backgroundColor: "{colors.osd-white}"
    textColor: "{colors.osd-black}"
  button-primary-day:
    backgroundColor: "{colors.signal-amber}"
    textColor: "{colors.lcd-ink}"
  button-primary-day-hover:
    backgroundColor: "{colors.lcd-ink}"
    textColor: "{colors.lcd-ground}"
  field:
    backgroundColor: "{colors.night-field}"
    textColor: "{colors.osd-white}"
    typography: "{typography.value}"
    rounded: "{rounded.none}"
    padding: "8px 10px"
  field-day:
    backgroundColor: "{colors.lcd-field}"
    textColor: "{colors.lcd-ink}"
  nav-item:
    backgroundColor: "transparent"
    textColor: "{colors.night-dim}"
    typography: "{typography.osd-strip}"
    padding: "0 12px"
    height: "44px"
  nav-item-current:
    backgroundColor: "{colors.osd-white}"
    textColor: "{colors.osd-black}"
  nav-item-current-day:
    backgroundColor: "{colors.lcd-ink}"
    textColor: "{colors.lcd-ground}"
  chip:
    backgroundColor: "transparent"
    textColor: "{colors.osd-white}"
    rounded: "{rounded.none}"
    padding: "0 10px"
    height: "32px"
  chip-pressed:
    backgroundColor: "{colors.osd-white}"
    textColor: "{colors.osd-black}"
  segment-filter-active:
    backgroundColor: "{colors.osd-white}"
    textColor: "{colors.osd-black}"
    typography: "{typography.label-button}"
    padding: "0 14px"
    height: "36px"
  switch:
    backgroundColor: "{colors.night-field}"
    textColor: "{colors.night-dim}"
    width: "92px"
    height: "32px"
  stamp:
    backgroundColor: "{colors.osd-black}"
    textColor: "{colors.osd-white}"
    typography: "{typography.stamp}"
    padding: "5px 7px"
  readout-hot:
    backgroundColor: "{colors.signal-amber}"
    textColor: "{colors.osd-black}"
    typography: "{typography.readout}"
    padding: "2px 5px"
---

# Design System: Birdy

## Overview

**Creative North Star: "The Camera's Own Screen"**

Birdy's web UI is the on-screen display of the feeder camera itself, the readout a trail cam burns onto its frames and the setup menu it shows when you open the back. It refuses the dashboard-of-cards admin panel: data is burned onto the photo, settings are menu rows, and the live state of the detector (motion score against the trigger) is always on screen. Nothing floats, nothing is decorated; every mark is either a reading or a control.

The world has two lights that come from the device, chosen by `prefers-color-scheme`. Night is the IR OSD: near-black ground, OSD-white text, signal amber for what is live or primary. Day is a reflective LCD setup screen: grey-green ground, dark segment ink, and no colour at all on the instrument except the one amber "sticker" on the primary button. Burned-in elements (the photo stamp bar, the viewfinder text, the ROI polygon) do not change with the light; they belong to the photo, not the page.

Density is instrument-like but readable by a non-technical household: every setting has a plain-language sans label and explanation, with the real `birdy.toml` key small in mono beside it. Numbers are never hidden.

**Key Characteristics:**
- Two device lights: IR OSD (night) and reflective LCD (day), same geometry.
- Martian Mono for every token, value, key and menu head; system sans only for plain-language explanation.
- Square corners, 1px hairlines, segmented meters, viewfinder corner brackets.
- Selection is an inverted row; amber means live or primary, never "chosen".
- Photos carry a burned-in stamp bar that is identical in both lights.

## Colors

A near-achromatic instrument palette (two grounds with a faint green cast) with one hot signal, amber, and one alarm, REC red.

### Primary
- **Signal Amber** (`signal-amber`): the live and primary colour. Night: the armed state lamp, lit walk-test segments, and the keyboard focus outline and focused-row pointer. Both lights: the SAVE / CLASSIFY primary button fill, walk-test segments past the trigger, the hot readout block when the score crosses the trigger, the unsaved-changes square, the ROI polygon and its handles, the LIVE dot and WATCH AREA text burned into the viewfinder, and the BIRD % on photo stamps.

### Secondary
- **REC Red** (`rec-red-night` / `rec-red-day`): only for the blinking REC lamp (score at or over the trigger) and for errors (camera-error lamp, ERR message line, invalid row label, error hint). The day value is darker so it reads as text on the LCD ground.

### Neutral
- **OSD Black** (`osd-black`): night ground, and the photo stamp bar in both lights.
- **Night Raised / Night Field** (`night-ground-raised`, `night-field`): one step off the ground for inputs and switch wells.
- **OSD White** (`osd-white`): night ink, burned-in viewfinder text, stamp text, and the night selection row fill.
- **Night Dim** (`night-dim`): secondary text, inactive nav, units, keys.
- **Night Rules** (`night-rule`, `night-rule-strong`): row hairlines and the stronger hairline under group heads, the strip, and around controls.
- **Night Segment Off** (`night-seg-off`): unlit meter and slider segments.
- **OSD Dim** (`osd-dim`): muted text burned onto dark frame areas: the stale-frame dot, NO SIGNAL on an empty viewfinder, "no thumbnail" in an empty tile.
- **LCD Ground / Raised / Field** (`lcd-ground`, `lcd-ground-raised`, `lcd-field`): day ground and its two neighbouring steps.
- **LCD Ink** (`lcd-ink`): day text, day "lit" segments and lamp, day selection row fill, day focus outline.
- **LCD Dim, Rules, Segment Off** (`lcd-dim`, `lcd-rule`, `lcd-rule-strong`, `lcd-seg-off`): the day counterparts of the night neutrals, same roles.

### Named Rules
**The Inverted Row Rule.** Selection (current nav item, active segmented filter, pressed watchlist chip, switch ON, button pressed, text selection) is an inverted row: ink-coloured fill with ground-coloured text (OSD white on OSD black at night, LCD ink on LCD ground by day). Amber is never used to mean "selected".

**The One Sticker Rule.** On the LCD ground amber appears only as a fill (the primary button, the hot readout block, meter segments past the trigger, the dirty square), never as text or a thin line on the grey-green. On the LCD, "lit" is dark ink.

**The Burned-In Rule.** Anything drawn onto a photo or the live frame (stamp bar, viewfinder text, ROI polygon) uses the fixed OSD colours (`osd-black`, `osd-white`, `signal-amber`, `osd-dim`) regardless of the page light.

## Typography

**Display / Label / Mono Font:** Martian Mono (self-hosted variable woff2, weight 100–800, width 75–112.5; fallback ui-monospace, SF Mono, Menlo, monospace)
**Body Font:** system-ui (with -apple-system, Segoe UI, Roboto, sans-serif)

**Character:** A condensed, square-shouldered mono speaks for the camera (readings, keys, menu heads, buttons); a plain system sans speaks for the household in full sentences. Numerals are tabular everywhere.

### Hierarchy
- **Verdict** (800, 22px, 1.1, width 87.5%, uppercase): the classify result call on the species page ("BIRD (93%)", or "NOT A BIRD" in dim).
- **Readout** (700, 20px, 1, width 87.5%): the live MOT / TRIG numbers on the walk test; turns into an amber block when hot.
- **Menu head** (700, 12px, 0.14em, width 87.5%, uppercase): the section headings themselves (MOTION, BIRD CHECK, WALK TEST, SPECIES SEEN), followed on the same baseline by a sans sentence in dim.
- **Row label** (sans 600, 15px): the plain-language name of a setting.
- **Body** (sans 400, 15px/1.45): default text.
- **Explain** (sans 400, 13px/1.4, dim, max 62ch): the sentence under each setting.
- **Value** (mono 500, 14px/1.3, width 87.5%): numbers and text inside fields and selects, right-aligned in numeric fields (9ch).
- **Button label** (mono 700, 12px, 0.1em, uppercase): buttons, segmented filters (11px), file picker.
- **OSD strip** (mono 500, 12px, 0.06em, uppercase): nav, lamp, score, clock; the brand is 800 at 13px, 0.12em, width 100%.
- **Key** (mono 400, 11px, 0.02em, dim): the `birdy.toml` key after each explanation, units, captions.
- **Stamp** (mono 500, 11px, 0.03em, width 81%, uppercase): the photo info bar; 75% width and 0 tracking at phone width.

### Named Rules
**The Two Voices Rule.** If the camera would print it (a number, a key, a state, a heading, a button), it is Martian Mono. If a person would say it (a label or an explanation), it is system sans. Never set explanations in mono or values in sans.

**The Narrow Mono Rule.** Martian Mono runs at width 87.5% by default (81% on stamps, 75% on phone stamps); full width only for the BIRDY brand.

## Layout

A sticky 44px OSD strip sits on top of every page (bottom hairline in `rule-strong`): brand, nav as full-height cells, then the live cluster pushed right (state lamp, MOT score, clock). At 640px and below the brand and clock drop out.

Page content is capped at 1500px (species page 960px) with a 16px gutter. Settings is a two-column instrument: viewfinder column (about 1.15fr) sticky under the strip on the left, separated by a hairline, setup menu (min 400px) scrolling on the right, with a sticky save bar at the bottom of the menu. At 900px and below it stacks: the viewfinder goes full-bleed and its column loses the sticky behaviour. At 420px the top-right burn label hides.

The capture grid is `auto-fill` tiles of min 220px with 14px gaps; at 640px it is exactly two columns with 8px gaps. Tables (species) use hairline rows with right-aligned mono numerics.

Rhythm: setting rows are 14px top / 12px bottom with a hairline between; groups are separated by 28px (36px on the species page) and open with a head on a `rule-strong` hairline. Control gaps are 6, 8, 12 or 16px.

## Elevation & Depth

Flat. There are no drop shadows and no layered surfaces; depth is expressed by hairlines (`rule` for rows, `rule-strong` for heads, strip, save bar and control borders) and by the two OSD layers: the page, and the dark photo/frame with text burned onto it. Sticky elements (strip, viewfinder, save bar) separate from scrolling content with a hairline and an opaque ground, never a shadow.

### Shadow Vocabulary
- **Burn halo** (`text-shadow: 0 1px 2px rgba(0,0,0,0.9), 0 0 1px #000`): only on text burned into the live viewfinder, for legibility over an unpredictable photo. It is part of the OSD, not elevation.
- **Field focus ring** (`box-shadow: inset 0 0 0 1px` ink): doubles a focused field's border to 2px. A ring, not a shadow.

### Named Rules
**The Hairline Rule.** Separation is a 1px line or an inverted fill. If something needs to stand out, it gets a stronger hairline or inverts, it does not lift.

## Shapes

Square everywhere: buttons, fields, chips, switches, meters, tiles, stamps and the slider thumb all have 0 radius. The only circles are the 9px status lamps and the 8px LIVE dot, because real indicator LEDs are round. Segmentation is the recurring geometry: the walk-test meter is discrete cells with 2px gaps, and slider tracks and probability bars are 8px tall with a repeating 6px-on / 2px-off cut so they read as LCD segments. The viewfinder carries corner brackets (22px arms, 2px thick, 10px inset; 16px / 8px on phones) instead of a frame border. Photo tiles use a 1px outline, 2px ink on hover.

## Components

### Buttons
Printed-label buttons: square, outlined, uppercase mono.
- **Shape:** square (0), min height 40px, 16px side padding, 1px `rule-strong` border.
- **Default:** transparent fill, ink label. Hover: border goes to ink. Active: the inverted selection row.
- **Primary:** signal-amber fill and border, dark label (SAVE, CLASSIFY). Hover inverts to an ink fill with ground-coloured text. One per view.
- **Disabled:** dim label, dashed border, struck through, `not-allowed` cursor.
- **Transitions:** background, colour and border at 150ms ease-out; all motion is removed under `prefers-reduced-motion`.
- **File picker:** the native file selector button is styled as the default button (40px, mono 700 uppercase).

### Chips
- **Style:** 32px square chips, 1px `rule-strong` border, mono 500 12px, transparent fill.
- **State:** pressed (`aria-pressed="true"`) is the inverted selection row. Used as add/remove toggles for watchlist species.

### Inputs / Fields
- **Style:** field ground (`night-field` / `lcd-field`), 1px `rule-strong` border, 0 radius, mono 500 14px, 8px 10px padding. Numeric fields are 9ch, right-aligned, with a 4ch uppercase mono unit after them.
- **Focus:** border goes to ink plus a 1px inset ring (2px total); the row's label gains a small signal-coloured pointer triangle on its left.
- **Select:** native appearance removed, a two-gradient chevron in ink at the right.
- **Error:** the row label turns REC red and a mono 600 12px REC-red hint line follows the row.

### Sliders
Every numeric setting pairs its field with a full-width range below the explanation. Track is 8px of LCD segments, filled in ink up to the value over `seg-off`; thumb is a square ink post (12×24px) with a ground-coloured border, turning signal on keyboard focus. Wide-range settings (`motion.min_area`, `telegram.cooldown_minutes`, `retention.non_bird_days`) map the track through log1p so the useful low end gets most of the travel.

### Switch
A 92×32px two-cell OFF | ON control drawn from a checkbox: the active half fills, OFF with `rule-strong`, ON with the inverted selection row.

### Navigation
The OSD strip: nav items are full-height cells (12px side padding, 10px on phones) in dim mono uppercase; hover goes to ink; the current page is an inverted row. Segmented filters on the grid (ALL / BIRDS / NOT BIRDS) are a bordered row of 36px cells split by hairlines, active cell inverted.

### Messages
An OSD line, not a toast: bordered in currentColor, mono 500 12px, prefixed with a bold status code: OK in ink, ERR in REC red.

### Photo Stamp (signature)
The info bar a trail cam burns under every photo: OSD-black bar, OSD-white mono text, items spread to both ends (time, BIRD nn% in amber bold or "no nn%" muted; second line MOT score and species). Identical in both lights. Non-bird tiles are desaturated (grayscale 0.6, brightness 0.8) until hover.

### Viewfinder and Walk Test (signature)
The live frame at full column width on a black ground, corner brackets, burned-in mono text in the four corners (LIVE · 1S with an amber dot that greys when the frame is stale, CAM · 320×240, WATCH AREA · N CORNERS in amber, timestamp). The ROI is a 2px amber polygon with a 10% amber fill and square amber handles (outlined in OSD black, 14px while dragged) labelled A, B, C... Under it the walk test: a 22px segmented meter on a log scale (10 / 100 / 1k / 10k) whose lit cells are `signal` at 55% opacity below the trigger and full amber past it, a 2px ink TRIG tick that moves with "Movement to trigger", and the MOT / TRIG readout that becomes an amber block when hot, while the strip lamp switches to a blinking REC.

### Save Bar
Sticky at the bottom of the setup menu, opaque ground with a `rule-strong` top hairline: a mono uppercase state line ("NO UNSAVED CHANGES" / "N UNSAVED CHANGES", with an 8px amber square when dirty) and an 11px dim mono line under it saying whether it applies live, then the primary SAVE button (min 120px).

## Do's and Don'ts

### Do:
- **Do** use the inverted row (ink fill, ground text) for every selected or current state, in both lights.
- **Do** keep amber for live and primary: the armed lamp at night, lit and over-trigger meter cells, the hot readout, the primary button, the ROI and burned-in live marks, BIRD %.
- **Do** pair every setting's plain-language sans label with its `birdy.toml` key in 11px dim mono.
- **Do** separate with 1px hairlines (`rule` for rows, `rule-strong` for heads, sticky bars and control borders).
- **Do** draw meters and bars as segments (6px on, 2px off; 2px gaps in the walk-test meter).
- **Do** give burned-in viewfinder text the burn halo so it reads over any frame.
- **Do** use a log1p slider track when a setting's range spans orders of magnitude.

### Don't:
- **Don't** use amber to mark selection, or as text or a thin line on the day LCD ground.
- **Don't** use REC red for anything but the REC lamp and errors.
- **Don't** round corners on any control, tile or container; only indicator lamps are round.
- **Don't** add drop shadows, cards or raised panels; the only shadow-like effects are the burn halo and the inset focus ring.
- **Don't** set values, keys or heads in the system sans, or explanations in mono.
- **Don't** load fonts or assets from the network; Martian Mono is self-hosted and everything else is system.
