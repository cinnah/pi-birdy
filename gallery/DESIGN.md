---
name: Birdy gallery
description: The public gallery's recording-sheet world; a daily ledger of birds at each collaborator's feeder, printed on eye-ease paper and typed in.
colors:
  paper: "#eef2e6"
  paper-on-desk: "#e5e9dc"
  desk-dark: "#1b1e22"
  ink: "#1f2a3a"
  rule: "#93a8cc"
  rule-ink: "#46608a"
  ledger-red: "#b5392f"
  tint: "rgba(70, 96, 138, 0.09)"
  selection: "#c9d6ea"
  photo-well: "#0b0b0a"
  evidence-backdrop: "rgba(18, 22, 28, 0.82)"
typography:
  masthead:
    fontFamily: "Libre Franklin, Franklin Gothic Medium, system-ui, sans-serif"
    fontSize: "13px"
    fontWeight: 700
    lineHeight: 1.3
    letterSpacing: "0.2em"
  label:
    fontFamily: "Libre Franklin, Franklin Gothic Medium, system-ui, sans-serif"
    fontSize: "11px"
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: "0.14em"
  grid-head:
    fontFamily: "Libre Franklin, Franklin Gothic Medium, system-ui, sans-serif"
    fontSize: "11px"
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: "0.06em"
  body:
    fontFamily: "Libre Franklin, Franklin Gothic Medium, system-ui, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.5
    fontFeature: "tabular-nums"
  small-print:
    fontFamily: "Libre Franklin, Franklin Gothic Medium, system-ui, sans-serif"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.4
  foot:
    fontFamily: "Libre Franklin, Franklin Gothic Medium, system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 400
    lineHeight: 1.5
  sheet-date:
    fontFamily: "Courier Prime, Courier New, Courier, monospace"
    fontSize: "clamp(26px, 4vw, 40px)"
    fontWeight: 400
    lineHeight: 1.1
    letterSpacing: "-0.01em"
  typed-name:
    fontFamily: "Courier Prime, Courier New, Courier, monospace"
    fontSize: "17px"
    fontWeight: 700
    lineHeight: 1.3
  typed:
    fontFamily: "Courier Prime, Courier New, Courier, monospace"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.5
  typed-compact:
    fontFamily: "Courier Prime, Courier New, Courier, monospace"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.3
  typed-entry:
    fontFamily: "Courier Prime, Courier New, Courier, monospace"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.4
  typed-note:
    fontFamily: "Courier Prime, Courier New, Courier, monospace"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.5
    fontStyle: "italic"
  typed-control:
    fontFamily: "Courier Prime, Courier New, Courier, monospace"
    fontSize: "16px"
    fontWeight: 400
rounded:
  none: "0"
spacing:
  hair: "2px"
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "24px"
  section: "36px"
  foot: "40px"
  gutter: "clamp(16px, 4vw, 48px)"
components:
  sheet:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "0 clamp(16px, 4vw, 48px) 40px"
    width: "1320px"
  masthead:
    textColor: "{colors.ink}"
    typography: "{typography.masthead}"
    padding: "16px 0 10px"
  masthead-nav:
    textColor: "{colors.rule-ink}"
    typography: "{typography.label}"
  label:
    textColor: "{colors.rule-ink}"
    typography: "{typography.label}"
  sheet-date:
    textColor: "{colors.ink}"
    typography: "{typography.sheet-date}"
    padding: "24px 0 12px"
  sheet-date-today:
    textColor: "{colors.ledger-red}"
    typography: "{typography.sheet-date}"
  feeder-head-name:
    textColor: "{colors.ink}"
    typography: "{typography.typed-name}"
  feeder-head-place:
    textColor: "{colors.rule-ink}"
    typography: "{typography.small-print}"
  species-name:
    textColor: "{colors.ink}"
    typography: "{typography.typed-name}"
    padding: "12px 0"
  species-tally:
    textColor: "{colors.rule-ink}"
    typography: "{typography.typed}"
  row-cell:
    rounded: "{rounded.none}"
    padding: "10px 12px 12px 0"
  row-hover:
    backgroundColor: "{colors.tint}"
  print:
    backgroundColor: "{colors.rule}"
    rounded: "{rounded.none}"
    width: "100%"
  entry:
    textColor: "{colors.ink}"
    typography: "{typography.typed-entry}"
    padding: "6px 0 0"
  entry-latest:
    textColor: "{colors.ledger-red}"
    typography: "{typography.typed-entry}"
  entry-who:
    textColor: "{colors.rule-ink}"
    typography: "{typography.typed-entry}"
  blank-cell:
    textColor: "{colors.rule-ink}"
    typography: "{typography.typed-entry}"
  totals-cell:
    textColor: "{colors.ink}"
    typography: "{typography.typed}"
    padding: "10px 12px 10px 0"
  note:
    textColor: "{colors.rule-ink}"
    typography: "{typography.typed-note}"
    padding: "9px 0"
  empty:
    textColor: "{colors.rule-ink}"
    typography: "{typography.typed-note}"
    padding: "16px 0"
  month-cell:
    textColor: "{colors.ink}"
    typography: "{typography.typed}"
    padding: "7px 0"
  month-cell-today:
    backgroundColor: "{colors.tint}"
    textColor: "{colors.ink}"
    typography: "{typography.typed}"
  month-head:
    textColor: "{colors.rule-ink}"
    typography: "{typography.grid-head}"
    padding: "10px 0 7px"
  month-head-today:
    textColor: "{colors.ledger-red}"
    typography: "{typography.grid-head}"
  month-species:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    typography: "{typography.typed}"
    padding: "7px 10px 7px 0"
  ledger-cell:
    textColor: "{colors.ink}"
    typography: "{typography.typed}"
    padding: "9px 12px 9px 0"
  ledger-head:
    textColor: "{colors.rule-ink}"
    typography: "{typography.label}"
    padding: "12px 12px 9px 0"
  section-heading:
    textColor: "{colors.rule-ink}"
    typography: "{typography.label}"
    padding: "0 0 8px"
  section-heading-typed:
    textColor: "{colors.ink}"
    typography: "{typography.typed}"
  logday:
    textColor: "{colors.ink}"
    typography: "{typography.typed-name}"
    padding: "0 0 6px"
  logday-today:
    textColor: "{colors.ledger-red}"
    typography: "{typography.typed-name}"
  logday-weekday:
    textColor: "{colors.rule-ink}"
    typography: "{typography.label}"
  scroll-hint:
    textColor: "{colors.rule-ink}"
    typography: "{typography.label}"
    padding: "6px 0 0"
  filter-label:
    textColor: "{colors.rule-ink}"
    typography: "{typography.label}"
  select:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    typography: "{typography.typed-control}"
    rounded: "{rounded.none}"
    padding: "4px 28px 4px 4px"
    height: "44px"
  foot:
    textColor: "{colors.rule-ink}"
    typography: "{typography.foot}"
    padding: "10px 0 0"
  evidence-dialog:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "10px 10px 6px"
  evidence-photo:
    backgroundColor: "{colors.photo-well}"
    rounded: "{rounded.none}"
  evidence-caption:
    textColor: "{colors.ink}"
    typography: "{typography.typed-entry}"
    padding: "4px 0 0"
    height: "44px"
  evidence-count:
    textColor: "{colors.rule-ink}"
    typography: "{typography.typed-entry}"
  dialog-button:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "0"
    size: "44px"
  thumb-button:
    backgroundColor: "{colors.rule}"
    rounded: "{rounded.none}"
    padding: "0"
    width: "48px"
    height: "36px"
  thumb-button-on:
    backgroundColor: "{colors.rule}"
    rounded: "{rounded.none}"
    width: "48px"
    height: "36px"
  error:
    textColor: "{colors.ledger-red}"
    typography: "{typography.typed}"
---

# Design System: Birdy gallery

This record covers the public, static gallery in `gallery/` (home sheet, feeder month, species card). The Pi's LAN interface is a different world ("trail-cam OSD", Martian Mono) recorded in the root `DESIGN.md`; nothing here applies to it.

## Overview

**Creative North Star: "The Recording Sheet"**

The gallery is a printed daily record, one sheet per day: species down the left, one ruled column per feeder, and in every cell the day's best photograph pasted on the paper with its count and last time typed beneath. The form itself is printed (Libre Franklin, tracked uppercase labels in blue rule ink on eye-ease green paper); what happened is typed on it (Courier Prime for every species name, count, time and date). The reader is a fellow collaborator comparing feeders during the day, so the sheet is dense, aligned and legible, never showy. The build refuses the photo-card grid and the stats dashboard: there are no cards, no boxes, no panels, only rules, columns and prints.

Hierarchy comes from the ledger's own devices, not from colour or weight. Rank is carried by row size (the busiest species gets the widest print), every number sits in a fixed-width digit column, and a single ledger red is held back for today and the latest entry. The sheet keeps one 1px ink weight for every rule and every print frame, and no greys: secondary text is simply the same blue as the printed rules, dark enough to read. Dark mode does not restyle the sheet; it puts the same paper on a dark desk and lets it cast a shadow.

Motion is a single gesture: the rows fill in once, top to bottom, when the sheet loads; the only other movement is a print sharpening from its thumbnail inside the evidence dialog. Nothing hovers, slides or glows. Under reduced motion the sheet is a single still.

**Key Characteristics:**
- One sheet, one paper (`paper`), one ink (`ink`), one rule colour (`rule`), one accent (`ledger-red`)
- Printed form in Libre Franklin, typed entries in Courier Prime; the role decides the face
- Square corners everywhere; prints wear a 1px ink frame like a pasted photograph
- Secondary text is rule ink, never a grey
- Rank by size, counts in digit columns, red only for today and the latest entry
- Photos are the content of every cell, never thumbnails in a sidebar

## Colors

A two-ink ledger palette: blue-black ink and blue rule ink printed on eye-ease green paper, with one ledger red held for the present moment.

### Primary
- **Ledger Red** (`ledger-red`): the only accent. Used for the masthead and foot double rule, the sheet date when it is today, the latest entry's time, today's column head on the monthly sheet, the day heading in the photo log when it is today, the marked thumbnail in the evidence strip, error text, and the focus ring. Nowhere else.

### Neutral
- **Eye-ease Paper** (`paper`): the sheet. In light mode it is also the page background (`--desk` equals `--paper`), so the sheet has no visible edge; the monthly species column uses it as a sticky backing.
- **Paper on Desk** (`paper-on-desk`): the sheet in dark mode, a shade duller so it reads as paper under lamplight rather than a white block.
- **Dark Desk** (`desk-dark`): the page behind the sheet in dark mode only. The sheet never goes dark.
- **Blue-black Ink** (`ink`): all primary text, every print frame, the select's underline, the dialog's button borders, and the cut edge of a sideways-scrolling month on a phone.
- **Rule Blue** (`rule`): every printed line (row rules, column rules, head and ledger rules) and the placeholder behind a loading print or thumbnail.
- **Rule Ink** (`rule-ink`): printed labels, column heads, the sheet number label, places and local times under feeder names, tallies, blank cells, notes, empties, the foot, the evidence count, and the feeder name typed under a print on a phone. This is the whole secondary register.
- **Finger Tint** (`tint`): a 9% wash of rule ink across a hovered row and down today's column on the monthly sheet.
- **Selection** (`selection`): text selection background, a lighter rule blue.
- **Photo Well** (`photo-well`): the near-black behind the full-size evidence print while it letterboxes.
- **Evidence Backdrop** (`evidence-backdrop`): the dimmed desk behind the open dialog.

### Named Rules
**The One Red Rule.** Ledger red marks the present and the focus, and nothing else: today, the latest entry, the current thumbnail, the keyboard focus ring, the two double rules that close the sheet. A new surface that wants a second accent does not get one.

**The No Grey Rule.** There is no grey in the palette. Anything that would be "muted" is rule ink (`rule-ink`); anything that would be a "light border" is rule blue (`rule`). Both are the same hue as the printed form.

**The One Ink Weight Rule.** Every rule, frame and underline is 1px solid. The only heavier line is the 3px double rule in ledger red at the masthead and the foot.

## Typography

**Display Font:** Courier Prime (with Courier New, Courier, monospace); self-hosted woff2 regular, bold, italic
**Body Font:** Libre Franklin (with Franklin Gothic Medium, system-ui, sans-serif); self-hosted variable woff2, 400–700
**Label/Mono Font:** the same two faces, split by role rather than by size

**Character:** The sheet is a printed form that has been filled in on a typewriter. Libre Franklin is the print: it carries the masthead, every tracked uppercase label, column heads, places, the foot and the body default. Courier Prime is the typing: every value a human or the camera entered, from the large sheet date down to a dash in an empty cell. If a string is data (a species, a count, a time, a date, a feeder's name as a heading), it is typed; if it names a field or a section, it is printed. Body text defaults to Libre Franklin 15px with tabular numerals, so even stray numbers line up.

### Hierarchy
- **Sheet date / display** (`sheet-date`; 400, clamp(26px, 4vw, 40px), 1.1, tracked -0.01em, Courier Prime): the date on the daily sheet, a feeder's name on its month, a species' name on its card. Red when the sheet is today. A 17px sub-line in ink beneath it carries place and "watching since".
- **Masthead** (`masthead`; 700, 13px, tracked 0.2em, uppercase, Libre Franklin): the word "birdy" at the top margin. The nav (Daily record · About, labelled "Site") beside it are labels.
- **Typed name / title** (`typed-name`; 700, 17px, 1.3, Courier Prime): species names in rows, feeder names in column heads, the day in the photo log, the bird's name under a log print.
- **Typed / body values** (`typed`; 400, 15px, 1.5, Courier Prime): counts, times, totals, monthly cells, ledger cells, the sheet number, the section heading's typed range. Entries under a print use the slightly tighter `typed-entry` (1.4).
- **Typed note** (`typed-note`; italic 15px Courier Prime, rule ink): the optional note under the head and every "nothing here" line.
- **Body** (`body`; 400, 15px, 1.5, Libre Franklin, tabular numerals): the page default; little text reaches it unstyled.
- **Label** (`label`; 600, 11px, 1.3, tracked 0.14em, uppercase, Libre Franklin, rule ink): printed field names: SPECIES, SHEET, section headings, ledger heads, weekday beside a log day, the scroll hint, the filter's "Show", the feeder name typed under a cell on a phone.
- **Typed, compact** (`typed-compact`; 400, 14px, 1.3, Courier Prime): typed text where a phone's width is short: species names down the monthly sheet, ledger cells, the caption in the photo dialog.
- **Grid head** (`grid-head`; 600, 11px, tracked 0.06em, Libre Franklin, rule ink): the day numbers and hours across the monthly sheet; their month abbreviation is uppercase and tracked 0.14em like a label.
- **Small print** (`small-print`; 13px/1.4 Libre Franklin, rule ink): place and local time under a feeder's name; the time itself is typed.
- **Foot** (`foot`; 12px/1.5 Libre Franklin, rule ink): the provenance line and "record kept since" under the closing red rule.

### Named Rules
**The Print-or-Typed Rule.** Every string is either printed (Libre Franklin) or typed (Courier Prime); the role decides, never the size. Values are typed. Field names are printed. No third face, and no system display face.

**The Two Weights Rule.** Courier Prime is 400 for values and 700 for names and totals; Libre Franklin is 400 for prose and 600–700 for labels and the masthead. Nothing is lighter than 400 and nothing is bolder than 700.

## Layout

The sheet is a single column of ruled rows, 1320px wide at most, centred, full viewport height, with side gutters of clamp(16px, 4vw, 48px) and 40px below the foot. There is no outer chrome: the masthead sits in the top margin above a 3px double red rule and the foot sits below one.

**Daily sheet (`.daily .row`).** Each row is a grid: a species column `minmax(150px, 230px)` then one `minmax(0, 1fr)` column per feeder (`--n`). Rows are separated by a 1px rule beneath; columns by a 1px rule to the left of every cell after the first. Cells pad `10px 12px 12px 0` (12px left when ruled). The column-head row is the same grid with 8px vertical padding; the totals row too, with 10px. Rows are ordered by visits and the top row is the tallest: print widths are capped by rank at 460 / 340 / 250px when there are several feeders, and at 100% / 76% / 58% of the column when there is one. Hovering a row washes it with `tint` across every feeder.

**Monthly sheet (`.mgrid`).** A grid of `minmax(150px, 190px)` species column, one `minmax(28px, 1fr)` column per day (`--d`, 30 by default) and a 56px total column, 1040px minimum, inside a horizontally scrolling `.month` that opens scrolled to today at the right edge. The species column is sticky on the left with the paper as backing and a 1px rule to its right; the total column has a 1px rule to its left. The hour grid is the same with 24 columns and 880px minimum. Count cells carry an **ink wash**: `data-w` 1–4 lays rule ink over the cell at 9 / 20 / 32 / 46 % (a `background-image`, so today's tint shows through), stepped by the cell's share of the busiest cell in its own row, so a rare bird's pattern reads as clearly as a common one's; the number stays in every cell and a `.washkey` line under the hour table explains the steps. **By hour (`.hgrid`)** is the same grid turned to hours: one row per species (from `stats.species[].hours`, all time) and an **All birds** row (`stats.per_hour`), only the hours between the first and last active one plus one on each side (`--h` columns), then a 72px **Busiest** column (the hour with the most visits, typed `7 h`) before the total. On a phone it opens on the morning while the days open on today. A manifest without per-species hours keeps the single **Visits** row.

**Ledger (`.ledger`).** Species card: `minmax(150px, 1fr) 7ch 10ch 10ch`, each cell ruled beneath, numbers right-aligned with a 32px right inset.

**Photo log (`.prints`).** `repeat(auto-fill, minmax(250px, 1fr))` with 18px/16px gaps under a ruled day heading; the latest-day strip on a feeder's month uses 230px minimums.

**Spacing rhythm.** The sheet works in small steps: 2px between a label and its value, 4–6px between a print and its entry, 8–12px inside rows, 16px at the masthead, 24px above the sheet date, 36px above a section heading, 40px to the foot. No step larger than 40px exists.

**Phone (max-width 719px).** The sheet keeps the same paper with 16px gutters. The head stacks (date, then meta left-aligned). Each species row becomes a section: the species line spans the full width with its tally inline, and the feeders' cells sit two-up beneath it (`1fr 1fr`, 12px gap, no column rules), so a species stays adjacent across feeders. A rank-1 cell spans both columns; prints lose their width cap. The feeder's name appears as a label above each cell when there are several feeders. Column heads become a wrapped line of feeder names. The monthly sheet narrows to a `minmax(120px, 140px)` species column, 24px day columns and 48px total (900px minimum), shows the scroll hint, and gets a 1px ink right border as the cut edge of the sheet continuing past the screen. Ledger columns tighten to `1fr 6ch 8ch 8ch` at 14px; print grids go to two columns.

**Dark mode (min-width 720px).** The sheet gets 24px of desk above and below (`min-height: calc(100vh - 48px)`) so the desk shows around it.

## Elevation & Depth

The sheet is flat. Depth is drawn, not lit: rules separate rows, a 1px ink frame makes a photograph a print, a sticky paper-backed column makes the month scroll under it. There are no shadows in light mode at all, and no raised or inset surfaces anywhere. Two shadows exist, both for a surface lifted off the desk rather than off the sheet.

### Shadow Vocabulary
- **Sheet on desk** (`box-shadow: 0 18px 50px rgba(0, 0, 0, .45)`): the sheet in dark mode at ≥720px only; the paper lying on the dark desk.
- **Evidence on desk** (`box-shadow: 0 24px 70px rgba(0, 0, 0, .5)`): the open dialog, over a backdrop of `evidence-backdrop`.

### Named Rules
**The Flat Sheet Rule.** Nothing on the sheet casts a shadow. The sheet itself casts one only when there is a desk beneath it (dark mode, wide viewport); the evidence dialog casts one because it is a second object on the desk.

## Shapes

Every corner is square: `border-radius: 0` is the only radius, including the select, every button and the dialog. Lines are 1px solid in one of two inks: rule blue for the printed form, ink for frames (prints, thumbnails, dialog buttons, the select's underline, the phone month's cut edge). The two double rules (3px double, ledger red) frame the sheet top and bottom. Prints are 4:3 at every size (`aspect-ratio: 4 / 3`, cover-cropped in cells, contained in the dialog), framed by 1px ink, with rule blue behind them while they load. Nothing is clipped, rounded, pill-shaped or outlined by a border-box; a group of content is a run of ruled rows, not a box. The one hairline drawn at a non-rule weight is the select's bottom border, still 1px ink.

## Components

### Masthead
- **Character:** the sheet's letterhead, printed in the top margin.
- **Layout:** flex, baseline-aligned, 16px gap, `16px 0 10px` padding, closed by a 3px double rule in ledger red.
- **Title:** "birdy" at `masthead` in ink, a link with the hit-area helper; **nav** at `label` in rule ink. Links underline on hover (1px, 3px offset).

### Sheet head
- **Character:** the form's own heading: a label, the date typed large, the sheet number at the right margin.
- **Layout:** two-column grid `1fr auto`, items aligned to the baseline end, `24px 0 12px` padding, 1px rule beneath. On a phone it becomes one column with the meta beneath.
- **Date:** `sheet-date` in ink; `.today` turns it ledger red. A `.sub` line (17px, untracked, ink) carries place and tenure on a feeder's month.
- **Meta field:** right-aligned `typed` with a `label` above (2px gap), for "Sheet", "Monthly record · all time" or "Species card · all feeders".
- **Note:** optional italic typed line in rule ink, `9px 0`, ruled beneath.

### Feeder head
- **Character:** a column heading that is also a link to the feeder's month.
- **Name** in `typed-name`; **place and local time** beneath in `small-print` rule ink, the time typed (`.now`). The whole head is a hit area. On a phone the heads wrap into a single line without column rules.

### Species cell
- **Name** as a link in `typed-name` with a 12px vertical hit area folded into negative margins; **tally** beneath in `typed` rule ink ("×12 · 2 feeders"). On a phone the cell spans the row and the tally sits inline at the right.

### Print and typed entry
- **Print:** a block link, 100% wide up to `--w` (rank cap), holding a 4:3 `object-fit: cover` image framed 1px ink on a rule-blue placeholder. No hover treatment beyond the cursor; the link does not underline.
- **Entry:** a wrapping flex line 6px beneath, `typed-entry`, 2px/14px gaps: a count `×5` right-aligned in a 3ch column, then the time; the latest entry's time is ledger red with a screen-reader note. In the log the species name leads in bold on its own line; on the species card the feeder (`.who`) follows in rule ink.
- **Blank cell:** a typed dash in rule ink, no frame, no placeholder.

### Totals row
- The same row grid, `typed` 15px, `10px 12px 10px 0`: species count at the left, "n species · n visits" per feeder.

### Monthly ledger
- **Cells** (`month-cell`): centred `typed`, `7px 0`, ruled beneath; a zero is left empty, not written. **Today's column** is washed with `tint` and its head is ledger red. **Totals** (`.tot`, `.mt`) are bold; the total column is ruled to its left.
- **Heads** (`month-head`): `grid-head` with a 10px uppercase month abbreviation above the first day and the first of each month.
- **Species column:** sticky, paper-backed, left-aligned, ellipsised on wide screens and wrapped at 14px on a phone; the name is a block link with the hit area.
- **Scroll hint** (`scroll-hint`): a label under the grid, shown only on a phone.

### Ledger
- A four-column grid with `label` heads padded 12px above, `typed` cells ruled beneath, right-aligned numbers; the empty line spans every column and drops its rule.

### Section heading
- `h3.sec`: a `label` in rule ink with 36px above, 8px below and a rule beneath; flex space-between so a typed value (date range, filter) can sit at the right margin in ink. No eyebrow above it, no decoration beside it.

### Log day
- `h4.logday`: the day in `typed-name` (ledger red when today) with the weekday as a `label` beside it, baseline-aligned, 22px above, 6px below, ruled.

### Filter select
- **Label** (`filter-label`): `label` text "Show" in rule ink, inline-flex with a 10px gap.
- **Select:** transparent, borderless except a 1px ink bottom border, `typed-control` 16px in ink, 44px minimum height, `4px 28px 4px 4px` padding, square. The browser's native arrow is kept. Focus shows the shared ring.

### Evidence dialog
- **Character:** the photo lifted off the sheet onto the desk, as large as the viewport allows, with nothing on it; the day's neighbours sit in the bar beneath, a tap or a key away.
- **Surface:** paper background, ink text, no border, no radius, up to the full viewport, the "evidence on desk" shadow over `evidence-backdrop`. The figure pads `10px 10px 6px` (`8px 8px 4px` on a phone).
- **Photo:** `width: min(100vw - 20px, (100vh - 20px - 54px) * 4/3)`, 4:3 contained, 1px ink frame on `photo-well`. While the full-size file loads (`aria-busy`) it shows the thumbnail blurred 4px and sharpens with `filter .3s` on the shared ease. The photo is the stepping control: its left third goes back, the rest forward; a lone photo closes on tap. Arrow keys step too.
- **Bar** (`figcaption`): one 44px row, `4px 0 0`, `typed-entry`: the caption (ellipsis on one line) grows, then the thumbnail strip, then the close control. "Photo n of m" is spoken only (`.sr`). On a phone the bar wraps: caption on its own line, strip and close on the next.
- **Thumbnail strip:** flex, 6px gaps; each thumb a 48×36px button with a 1px ink border on rule blue, cover-cropped image. The current one (`.on`) swaps its border to ledger red and adds a 1px red outline, offset 0. Hidden when the cell has one photo.
- **Close** (`dialog-button`): 44×44px, borderless on the paper, a 24px 1.25-stroke inline SVG cross in ink that turns ledger red on hover. There are no previous/next buttons and nothing is ever placed over the photo.

### Helpers
- **`.hit`:** inline-block with `12px 0` padding and matching negative margins, giving text links a 44px-class touch target without moving the type.
- **`.sr`:** the visually-hidden pattern (1px, clip-path inset 50%).
- **Focus:** `outline: 2px solid ledger-red; outline-offset: 2px` on every focusable element.
- **Error** (`error`): `typed` in ledger red.

### Motion
- **Fill:** `.daily .row`, `.log`, `.mgrid` and `.ledger` animate `fill` once (`from { opacity: 0; transform: translateY(4px) }`), 0.4s on `cubic-bezier(.2, .7, .2, 1)`, `both`, delayed `--i × 40ms` where `--i` is the row index capped at 12.
- **Sharpen:** the dialog image's blur-to-sharp, `filter .3s` on the same ease.
- **Reduced motion:** both are removed; nothing else in the build moves.

## Do's and Don'ts

### Do:
- **Do** type every value in Courier Prime (species, counts, times, dates, feeder names as headings) and print every field name in Libre Franklin at `label`: 600, 11px, tracked 0.14em, uppercase, rule ink.
- **Do** keep every rule, frame and underline at 1px solid, in rule blue for the printed form and ink for frames; reserve the 3px double red rule for the masthead and the foot.
- **Do** use ledger red only for today, the latest entry, the marked thumbnail, errors and the focus ring.
- **Do** set secondary text in rule ink (`rule-ink`) and loading placeholders in rule blue (`rule`).
- **Do** carry rank by size (print width caps 460 / 340 / 250px, or 100% / 76% / 58% for a single feeder) and keep counts in a right-aligned 3ch digit column.
- **Do** frame every photograph 1px ink at 4:3 and let it fill its cell; on a phone, keep a species' cells adjacent (two-up under the species line).
- **Do** give every text link the `.hit` 12px vertical hit area and every control a 44px minimum.
- **Do** keep dark mode as the same sheet (`paper-on-desk`) on a dark desk (`desk-dark`) with the sheet-on-desk shadow at ≥720px; never invert the paper.
- **Do** let rows fill in once on load (0.4s, 40ms stagger, capped at the twelfth row) and hold everything still under reduced motion.

### Don't:
- **Don't** use grey anywhere: no grey text, grey borders, grey placeholders or grey disabled states.
- **Don't** round a corner, add a card, a box, a panel, a pill or a filled button; content is a run of ruled rows and a print on the paper.
- **Don't** put an eyebrow or kicker above a heading; a section heading is the label itself, with its typed value at the right margin.
- **Don't** make photos secondary: no thumbnail rails, no icon-sized prints, no cropping to a square.
- **Don't** introduce a second accent, a tinted background other than `tint`, or a coloured fill for state; today's column is a 9% wash, not a highlight bar.
- **Don't** add a system or third typeface, a weight below 400 or above 700, or a glyph icon font; the three dialog icons are inline SVG strokes.
- **Don't** add hover motion, transitions on rules or rows, or a second load animation; the fill and the sharpen are the whole motion grammar.
- **Don't** cast shadows on the sheet; only the sheet itself (dark desk) and the evidence dialog cast one.

## About page (`about.html`)

A typed note tucked into the record: masthead and sheet head as on the other pages, then `article.about` whose sections are form rows. On sheets 1000px and wider each `section` is a grid `230px minmax(0, 62ch)`: the printed `.label` (h3) sits in the margin column that the daily sheet gives to species, the entry beside it. Narrower, the label sits above and the article is capped at 62ch (measured in Courier, so `article` sets `font-family: var(--type)`). Body copy, list items and `dd`s are Courier Prime 15/1.65; sections are divided by the 1px rule with 28px above. Definition lists (`.about dl`) are two ruled columns (`minmax(120px,160px) 1fr`), one column under 720px. The flow diagram is a `pre` (13px Courier) in a 1px rule box that fits the column on desktop and scrolls sideways on phones. The rig photo slot is a `figure`: a ruled 4:3 `.plate` in `--tint` with a typed line until `about/rig.jpg` exists. In-text links underline in rule ink; the index line and the footer carry the 44px targets. No JavaScript, no greys, red only on the double rules and focus. Every masthead carries an About link: `.mast nav` (labelled "Site") is a flex row with a 20px gap, and `.hit` pads links to 44px.
