---
version: 1
slug: "gallery-index-html"
primary_target: "gallery/index.html"
related_targets: ["gallery/site.html","gallery/species.html","gallery/app.js","gallery/style.css"]
---

# Public gallery (birdy.akibeo.com)

Scope: the three static pages in `gallery/` (home, feeder sheet, species card) plus `app.js`/`style.css`. Visitor mode: Experience. Audience: fellow collaborators who run their own feeder camera, checking on a phone or laptop during the day; comparing feeders side by side matters most. Job: see what came to each feeder today, compare, open the evidence, turn to a feeder's month. Content: `sites.json` + each site's `manifest.json` (published photos, stats). Constraints: static files, no build step, nothing runs on the server; self-hosted fonts only; light and dark; never loud or gimmicky; photos are never secondary.

## Direction contract

THESIS: The gallery is a daily recording sheet: one printed form per day, species down the side, one column per feeder, every cell the day's best photo with its count and last time. It refuses the photo-card grid and the stats dashboard.

OWN-WORLD: Eye-ease ledger paper `#eef2e6`; horizontal rules in blue rule ink `#93a8cc` (printed labels in the same blue, darkened to `#46608a` so they read at 4.5:1); one red double rule and a single ledger-red accent `#b5392f` reserved for today and the latest entry; blue-black ink `#1f2a3a`. Form print in Libre Franklin (tracked small caps labels and heads); entries typed in Courier Prime (species, counts, times, dates). One 1px ink weight for every rule and frame. No greys: secondary text is rule ink. Photos sit in ruled cells with square corners and a hairline ink frame, like prints pasted on the sheet. Dark mode is the same sheet on a dark desk.

STORY: A collaborator opens it at the window and reads the sheet row by row: who came, to which feeder, how often, when. They compare cells across feeders, tap a cell to unfold the day's evidence, and turn to a feeder's monthly sheet or a species card.

FIRST VIEWPORT: Head on the top rule: DAILY RECORD (small caps, left), the date typed large, feeder count right. Column heads: SPECIES, then one head per feeder (name in ink; place and local time in rule ink; the head links to the monthly sheet). Rows ordered by visits: species and total at the left (3/12), one cell per feeder with the photo filling the cell and `×5  18:51` typed beneath; an empty cell is blank ruled paper with a typed dash. The tallest row is the first. Totals rule at the foot. Phone: each species keeps its row and the feeders' cells stack inside it, two-up, so the same species stays adjacent across feeders (the user's answer: comparing feeders side by side matters most); a single feeder's prints fill their column at every width.

FORM: The Recording Sheet, candidate 4 of 7 on the ordered list (contact sheet, hide sightings board, barograph day strip, recording sheet, specimen drawer, observatory sightings log, field-guide plate); seed key 15432a21. Signature interaction: tapping a cell opens its evidence in a dialog, the day's keepers of that species as a row of framed thumbnails under the print with the current one marked in red, previous/next and arrow keys stepping through them (a dialog rather than an inline unfold because the full-resolution evidence needs the whole viewport and the keyboard flow stays protected); hovering a row runs a finger along the ledger line across every feeder. Motion grammar: rows fill in once with a 40 ms stagger on load; nothing else moves; reduced motion holds the sheet as a single still. Raises kept from the declined hand: rank carried by row size; counts in fixed-width digit columns; one tally voice for every number; one accent for the active entry; one ink weight; no greys.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance.
