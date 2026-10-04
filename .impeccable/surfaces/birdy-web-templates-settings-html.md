---
version: 1
slug: "birdy-web-templates-settings-html"
primary_target: "birdy/web/templates/settings.html"
related_targets: ["birdy/web/templates/index.html","birdy/web/templates/species.html"]
---

# Birdy web UI: settings (first surface), grid, species

Mode: Operate. Audience: the household (some non-technical people) on phones at the window and on laptops. Jobs: live tuning (ROI, motion, thresholds, with the live score) plus occasional admin (watchlist, location, retention). Constraints: served from the Pi, no remote assets, minimal JS, light + dark. User rejects: twee/cute, heavier/slower, hiding the numbers. Labels: plain language first, with the birdy.toml key small in mono.

## Direction contract

THESIS: Birdy's UI is the camera's own on-screen display, like a trail-cam readout. It refuses the dashboard-of-cards admin panel. Data is burned onto the frame and settings are a setup menu.

OWN-WORLD: Two lights from the device itself. Night = IR OSD: near-black #0C0D0C ground, OSD white #F2F2EE, signal amber #FFB000 only for live/triggered/primary, REC red #FF4D3D for errors. Day = reflective LCD setup screen: grey-green #C9CEBD ground, segment ink #141812, inverted ink rows for the selection. Martian Mono (self-hosted) for every OSD token, value, key and menu head; system sans only for the plain-language explanations. 1px hairlines, square corners, viewfinder corner brackets, segmented meters, no cards, no shadows.

STORY: The household sees exactly what the camera sees and what it's thinking (live score vs trigger mark), changes a value in plain words, and knows whether it applied live or needs a restart.

FIRST VIEWPORT: Sticky OSD strip on top (BIRDY, the GRID/SPECIES/SETUP nav, a state lamp, the clock). Below it the viewfinder: the live frame at full column width with corner brackets, an amber ROI polygon with square corner handles labelled A–D, and burned-in text (LIVE 1s top-left, timestamp bottom-right). A WALK TEST segmented meter under the frame shows the motion score against the trigger tick. On desktop the viewfinder is sticky on the left and the setup menu scrolls on the right; on a phone it stacks. A sticky save bar at the bottom holds the amber SAVE button and the unsaved-changes count.

FORM: Trail-cam OSD / setup menu. My list ranked it #1 (taken as IMPECCABLE'S PICK). Seed d5854ddb. Signature interaction: the walk test. The meter's segments light amber as the live score climbs, the trigger tick moves as you drag "movement to trigger", and a REC lamp blinks when the score crosses it.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
