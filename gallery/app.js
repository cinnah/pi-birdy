/* birdy gallery: renders the daily record, each feeder's monthly sheet and the species cards
   from sites.json + sites/<id>/manifest.json. No build step. */
const BIRDY = (() => {
  const main = () => document.getElementById("main");
  // Every string that comes from sites.json, a manifest or the URL goes through esc() before it is injected as HTML.
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const q = (name) => new URLSearchParams(location.search).get(name);
  const day = (iso) => String(iso ?? "").slice(0, 10);   // captured_at carries the site's offset: the local date is the prefix
  const hm = (iso) => String(iso ?? "").slice(11, 16);
  const instant = (iso) => Date.parse(iso) || 0;          // compare moments, not strings (offsets may differ between sites)
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const MONTHS_LONG = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
  const DOW = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
  // Dates are read from the string so the site's own calendar day is kept, whatever the viewer's zone.
  const parts = (iso) => { const d = day(iso); return d.length === 10 ? [+d.slice(0, 4), +d.slice(5, 7), +d.slice(8, 10)] : null; };
  const utc = (p) => Date.UTC(p[0], p[1] - 1, p[2]);
  const dayShort = (iso) => { const p = parts(iso); return p ? `${p[2]} ${MONTHS[p[1] - 1]}` : day(iso); };
  const dayLong = (iso) => { const p = parts(iso); return p ? `${dayShort(iso)} ${p[0]}` : day(iso); };
  const dayFull = (iso) => { const p = parts(iso); return p ? `${DOW[new Date(utc(p)).getUTCDay()]} ${p[2]} ${MONTHS_LONG[p[1] - 1]} ${p[0]}` : day(iso); };
  const dow3 = (iso) => { const p = parts(iso); return p ? DOW[new Date(utc(p)).getUTCDay()].slice(0, 3) : ""; };
  const dayDiff = (a, b) => { const pa = parts(a), pb = parts(b); return pa && pb ? Math.round((utc(pb) - utc(pa)) / 86400000) : 0; };
  const name = (p) => p.species || "Unidentified bird";
  // The site's current local date and time: shift "now" by the site's UTC offset, then read the UTC fields.
  const siteClock = (site) => new Date(Date.now() + (site.utc_offset_minutes || 0) * 60000).toISOString();
  const siteToday = (site) => siteClock(site).slice(0, 10);
  const siteNow = (site) => siteClock(site).slice(11, 16);
  const localToday = () => new Date(Date.now() - new Date().getTimezoneOffset() * 60000).toISOString().slice(0, 10);   // the viewer's own calendar day
  const speciesHref = (s) => "species.html?name=" + encodeURIComponent(s);
  const siteHref = (id) => "site.html?id=" + encodeURIComponent(id);
  const siteBase = (id) => "sites/" + encodeURIComponent(id) + "/";
  const plural = (n, w) => `${n} ${w}${n === 1 || w.endsWith("s") ? "" : "s"}`;   // "1 visit", "3 visits", "3 species"

  async function loadSites() {
    const r = await fetch("sites.json", { cache: "no-cache" });
    if (!r.ok) throw new Error(`The list of feeders couldn't be loaded (HTTP ${r.status}). Try again in a minute.`);
    const sites = await r.json();
    if (!Array.isArray(sites)) throw new Error("The list of feeders is damaged. Try again in a minute.");
    return sites;
  }
  // The manifest shape is a contract (version 1); anything else makes the site "unavailable" instead of breaking the page.
  const validManifest = (m) =>
    !!m && m.version === 1 && !!m.site && Array.isArray(m.photos) && !!m.stats &&
    Array.isArray(m.stats.species) && Array.isArray(m.stats.per_day) && Array.isArray(m.stats.per_hour);
  async function loadManifest(id) {
    try {
      const r = await fetch("sites/" + encodeURIComponent(id) + "/manifest.json", { cache: "no-cache" });
      if (!r.ok) return null;
      const m = await r.json();
      return validManifest(m) ? m : null;
    } catch { return null; }
  }
  async function loadAll() {
    const sites = await loadSites();
    const manifests = await Promise.all(sites.map((s) => loadManifest(s.id)));
    return sites.map((s, i) => ({ ...s, manifest: manifests[i] }));
  }

  // --- counting -------------------------------------------------------------------
  // Visits of a species on a local day: the Pi's own count when the manifest carries one (stats.species[].days),
  // otherwise the published photos of that day (a lower bound: at most a few per species are published).
  const visitsOn = (m, species, d) => m.stats.species.find((x) => x.name === species)?.days?.[d] ?? null;
  const hasDayCounts = (m) => m.stats.species.some((x) => x.days);
  // Group one day's photos by species: best photo, every photo of the strip (newest first), the count, the latest moment.
  function groupByDay(m, d) {
    const groups = new Map();
    for (const p of m.photos.filter((p) => day(p.captured_at) === d)) {
      const g = groups.get(name(p)) || { species: name(p), best: p, items: [], latest: 0 };
      g.items.push(p);
      if ((p.confidence ?? 0) > (g.best.confidence ?? 0)) g.best = p;
      g.latest = Math.max(g.latest, instant(p.captured_at));
      groups.set(name(p), g);
    }
    for (const g of groups.values()) {
      g.items.sort((a, b) => instant(b.captured_at) - instant(a.captured_at));
      g.count = visitsOn(m, g.species, d) ?? g.items.length;
      g.last = g.items[0].captured_at;
    }
    return groups;
  }

  // --- prints and the evidence strip ----------------------------------------------------
  const STRIPS = [];   // each print links to one strip: the photos it belongs with, for the dialog's previous/next
  const stripItem = (site, p, who) => ({
    src: siteBase(site.id) + p.photo, thumb: siteBase(site.id) + p.thumb,
    caption: `${name(p)} · ${dayLong(p.captured_at)}, ${hm(p.captured_at)}${p.confidence == null ? "" : " · " + Math.round(p.confidence * 100) + "%"}${who ? " · " + who : ""}`,
  });
  // A print pasted on the sheet with its typed entry. `entry` is already-escaped HTML.
  // `sizes` tells the browser how wide the print renders, so a wide cell gets the full photo instead of a blurred thumbnail.
  // On the feeder and species pages (TO_SHEET) a print goes to its day's sheet, which opens it in the lightbox (?site&photo).
  let TO_SHEET = false;
  function print(site, p, strip, idx, entry, sizes = "(max-width: 719px) 45vw, 320px", cls = "") {
    const base = siteBase(site.id);
    const href = TO_SHEET ? `${sheetHref(day(p.captured_at))}&site=${encodeURIComponent(site.id)}&photo=${encodeURIComponent(p.photo)}` : base + p.photo;
    // alt is empty on purpose: the entry beneath names the bird, so the link isn't read twice.
    return `<figure class="cell ${cls}">
      <a class="print" href="${esc(href)}" ${TO_SHEET ? "data-sheet" : `data-strip="${strip}" data-i="${idx}"`}><img src="${esc(base + p.thumb)}" srcset="${esc(base + p.thumb)} 400w, ${esc(base + p.photo)} 1600w" sizes="${esc(sizes)}" alt="" loading="lazy"></a>
      <figcaption class="entry">${entry}</figcaption></figure>`;
  }
  function evidence() {
    const dlg = document.getElementById("lightbox");
    if (!dlg) return () => false;
    // the print is made here: it has no source until a cell is opened
    const img = dlg.querySelector("img") || dlg.querySelector("figure").insertAdjacentElement("afterbegin", Object.assign(document.createElement("img"), { alt: "" }));
    const cap = dlg.querySelector(".cap"), cnt = dlg.querySelector(".count");
    const thumbs = dlg.querySelector(".thumbs");
    let items = [], i = 0, token = 0;
    const show = () => {
      const it = items[i], t = ++token;
      // The thumbnail is already loaded: show it at full size now and sharpen into the real photo when it lands.
      img.src = it.thumb; img.alt = it.caption; cap.textContent = it.caption;
      cnt.textContent = items.length > 1 ? `Photo ${i + 1} of ${items.length}` : "";
      thumbs.hidden = items.length < 2;
      // the strip: every photo of the cell as a framed thumbnail, the current one marked in red
      thumbs.innerHTML = items.length > 1 ? items.map((x, k) => `<button type="button" class="${k === i ? "on" : ""}" aria-label="Photo ${k + 1} of ${items.length}" aria-current="${k === i}"><img src="${esc(x.thumb)}" alt=""></button>`).join("") : "";
      dlg.setAttribute("aria-busy", "true");
      const full = new Image();
      full.onload = () => { if (t === token && dlg.open) { img.src = it.src; dlg.removeAttribute("aria-busy"); } };
      full.onerror = () => { if (t === token) { dlg.removeAttribute("aria-busy"); cap.textContent = it.caption + " · full-size photo didn't load"; } };
      full.src = it.src;
    };
    const step = (d) => { if (items.length > 1) { i = (i + d + items.length) % items.length; show(); } };
    document.addEventListener("click", (e) => {
      const a = e.target.closest("a.print");
      if (!a || "sheet" in a.dataset || e.metaKey || e.ctrlKey || e.shiftKey || e.button) return;   // let modifier/middle clicks open the photo in a tab; a data-sheet print navigates
      e.preventDefault();
      open(STRIPS[+a.dataset.strip], +a.dataset.i);
    });
    const open = (strip, idx) => {
      items = strip || [];
      i = Math.min(idx || 0, Math.max(0, items.length - 1));
      if (!items.length) return false;
      show();
      dlg.showModal();
      return true;
    };
    thumbs.addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; i = [...thumbs.children].indexOf(b); show(); });
    // Nothing sits on the photo: its left third steps back, the rest steps forward; a lone photo closes on tap.
    img.addEventListener("click", (e) => { if (items.length < 2) return dlg.close(); const r = img.getBoundingClientRect(); step(e.clientX - r.left < r.width / 3 ? -1 : 1); });
    dlg.addEventListener("click", (e) => { if (e.target === dlg || e.target.closest(".close")) dlg.close(); });
    dlg.addEventListener("keydown", (e) => { if (e.key === "ArrowLeft") step(-1); else if (e.key === "ArrowRight") step(1); });
    dlg.addEventListener("close", () => { token++; img.removeAttribute("src"); img.alt = ""; dlg.removeAttribute("aria-busy"); });
    return open;
  }

  // --- shared pieces ------------------------------------------------------------------
  const foot = (left, right) => `<footer class="foot"><span>${left}</span><span>${right}</span></footer>`;
  const sheetHref = (d) => "index.html?day=" + encodeURIComponent(d);   // the daily record of that day
  const RECORDED = "Recorded automatically by a camera at each window · the best photos of each species per day";
  // The photo log: every published photo, grouped by day, newest first.
  function photoLog(entries, todayDay, withWho) {
    // entries: [{site, p}] sorted newest first
    const days = new Map();
    for (const e of entries) { const d = day(e.p.captured_at); (days.get(d) || days.set(d, []).get(d)).push(e); }
    let i = 0;
    return [...days.entries()].map(([d, list]) => {
      const strip = STRIPS.push(list.map((e) => stripItem(e.site, e.p, withWho ? e.site.name : ""))) - 1;
      const prints = list.map((e, k) => print(e.site, e.p, strip, k,
        `<b>${esc(name(e.p))}</b><span class="t">${esc(hm(e.p.captured_at))}</span>${withWho ? `<span class="who">${esc(e.site.name)}</span>` : ""}`)).join("");
      return `<section class="log" style="--i:${Math.min(i++, 12)}"><h4 class="logday"><b class="${d === todayDay ? "today" : ""}"><a href="${sheetHref(d)}">${esc(dayLong(d))}</a></b><span>${esc(dow3(d))}</span></h4><div class="prints">${prints}</div></section>`;
    }).join("");
  }

  // --- pages ----------------------------------------------------------------------
  // Home: the daily record. Species down the side, a column per feeder, each cell the day's best photo and count.
  async function home() {
    const sites = await loadAll();
    if (!sites.length) { main().innerHTML = `<p class="empty">No feeders configured yet.</p>`; return; }
    const live = sites.filter((s) => s.manifest);
    const todays = live.map((s) => siteToday(s.manifest.site));
    let sheetDay = todays[0] || localToday(), note = "";
    // Red is for today only: a feeder's own today. The fallback sheet (last day with entries) is never red.
    const isToday = (d) => (todays.length ? todays : [localToday()]).includes(d);
    // Every local day that has entries at some feeder, oldest first: the sheets the pager steps through.
    const entryDays = [...new Set(live.flatMap((s) => s.manifest.photos.map((p) => day(p.captured_at))))].sort();
    const asked = q("day");
    if (asked && /^\d{4}-\d{2}-\d{2}$/.test(asked) && !isToday(asked)) sheetDay = asked;
    else if (!live.some((s, i) => s.manifest.photos.some((p) => day(p.captured_at) === todays[i]))) {
      // Nothing anywhere today: the sheet shows the last day that has entries, and says so.
      const last = entryDays.at(-1);
      if (last) { sheetDay = last; note = "No entries yet today. This is the last sheet with entries."; }
    }
    // Previous/next sheet with entries; past the last one, the way forward is today's sheet (the page without ?day).
    const prevDay = entryDays.filter((d) => d < sheetDay).at(-1);
    const nextDay = entryDays.find((d) => d > sheetDay && !isToday(d));
    const onToday = !q("day") || isToday(sheetDay);
    const prevLink = prevDay ? `<a class="hit" href="${sheetHref(prevDay)}" rel="prev"><span aria-hidden="true">← </span>${esc(dayShort(prevDay))}</a>` : "";
    const nextLink = nextDay ? `<a class="hit" href="${sheetHref(nextDay)}" rel="next">${esc(dayShort(nextDay))}<span aria-hidden="true"> →</span></a>`
      : onToday ? "" : `<a class="hit" href="index.html">Today<span aria-hidden="true"> →</span></a>`;
    const pager = prevLink || nextLink ? `<nav class="pager typed" aria-label="Other sheets"><span>${prevLink}</span><span>${nextLink}</span></nav>` : "";
    const groups = sites.map((s) => (s.manifest ? groupByDay(s.manifest, sheetDay) : null));
    const rows = new Map();
    groups.forEach((g, i) => {
      if (!g) return;
      for (const [sp, grp] of g) {
        const r = rows.get(sp) || { species: sp, total: 0, latest: 0, cells: Array(sites.length).fill(null), feeders: 0 };
        r.cells[i] = grp; r.total += grp.count; r.latest = Math.max(r.latest, grp.latest); r.feeders++;
        rows.set(sp, r);
      }
    });
    const ordered = [...rows.values()].sort((a, b) => b.total - a.total || a.species.localeCompare(b.species));
    const latest = Math.max(0, ...ordered.map((r) => r.latest));
    const since = live.map((s) => s.manifest.site.since).filter(Boolean).sort()[0];
    const sheetNo = since && sheetDay >= since ? dayDiff(since, sheetDay) + 1 : null;   // photos ranked from before "since" have no sheet number
    const multi = sites.length > 1;
    document.title = `birdy — daily record, ${dayLong(sheetDay)}`;

    const heads = sites.map((s) => `<a class="feeder hit" href="${siteHref(s.id)}"><b>${esc(s.name)}</b><span>${esc(s.place || "")}${s.manifest ? `${s.place ? " · " : ""}<span class="now">${esc(siteNow(s.manifest.site))}</span> local` : `${s.place ? " · " : ""}not available right now`}</span></a>`).join("");
    const rowHtml = (r, i) => {
      const rank = i === 0 ? 1 : i < 3 ? 2 : 3;
      const cells = r.cells.map((g, j) => {
        const s = sites[j];
        if (!g) return `<div class="cell blank"><span class="feederName">${esc(s.name)}</span>—</div>`;
        const strip = STRIPS.push(g.items.map((p) => stripItem(s, p, multi ? s.name : ""))) - 1;
        const idx = g.items.indexOf(g.best);
        const hot = g.latest === latest;
        const w = multi ? [460, 340, 250][rank - 1] : [980, 740, 560][rank - 1];
        return print(s, g.best, strip, idx,
          `<span class="feederName">${esc(s.name)}</span><span class="n">×${esc(g.count)}</span><span class="t ${hot ? "latest" : ""}">${esc(hm(g.last))}${hot ? `<span class="sr"> (latest entry)</span>` : ""}</span>`,
          `(max-width: 719px) ${rank === 1 ? "calc(100vw - 32px)" : "45vw"}, ${w}px`);
      }).join("");
      return `<div class="row rank-${rank}" style="--i:${Math.min(i, 12)}">
        <div class="species"><a href="${speciesHref(r.species)}">${esc(r.species)}</a><span class="tally">×${esc(r.total)}${multi ? ` · ${plural(r.feeders, "feeder")}` : ""}</span></div>${cells}</div>`;
    };
    const totals = sites.map((s, j) => {
      if (!s.manifest) return `<div><span class="feederName">${esc(s.name)}</span>—</div>`;
      const g = groups[j];
      let visits = 0; for (const x of g.values()) visits += x.count;
      return `<div><span class="feederName">${esc(s.name)}</span>${plural(g.size, "species")} · ${plural(visits, "visit")}</div>`;
    }).join("");

    main().innerHTML = `
      <section class="daily ${multi ? "multi" : "single"}" style="--n:${sites.length}">
        <header class="head">
          <h2 class="${isToday(sheetDay) ? "today" : ""}">${esc(dayFull(sheetDay))}</h2>
          <div class="meta">${sheetNo ? `<p class="label">Sheet</p>${esc(sheetNo)}` : ""}</div>
          ${pager}
        </header>
        ${note ? `<p class="note">${esc(note)}</p>` : ""}
        <div class="row heads"><p class="label">Species</p>${heads}</div>
        ${ordered.map(rowHtml).join("") || `<p class="empty">No entries on this sheet.</p>`}
        <div class="row totals"><div>${plural(ordered.length, "species")}</div>${totals}</div>
      </section>
      ${foot(RECORDED, since ? `Record kept since ${esc(dayLong(since))}` : "")}`;
    const open = evidence();
    // Arriving from a print on a feeder or species page: open that photo, then drop it from the address so a reload shows the plain sheet.
    const photo = q("photo"), from = q("site");
    if (photo && from) {
      const src = siteBase(from) + photo;
      const strip = STRIPS.find((items) => items.some((x) => x.src === src));
      if (strip) open(strip, strip.findIndex((x) => x.src === src));
      history.replaceState(null, "", q("day") ? sheetHref(q("day")) : "index.html");
    }
  }

  // Feeder page: the monthly sheet (species × days), the hour tally, and the photo log.
  async function site() {
    const id = q("id");
    const sites = await loadSites();
    const s = sites.find((x) => x.id === id);
    if (!s) throw new Error("There is no feeder with that name.");
    const m = await loadManifest(id);
    TO_SHEET = true;
    if (!m) throw new Error(`${s.name} hasn't published anything yet, or the server is unreachable. Try again in a minute.`);
    document.title = `birdy — ${s.name}, monthly record`;
    const st = m.stats, todayDay = siteToday(m.site);
    let days = st.per_day.map((d) => d.day).filter((d) => d <= todayDay);
    if (!days.length) days = st.per_day.map((d) => d.day);   // a manifest generated ahead of the viewer's clock: show it as is
    const counted = hasDayCounts(m);
    // per species per day: the Pi's counts, or published photos as a lower bound on an older manifest
    const photosOn = {};
    for (const p of m.photos) { const k = name(p) + "\n" + day(p.captured_at); photosOn[k] = (photosOn[k] || 0) + 1; }
    const cellCount = (sp, d) => (counted ? sp.days?.[d] : photosOn[sp.name + "\n" + d]) || 0;
    const species = st.species.map((sp) => ({ ...sp, month: days.reduce((a, d) => a + cellCount(sp, d), 0) }))
      .filter((sp) => sp.month > 0).sort((a, b) => b.month - a.month || a.name.localeCompare(b.name));
    // A day's column head links to that day's sheet when the sheet has entries (published photos); an empty sheet gets no link.
    const photoDays = new Set(m.photos.map((p) => day(p.captured_at)));
    const dayHead = (d, i) => {
      const p = parts(d); const first = i === 0 || p[2] === 1;
      const inner = `<small>${first ? esc(MONTHS[p[1] - 1]) : ""}</small>${p[2]}`;
      return `<div class="mh ${d === todayDay ? "today" : ""}">${photoDays.has(d) ? `<a href="${sheetHref(d)}" aria-label="${esc(dayLong(d))}, daily record">${inner}</a>` : inner}</div>`;
    };
    // A row of count cells with an ink wash: four steps, darker = busier within that row, so a
    // rare bird's pattern reads as clearly as a common one's. The number stays in every cell.
    const washed = (counts, cls = "", extra = () => "") => {
      const top = Math.max(0, ...counts);
      return counts.map((c, i) => `<div class="mc ${cls} ${extra(i)}"${c ? ` data-w="${Math.max(1, Math.ceil((c / top) * 4))}"` : ""}>${c || ""}</div>`).join("");
    };
    const isToday = (i) => (days[i] === todayDay ? "today" : "");
    const colTotals = days.map((d) => species.reduce((a, sp) => a + cellCount(sp, d), 0));
    const mgrid = species.length ? `<div class="mgrid" style="--d:${days.length}">
        <div class="mh sp"><small>Species</small>${counted ? "visits per day" : "published photos per day"}</div>${days.map(dayHead).join("")}<div class="mh mt"><small>&nbsp;</small>Total</div>
        ${species.map((sp) => `<div class="sp"><a href="${speciesHref(sp.name)}">${esc(sp.name)}</a></div>${washed(days.map((d) => cellCount(sp, d)), "", isToday)}<div class="mt">${esc(sp.month)}</div>`).join("")}
        <div class="sp tot">All species</div>${washed(colTotals, "tot", isToday)}<div class="mt">${esc(colTotals.reduce((a, b) => a + b, 0))}</div>
      </div>` : `<p class="empty">Nothing recorded in the last 30 days.</p>`;
    // By hour: species × hour of day over all time, so each bird's habits show. Hours nobody came
    // at either end of the day are left off; an older Pi's manifest has no per-species hours and
    // gets the single row it always had.
    const active = st.per_hour.map((v, h) => (v ? h : -1)).filter((h) => h >= 0);
    const h0 = active.length ? Math.max(0, active[0] - 1) : 0, h1 = active.length ? Math.min(23, active.at(-1) + 1) : 23;
    const hours = Array.from({ length: h1 - h0 + 1 }, (_, i) => h0 + i);
    const busiest = (row) => { const top = Math.max(...row); return top ? `${row.indexOf(top)} h` : ""; };
    const hourRow = (label, row, cls = "") => `<div class="sp ${cls}">${label}</div>${washed(hours.map((h) => row[h]), cls)}<div class="mt mb">${esc(busiest(row))}</div><div class="mt">${esc(row.reduce((a, b) => a + b, 0))}</div>`;
    const hourly = st.species.filter((sp) => Array.isArray(sp.hours) && sp.hours.length === 24);
    const hgrid = `<div class="mgrid hgrid" style="--i:1;--h:${hours.length}">
        <div class="mh sp"><small>Species</small>visits per hour of day</div>${hours.map((h) => `<div class="mh">${h}</div>`).join("")}<div class="mh mt mb"><small>&nbsp;</small>Busiest</div><div class="mh mt"><small>&nbsp;</small>Total</div>
        ${hourly.map((sp) => hourRow(`<a href="${speciesHref(sp.name)}">${esc(sp.name)}</a>`, sp.hours)).join("")}
        ${hourRow(hourly.length ? "All birds" : "Visits", st.per_hour, hourly.length ? "tot" : "")}
      </div>`;
    // the latest day's entries, as prints
    const latestDay = m.photos.length ? day(m.photos[0].captured_at) : null;
    let latest = "";
    if (latestDay) {
      const g = [...groupByDay(m, latestDay).values()].sort((a, b) => b.count - a.count || a.species.localeCompare(b.species));
      latest = `<h3 class="sec"><span>${latestDay === todayDay ? "Today" : "Last entries"}</span><a class="typed" href="${sheetHref(latestDay)}">${esc(dayFull(latestDay))}</a></h3>
        <div class="strip">${g.map((grp) => { const strip = STRIPS.push(grp.items.map((p) => stripItem(s, p))) - 1;
          return print(s, grp.best, strip, grp.items.indexOf(grp.best), `<b>${esc(grp.species)}</b><span class="n">×${esc(grp.count)}</span><span class="t">${esc(hm(grp.last))}</span>`, "(max-width: 719px) 45vw, 440px"); }).join("")}</div>`;
    }
    const options = [...new Set(m.photos.map(name))].sort().map((n) => `<option>${esc(n)}</option>`).join("");
    main().innerHTML = `
      <header class="head">
        <h2>${esc(s.name)}<span class="sub">${esc(s.place || "")}${m.site.since ? `${s.place ? " · " : ""}watching since ${esc(dayLong(m.site.since))}` : ""}</span></h2>
        <div class="meta"><p class="label">Monthly record · all time</p>${plural(st.species_count, "species")} · ${plural(st.total_birds, "visit")}${st.last_capture_at ? `<br>last entry ${esc(dayShort(st.last_capture_at))}, ${esc(hm(st.last_capture_at))}` : ""}</div>
      </header>
      ${latest}
      <h3 class="sec"><span>Last 30 days</span><span class="typed">${esc(dayShort(days[0]))} – ${esc(dayShort(days.at(-1)))}</span></h3>
      <div class="month">${mgrid}</div><p class="scrollhint">Today is at the right edge · scroll sideways for earlier days</p>
      <h3 class="sec"><span>By hour</span><span class="typed">all time · ${esc(hours[0])} h – ${esc(hours.at(-1))} h</span></h3>
      <div class="month">${hgrid}</div><p class="scrollhint">Scroll sideways for the other hours</p>
      <p class="washkey"><span>Shading</span><i data-w="1"></i><i data-w="2"></i><i data-w="3"></i><i data-w="4"></i><span class="typed">darker = busier, within each row</span></p>
      <h3 class="sec"><span>Photo log</span><label class="filter">Show <select id="flt"><option value="">all species</option>${options}</select></label></h3>
      <div id="log"></div>
      ${foot(RECORDED, `<a class="hit" href="index.html">Daily record</a>`)}`;
    const log = document.getElementById("log");
    const draw = (f) => { log.innerHTML = photoLog(m.photos.filter((p) => !f || name(p) === f).map((p) => ({ site: s, p })), todayDay, false) || `<p class="empty">Nothing published yet.</p>`; };
    document.getElementById("flt").addEventListener("change", (e) => draw(e.target.value));
    draw("");
    const monthEl = document.querySelector(".month");
    if (monthEl) monthEl.scrollLeft = monthEl.scrollWidth;   // the days open on today, not on the empty left edge; the hours open on the morning
    evidence();
  }

  // Species card: the species across every feeder, then its photo log.
  async function species() {
    const sp = q("name");
    if (!sp) throw new Error("No species was given.");
    document.title = `birdy — ${sp}`;
    const sites = await loadAll();
    TO_SHEET = true;
    const rows = [], entries = [];
    let total = 0;
    for (const s of sites) {
      if (!s.manifest) continue;
      const st = s.manifest.stats.species.find((x) => x.name === sp);
      if (st) { total += st.count; rows.push(`<div><a href="${siteHref(s.id)}">${esc(s.name)}</a></div><div class="n">${esc(st.count)}</div><div>${esc(dayShort(st.first))}</div><div>${esc(dayShort(st.last))}</div>`); }
      for (const p of s.manifest.photos.filter((p) => p.species === sp)) entries.push({ site: s, p });
    }
    entries.sort((a, b) => instant(b.p.captured_at) - instant(a.p.captured_at));
    const multi = sites.length > 1;
    main().innerHTML = `
      <header class="head">
        <h2>${esc(sp)}</h2>
        <div class="meta"><p class="label">Species card · all feeders</p>${plural(total, "visit")}</div>
      </header>
      <div class="ledger"><div class="lh">Feeder</div><div class="lh n">Visits</div><div class="lh">First</div><div class="lh">Last</div>
        ${rows.join("") || `<p class="empty">Not seen at any feeder yet.</p>`}</div>
      <h3 class="sec"><span>Photo log</span></h3>
      ${photoLog(entries, localToday(), multi) || `<p class="empty">None published.</p>`}
      ${foot(RECORDED, `<a class="hit" href="index.html">Daily record</a>`)}`;
    evidence();
  }

  return { home, site, species };
})();

document.addEventListener("DOMContentLoaded", () => {
  const page = document.body.dataset.page;
  const run = BIRDY[page];
  const el = document.getElementById("main");
  if (run) run().then(() => el.removeAttribute("aria-busy"), (e) => {
    const p = document.createElement("p");
    p.className = "err";
    p.textContent = e.message;
    const back = document.createElement("p");
    back.innerHTML = `<a class="hit" href="index.html">Daily record</a>`;
    el.replaceChildren(p, back);
    el.removeAttribute("aria-busy");
  });
});
