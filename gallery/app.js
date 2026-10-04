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
  function print(site, p, strip, idx, entry, sizes = "(max-width: 719px) 45vw, 220px", cls = "") {
    const base = siteBase(site.id);
    // alt is empty on purpose: the entry beneath names the bird, so the link isn't read twice.
    return `<figure class="cell ${cls}">
      <a class="print" href="${esc(base + p.photo)}" data-strip="${strip}" data-i="${idx}"><img src="${esc(base + p.thumb)}" srcset="${esc(base + p.thumb)} 400w, ${esc(base + p.photo)} 1600w" sizes="${esc(sizes)}" alt="" loading="lazy"></a>
      <figcaption class="entry">${entry}</figcaption></figure>`;
  }
  function evidence() {
    const dlg = document.getElementById("lightbox");
    if (!dlg) return;
    const img = dlg.querySelector("img"), cap = dlg.querySelector(".cap"), cnt = dlg.querySelector(".count");
    const prev = dlg.querySelector(".prev"), next = dlg.querySelector(".next"), thumbs = dlg.querySelector(".thumbs");
    let items = [], i = 0, token = 0;
    const show = () => {
      const it = items[i], t = ++token;
      // The thumbnail is already loaded: show it at full size now and sharpen into the real photo when it lands.
      img.src = it.thumb; img.alt = it.caption; cap.textContent = it.caption;
      cnt.textContent = items.length > 1 ? `${i + 1} / ${items.length}` : "";
      prev.hidden = next.hidden = thumbs.hidden = items.length < 2;
      dlg.classList.toggle("has-strip", items.length > 1);
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
      if (!a || e.metaKey || e.ctrlKey || e.shiftKey || e.button) return;   // let modifier/middle clicks open the photo in a tab
      e.preventDefault();
      items = STRIPS[+a.dataset.strip] || [];
      i = Math.min(+a.dataset.i || 0, Math.max(0, items.length - 1));
      if (!items.length) return;
      show();
      dlg.showModal();
    });
    prev.addEventListener("click", (e) => { e.stopPropagation(); step(-1); });
    thumbs.addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; e.stopPropagation(); i = [...thumbs.children].indexOf(b); show(); });
    next.addEventListener("click", (e) => { e.stopPropagation(); step(1); });
    dlg.addEventListener("click", (e) => { if (!e.target.closest(".prev, .next, .thumbs")) dlg.close(); });
    dlg.addEventListener("keydown", (e) => { if (e.key === "ArrowLeft") step(-1); else if (e.key === "ArrowRight") step(1); });
    dlg.addEventListener("close", () => { token++; img.removeAttribute("src"); img.alt = ""; dlg.removeAttribute("aria-busy"); });
  }

  // --- shared pieces ------------------------------------------------------------------
  const foot = (left, right) => `<footer class="foot"><span>${left}</span><span>${right}</span></footer>`;
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
      return `<section class="log" style="--i:${Math.min(i++, 12)}"><h4 class="logday"><b class="${d === todayDay ? "today" : ""}">${esc(dayLong(d))}</b><span>${esc(dow3(d))}</span></h4><div class="prints">${prints}</div></section>`;
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
    if (!live.some((s, i) => s.manifest.photos.some((p) => day(p.captured_at) === todays[i]))) {
      // Nothing anywhere today: the sheet shows the last day that has entries, and says so.
      const last = live.flatMap((s) => s.manifest.photos.map((p) => day(p.captured_at))).sort().at(-1);
      if (last) { sheetDay = last; note = "No entries yet today. This is the last sheet with entries."; }
    }
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
    const sheetNo = since ? dayDiff(since, sheetDay) + 1 : null;
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
        </header>
        ${note ? `<p class="note">${esc(note)}</p>` : ""}
        <div class="row heads"><p class="label">Species</p>${heads}</div>
        ${ordered.map(rowHtml).join("") || `<p class="empty">No entries on this sheet.</p>`}
        <div class="row totals"><div>${plural(ordered.length, "species")}</div>${totals}</div>
      </section>
      ${foot(RECORDED, since ? `Record kept since ${esc(dayLong(since))}` : "")}`;
    evidence();
  }

  // Feeder page: the monthly sheet (species × days), the hour tally, and the photo log.
  async function site() {
    const id = q("id");
    const sites = await loadSites();
    const s = sites.find((x) => x.id === id);
    if (!s) throw new Error("There is no feeder with that name.");
    const m = await loadManifest(id);
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
    const dayHead = (d, i) => { const p = parts(d); const first = i === 0 || p[2] === 1; return `<div class="mh ${d === todayDay ? "today" : ""}"><small>${first ? esc(MONTHS[p[1] - 1]) : ""}</small>${p[2]}</div>`; };
    const colTotals = days.map((d) => species.reduce((a, sp) => a + cellCount(sp, d), 0));
    const mgrid = species.length ? `<div class="mgrid" style="--d:${days.length}">
        <div class="mh sp"><small>Species</small>${counted ? "visits per day" : "published photos per day"}</div>${days.map(dayHead).join("")}<div class="mh mt"><small>&nbsp;</small>Total</div>
        ${species.map((sp) => `<div class="sp"><a href="${speciesHref(sp.name)}">${esc(sp.name)}</a></div>${days.map((d) => { const c = cellCount(sp, d); return `<div class="mc ${d === todayDay ? "today" : ""}">${c || ""}</div>`; }).join("")}<div class="mt">${esc(sp.month)}</div>`).join("")}
        <div class="sp tot">All species</div>${colTotals.map((c, i) => `<div class="mc tot ${days[i] === todayDay ? "today" : ""}">${c || ""}</div>`).join("")}<div class="mt">${esc(colTotals.reduce((a, b) => a + b, 0))}</div>
      </div>` : `<p class="empty">Nothing recorded in the last 30 days.</p>`;
    const hgrid = `<div class="mgrid hgrid" style="--i:1">
        <div class="mh sp"><small>Hour of day</small>all time</div>${st.per_hour.map((_, h) => `<div class="mh">${h}</div>`).join("")}<div class="mh mt"><small>&nbsp;</small>Total</div>
        <div class="sp">Visits</div>${st.per_hour.map((v) => `<div class="mc">${v || ""}</div>`).join("")}<div class="mt">${esc(st.per_hour.reduce((a, b) => a + b, 0))}</div>
      </div>`;
    // the latest day's entries, as prints
    const latestDay = m.photos.length ? day(m.photos[0].captured_at) : null;
    let latest = "";
    if (latestDay) {
      const g = [...groupByDay(m, latestDay).values()].sort((a, b) => b.count - a.count || a.species.localeCompare(b.species));
      latest = `<h3 class="sec"><span>${latestDay === todayDay ? "Today" : "Last entries"}</span><span class="typed ${latestDay === todayDay ? "" : ""}">${esc(dayFull(latestDay))}</span></h3>
        <div class="strip">${g.map((grp) => { const strip = STRIPS.push(grp.items.map((p) => stripItem(s, p))) - 1;
          return print(s, grp.best, strip, grp.items.indexOf(grp.best), `<b>${esc(grp.species)}</b><span class="n">×${esc(grp.count)}</span><span class="t">${esc(hm(grp.last))}</span>`, "(max-width: 719px) 45vw, 300px"); }).join("")}</div>`;
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
      <h3 class="sec"><span>By hour</span></h3>
      <div class="month">${hgrid}</div><p class="scrollhint">Scroll sideways for the other hours</p>
      <h3 class="sec"><span>Photo log</span><label class="filter">Show <select id="flt"><option value="">all species</option>${options}</select></label></h3>
      <div id="log"></div>
      ${foot(RECORDED, `<a class="hit" href="index.html">Daily record</a>`)}`;
    const log = document.getElementById("log");
    const draw = (f) => { log.innerHTML = photoLog(m.photos.filter((p) => !f || name(p) === f).map((p) => ({ site: s, p })), todayDay, false) || `<p class="empty">Nothing published yet.</p>`; };
    document.getElementById("flt").addEventListener("change", (e) => draw(e.target.value));
    draw("");
    for (const el of document.querySelectorAll(".month")) el.scrollLeft = el.scrollWidth;   // open on today, not on the empty left edge
    evidence();
  }

  // Species card: the species across every feeder, then its photo log.
  async function species() {
    const sp = q("name");
    if (!sp) throw new Error("No species was given.");
    document.title = `birdy — ${sp}`;
    const sites = await loadAll();
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
