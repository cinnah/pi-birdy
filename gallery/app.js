/* birdy gallery: renders index/site/species pages from sites.json + sites/<id>/manifest.json. No build step. */
const BIRDY = (() => {
  const main = () => document.getElementById("main");
  // Every string that comes from sites.json, a manifest or the URL goes through esc() before it is injected as HTML.
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const q = (name) => new URLSearchParams(location.search).get(name);
  const day = (iso) => String(iso ?? "").slice(0, 10);   // captured_at carries the site's offset: the local date is the prefix
  const hm = (iso) => String(iso ?? "").slice(11, 16);
  const instant = (iso) => Date.parse(iso) || 0;          // compare moments, not strings (offsets may differ between sites)
  const pct = (c) => (c == null ? "" : Math.round(c * 100) + "%");
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  // "2026-10-04" -> "4 Oct"; dates are read from the string so the site's own calendar day is kept.
  const dayShort = (iso) => { const d = day(iso); return d.length === 10 ? `${+d.slice(8, 10)} ${MONTHS[+d.slice(5, 7) - 1]}` : d; };
  const dayLong = (iso) => { const d = day(iso); return d.length === 10 ? `${dayShort(d)} ${d.slice(0, 4)}` : d; };
  const name = (p) => p.species || "Unidentified bird";
  // The site's current local date: shift "now" by the site's UTC offset, then read the UTC date.
  const siteToday = (site) => new Date(Date.now() + (site.utc_offset_minutes || 0) * 60000).toISOString().slice(0, 10);
  const speciesHref = (s) => "species.html?name=" + encodeURIComponent(s);
  const siteHref = (id) => "site.html?id=" + encodeURIComponent(id);
  const siteBase = (id) => "sites/" + encodeURIComponent(id) + "/";

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

  // --- rendering helpers ------------------------------------------------------
  // `extra` is already-escaped HTML for the right side of the caption.
  function photoCard(siteId, p, extra = "") {
    const base = siteBase(siteId);
    const caption = `${esc(name(p))} · ${esc(dayLong(p.captured_at))}, ${esc(hm(p.captured_at))}${p.confidence == null ? "" : " · " + esc(pct(p.confidence))}`;
    // alt is empty on purpose: the caption below names the bird, so the link isn't read twice.
    return `<a class="card" href="${esc(base + p.photo)}" data-caption="${caption}">
      <img src="${esc(base + p.thumb)}" alt="" loading="lazy">
      <div class="cap"><b>${esc(name(p))}</b><span class="muted">${esc(hm(p.captured_at))} ${extra}</span></div></a>`;
  }
  function lightbox() {
    const dlg = document.getElementById("lightbox");
    if (!dlg) return;
    const img = dlg.querySelector("img");
    document.addEventListener("click", (e) => {
      const a = e.target.closest("a.card");
      if (!a || e.metaKey || e.ctrlKey || e.shiftKey || e.button) return;   // let modifier/middle clicks open the photo in a tab
      e.preventDefault();
      const caption = a.dataset.caption || "", href = a.getAttribute("href");
      // The thumbnail is already loaded: show it at full size now and fade the real photo in over it.
      img.src = a.querySelector("img").src;
      img.alt = caption;
      dlg.querySelector("p").textContent = caption;
      dlg.setAttribute("aria-busy", "true");
      dlg.showModal();
      const full = new Image();
      full.onload = () => { if (dlg.open && dlg.dataset.href === href) { img.src = href; dlg.removeAttribute("aria-busy"); } };
      full.onerror = () => { if (dlg.dataset.href === href) { dlg.removeAttribute("aria-busy"); dlg.querySelector("p").textContent = caption + " · full-size photo didn't load"; } };
      dlg.dataset.href = href;
      full.src = href;
    });
    dlg.addEventListener("click", () => dlg.close());
    dlg.addEventListener("close", () => { img.removeAttribute("src"); img.alt = ""; delete dlg.dataset.href; dlg.removeAttribute("aria-busy"); });
  }
  // Inline SVG bars: x in percent so the chart fills its box without stretching the labels; y in px.
  function bars(title, values, labels, every) {
    const H = 120, top = 14, bottom = 16, n = Math.max(1, values.length), max = Math.max(1, ...values);
    const bw = 100 / n, inner = H - top - bottom;
    const rects = values.map((v, i) => {
      const h = Math.round((v / max) * inner);
      return `<rect x="${(i * bw).toFixed(2)}%" y="${H - bottom - h}" width="${(bw * 0.8).toFixed(2)}%" height="${h}" rx="1"><title>${esc(labels[i])}: ${esc(v)}</title></rect>`;
    });
    const texts = labels.map((l, i) => (i % every ? "" : `<text x="${(i * bw).toFixed(2)}%" y="${H - 3}">${esc(l)}</text>`));
    // A baseline, a hairline at the top of the tallest bar and its value: the whole scale in two lines.
    const axes = `<line x1="0" x2="100%" y1="${H - bottom + 0.5}" y2="${H - bottom + 0.5}"/>` +
      `<line x1="0" x2="100%" y1="${top + 0.5}" y2="${top + 0.5}" stroke-dasharray="2 3"/>` +
      `<text x="100%" y="${top - 4}" text-anchor="end">${esc(max)}</text>`;
    const data = values.map((v, i) => (v ? `<tr><th scope="row">${esc(labels[i])}</th><td>${esc(v)}</td></tr>` : "")).join("");
    return `<svg height="${H}" aria-hidden="true">${axes}${rects.join("")}${texts.join("")}</svg>` +
      `<table class="sr"><caption>${esc(title)}</caption><tbody>${data || `<tr><td>none</td></tr>`}</tbody></table>`;
  }
  function groupByDay(photos, d) {
    const groups = new Map();
    for (const p of photos.filter((p) => day(p.captured_at) === d)) {
      const g = groups.get(name(p)) || { best: p, count: 0, latest: instant(p.captured_at) };
      g.count += 1;
      if ((p.confidence ?? 0) > (g.best.confidence ?? 0)) g.best = p;
      g.latest = Math.max(g.latest, instant(p.captured_at));
      groups.set(name(p), g);
    }
    return [...groups.values()].sort((a, b) => b.latest - a.latest);
  }
  function unavailable(site) {
    return `<section class="col"><h2>${esc(site.name)}</h2><p class="place">${esc(site.place || "")}</p><p class="empty">Not available right now.</p></section>`;
  }
  // head: [label, class] pairs; class "n" right-aligns numbers, "d" keeps dates on one line.
  const statsTable = (head, rows, empty) =>
    `<table><thead><tr>${head.map(([h, c]) => `<th scope="col" class="${c}">${esc(h)}</th>`).join("")}</tr></thead>
     <tbody>${rows.join("") || `<tr><td colspan="${head.length}" class="muted">${esc(empty)}</td></tr>`}</tbody></table>`;

  // --- pages ----------------------------------------------------------------------
  async function home() {
    const sites = await loadAll();
    if (!sites.length) { main().innerHTML = `<p class="muted">No feeders configured yet.</p>`; return; }
    main().innerHTML = `<div class="columns">${sites.map((s) => {
      const m = s.manifest;
      if (!m) return unavailable(s);
      const today = siteToday(m.site);
      let groups = groupByDay(m.photos, today), heading = "", past = "";
      if (!groups.length) {
        const last = m.photos.length ? day(m.photos[0].captured_at) : null;
        heading = `<p class="empty">No visitors yet today.</p>`;
        if (last) { groups = groupByDay(m.photos, last); past = " past"; heading += `<p class="muted">Last visit ${esc(dayLong(last))}</p>`; }
      }
      const cards = groups.map((g) => photoCard(s.id, g.best, g.count > 1 ? `· ${esc(g.count)} visits` : "")).join("");
      return `<section class="col"><h2><a class="hit" href="${siteHref(s.id)}">${esc(s.name)}</a></h2>
        <p class="place">${esc(s.place || "")}${s.place ? " · " : ""}<a class="hit" href="${siteHref(s.id)}">Dashboard</a></p>${heading}<div class="cards${past}">${cards}</div></section>`;
    }).join("")}</div>`;
    lightbox();
  }

  async function site() {
    const id = q("id");
    const sites = await loadSites();
    const s = sites.find((x) => x.id === id);
    if (!s) throw new Error("There is no feeder with that name.");
    const m = await loadManifest(id);
    if (!m) throw new Error(`${s.name} hasn't published anything yet, or the server is unreachable. Try again in a minute.`);
    document.title = `birdy — ${s.name}`;
    const st = m.stats, todayDay = siteToday(m.site);
    // per_day ends on the Pi's day at generation time; a stale manifest must not present an old day as "today".
    const pd = st.per_day.filter((d) => d.day <= todayDay);
    const today = pd.at(-1)?.day === todayDay ? pd.at(-1).birds : 0;
    const week = pd.slice(-7).reduce((a, d) => a + d.birds, 0);
    const nums = [[st.species_count, "species"], [today, "today"], [week, "this week"], [st.total_birds, "in total"],
      [st.last_capture_at ? hm(st.last_capture_at) : "–", st.last_capture_at ? "last visit, " + dayShort(st.last_capture_at) : "no visits yet"]];
    const top = st.species.map((x) => `<tr><td><a href="${speciesHref(x.name)}">${esc(x.name)}</a></td><td class="n">${esc(x.count)}</td><td class="d">${esc(dayShort(x.first))}</td><td class="d">${esc(dayShort(x.last))}</td></tr>`);
    const options = [...new Set(m.photos.map(name))].sort().map((n) => `<option>${esc(n)}</option>`).join("");
    main().innerHTML = `
      <h2 class="first">${esc(s.name)}<span class="sub">${esc(s.place || "")}${m.site.since ? (s.place ? " · " : "") + "watching since " + esc(dayLong(m.site.since)) : ""}</span></h2>
      <ul class="stats">${nums.map(([v, l]) => `<li><b>${esc(v)}</b><span>${esc(l)}</span></li>`).join("")}</ul>
      <div class="charts">
        <div class="chart"><p class="title">Birds per day, last 30 days</p>${bars("Birds per day, last 30 days", st.per_day.map((d) => d.birds), st.per_day.map((d) => dayShort(d.day)), 5)}</div>
        <div class="chart"><p class="title">Visits by hour of day</p>${bars("Visits by hour of day", st.per_hour, st.per_hour.map((_, h) => String(h)), 3)}</div>
      </div>
      <h2>Species</h2>
      ${statsTable([["Species", ""], ["Visits", "n"], ["First", "d"], ["Last", "d"]], top, "None yet")}
      <h2>Photos</h2>
      <label class="filter">Show <select id="flt"><option value="">all species</option>${options}</select></label>
      <div class="grid" id="grid"></div>`;
    const grid = document.getElementById("grid");
    const draw = (f) => { grid.innerHTML = m.photos.filter((p) => !f || name(p) === f).map((p) => photoCard(id, p, `· ${esc(dayShort(p.captured_at))}`)).join("") || `<p class="muted">Nothing published yet</p>`; };
    document.getElementById("flt").addEventListener("change", (e) => draw(e.target.value));
    draw("");
    lightbox();
  }

  async function species() {
    const sp = q("name");
    if (!sp) throw new Error("No species was given.");
    document.title = `birdy — ${sp}`;
    const sites = await loadAll();
    const rows = [], cards = [];
    for (const s of sites) {
      if (!s.manifest) continue;
      const st = s.manifest.stats.species.find((x) => x.name === sp);
      if (st) rows.push(`<tr><td><a href="${siteHref(s.id)}">${esc(s.name)}</a></td><td class="n">${esc(st.count)}</td><td class="d">${esc(dayShort(st.first))}</td><td class="d">${esc(dayShort(st.last))}</td></tr>`);
      for (const p of s.manifest.photos.filter((p) => p.species === sp)) cards.push({ s, p });
    }
    cards.sort((a, b) => instant(b.p.captured_at) - instant(a.p.captured_at));
    main().innerHTML = `<h2 class="first">${esc(sp)}</h2>
      ${statsTable([["Feeder", ""], ["Visits", "n"], ["First", "d"], ["Last", "d"]], rows, "Not seen yet")}
      <h2>Photos</h2><div class="grid">${cards.map(({ s, p }) => photoCard(s.id, p, `· ${esc(dayShort(p.captured_at))}${sites.length > 1 ? " · " + esc(s.name) : ""}`)).join("") || `<p class="muted">None published</p>`}</div>`;
    lightbox();
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
    back.innerHTML = `<a class="hit" href="index.html">All feeders</a>`;
    el.replaceChildren(p, back);
    el.removeAttribute("aria-busy");
  });
});
