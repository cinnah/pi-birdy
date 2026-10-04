/* birdy gallery: renders index/site/species pages from sites.json + sites/<id>/manifest.json. No build step. */
const BIRDY = (() => {
  const main = () => document.getElementById("main");
  // Every string that comes from sites.json, a manifest or the URL goes through esc() before it is injected as HTML.
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const q = (name) => new URLSearchParams(location.search).get(name);
  const day = (iso) => String(iso ?? "").slice(0, 10);   // captured_at carries the site's offset: the local date is the prefix
  const hm = (iso) => String(iso ?? "").slice(11, 16);
  const pct = (c) => (c == null ? "" : Math.round(c * 100) + "%");
  const name = (p) => p.species || "Unidentified bird";
  // The site's current local date: shift "now" by the site's UTC offset, then read the UTC date.
  const siteToday = (site) => new Date(Date.now() + (site.utc_offset_minutes || 0) * 60000).toISOString().slice(0, 10);
  const speciesHref = (s) => "species.html?name=" + encodeURIComponent(s);
  const siteHref = (id) => "site.html?id=" + encodeURIComponent(id);
  const siteBase = (id) => "sites/" + encodeURIComponent(id) + "/";

  async function loadSites() {
    const r = await fetch("sites.json", { cache: "no-cache" });
    if (!r.ok) throw new Error("sites.json: HTTP " + r.status);
    return r.json();
  }
  async function loadManifest(id) {
    try {
      const r = await fetch("sites/" + encodeURIComponent(id) + "/manifest.json", { cache: "no-cache" });
      return r.ok ? await r.json() : null;
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
    const caption = `${esc(name(p))} · ${esc(day(p.captured_at))} ${esc(hm(p.captured_at))}${p.confidence == null ? "" : " · " + esc(pct(p.confidence))}`;
    return `<a class="card" href="${esc(base + p.photo)}" data-caption="${caption}">
      <img src="${esc(base + p.thumb)}" alt="${esc(name(p))}" loading="lazy">
      <div class="cap"><b>${esc(name(p))}</b><span class="muted">${esc(hm(p.captured_at))} ${extra}</span></div></a>`;
  }
  function lightbox() {
    const dlg = document.getElementById("lightbox");
    if (!dlg) return;
    document.addEventListener("click", (e) => {
      const a = e.target.closest("a.card");
      if (!a) return;
      e.preventDefault();
      dlg.querySelector("img").src = a.getAttribute("href");
      dlg.querySelector("p").textContent = a.dataset.caption || "";
      dlg.showModal();
    });
    dlg.addEventListener("click", () => dlg.close());
    dlg.addEventListener("close", () => dlg.querySelector("img").removeAttribute("src"));
  }
  // Inline SVG bars: x in percent so the chart fills its box without stretching the labels; y in px.
  function bars(values, labels, every) {
    const H = 120, top = 14, bottom = 16, n = Math.max(1, values.length), max = Math.max(1, ...values);
    const bw = 100 / n, inner = H - top - bottom;
    const rects = values.map((v, i) => {
      const h = Math.round((v / max) * inner);
      return `<rect x="${(i * bw).toFixed(2)}%" y="${H - bottom - h}" width="${(bw * 0.8).toFixed(2)}%" height="${h}"><title>${esc(labels[i])}: ${esc(v)}</title></rect>`;
    });
    const texts = labels.map((l, i) => (i % every ? "" : `<text x="${(i * bw).toFixed(2)}%" y="${H - 3}">${esc(l)}</text>`));
    return `<svg height="${H}" role="img"><text x="0" y="10">max ${esc(max)}</text>${rects.join("")}${texts.join("")}</svg>`;
  }
  function groupByDay(photos, d) {
    const groups = new Map();
    for (const p of photos.filter((p) => day(p.captured_at) === d)) {
      const g = groups.get(name(p)) || { best: p, count: 0, latest: p.captured_at };
      g.count += 1;
      if ((p.confidence ?? 0) > (g.best.confidence ?? 0)) g.best = p;
      if (p.captured_at > g.latest) g.latest = p.captured_at;
      groups.set(name(p), g);
    }
    return [...groups.values()].sort((a, b) => (a.latest < b.latest ? 1 : -1));
  }
  function unavailable(site) {
    return `<section class="col"><h2>${esc(site.name)}</h2><p class="place muted">${esc(site.place || "")}</p><p class="empty">unavailable</p></section>`;
  }

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
        heading = `<p class="empty">No visitors yet today</p>`;
        if (last) { groups = groupByDay(m.photos, last); past = " past"; heading += `<p class="muted">last visit ${esc(last)}</p>`; }
      }
      const cards = groups.map((g) => photoCard(s.id, g.best, `· ${esc(g.count)} ${past ? "photos" : "today"}`)).join("");
      return `<section class="col"><h2><a href="${siteHref(s.id)}">${esc(s.name)}</a></h2>
        <p class="place muted">${esc(s.place || "")} · <a href="${siteHref(s.id)}">dashboard</a></p>${heading}<div class="cards${past}">${cards}</div></section>`;
    }).join("")}</div>`;
    lightbox();
  }

  async function site() {
    const id = q("id");
    const sites = await loadSites();
    const s = sites.find((x) => x.id === id);
    if (!s) throw new Error("unknown site");
    const m = await loadManifest(id);
    if (!m) throw new Error("manifest unavailable");
    document.title = `birdy — ${s.name}`;
    const st = m.stats, pd = st.per_day, today = pd[pd.length - 1]?.birds ?? 0;
    const week = pd.slice(-7).reduce((a, d) => a + d.birds, 0);
    const nums = [[st.species_count, "species seen"], [today, "birds today"], [week, "last 7 days"], [st.total_birds, "birds in total"],
      [st.last_capture_at ? hm(st.last_capture_at) : "–", st.last_capture_at ? "last visit " + day(st.last_capture_at) : "no visits yet"]];
    const top = st.species.map((x) => `<tr><td><a href="${speciesHref(x.name)}">${esc(x.name)}</a></td><td>${esc(x.count)}</td><td>${esc(x.first)}</td><td>${esc(day(x.last))}</td></tr>`).join("");
    const options = [...new Set(m.photos.map(name))].sort().map((n) => `<option>${esc(n)}</option>`).join("");
    main().innerHTML = `
      <h2 style="margin-top:0">${esc(s.name)} <span class="muted" style="font-weight:normal">${esc(s.place || "")}${m.site.since ? " · watching since " + esc(m.site.since) : ""}</span></h2>
      <div class="nums">${nums.map(([v, l]) => `<div class="num"><div class="v">${esc(v)}</div><div class="l">${esc(l)}</div></div>`).join("")}</div>
      <div class="charts" style="margin-top:12px">
        <div class="chart"><div class="muted">birds per day, last 30 days</div>${bars(pd.map((d) => d.birds), pd.map((d) => day(d.day).slice(5)), 5)}</div>
        <div class="chart"><div class="muted">visits by hour of day</div>${bars(st.per_hour, st.per_hour.map((_, h) => String(h)), 3)}</div>
      </div>
      <h2>species</h2>
      <table><tr><th>species</th><th>visits</th><th>first</th><th>last</th></tr>${top || "<tr><td colspan=4 class=muted>none yet</td></tr>"}</table>
      <h2>published photos <select id="flt" aria-label="filter by species"><option value="">all species</option>${options}</select></h2>
      <div class="grid" id="grid"></div>`;
    const grid = document.getElementById("grid");
    const draw = (f) => { grid.innerHTML = m.photos.filter((p) => !f || name(p) === f).map((p) => photoCard(id, p, esc(day(p.captured_at)))).join("") || "<p class=muted>nothing published yet</p>"; };
    document.getElementById("flt").addEventListener("change", (e) => draw(e.target.value));
    draw("");
    lightbox();
  }

  async function species() {
    const sp = q("name");
    if (!sp) throw new Error("no species given");
    document.title = `birdy — ${sp}`;
    const sites = await loadAll();
    const rows = [], cards = [];
    for (const s of sites) {
      if (!s.manifest) continue;
      const st = s.manifest.stats.species.find((x) => x.name === sp);
      if (st) rows.push(`<tr><td><a href="${siteHref(s.id)}">${esc(s.name)}</a></td><td>${esc(st.count)}</td><td>${esc(st.first)}</td><td>${esc(day(st.last))}</td></tr>`);
      for (const p of s.manifest.photos.filter((p) => p.species === sp)) cards.push({ s, p });
    }
    cards.sort((a, b) => (a.p.captured_at < b.p.captured_at ? 1 : -1));
    main().innerHTML = `<h2 style="margin-top:0">${esc(sp)}</h2>
      <table><tr><th>feeder</th><th>visits</th><th>first</th><th>last</th></tr>${rows.join("") || "<tr><td colspan=4 class=muted>not seen yet</td></tr>"}</table>
      <h2>photos</h2><div class="grid">${cards.map(({ s, p }) => photoCard(s.id, p, `${esc(day(p.captured_at))} · ${esc(s.name)}`)).join("") || "<p class=muted>none published</p>"}</div>`;
    lightbox();
  }

  return { home, site, species };
})();

document.addEventListener("DOMContentLoaded", () => {
  const page = document.body.dataset.page;
  const run = BIRDY[page];
  if (run) run().catch((e) => {
    const p = document.createElement("p");
    p.className = "err";
    p.textContent = e.message;
    document.getElementById("main").replaceChildren(p);
  });
});
