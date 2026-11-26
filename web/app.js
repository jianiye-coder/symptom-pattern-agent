const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
// One random id per browser, so visitors of the public demo each get their own synthetic diary.
const USER = (() => {
  try {
    let id = localStorage.getItem("spa-user");
    if (!id) {
      id = "v" + crypto.getRandomValues(new Uint32Array(2)).join("").slice(0, 16);
      localStorage.setItem("spa-user", id);
    }
    return id;
  } catch {
    return "demo";
  }
})();
const history = [];
let highlight = null;
let flagged = new Map(); // date -> note
let editing = null; // date whose flag note is being edited

async function api(path, opts = {}) {
  const r = await fetch(path, {
    ...opts,
    headers: { "Content-Type": "application/json", "X-User-Id": USER, ...(opts.headers || {}) },
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
// minimal markdown: **bold** only, after escaping
const md = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
const pretty = (k) => k.replace(/_/g, " ");
const day = (iso, opts) => new Date(iso + "T12:00:00").toLocaleDateString("en-GB", opts);

// ------------------------------------------------------------------ boot
async function init() {
  const now = new Date();
  const today = new Date(now.getTime() - now.getTimezoneOffset() * 6e4).toISOString().slice(0, 10); // local date
  $("#report-link").href = `/report?user=${encodeURIComponent(USER)}`;
  $("#today-label").textContent = day(today, { day: "numeric", month: "long", year: "numeric" });
  $("#log-date").value = today;
  setLongDate();
  const [{ personas }, h, { entries }] = await Promise.all([api("/api/personas"), api("/api/health"), api("/api/entries")]);
  $("#persona").innerHTML = Object.entries(personas)
    .map(([k, v]) => `<option value="${esc(k)}" title="${esc(v)}">${esc(pretty(k))}</option>`)
    .join("");
  $("#mode").textContent = `Model: ${h.llm}.`;
  if (!entries.length) await loadPersona("pem");
  else note("Your record is open. Ask a question, or add today's entry.");
  await refresh();
}

function setLongDate() {
  const v = $("#log-date").value;
  $("#long-date").textContent = v ? day(v, { weekday: "long", day: "numeric", month: "long" }) : "";
}

async function loadPersona(p) {
  const r = await api(`/api/demo/${p}`, { method: "POST" });
  $("#persona").value = p;
  history.length = 0;
  highlight = null;
  editing = null;
  $("#messages").innerHTML = "";
  note(`Opened the synthetic diary “${pretty(p)}”: ${r.loaded} days. ${r.description}`);
}

async function refresh() {
  const { flags } = await api("/api/flags");
  flagged = new Map(flags.map((f) => [f.date, f.note]));
  await Promise.all([renderMemory(), renderChart(), renderLedger(), renderProgress()]);
}

// ------------------------------------------------------------------ progress
async function renderProgress() {
  const p = await api("/api/progress");
  if (p.status !== "ok") {
    $("#progress").innerHTML = `Nothing recorded yet. Patterns usually take 45–60 days of entries to show.`;
    return;
  }
  const max = 120;
  const pos = (d) => `${Math.min(100, (d / max) * 100)}%`;
  const ticks = [30, 45, 60, 90, 120].map((d) => `<i class="tick" style="left:${pos(d)}"><span>${d}d</span></i>`).join("");
  const next = p.next_milestone_days
    ? ` By day ${p.next_milestone_days}: about ${p.detection_chance_at_milestone_pct}%.`
    : "";
  const gap = p.days_since_last_entry;
  const nudge =
    gap >= 2 && gap <= 30
      ? `<p class="nudge">It's been ${gap} days since your last entry. One line is enough to keep the record useful.</p>`
      : "";
  $("#progress").innerHTML = `<b>Day ${p.span_days}</b> of your record · ${p.n_days} entries (${p.coverage_pct}% of days), longest gap ${p.longest_gap_days} ${p.longest_gap_days === 1 ? "day" : "days"}.
    <div class="ruler" aria-hidden="true"><i class="track"></i><i class="fill" style="width:${pos(p.span_days)}"></i>${ticks}<i class="you" style="left:${pos(p.span_days)}"></i></div>
    A strong, clean pattern would show up about <b>${p.detection_chance_pct}%</b> of the time with this much data.${next}
    Real effects are usually subtler and take longer.${nudge}`;
}

// ------------------------------------------------------------------ chart
async function renderChart() {
  const q = highlight ? `?exposure=${encodeURIComponent(highlight)}` : "";
  const d = await api(`/api/chart${q}`);
  const pts = d.points;
  $("#chart-sub").textContent = `flare = ${d.flare_definition}`;
  if (!pts.length) {
    $("#chart").innerHTML = `<p class="caption">No entries yet.</p>`;
    return;
  }
  const W = 760, H = 250, L = 28, R = 6, T = 18, B = 26;
  const t0 = new Date(pts[0].date), t1 = new Date(pts[pts.length - 1].date);
  const span = Math.max(1, (t1 - t0) / 864e5);
  const x = (s) => L + ((new Date(s) - t0) / 864e5 / span) * (W - L - R);
  const y = (v) => T + (1 - v / 10) * (H - T - B);
  const step = (W - L - R) / (span + 1);
  const bw = Math.max(1.5, step * 0.38);
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Daily worst symptom severity with flare days and exertion">`;
  for (const g of [0, 5, 10])
    svg += `<line x1="${L}" x2="${W - R}" y1="${y(g)}" y2="${y(g)}" stroke="${g ? "#d6cdbd" : "#1d1c1a"}" stroke-width="${g ? 1 : 1}" ${g ? 'stroke-dasharray="1 3"' : ""}/>
      <text x="${L - 8}" y="${y(g) + 3.5}" font-family="IBM Plex Mono, monospace" font-size="10" fill="#8a8378" text-anchor="end">${g}</text>`;
  for (const p of pts) {
    if (p.exposed)
      svg += `<rect x="${x(p.date) - step / 2}" y="${T}" width="${step}" height="${y(0) - T}" fill="#f2e3c3"/>`;
  }
  for (const p of pts)
    if (p.exertion != null)
      svg += `<rect x="${x(p.date) - bw / 2}" y="${y(p.exertion)}" width="${bw}" height="${y(0) - y(p.exertion)}" fill="${p.exposed ? "#c48a1e" : "#c9c1b3"}"/>`;
  const line = pts.filter((p) => p.severity != null).map((p) => `${x(p.date).toFixed(1)},${y(p.severity).toFixed(1)}`).join(" ");
  svg += `<polyline points="${line}" fill="none" stroke="#1d1c1a" stroke-width="1.6" stroke-linejoin="round"/>`;
  for (const p of pts)
    if (p.flare)
      svg += `<circle cx="${x(p.date)}" cy="${y(p.severity)}" r="3.6" fill="#b23a2b" stroke="#f4f0e8" stroke-width="1.5"/>`;
  for (const p of pts)
    if (flagged.has(p.date))
      svg += `<line x1="${x(p.date)}" x2="${x(p.date)}" y1="${T - 12}" y2="${y(0)}" stroke="#b23a2b" stroke-width="1"/><path d="M${x(p.date)},${T - 12} l8,4 l-8,4 z" fill="#b23a2b"/>`;
  // month labels on the baseline
  let lastMonth = "";
  for (const p of pts) {
    const m = p.date.slice(0, 7);
    if (m !== lastMonth) {
      lastMonth = m;
      svg += `<line x1="${x(p.date)}" x2="${x(p.date)}" y1="${y(0)}" y2="${y(0) + 5}" stroke="#1d1c1a"/>
        <text x="${x(p.date) + 3}" y="${H - 6}" font-family="Public Sans, sans-serif" font-size="10.5" fill="#57524a">${day(p.date, { month: "short", day: "numeric" })}</text>`;
    }
  }
  for (const p of pts)
    svg += `<rect class="hit" data-date="${p.date}" x="${x(p.date) - step / 2}" y="${T - 12}" width="${step}" height="${y(0) - T + 12}" fill="transparent"><title>${day(p.date, { weekday: "short", day: "numeric", month: "short" })} · worst ${p.severity ?? "–"}${p.exertion != null ? ` · exertion ${p.exertion}` : ""}${p.sleep_hours != null ? ` · slept ${p.sleep_hours}h` : ""}${flagged.has(p.date) ? " · flagged" : ""}\nClick to ${flagged.has(p.date) ? "see the note" : "flag for your doctor"}</title></rect>`;
  svg += `</svg>`;
  $("#chart").innerHTML = svg;
  $$("#chart .hit").forEach((r) => r.addEventListener("click", () => openDay(r.dataset.date)));
  $("#legend-exp").hidden = !highlight;
  if (highlight) {
    $("#legend-exp-label").textContent = d.exposure_label;
    $("#chart-note").textContent = `Shaded: days with ${d.exposure_label}. Watch for red dots in the days that follow. Click any day to flag it.`;
  }
}

// ------------------------------------------------------------------ ledger
let ledgerEntries = [];
let ledgerLimit = 21;

async function renderLedger() {
  const { entries } = await api("/api/entries");
  const fd = await api("/api/chart"); // flare flags per day, using the saved definition
  const flare = new Map(fd.points.map((p) => [p.date, p.flare]));
  ledgerEntries = entries.slice().reverse();
  const rows = ledgerEntries.slice(0, ledgerLimit).map((e) => {
    const sy = Object.entries(e.symptoms);
    const worst = sy.length ? Math.max(...sy.map(([, v]) => v)) : null;
    const isF = flagged.has(e.date);
    const food = [...e.foods, ...e.meds.map((m) => `<em>${esc(m)}</em>`)];
    let html = `<tr data-date="${e.date}" class="${flare.get(e.date) ? "flare" : ""} ${isF ? "is-flagged" : ""}">
      <td class="date">${day(e.date, { day: "2-digit", month: "short" })}<small>${day(e.date, { weekday: "short" })}</small></td>
      <td class="num ${worst >= 7 ? "worst-hi" : ""}">${worst ?? "–"}</td>
      <td class="syms">${sy.map(([k, v]) => `${esc(pretty(k))} ${v}`).join(", ") || "–"}</td>
      <td class="num">${e.sleep_hours != null ? e.sleep_hours + "h" : "–"}</td>
      <td class="num">${e.exertion ?? "–"}</td>
      <td class="food">${food.map((f) => (f.startsWith("<em>") ? f : esc(f))).join(", ") || "–"}</td>
      <td class="flag-col"><button class="flag-btn ${isF ? "on" : ""}" data-date="${e.date}" aria-pressed="${isF}" title="${isF ? "Flagged for your doctor" : "Flag for your doctor"}">⚑</button></td>
    </tr>`;
    if (editing === e.date)
      html += `<tr class="note-row"><td colspan="7"><form class="flag-editor" data-date="${e.date}">
        <input name="note" value="${esc(flagged.get(e.date) || "")}" placeholder="A note for your doctor (optional)" maxlength="300" />
        <button class="text-btn" type="submit">${isF ? "Save" : "Flag day"}</button>
        ${isF ? `<button class="text-btn unflag" type="button">Remove flag</button>` : ""}
        <button class="text-btn cancel" type="button">Cancel</button></form></td></tr>`;
    else if (isF && flagged.get(e.date))
      html += `<tr class="note-row"><td></td><td colspan="6" class="note">“${esc(flagged.get(e.date))}”</td></tr>`;
    return html;
  });
  if (ledgerEntries.length > ledgerLimit)
    rows.push(`<tr><td colspan="7"><button class="text-btn ledger-more">Show ${Math.min(21, ledgerEntries.length - ledgerLimit)} earlier days</button></td></tr>`);
  $("#ledger").innerHTML = rows.join("") || `<tr><td colspan="7" class="caption">No entries yet.</td></tr>`;

  $$("#ledger .flag-btn").forEach((b) => b.addEventListener("click", () => {
    editing = editing === b.dataset.date ? null : b.dataset.date;
    renderLedger().then(focusEditor);
  }));
  const form = $("#ledger .flag-editor");
  if (form) {
    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      await api("/api/flags", { method: "POST", body: JSON.stringify({ date: form.dataset.date, note: form.note.value }) });
      editing = null;
      await refresh();
    });
    form.querySelector(".cancel").addEventListener("click", () => { editing = null; renderLedger(); });
    form.querySelector(".unflag")?.addEventListener("click", async () => {
      await api(`/api/flags/${form.dataset.date}`, { method: "DELETE" });
      editing = null;
      await refresh();
    });
  }
  $("#ledger .ledger-more")?.addEventListener("click", () => { ledgerLimit += 21; renderLedger(); });
}

function focusEditor() {
  $("#ledger .flag-editor input")?.focus();
}

// Clicking a day on the chart (or a cited date) opens that row in the ledger with the flag editor.
async function openDay(date) {
  const idx = ledgerEntries.findIndex((e) => e.date === date);
  if (idx < 0) return;
  if (idx >= ledgerLimit) ledgerLimit = idx + 7;
  editing = date;
  await renderLedger();
  const row = $(`#ledger tr[data-date="${date}"]`);
  row?.scrollIntoView({ block: "center", behavior: "smooth" });
  row?.classList.add("pulse");
  focusEditor();
}

// ------------------------------------------------------------------ memory
async function renderMemory() {
  const { facts, events } = await api("/api/memory");
  $("#facts").innerHTML = facts.length
    ? facts
        .map((f) => {
          const prov = f.provenance.user_quote
            ? `You said: “${esc(f.provenance.user_quote.slice(0, 90))}”`
            : f.provenance.tool
            ? `Inferred from the data (run ${esc(f.provenance.run_id)}), e.g. flare on ${esc(
                (f.provenance.evidence || []).slice(0, 2).map((e) => e.flare_date).join(", ")
              )}. Not confirmed by you.`
            : f.provenance.via === "ui" ? "Added by you." : "";
          const cat = f.category.replace(/_/g, " ");
          return `<li><span class="cat ${f.source}">${esc(cat)}</span>${esc(f.statement)}
            <span class="prov">${prov} ${esc(f.created_at.slice(0, 10))}<button class="forget" data-id="${f.id}">forget</button></span></li>`;
        })
        .join("")
    : `<li class="empty">Nothing yet. Tell it how you define a bad day, or what you've already tried.</li>`;
  $("#events").innerHTML = events
    .slice()
    .reverse()
    .map((e) => `<li>${esc(e.at.slice(0, 16).replace("T", " "))} · #${esc(e.fact_id)} ${esc(e.action)}</li>`)
    .join("");
  $$(".facts .forget").forEach((b) =>
    b.addEventListener("click", async () => {
      await api(`/api/memory/${b.dataset.id}`, { method: "DELETE" });
      await refresh();
    })
  );
}

// ------------------------------------------------------------------ questions
function note(text) {
  const el = document.createElement("div");
  el.className = "qa note";
  el.textContent = text;
  $("#messages").appendChild(el);
}

function addQA(question) {
  const el = document.createElement("article");
  el.className = "qa pending";
  el.innerHTML = `<p class="q">${esc(question)}</p><div class="a">Working it out from your record</div>`;
  $("#messages").appendChild(el);
  el.scrollIntoView({ block: "nearest", behavior: "smooth" });
  return el;
}

function fillQA(el, res) {
  el.classList.remove("pending");
  if (res.safety_flags.some((f) => f.startsWith("red_flag"))) el.classList.add("alert");
  const days = res.citations.slice(0, 10).map((c) => `<span class="day" data-date="${esc(c)}">${esc(c.slice(5))}</span>`).join("");
  const tools = [...new Set(res.trace.filter((t) => t.ok).map((t) => t.tool.replace(/_/g, " ")))].join(", ");
  const bindings = (res.bindings || []).length
    ? `<details><summary>Where each number came from</summary><pre>${esc(
        res.bindings.map((b) => `${b.value}  ←  ${b.ref}.${b.path}${b.filter ? " | " + b.filter : ""}`).join("\n")
      )}</pre></details>`
    : "";
  el.querySelector(".a").innerHTML = md(res.answer);
  el.insertAdjacentHTML("beforeend", `<div class="meta">
      ${days ? `Days used: ${days}<br>` : ""}
      ${tools ? `Computed with ${esc(tools)}. ` : ""}${res.grounded ? `<span class="ok">Numbers checked against the computation.</span>` : "Some numbers could not be checked."}
      ${res.safety_flags.includes("blocked_unsafe_output") ? " A draft was withheld for giving medical advice." : ""}
    </div>${bindings}
    <details><summary>Full computation trace</summary><pre>${esc(JSON.stringify(res.trace, null, 1))}</pre></details>`);
  el.querySelectorAll(".day").forEach((s) => s.addEventListener("click", () => openDay(s.dataset.date)));
}

async function send(text) {
  if (!text.trim()) return;
  const el = addQA(text);
  try {
    const res = await api("/api/chat", { method: "POST", body: JSON.stringify({ message: text, history }) });
    history.push({ role: "user", content: text }, { role: "assistant", content: res.answer });
    fillQA(el, res);
    for (const t of res.trace) {
      const f = (t.result && t.result.findings && t.result.findings[0]) || null;
      if (f) highlight = f.exposure;
    }
    await refresh();
  } catch (e) {
    el.classList.remove("pending");
    el.querySelector(".a").textContent = `Something went wrong: ${e.message}`;
  }
}

// ------------------------------------------------------------------ events
$("#chat-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const v = $("#chat-input").value;
  $("#chat-input").value = "";
  send(v);
});
$("#suggestions").addEventListener("click", (e) => {
  if (e.target.tagName === "BUTTON") send(e.target.textContent);
});
$("#load").addEventListener("click", async () => {
  await loadPersona($("#persona").value);
  await refresh();
});
$("#log-date").addEventListener("change", setLongDate);
$("#log-btn").addEventListener("click", async () => {
  const text = $("#log-text").value.trim();
  if (!text) {
    $("#log-text").focus();
    return;
  }
  try {
    const r = await api("/api/entries", { method: "POST", body: JSON.stringify({ text, date: $("#log-date").value || null }) });
    const e = r.entry;
    const sym = Object.entries(e.symptoms).map(([k, v]) => `${pretty(k)} ${v}`).join(", ") || "no symptoms";
    $("#log-result").innerHTML =
      (r.care_notice ? `<p class="nudge">${md(r.care_notice)}</p>` : "") +
      `Recorded ${esc(day(e.date, { day: "numeric", month: "short" }))}: <span class="mono">${esc(sym)} · sleep ${e.sleep_hours ?? "?"}h · exertion ${
        e.exertion ?? "?"
      }${e.foods.length ? " · " + esc(e.foods.join(", ")) : ""}${e.meds.length ? " · " + esc(e.meds.join(", ")) : ""}</span>`;
    $("#log-text").value = "";
    await refresh();
  } catch (err) {
    $("#log-result").textContent = `Couldn't save: ${err.message}`;
  }
});

init();
