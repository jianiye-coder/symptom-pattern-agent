const $ = (s) => document.querySelector(s);
const USER = "demo";
const history = [];
let highlight = null;

async function api(path, opts = {}) {
  const r = await fetch(path, {
    ...opts,
    headers: { "Content-Type": "application/json", "X-User-Id": USER, ...(opts.headers || {}) },
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
// minimal markdown: **bold** only, after escaping
const md = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");

async function init() {
  const { personas } = await api("/api/personas");
  $("#persona").innerHTML = Object.entries(personas)
    .map(([k, v]) => `<option value="${esc(k)}" title="${esc(v)}">${esc(k)}</option>`)
    .join("");
  const h = await api("/api/health");
  $("#mode").textContent = `LLM: ${h.llm}`;
  $("#log-date").value = new Date().toISOString().slice(0, 10);
  const { entries } = await api("/api/entries");
  if (!entries.length) await loadPersona("pem");
  await refresh();
}

async function loadPersona(p) {
  const r = await api(`/api/demo/${p}`, { method: "POST" });
  $("#persona").value = p;
  history.length = 0;
  highlight = null;
  $("#messages").innerHTML = "";
  addMsg("agent", `Loaded synthetic user "${p}" (${r.loaded} days). ${r.description}`);
}

async function refresh() {
  await Promise.all([renderMemory(), renderChart()]);
}

async function renderMemory() {
  const { facts, events } = await api("/api/memory");
  $("#facts").innerHTML = facts.length
    ? facts
        .map((f) => {
          const prov = f.provenance.user_quote
            ? `you said: “${esc(f.provenance.user_quote.slice(0, 90))}”`
            : f.provenance.tool
            ? `from ${esc(f.provenance.tool)} (run ${esc(f.provenance.run_id)}), evidence ${esc(
                (f.provenance.evidence || []).slice(0, 2).map((e) => e.flare_date).join(", ")
              )}`
            : esc(JSON.stringify(f.provenance));
          return `<li><div class="cat">#${f.id} · ${f.category} · ${f.source}</div>${esc(f.statement)}
            <div class="prov">${prov} · ${f.created_at.slice(0, 16).replace("T", " ")}</div>
            <button class="x" title="Retract" data-id="${f.id}">×</button></li>`;
        })
        .join("")
    : `<li class="muted">Nothing yet. Tell the agent how you define a bad day, or what you've tried.</li>`;
  $("#events").innerHTML = events
    .slice()
    .reverse()
    .map((e) => `<li>${esc(e.at.slice(11, 19))} #${esc(e.fact_id)} ${esc(e.action)}</li>`)
    .join("");
  document.querySelectorAll(".facts .x").forEach((b) =>
    b.addEventListener("click", async () => {
      await api(`/api/memory/${b.dataset.id}`, { method: "DELETE" });
      await refresh();
    })
  );
}

async function renderChart() {
  const q = highlight ? `?exposure=${encodeURIComponent(highlight)}` : "";
  const d = await api(`/api/chart${q}`);
  const pts = d.points;
  $("#chart-sub").textContent = `${d.symptom} · flare = ${d.flare_definition}`;
  if (!pts.length) {
    $("#chart").innerHTML = `<p class="muted">No entries yet.</p>`;
    return;
  }
  const W = 440, H = 300, L = 26, R = 8, T = 10, B = 26;
  const t0 = new Date(pts[0].date), t1 = new Date(pts[pts.length - 1].date);
  const span = Math.max(1, (t1 - t0) / 864e5);
  const x = (s) => L + ((new Date(s) - t0) / 864e5 / span) * (W - L - R);
  const y = (v) => T + (1 - v / 10) * (H - T - B);
  const bw = Math.max(2, ((W - L - R) / (span + 1)) * 0.7);
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="symptom timeline">`;
  for (const g of [0, 5, 10])
    svg += `<line x1="${L}" x2="${W - R}" y1="${y(g)}" y2="${y(g)}" stroke="#ece6dc"/><text x="4" y="${y(g) + 4}" font-size="10" fill="#8a9396">${g}</text>`;
  for (const p of pts) {
    if (p.exposed)
      svg += `<rect x="${x(p.date) - bw / 2 - 1}" y="${T}" width="${bw + 2}" height="${H - T - B}" fill="#e0a43a" opacity="0.22"/>`;
    if (p.exertion != null)
      svg += `<rect x="${x(p.date) - bw / 2}" y="${y(p.exertion)}" width="${bw}" height="${y(0) - y(p.exertion)}" fill="${p.exposed ? "#e0a43a" : "#cfd8d6"}" rx="1"><title>${p.date} exertion ${p.exertion}</title></rect>`;
  }
  const line = pts.filter((p) => p.severity != null).map((p) => `${x(p.date)},${y(p.severity)}`).join(" ");
  svg += `<polyline points="${line}" fill="none" stroke="#2f6f6a" stroke-width="2" stroke-linejoin="round"/>`;
  for (const p of pts)
    if (p.flare)
      svg += `<circle cx="${x(p.date)}" cy="${y(p.severity)}" r="4" fill="#c8553d"><title>${p.date} flare (${p.severity})</title></circle>`;
  const ticks = [pts[0], pts[Math.floor(pts.length / 2)], pts[pts.length - 1]];
  for (const p of ticks)
    svg += `<text x="${x(p.date)}" y="${H - 8}" font-size="10" fill="#8a9396" text-anchor="middle">${p.date.slice(5)}</text>`;
  svg += `</svg>`;
  $("#chart").innerHTML = svg;
  $("#legend-exp").hidden = !highlight;
  if (highlight) {
    $("#legend-exp-label").textContent = d.exposure_label;
    $("#chart-note").textContent = `Highlighted days: ${d.exposure_label}. Look for red dots in the following days.`;
  }
}

function addMsg(role, text, res) {
  const el = document.createElement("div");
  el.className = `msg ${role}` + (res && res.safety_flags.some((f) => f.startsWith("red_flag")) ? " alert" : "");
  el.innerHTML = md(text);
  if (res) {
    const cites = res.citations.slice(0, 12).map((c) => `<span class="chip">${esc(c)}</span>`).join("");
    const tools = res.trace.map((t) => `${t.tool}${t.ok ? "" : " (rejected)"}`).join(" → ") || "no tools";
    el.innerHTML += `<div class="meta">${cites ? "Cited days: " + cites + "<br>" : ""}Tools: ${esc(tools)} · ${
      res.grounded ? "grounded ✓" : "ungrounded ✗"
    } · ${res.latency_ms} ms${res.safety_flags.length ? " · " + esc(res.safety_flags.join(", ")) : ""}</div>
    <details><summary>Tool trace</summary><pre>${esc(JSON.stringify(res.trace, null, 1))}</pre></details>`;
  }
  $("#messages").appendChild(el);
  $("#messages").scrollTop = 1e9;
}

async function send(text) {
  if (!text.trim()) return;
  addMsg("user", text);
  try {
    const res = await api("/api/chat", { method: "POST", body: JSON.stringify({ message: text, history }) });
    history.push({ role: "user", content: text }, { role: "assistant", content: res.answer });
    addMsg("agent", res.answer, res);
    for (const t of res.trace) {
      const r = t.result || {};
      const f = (r.findings && r.findings[0]) || null;
      if (f) highlight = f.exposure;
    }
    await refresh();
  } catch (e) {
    addMsg("agent", `Error: ${e.message}`);
  }
}

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
$("#log-btn").addEventListener("click", async () => {
  const text = $("#log-text").value.trim();
  if (!text) return;
  try {
    const r = await api("/api/entries", { method: "POST", body: JSON.stringify({ text, date: $("#log-date").value || null }) });
    const e = r.entry;
    const sym = Object.entries(e.symptoms).map(([k, v]) => `${k} ${v}`).join(", ") || "none";
    $("#log-result").innerHTML =
      (r.care_notice ? `<div class="msg agent alert">${md(r.care_notice)}</div>` : "") +
      `Saved ${esc(e.date)} (${esc(r.extraction.backend)}): symptoms ${esc(sym)} · sleep ${e.sleep_hours ?? "?"}h · exertion ${
        e.exertion ?? "?"
      } · foods ${esc(e.foods.join(", ") || "-")} · meds ${esc(e.meds.join(", ") || "-")}`;
    $("#log-text").value = "";
    await refresh();
  } catch (err) {
    $("#log-result").textContent = `Error: ${err.message}`;
  }
});

init();
