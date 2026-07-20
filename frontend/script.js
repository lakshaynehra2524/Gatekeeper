/* Edge Gatekeeper console — vanilla JS, no external dependencies. */

const $ = (id) => document.getElementById(id);
const stream = $("stream");
const emptyState = $("empty-state");
const results = []; // decisions in arrival order, for the inspector
let replaying = false;

/*  bootstrap */
async function init() {
  const [info, scenarios] = await Promise.all([
    fetch("/api/model-info").then((r) => r.json()),
    fetch("/api/scenarios").then((r) => r.json()),
  ]);

  $("chip-size").textContent = `${info.model_assets_mb} MB`;
  const lat = info.meta?.latency_ms_per_utterance_laptop;
  $("chip-latency").textContent = lat ? `~${lat} ms` : "~2 ms";

  const select = $("scenario-select");
  scenarios.forEach((s, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = `${s.name} — ${s.turns.length} turns`;
    select.appendChild(opt);
  });
  select.dataset.scenarios = JSON.stringify(scenarios);
  updateStats(info.stats || {});
}
init();

/* evaluation */
async function evaluateText(text) {
  const res = await fetch("/api/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error("evaluate failed");
  const decision = await res.json();
  results.push(decision);
  addRow(decision, results.length - 1);
  showDecision(decision, results.length - 1);
  bumpStats(decision.decision);
  return decision;
}

/* render */
function addRow(d, idx) {
  emptyState?.remove();
  const row = document.createElement("div");
  row.className = `row ${d.decision}`;
  row.dataset.idx = idx;
  row.innerHTML = `
    <span class="badge ${d.decision}">${d.decision}</span>
    <span class="utt"></span>
    <span class="mini-meter"><i style="left:${d.p_meaningful * 100}%"></i></span>`;
  row.querySelector(".utt").textContent = d.text;
  row.addEventListener("click", () => showDecision(results[idx], idx));
  stream.appendChild(row);
  stream.scrollTop = stream.scrollHeight;
}

function showDecision(d, idx) {
  document.querySelectorAll(".row.selected").forEach((r) => r.classList.remove("selected"));
  document.querySelector(`.row[data-idx="${idx}"]`)?.classList.add("selected");

  $("inspect-empty").classList.add("hidden");
  $("inspect-body").classList.remove("hidden");

  const banner = $("decision-banner");
  banner.textContent = d.decision;
  banner.className = `decision-banner ${d.decision}`;
  $("inspect-text").textContent = `"${d.text}"`;
  $("meter-needle").style.left = `${d.p_meaningful * 100}%`;

  const bars = $("prob-bars");
  bars.innerHTML = "";
  Object.entries(d.probs)
    .sort((a, b) => b[1] - a[1])
    .forEach(([label, p], i) => {
      const row = document.createElement("div");
      row.className = `bar-row${i === 0 ? " top" : ""}`;
      row.innerHTML = `
        <span>${label}</span>
        <span class="bar-track"><span class="bar-fill" style="width:${p * 100}%"></span></span>
        <span class="val">${(p * 100).toFixed(0)}%</span>`;
      bars.appendChild(row);
    });

  $("inspect-reason").textContent = d.reason;
  const dup = $("inspect-dup");
  if (d.similar_to) {
    dup.textContent = `similar to already-forwarded: "${d.similar_to}"`;
    dup.classList.remove("hidden");
  } else {
    dup.classList.add("hidden");
  }
  $("inspect-latency").textContent = `${d.latency_ms} ms`;
}

/* stats */
const statEls = {};
document.querySelectorAll(".stat").forEach((el) => {
  const key = el.classList[1].toUpperCase();
  statEls[key] = el.querySelector("b");
});
function updateStats(stats) {
  Object.entries(statEls).forEach(([k, el]) => (el.textContent = stats[k] ?? 0));
}
function bumpStats(decision) {
  const el = statEls[decision];
  if (el) el.textContent = Number(el.textContent) + 1;
}

/* actions */
$("composer").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("utterance-input");
  const text = input.value.trim();
  if (!text || replaying) return;
  input.value = "";
  await evaluateText(text);
});

$("btn-replay").addEventListener("click", async () => {
  if (replaying) return;
  const select = $("scenario-select");
  const scenarios = JSON.parse(select.dataset.scenarios || "[]");
  const scenario = scenarios[Number(select.value)];
  if (!scenario) return;

  replaying = true;
  $("btn-replay").disabled = true;
  await resetSession();
  for (const turn of scenario.turns) {
    await evaluateText(turn.text);
    await new Promise((r) => setTimeout(r, 650)); // fragments arrive incrementally
  }
  replaying = false;
  $("btn-replay").disabled = false;
});

$("btn-reset").addEventListener("click", async () => {
  if (replaying) return;
  await resetSession();
});

async function resetSession() {
  await fetch("/api/reset", { method: "POST" });
  results.length = 0;
  stream.innerHTML =
    '<div class="empty">Session reset — the gate has forgotten all context and duplicates.</div>';
  updateStats({});
  $("inspect-body").classList.add("hidden");
  $("inspect-empty").classList.remove("hidden");
}
