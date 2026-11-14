const WINDOW_S = 60;
const $ = (id) => document.getElementById(id);
const BED = decodeURIComponent(location.pathname.split("/").filter(Boolean)[1] || "");
const API = `/api/beds/${encodeURIComponent(BED)}`;
$("video").src = `${API}/video`;
const banner = $("banner"), bannerText = $("banner-text");
let lastState = null, t0 = null, lastChartUpdate = 0;

// ---------------------------------------------------------------- chart
const chart = new Chart($("chart"), {
  type: "line",
  data: {
    datasets: [
      { label: "score", data: [], borderColor: "#4f8ef7", borderWidth: 2, pointRadius: 0, spanGaps: false },
      { label: "threshold", data: [], borderColor: "#c62828", borderDash: [6, 4], borderWidth: 1.5, pointRadius: 0 },
    ],
  },
  options: {
    animation: false, maintainAspectRatio: false, parsing: false,
    scales: {
      x: { type: "linear", ticks: { color: "#8a919e", callback: (v) => `${Math.round(v)}s` }, grid: { color: "#262a33" } },
      y: { min: 0, suggestedMax: 6, ticks: { color: "#8a919e" }, grid: { color: "#262a33" } },
    },
    plugins: { legend: { labels: { color: "#e8eaed" } } },
  },
});

function pushChart(st) {
  if (t0 === null) t0 = st.t;
  const x = st.t - t0;
  const [score, thr] = chart.data.datasets;
  // gap in the line while baselining or skipping frames
  score.data.push({ x, y: st.state !== "baselining" && st.valid ? st.score : null });
  thr.data.push({ x, y: st.threshold });
  for (const ds of chart.data.datasets) {
    while (ds.data.length && ds.data[0].x < x - WINDOW_S) ds.data.shift();
  }
  chart.options.scales.x.min = Math.max(0, x - WINDOW_S);
  chart.options.scales.x.max = Math.max(WINDOW_S, x);
  const now = Date.now();
  if (now - lastChartUpdate > 100) { chart.update("none"); lastChartUpdate = now; }
}

// ---------------------------------------------------------------- UI
const fmtTime = (epoch) => new Date(epoch * 1000).toLocaleTimeString([], { hour12: false });

function render(st) {
  $("bed-name").textContent = st.bed_name;
  document.title = `StrokeWatch: ${st.bed_name}`;
  $("bed-kind").textContent = { sim: "SIM", phone: "PHONE" }[st.kind] || "";
  $("bed-kind").hidden = !$("bed-kind").textContent;
  $("phone-actions").hidden = st.kind !== "phone";
  banner.className = `banner ${st.state}`;
  if (st.state === "offline") {
    const age = st.last_frame_age == null ? "" : ` (last frame ${Math.round(st.last_frame_age)} s ago)`;
    const what = st.kind === "phone" && !st.phone_connected ? "Phone disconnected" : "Camera offline";
    bannerText.textContent = `${what}${age}`;
    lastState = st.state;
    return;
  }
  if (st.state === "starting") {
    bannerText.textContent = "Starting camera…";
    return;
  }
  if (st.state === "baselining") {
    const pct = Math.round(st.baseline_progress * 100);
    bannerText.textContent = st.calibrating
      ? `Calibrating ${pct}%: ${st.prompt || ""}` : `Building baseline… ${pct}%`;
    $("progress").style.width = `${st.baseline_progress * 100}%`;
  } else if (st.state === "monitoring") {
    const cal = st.calibrated_range ? ` (calibrated for turns ${st.calibrated_range[0]}° to ${st.calibrated_range[1]}°)` : "";
    bannerText.textContent = `Monitoring: no new droop${cal}`;
  } else {
    const onset = st.onset_time ? fmtTime(st.onset_time) : "?";
    const ack = st.alert && st.alert.status === "acknowledged" ? " (acknowledged)" : "";
    bannerText.textContent = `STROKE ALERT: possible new ${st.side} facial droop, onset ${onset}${ack}`;
  }
  if (st.state === "ALERT" && lastState !== "ALERT") beep();
  lastState = st.state;

  const skip = $("skip");
  if (!st.valid) { skip.className = "chip skip"; skip.textContent = `Skipping frames: ${st.invalid_reason}`; }
  else { skip.className = "chip ok"; skip.textContent = st.invalid_reason ? `Tracking (${st.invalid_reason})` : "Tracking face"; }

  $("score").textContent = st.state === "baselining" ? "–" : st.score.toFixed(2);
  $("above").textContent = `${st.above_seconds.toFixed(1)} / ${st.alert_seconds}s`;
  $("fps").textContent = st.fps ?? "–";
  renderZ(st);
  pushChart(st);
}

function renderZ(st) {
  const names = { mouth_diff: "mouth", eye_ratio: "eye", brow_diff: "brow" };
  const box = $("zbars");
  if (!st.z || !Object.keys(st.z).length) { box.innerHTML = '<span class="muted">available after baseline</span>'; return; }
  box.innerHTML = Object.entries(st.z).map(([k, z]) => {
    const pct = Math.min(Math.abs(z) / (st.threshold * 1.5), 1) * 50;
    const left = z >= 0 ? 50 : 50 - pct;
    const hot = Math.abs(z) >= st.threshold ? "hot" : "";
    return `<div class="zrow"><span>${names[k] || k}</span>
      <div class="ztrack"><div class="zfill ${hot}" style="left:${left}%;width:${pct}%"></div></div>
      <span class="zval">${z.toFixed(1)}</span></div>`;
  }).join("");
}

let audio;
function beep() {
  try {
    audio = audio || new AudioContext();
    [0, 0.35, 0.7].forEach((d) => {
      const o = audio.createOscillator(), g = audio.createGain();
      o.frequency.value = 880; o.connect(g); g.connect(audio.destination);
      g.gain.setValueAtTime(0.25, audio.currentTime + d);
      o.start(audio.currentTime + d); o.stop(audio.currentTime + d + 0.2);
    });
  } catch (e) { /* sound is optional */ }
}
document.addEventListener("click", () => { audio = audio || new AudioContext(); }, { once: true });

async function rebuild(action) {
  await fetch(`${API}/${action}`, { method: "POST" });
  for (const ds of chart.data.datasets) ds.data = [];
  t0 = null; lastState = null;
}
$("reset").addEventListener("click", () => rebuild("reset"));
$("calibrate").addEventListener("click", () => rebuild("calibrate"));

$("say-test").addEventListener("click", async () => {
  const r = await fetch(`${API}/say`, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: "This is a test from the StrokeWatch dashboard." }) });
  if (!r.ok) alert("The phone is not connected.");
});
$("remove").addEventListener("click", async () => {
  if (!confirm("Remove this phone from the dashboard? Its baseline and alerts are dropped.")) return;
  await fetch(API, { method: "DELETE" });
  location.href = "/";
});

// ---------------------------------------------------------------- websocket
function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}${API}/ws`);
  ws.onmessage = (e) => render(JSON.parse(e.data));
  ws.onclose = (e) => {
    if (e.code === 4404) {
      banner.className = "banner offline";
      bannerText.textContent = "This bed no longer exists (phone removed or server restarted).";
      return;
    }
    banner.className = "banner offline";
    bannerText.textContent = "Disconnected: retrying…";
    setTimeout(connect, 1000);
  };
}
connect();
