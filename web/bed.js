const WINDOW_S = 60;
const $ = (id) => document.getElementById(id);
const BED = decodeURIComponent(location.pathname.split("/").filter(Boolean)[1] || "");
const API = `/api/beds/${encodeURIComponent(BED)}`;
// Landmark lines and dots on the video: on each time the page opens, toggled with the button.
let overlay = true;
function setVideo() {
  $("video").src = `${API}/video?overlay=${overlay ? 1 : 0}`;
  $("overlay-toggle").textContent = overlay ? "Hide landmarks" : "Show landmarks";
  $("overlay-toggle").setAttribute("aria-pressed", String(overlay));
}
$("overlay-toggle").addEventListener("click", () => { overlay = !overlay; setVideo(); });
setVideo();
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
  // combined detection scores while StrokeWatch is still building its baseline; StrokeWatch alone does not
  const scored = st.combined ? st.combined.state === "scoring" || st.state !== "baselining" : st.state !== "baselining" && st.valid;
  score.data.push({ x, y: scored ? st.score : null });
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

// Short text for an open alert: its findings (face droop and/or BE-FAST check signs).
function alertSigns(st) {
  const f = (st.alert && st.alert.findings) || [];
  return f.length ? f : [`Face: possible new ${st.side} facial droop`];
}

function render(st) {
  renderExam(st);
  $("banner-sub").hidden = true;
  $("bed-name").textContent = st.bed_name;
  document.title = `StrokeWatch: ${st.bed_name}`;
  $("bed-kind").textContent = { sim: "SIM", phone: "PHONE", demo: "DEMO" }[st.kind] || "";
  $("bed-kind").hidden = !$("bed-kind").textContent;
  $("phone-actions").hidden = st.kind !== "phone" && st.kind !== "demo";
  $("say-test").hidden = st.kind !== "phone";
  $("remove").textContent = st.kind === "demo" ? "Remove this demo" : "Remove this phone";
  $("sound-row").hidden = st.kind !== "camera" && st.kind !== "demo";
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
  } else if (st.state === "monitoring" && st.exam && st.exam.active) {
    banner.className = "banner exam";
    bannerText.textContent = `${st.exam.only === "arms" ? "Hand test" : "BE-FAST check"}: ${st.exam.label}`;
  } else if (st.state === "monitoring") {
    const cal = st.calibrated_range ? ` (calibrated for turns ${st.calibrated_range[0]}° to ${st.calibrated_range[1]}°)` : "";
    bannerText.textContent = `Monitoring: no new droop${cal}`;
  } else {
    const onset = st.alert && st.alert.onset_time ? fmtTime(st.alert.onset_time)
      : st.onset_time ? fmtTime(st.onset_time) : "?";
    const ack = st.alert && st.alert.status === "acknowledged" ? " (acknowledged)" : "";
    const signs = alertSigns(st);
    // st.side is the patient's WEAK side, not the side of the brain: the hemisphere wording
    // ("may be a right hemispheric stroke") comes from the server's localization.
    const loc = st.alert && st.alert.localization;
    const faceOnly = signs.length === 1 && signs[0].startsWith("Face:");
    bannerText.textContent = faceOnly
      ? `STROKE ALERT: possible new ${st.side} facial droop, onset ${onset}${ack}`
      : `STROKE ALERT: possible stroke signs${loc ? `, ${loc.short}` : ""}, onset ${onset}${ack}`;
    const sub = faceOnly ? [] : [signs.join(" · ")];
    if (loc) sub.push(`${loc.text} ${loc.caveat}`);
    $("banner-sub").textContent = sub.join("\n");
    $("banner-sub").style.whiteSpace = "pre-line";
    $("banner-sub").hidden = !sub.length;
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

// ---------------------------------------------------------------- BE-FAST check
const COMP_NAMES = { eyes: "Eyes", arms: "Arm drift" };
const ST_TEXT = { normal: "normal", abnormal: "ABNORMAL", no_response: "no response",
                  inconclusive: "INCONCLUSIVE", not_assessed: "not assessed", skipped: "skipped" };
let examSig = "";

function renderExam(st) {
  const ex = st.exam || { active: false };
  $("exam-live").hidden = !ex.active;
  $("exam-start").hidden = ex.active;
  $("hand-start").hidden = ex.active;
  $("exam-cancel").hidden = !ex.active;
  $("exam-start").disabled = $("hand-start").disabled = st.state === "offline";
  if (ex.active) {
    $("exam-idx").textContent = `step ${ex.index}/${ex.count} · ${ex.phase}`;
    $("exam-label").textContent = ex.label;
    $("exam-say").textContent = `Spoken: "${ex.say}"`;
    $("exam-prog").style.width = `${ex.progress * 100}%`;
    $("exam-trigger").textContent = `Started ${fmtTime(ex.started_at)} · reason: ${ex.trigger}`;
  }
  const r = st.last_exam;
  const sig = JSON.stringify(r && [r.finished_at, r.summary]);
  if (sig !== examSig) {
    examSig = sig;
    const box = $("exam-result");
    const what = r && r.only === "arms" ? "Last hand test" : "Last check";
    if (!r) box.innerHTML = '<span class="muted">Starts by itself on facial droop or a fixed gaze, or run it now.</span>';
    else if (r.cancelled) box.innerHTML = `<div class="exam-head">${what} ${fmtTime(r.finished_at)}: stopped (${esc(r.cancelled)})</div>`;
    else {
      const rows = Object.entries(r.components).map(([k, c]) =>
        `<div class="comp"><span>${COMP_NAMES[k] || k}</span><span class="st ${c.status}">${ST_TEXT[c.status] || c.status}</span><span>${esc(c.detail)}</span></div>`).join("");
      const head = r.flagged ? "flagged" : (r.inconclusive || []).length ? "inconclusive" : "";
      box.innerHTML = `<div class="exam-head ${head}">${what} ${fmtTime(r.finished_at)}: ${esc(r.summary)}</div>
        <div class="muted" style="font-size:14px;margin-bottom:4px">reason: ${esc(r.trigger)}${r.localization ? ` · ${esc(r.localization.text)} ${esc(r.localization.caveat)}` : ""}</div>${rows}`;
    }
  }
  renderGaze(st);
}

function renderGaze(st) {
  const g = st.gaze, f = st.features || {};
  const RANGE = 0.3;  // track spans -0.3 .. +0.3 (patient's right .. left)
  const pos = (v) => 50 + Math.max(-1, Math.min(1, v / RANGE)) * 50;
  if (!g) { $("gaze-text").textContent = "–"; return; }
  $("gthr-l").style.left = `${pos(g.threshold)}%`;
  $("gthr-r").style.left = `${pos(-g.threshold)}%`;
  if (!g.ready) {
    $("gdot").hidden = true;
    $("gaze-text").textContent = `learning ${Math.round(g.progress * 100)}%`;
    return;
  }
  const dot = $("gdot");
  const now = f.gaze_h == null ? null : f.gaze_h - g.center;
  dot.hidden = now == null;
  if (now != null) { dot.style.left = `${pos(now)}%`; dot.classList.toggle("hot", !!g.deviation); }
  if (st.exam && st.exam.active) {
    $("gaze-text").textContent = now == null ? "check running" : `${Math.abs(now).toFixed(2)} ${now >= 0 ? "L" : "R"}`;
  } else if (g.deviation) {
    const held = g.since ? Math.round(st.t - g.since) : 0;
    $("gaze-text").textContent = `held ${g.deviation} ${held}s`;
  } else {
    $("gaze-text").textContent = now == null ? "eyes not seen" : `${Math.abs(now).toFixed(2)} ${now >= 0 ? "L" : "R"}`;
  }
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

$("exam-start").addEventListener("click", () => fetch(`${API}/exam`, { method: "POST" }));
$("hand-start").addEventListener("click", () => fetch(`${API}/exam?only=arms`, { method: "POST" }));
$("exam-cancel").addEventListener("click", () => fetch(`${API}/exam/cancel`, { method: "POST" }));

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

// Webcam beds: check prompts play on this page, which needs one click first (browser rule).
function renderSound(on) {
  $("sound-state").textContent = on ? "Voice prompts play on this computer"
                                    : "Click anywhere on this page to turn on voice prompts";
  $("sound-state").className = on ? "muted" : "sound-off";
}
renderSound(Voice.unlocked);
Voice.onUnlockChange(renderSound);
$("sound-test").addEventListener("click", () => Voice.say("This is a voice test from StrokeWatch."));
$("say-test").addEventListener("click", async () => {
  const r = await fetch(`${API}/say`, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: "This is a test from the StrokeWatch dashboard." }) });
  if (!r.ok) alert("The phone is not connected.");
});
$("remove").addEventListener("click", async () => {
  const what = $("remove").textContent.replace("Remove ", "");
  if (!confirm(`Remove ${what} from the dashboard? Its baseline and alerts are dropped.`)) return;
  await fetch(API, { method: "DELETE" });
  location.href = "/";
});

// ---------------------------------------------------------------- websocket
function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}${API}/ws`);
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.type === "say") Voice.say(m.text);  // webcam bed: this page is the patient's speaker
    else render(m);
  };
  ws.onclose = (e) => {
    if (e.code === 4404) {
      banner.className = "banner offline";
      bannerText.textContent = "This bed no longer exists (removed or server restarted).";
      return;
    }
    banner.className = "banner offline";
    bannerText.textContent = "Disconnected: retrying…";
    setTimeout(connect, 1000);
  };
}
connect();
