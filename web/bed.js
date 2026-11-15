const WINDOW_S = 60;
const $ = (id) => document.getElementById(id);
const BED = decodeURIComponent(location.pathname.split("/").filter(Boolean)[1] || "");
const API = `/api/beds/${encodeURIComponent(BED)}`;
$("topbar").prepend(brandBar());
$("unit-name").textContent = UNIT;
bindStaffInput($("staff"));
const banner = $("banner"), bannerText = $("banner-text");
let st = null, lastState = null, t0 = null, lastChartUpdate = 0;
let ptSig = "", alertSig = "", histSig = "", logSig = "", examSig = "";

// ---------------------------------------------------------------- chart
const chart = new Chart($("chart"), {
  type: "line",
  data: {
    datasets: [
      { label: "Change score", data: [], borderColor: "#1068c4", backgroundColor: "rgba(16,104,196,0.08)", fill: true, borderWidth: 2, pointRadius: 0, spanGaps: false },
      { label: "Alert threshold", data: [], borderColor: "#c4262e", borderDash: [6, 4], borderWidth: 1.5, pointRadius: 0 },
      // the two parts that add up to the change score (combined detection only)
      { label: "Eye-CU detector part", data: [], borderColor: "#7d8ea0", borderWidth: 1.5, pointRadius: 0, spanGaps: false },
      { label: "Palsy model part", data: [], borderColor: "#3a90a2", borderWidth: 1.5, pointRadius: 0, spanGaps: false },
    ],
  },
  options: {
    animation: false, maintainAspectRatio: false, parsing: false,
    scales: {
      x: { type: "linear", ticks: { color: "#5f7489", callback: (v) => `${Math.round(v)}s` }, grid: { color: "#e6edf4" } },
      y: { suggestedMin: 0, suggestedMax: 6, ticks: { color: "#5f7489" }, grid: { color: "#e6edf4" } },
    },
    plugins: { legend: { labels: { color: "#13293d", boxWidth: 18 } } },
  },
});

function pushChart(s) {
  if (t0 === null) t0 = s.t;
  const x = s.t - t0;
  const [score, thr, swPart, mdPart] = chart.data.datasets;
  // combined detection scores while Eye-CU's detector is still building its baseline; the detector alone does not
  const scored = s.combined ? s.combined.state === "scoring" || s.state !== "baselining" : s.state !== "baselining" && s.valid;
  score.data.push({ x, y: scored ? s.score : null });
  thr.data.push({ x, y: s.threshold });
  const c = s.combined;
  swPart.data.push({ x, y: c && scored ? c.sw_term : null });
  mdPart.data.push({ x, y: c && scored ? c.model_term : null });
  for (const ds of chart.data.datasets) {
    while (ds.data.length && ds.data[0].x < x - WINDOW_S) ds.data.shift();
  }
  chart.options.scales.x.min = Math.max(0, x - WINDOW_S);
  chart.options.scales.x.max = Math.max(WINDOW_S, x);
  const now = Date.now();
  if (now - lastChartUpdate > 100) { chart.update("none"); lastChartUpdate = now; }
}

// ---------------------------------------------------------------- patient banner
function lkwLocal(epoch) {
  if (!epoch) return "";
  const d = new Date(epoch * 1000), p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}`;
}
function renderPatient(s) {
  const p = s.patient || {};
  const sig = JSON.stringify([s.bed_name, s.kind, s.label, p]);
  if (sig === ptSig) return;
  ptSig = sig;
  $("bed-name").textContent = s.bed_name;
  $("crumb-bed").textContent = s.bed_name;
  $("pt-modal-bed").textContent = s.bed_name;
  $("bed-kind").textContent = KIND_BADGE[s.kind] || "";
  $("bed-kind").className = `kind-badge ${s.kind}`;
  $("bed-kind").hidden = !$("bed-kind").textContent;
  $("bed-label").textContent = s.label || (s.demo_clip ? `Demo video: ${s.demo_clip}` : "");
  const f = (k, v) => `<div class="pt-f"><div class="k">${k}</div><div class="v${v ? "" : " none"}">${v ? esc(v) : "not entered"}</div></div>`;
  $("pt-fields").innerHTML = [
    f("Patient", p.name), f("MRN", p.mrn), f("Age / sex", [p.age, p.sex].filter(Boolean).join(" ")),
    f("Diagnosis", p.dx), f("Attending", p.attending),
    f("Last known well", p.lkw ? `${fmtTime(p.lkw)} · ${new Date(p.lkw * 1000).toLocaleDateString()}` : ""),
  ].join("");
  $("pt-fields").title = p.notes ? `Notes: ${p.notes}` : "";
}
$("pt-edit").onclick = () => {
  const p = (st && st.patient) || {};
  const form = $("pt-form");
  for (const k of ["name", "mrn", "age", "sex", "dx", "attending", "notes"]) form.elements[k].value = p[k] || "";
  form.elements.lkw.value = lkwLocal(p.lkw);
  $("pt-modal").hidden = false;
  form.elements.name.focus();
};
const closePt = () => { $("pt-modal").hidden = true; };
$("pt-close").onclick = closePt;
$("pt-cancel").onclick = closePt;
$("pt-modal").onclick = (e) => { if (e.target === $("pt-modal")) closePt(); };
$("pt-form").onsubmit = async (e) => {
  e.preventDefault();
  const form = e.target, data = {};
  for (const k of ["name", "mrn", "age", "sex", "dx", "attending", "notes"]) data[k] = form.elements[k].value;
  const lkw = form.elements.lkw.value;
  data.lkw = lkw ? new Date(lkw).getTime() / 1000 : null;
  if (await post(`${API}/patient`, data)) closePt();
  else alert("Could not save the patient details.");
};

// ---------------------------------------------------------------- status banner
function render(s) {
  st = s;
  setServerTime(s.server_time);
  renderPatient(s);
  renderAlert(s);
  renderExam(s);
  renderHistory(s);
  renderLog(s);
  $("phone-actions").hidden = s.kind !== "phone" && s.kind !== "demo";
  $("say-test").hidden = s.kind !== "phone";
  $("remove").textContent = s.kind === "demo" ? "Remove this demo" : "Remove this phone";
  $("sound-row").hidden = !["camera", "demo", "video"].includes(s.kind);
  document.title = `${s.state === "ALERT" ? "ALERT · " : ""}${s.bed_name} · Azure Medical Center`;

  const exam = s.exam && s.exam.active;
  const acked = s.alert && s.alert.status === "acknowledged";
  banner.className = `banner ${exam && s.state === "monitoring" ? "exam" : s.state}${acked ? " acked" : ""}`;
  $("banner-sub").hidden = true;
  let since = null;
  if (s.state === "offline") {
    const age = s.last_frame_age == null ? "" : ` (last frame ${Math.round(s.last_frame_age)} s ago)`;
    const what = s.kind === "phone" && !s.phone_connected ? "Phone camera disconnected" : "Camera offline: not monitoring";
    bannerText.textContent = `${what}${age}`;
    $("banner-time").textContent = "";
    delete $("banner-time").dataset.since;
    lastState = s.state;
    return;
  }
  if (s.state === "starting") {
    bannerText.textContent = "Starting camera…";
    return;
  }
  if (s.state === "baselining") {
    const pct = Math.round(s.baseline_progress * 100);
    bannerText.textContent = s.calibrating ? `Calibrating head turns · ${pct}%: ${s.prompt || ""}`
      : `Learning this patient's baseline · ${pct}%`;
    $("progress").style.width = `${s.baseline_progress * 100}%`;
  } else if (s.state === "monitoring" && exam) {
    bannerText.textContent = `${s.exam.only === "arms" ? "Hand test" : "BE-FAST check"} in progress: ${s.exam.label}`;
  } else if (s.state === "monitoring") {
    const cal = s.calibrated_range ? ` · calibrated for head turns ${s.calibrated_range[0]}° to ${s.calibrated_range[1]}°` : "";
    bannerText.textContent = `Monitoring · no new facial asymmetry${cal}`;
  } else {
    const signs = s.alert ? alertFindings(s.alert) : [`Face: possible new ${s.side} facial droop`];
    // side = the patient's WEAK side; the hemisphere wording comes from the server (localize.py)
    const side = (s.alert && s.alert.side) || s.side;
    const loc = s.alert && s.alert.localization;
    const what = loc ? loc.short : side ? `${side}-sided signs` : "possible stroke signs";
    bannerText.textContent = `STROKE ALERT: ${what}${acked ? " · acknowledged" : ""}`;
    const sub = [signs.map(stripTag).join(" · ")];
    if (loc) sub.push(locText(loc));
    $("banner-sub").textContent = sub.join("\n");
    $("banner-sub").style.whiteSpace = "pre-line";
    $("banner-sub").hidden = false;
    since = s.alert ? s.alert.onset_time || s.alert.raised_at : s.onset_time;
  }
  const bt = $("banner-time");
  if (since) { bt.dataset.since = since; bt.textContent = `since onset ${fmtDur(serverNow() - since)}`; }
  else { delete bt.dataset.since; bt.textContent = ""; }
  if (s.state === "ALERT" && lastState !== "ALERT") beep();
  lastState = s.state;

  const skip = $("skip");
  if (!s.valid) { skip.className = "pill warn"; skip.textContent = `Paused: ${s.invalid_reason}`; }
  else { skip.className = "pill ok"; skip.textContent = s.invalid_reason ? `Tracking (${s.invalid_reason})` : "Face tracked"; }

  $("score").textContent = s.state === "baselining" ? "–" : s.score.toFixed(2);
  const cb = s.combined;
  $("sw-part").textContent = cb ? cb.sw_term.toFixed(2) : "–";
  $("model-part").textContent = !cb ? "–" : cb.head_turned ? "paused (head turned)"
    : cb.relative == null ? (cb.state || "–") : cb.model_term.toFixed(2);
  $("above").textContent = `${s.above_seconds.toFixed(1)} / ${s.alert_seconds} s`;
  const f = s.features || {};
  $("pose").textContent = f.yaw == null ? "–" : `${Math.round(f.yaw)}° / ${Math.round(f.pitch)}°`;
  $("fps").textContent = s.fps ? `${Math.round(s.fps)} fps` : "–";
  renderZ(s);
  pushChart(s);
}

function renderZ(s) {
  const names = { mouth_diff: "Mouth", eye_ratio: "Eye", brow_diff: "Brow" };
  const box = $("zbars");
  if (!s.z || !Object.keys(s.z).length) { box.innerHTML = '<span class="muted">Available once the baseline is learned.</span>'; return; }
  box.innerHTML = Object.entries(s.z).map(([k, z]) => {
    const pct = Math.min(Math.abs(z) / (s.threshold * 1.5), 1) * 50;
    const left = z >= 0 ? 50 : 50 - pct;
    const hot = Math.abs(z) >= s.threshold ? "hot" : "";
    return `<div class="zrow"><span>${names[k] || k}</span>
      <div class="ztrack"><div class="zfill ${hot}" style="left:${left}%;width:${pct}%"></div></div>
      <span class="zval">${z.toFixed(1)}</span></div>`;
  }).join("");
}

// ---------------------------------------------------------------- alert panel
function renderAlert(s) {
  const a = s.alert;
  $("alert-panel").hidden = !a;
  if (!a) { alertSig = ""; return; }
  const sig = JSON.stringify([a.id, a.status, a.findings, a.notes, a.localization, a.checklist, (s.patient || {}).lkw]);
  if (sig === alertSig) return;
  alertSig = sig;
  $("alert-pill").className = `pill ${a.status === "active" ? "alert" : "alert-soft"}`;
  $("alert-pill").textContent = a.status === "active" ? "Unacknowledged" : `Ack ${fmtTime(a.acked_at)}${a.acked_by ? ` · ${a.acked_by}` : ""}`;
  $("alert-findings").replaceChildren(...alertFindings(a).map((f) => el("li", null, stripTag(f))));
  $("alert-loc").hidden = !a.localization;
  $("alert-loc").textContent = locText(a.localization);
  $("alert-notes").replaceChildren(...(a.notes || []).map((n) => el("div", "alert-note", n)));
  $("alert-meta").innerHTML = `Onset ${a.onset_time ? fmtTime(a.onset_time) : "?"} · raised ${fmtTime(a.raised_at)} (<span data-ago="${a.raised_at}"></span>)`;
  $("alert-timers").replaceChildren(timersBlock(a, s.patient));
  $("alert-checklist").replaceChildren(checklistBlock(a));
  $("alert-ack").hidden = a.status !== "active";
  tickTimers($("alert-panel"));
}
$("alert-ack").onclick = () => st && st.alert && post(`/api/alerts/${st.alert.id}/ack`);
$("alert-false").onclick = () => st && st.alert && post(`/api/alerts/${st.alert.id}/resolve`, { resolution: "false_alarm" });
$("alert-confirm").onclick = () => st && st.alert && post(`/api/alerts/${st.alert.id}/resolve`, { resolution: "confirmed" });

// ---------------------------------------------------------------- BE-FAST check
const COMP_NAMES = { eyes: "Eyes", arms: "Arm drift" };
const ST_TEXT = { normal: "normal", abnormal: "abnormal", no_response: "no response",
                  inconclusive: "inconclusive", not_assessed: "not assessed", skipped: "skipped" };

function renderExam(s) {
  const ex = s.exam || { active: false };
  $("exam-live").hidden = !ex.active;
  $("exam-start").hidden = ex.active;
  $("hand-start").hidden = ex.active;
  $("exam-cancel").hidden = !ex.active;
  $("exam-start").disabled = $("hand-start").disabled = s.state === "offline";
  if (ex.active) {
    $("exam-idx").textContent = `Step ${ex.index}/${ex.count} · ${ex.phase}`;
    $("exam-label").textContent = ex.label;
    $("exam-say").textContent = `Spoken: "${ex.say}"`;
    $("exam-prog").style.width = `${ex.progress * 100}%`;
    $("exam-trigger").textContent = `Started ${fmtTimeS(ex.started_at)} · reason: ${ex.trigger}`;
  }
  const r = s.last_exam;
  const sig = JSON.stringify(r && [r.finished_at, r.summary]);
  if (sig !== examSig) {
    examSig = sig;
    const box = $("exam-result");
    const what = r && r.only === "arms" ? "Last hand test" : "Last check";
    if (!r) box.innerHTML = '<span class="muted">Starts by itself on a fixed gaze, or run it now.</span>';
    else if (r.cancelled) box.innerHTML = `<div class="exam-head">${what} ${fmtTime(r.finished_at)}: stopped (${esc(r.cancelled)})</div>`;
    else {
      const rows = Object.entries(r.components).map(([k, c]) =>
        `<div class="comp"><span>${COMP_NAMES[k] || k}</span><span class="st ${c.status}">${ST_TEXT[c.status] || c.status}</span><span>${esc(c.detail)}</span></div>`).join("");
      const head = r.flagged ? "flagged" : inconclusive(r) ? "inconclusive" : "";
      box.innerHTML = `<div class="exam-head ${head}">${what} ${fmtTime(r.finished_at)}: ${esc(r.summary)}</div>
        <div class="muted" style="font-size:12px;margin-bottom:4px">Reason: ${esc(r.trigger)}${r.localization ? ` · ${esc(locText(r.localization))}` : ""}</div>${rows}`;
    }
  }
  renderGaze(s);
}

function renderGaze(s) {
  const g = s.gaze, f = s.features || {};
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
  if (s.exam && s.exam.active) {
    $("gaze-text").textContent = now == null ? "check running" : `${Math.abs(now).toFixed(2)} ${now >= 0 ? "L" : "R"}`;
  } else if (g.deviation) {
    const held = g.since ? Math.round(s.t - g.since) : 0;
    $("gaze-text").textContent = `held ${g.deviation.toLowerCase()} ${held}s`;
  } else {
    $("gaze-text").textContent = now == null ? "eyes not seen" : `${Math.abs(now).toFixed(2)} ${now >= 0 ? "L" : "R"}`;
  }
}

function renderHistory(s) {
  const h = s.exam_history || [];
  const sig = JSON.stringify(h.map((r) => r.finished_at));
  if (sig === histSig) return;
  histSig = sig;
  const box = $("exam-history");
  box.replaceChildren();
  if (!h.length) { box.append(el("div", "log-empty", "No checks yet on this bed.")); return; }
  for (const r of h) {
    const row = el("div", "hist");
    row.append(el("span", "when", fmtTime(r.finished_at)));
    const right = el("div");
    right.append(el("div", `what${r.flagged ? " flagged" : ""}`, `${r.only === "arms" ? "Hand test: " : ""}${r.summary}`),
      el("div", "why", `Reason: ${r.trigger}`));
    row.append(right);
    box.append(row);
  }
}

function renderLog(s) {
  const evs = s.events || [];
  const sig = `${evs.length && evs[0].id}|${evs.length}`;
  if (sig === logSig) return;
  logSig = sig;
  const box = $("bed-log");
  box.replaceChildren();
  if (!evs.length) box.append(el("div", "log-empty", "No events yet."));
  for (const e of evs) box.append(eventRow(e, false));
}

$("exam-start").addEventListener("click", () => post(`${API}/exam`));
$("hand-start").addEventListener("click", () => post(`${API}/exam?only=arms`));
$("exam-cancel").addEventListener("click", () => post(`${API}/exam/cancel`));

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
  if (st && st.alert && !confirm("This closes the open alert and re-learns this patient's baseline. Continue?")) return;
  await post(`${API}/${action}`);
  for (const ds of chart.data.datasets) ds.data = [];
  t0 = null; lastState = null;
}
$("reset").addEventListener("click", () => rebuild("reset"));
$("calibrate").addEventListener("click", () => rebuild("calibrate"));

// Webcam / demo beds: check prompts play on this page, which needs one click first (browser rule).
function renderSound(on) {
  $("sound-state").textContent = on ? "Voice prompts play on this computer"
                                    : "Click anywhere on this page to turn on voice prompts";
  $("sound-state").className = on ? "muted" : "sound-off";
}
renderSound(Voice.unlocked);
Voice.onUnlockChange(renderSound);
$("sound-test").addEventListener("click", () => Voice.say("This is a voice test from Eye-CU."));
$("say-test").addEventListener("click", async () => {
  const r = await fetch(`${API}/say`, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: "This is a test from the Eye-CU dashboard." }) });
  if (!r.ok) alert("The phone is not connected.");
});
$("remove").addEventListener("click", async () => {
  const what = $("remove").textContent.replace("Remove ", "");
  if (!confirm(`Remove ${what} from the ward? Its baseline and alerts are dropped.`)) return;
  await fetch(API, { method: "DELETE" });
  location.href = "/";
});

setInterval(() => {
  const now = serverNow();
  $("clock").textContent = fmtTimeS(now);
  $("date").textContent = fmtDate(now);
  const bt = $("banner-time");
  if (bt.dataset.since) bt.textContent = `since onset ${fmtDur(now - +bt.dataset.since)}`;
  tickTimers(document);
}, 1000);

// Live video: JPEG frames arrive on the bed's WebSocket after each status message.
let frameUrl = null;
function showFrame(blob) {
  const img = $("video");
  const url = URL.createObjectURL(blob);
  const old = frameUrl;
  frameUrl = url;
  img.onload = () => { if (old) URL.revokeObjectURL(old); };
  img.src = url;
  img.hidden = false;
  $("video-wait").hidden = true;
}

// ---------------------------------------------------------------- websocket
// Landmark lines and dots on the video: on each time the page opens, toggled with the button
// (the video stream reconnects with the other setting).
let overlay = true, ws = null;
function renderOverlayButton() {
  $("overlay-toggle").textContent = overlay ? "Hide landmarks" : "Show landmarks";
  $("overlay-toggle").setAttribute("aria-pressed", String(overlay));
}
$("overlay-toggle").addEventListener("click", () => {
  overlay = !overlay;
  renderOverlayButton();
  if (ws) { ws.onclose = null; ws.close(); }
  connect();
});
renderOverlayButton();

function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}${API}/ws?overlay=${overlay ? 1 : 0}`);
  ws.binaryType = "blob";
  ws.onmessage = (e) => {
    if (typeof e.data !== "string") { showFrame(e.data); return; }
    const m = JSON.parse(e.data);
    // webcam bed: this page is the patient's speaker (an ESP32 bed has its own speaker)
    if (m.type === "say" && m.speaker !== "esp32") Voice.say(m.text);
    else render(m);
  };
  ws.onclose = (e) => {
    banner.className = "banner offline";
    if (e.code === 4404) {
      bannerText.textContent = "This bed no longer exists (removed or server restarted).";
      return;
    }
    bannerText.textContent = "Connection to monitoring server lost: retrying…";
    setTimeout(connect, 1000);
  };
}
connect();
