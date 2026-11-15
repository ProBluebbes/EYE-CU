// Ward dashboard: one WebSocket carries JSON state (text frames) and thumbnails (binary).
const $ = (id) => document.getElementById(id);
const tiles = new Map();     // bed_id -> tile parts
const history = new Map();   // bed_id -> [{t, s}] change score over the last 10 min (sparklines)
const HIST_S = 600, HIST_STEP_S = 2;
let state = null;
let filter = "all", query = "", sortBy = "bed", logFilter = "all";
let alertSig = "", closedSig = "", logSig = "", orderSig = "";

$("topbar").prepend(brandBar());
$("unit-name").textContent = UNIT;
bindStaffInput($("staff"));

const bedNum = (b) => +(b.bed_id.match(/\d+/) || [0])[0];
const isExam = (b) => !!(b.exam && b.exam.active);
const isAcked = (b) => b.alert && b.alert.status === "acknowledged";
function priority(b) {
  if (b.state === "ALERT") return isAcked(b) ? 1 : 0;
  if (isExam(b)) return 2;
  if (b.state === "offline") return 3;
  if (b.state === "baselining" || b.state === "starting") return 4;
  return 5;
}
const FILTERS = {
  all: () => true,
  alert: (b) => b.state === "ALERT",
  check: isExam,
  calib: (b) => b.state === "baselining" || b.state === "starting",
  offline: (b) => b.state === "offline",
};
function patientText(b) {
  const p = b.patient || {};
  return [p.name, p.mrn && `MRN ${p.mrn}`, [p.age, p.sex].filter(Boolean).join(" ")].filter(Boolean).join(" · ");
}

// ---------------------------------------------------------------- tiles
function makeTile(bed) {
  const a = el("a", "tile");
  a.href = `/bed/${encodeURIComponent(bed.bed_id)}`;
  const head = el("div", "tile-head");
  const name = el("span", "tile-name");
  const kind = el("span", "kind-badge");
  const pill = el("span", "pill");
  head.append(name, kind, pill);
  const patient = el("div", "tile-patient");
  const strip = el("div", "alarm-strip");
  const stripText = el("span"), stripT = el("span", "t");
  strip.append(stripText, stripT);
  const imgWrap = el("div", "tile-img");
  const img = el("img");
  img.alt = `${bed.bed_name} camera`;
  const overlay = el("div", "tile-overlay");
  imgWrap.append(img, overlay);

  const body = el("div", "tile-body");
  const status = el("div", "tile-status");
  const metric = el("div", "metric");
  const bar = el("div", "scorebar");
  const fill = el("div", "fill"), pre = el("div", "mark pre"), thr = el("div", "mark thr");
  bar.append(fill, pre, thr);
  const mval = el("span", "metric-val");
  metric.append(el("span", "metric-label", "Droop score"), bar, mval);
  const spark = el("canvas", "spark");
  spark.title = "Facial change score, last 10 minutes (red line = alert threshold)";
  const foot = el("div", "tile-foot");
  body.append(status, metric, spark, foot);

  a.append(head, patient, strip, imgWrap, body);
  $("grid").append(a);
  const t = { el: a, img, name, kind, pill, patient, stripText, stripT, overlay, status, bar, fill, pre, thr, mval, spark, foot, url: null };
  tiles.set(bed.bed_id, t);
  return t;
}

const SCALE = 1.5; // score bar spans 0 .. 1.5 x threshold

function renderTile(bed) {
  const t = tiles.get(bed.bed_id) || makeTile(bed);
  const exam = isExam(bed), acked = isAcked(bed);
  const st = exam && bed.state !== "ALERT" ? "exam" : bed.state;
  t.el.className = `tile st-${st}${acked ? " acked" : ""}`;
  t.name.textContent = bed.bed_name;
  t.kind.className = `kind-badge ${bed.kind}`;
  t.kind.textContent = KIND_BADGE[bed.kind] || "";
  t.kind.title = bed.demo_clip ? `Demo video: ${bed.demo_clip}` : "";

  const [ptxt, pcls] = bed.state === "ALERT" ? (acked ? ["Alert · acknowledged", "alert-soft"] : ["Stroke alert", "alert"])
    : exam ? ["BE-FAST check", "check"]
    : { monitoring: ["Monitoring", "ok"], baselining: ["Calibrating", "info"], starting: ["Starting", ""],
        offline: ["Offline", "warn"] }[bed.state] || [bed.state, ""];
  t.pill.className = `pill ${pcls}`;
  t.pill.textContent = ptxt;

  const pt = patientText(bed);
  t.patient.innerHTML = pt ? `<b>${esc(pt)}</b>${bed.patient.dx ? `<div class="muted">${esc(bed.patient.dx)}</div>` : ""}`
    : `<span class="muted">${esc(bed.label || (bed.demo_clip && `Demo video: ${bed.demo_clip}`) || "No patient details entered")}</span>`;

  if (bed.state === "ALERT" && bed.alert) {
    const side = bed.alert.side || bed.side;
    t.stripText.textContent = `STROKE ALERT${side ? ` · ${side} side` : ""}${acked ? ` · ack ${bed.alert.acked_by || ""}` : ""}`;
    t.stripT.dataset.since = bed.alert.onset_time || bed.alert.raised_at;
  } else {
    delete t.stripT.dataset.since;
  }

  t.overlay.hidden = bed.state !== "offline";
  if (bed.state === "offline") {
    const what = bed.kind === "phone" && !bed.phone_connected ? "Phone disconnected" : "Camera offline";
    t.overlay.textContent = bed.last_frame_age == null ? `${what} · no video yet`
      : `${what} · last frame ${ago(serverNow() - bed.last_frame_age)}`;
  }

  if (exam) t.status.textContent = `Check step ${bed.exam.index}/${bed.exam.count}: ${bed.exam.label}`;
  else if (bed.state === "ALERT" && bed.alert) {
    const h = alertHeadline(bed.alert);
    t.status.textContent = h.charAt(0).toUpperCase() + h.slice(1);
  } else if (bed.state === "ALERT") t.status.textContent = `Possible new ${bed.side} facial droop`;
  else if (bed.state === "baselining") t.status.textContent = `Learning this patient's baseline · ${Math.round(bed.baseline_progress * 100)}%`;
  else if (bed.state === "monitoring") t.status.textContent = "No new facial asymmetry";
  else if (bed.state === "offline") t.status.textContent = "Not monitoring: camera offline";
  else t.status.textContent = "Starting camera…";

  // score bar (baseline progress while calibrating)
  const thrPct = 100 / SCALE;
  t.thr.style.left = `${thrPct}%`;
  t.pre.style.left = `${(bed.pre_threshold / bed.threshold) * thrPct}%`;
  let pct, level;
  if (bed.state === "baselining" || bed.state === "starting") {
    pct = bed.baseline_progress * 100; level = "progress";
    t.mval.textContent = `${Math.round(pct)}%`;
  } else {
    pct = Math.min(bed.score / (bed.threshold * SCALE), 1) * 100;
    level = bed.score >= bed.threshold ? "hot" : bed.score >= bed.pre_threshold ? "warm" : "ok";
    t.mval.textContent = `${bed.score.toFixed(1)} / ${bed.threshold}`;
  }
  t.fill.style.width = `${pct}%`;
  t.fill.className = `fill ${level}`;
  t.bar.classList.toggle("dim", bed.state === "offline");

  // footer facts
  const parts = [];
  if (bed.state !== "offline" && bed.state !== "starting") {
    parts.push(!bed.valid && bed.invalid_reason ? [`Paused: ${bed.invalid_reason}`, "warn"] : ["Face tracked", "ok"]);
    if (bed.gaze && bed.gaze.deviation) parts.push([`Gaze held ${bed.gaze.deviation.toLowerCase()}`, "bad"]);
    const le = bed.last_exam;
    if (!exam && le && !le.cancelled) {
      const when = `Last ${le.only === "arms" ? "hand test" : "check"} ${fmtTime(le.finished_at)}`;
      if (le.flagged) parts.push([`${when}: abnormal`, "bad"]);
      if (inconclusive(le)) parts.push([`${when}: ${inconclusiveText(le)}`, "warn"]);
      if (!le.flagged && !inconclusive(le)) parts.push([`${when}: normal`, ""]);
    }
  }
  const le = bed.last_exam;
  t.foot.title = !exam && inconclusive(le) ? (le.notes || []).join("\n") : "";
  if (bed.fps && bed.state !== "offline") parts.push([`${Math.round(bed.fps)} fps`, ""]);
  t.foot.innerHTML = parts.map(([s, c]) => `<span class="${c}">${esc(s)}</span>`).join("");

  recordHistory(bed);
  drawSpark(t.spark, history.get(bed.bed_id) || [], bed);
}

function recordHistory(bed) {
  const h = history.get(bed.bed_id) || [];
  history.set(bed.bed_id, h);
  const now = serverNow();
  if (h.length && now - h[h.length - 1].t < HIST_STEP_S) return;
  const live = bed.valid && (bed.state === "monitoring" || bed.state === "ALERT");
  h.push({ t: now, s: live ? bed.score : null });
  while (h.length && h[0].t < now - HIST_S) h.shift();
}

function drawSpark(cv, h, bed) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, hgt = cv.clientHeight;
  if (!w) return;
  if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(hgt * dpr); }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, hgt);
  const max = bed.threshold * SCALE, now = serverNow();
  const x = (t) => w - ((now - t) / HIST_S) * w, y = (s) => hgt - 2 - (Math.min(s, max) / max) * (hgt - 4);
  g.fillStyle = "#f6f9fc"; g.fillRect(0, 0, w, hgt);
  g.strokeStyle = "#c4262e"; g.setLineDash([4, 3]); g.lineWidth = 1;
  g.beginPath(); g.moveTo(0, y(bed.threshold)); g.lineTo(w, y(bed.threshold)); g.stroke();
  g.setLineDash([]);
  g.strokeStyle = "#1068c4"; g.lineWidth = 1.5; g.beginPath();
  let pen = false;
  for (const p of h) {
    if (p.s == null) { pen = false; continue; }
    if (pen) g.lineTo(x(p.t), y(p.s)); else { g.moveTo(x(p.t), y(p.s)); pen = true; }
  }
  g.stroke();
  g.fillStyle = "#8da0b3"; g.font = "10px sans-serif";
  g.fillText("10 min", 3, 10);
}

function onThumb(buf) {
  const u8 = new Uint8Array(buf);
  const n = u8[0];
  const id = new TextDecoder().decode(u8.subarray(1, 1 + n));
  const t = tiles.get(id);
  if (!t) return;
  const url = URL.createObjectURL(new Blob([u8.subarray(1 + n)], { type: "image/jpeg" }));
  const old = t.url;
  t.url = url;
  t.img.onload = () => { if (old) URL.revokeObjectURL(old); };
  t.img.src = url;
}

function layoutGrid(beds) {
  const q = query.toLowerCase();
  const shown = beds.filter((b) => FILTERS[filter](b) &&
    (!q || `${b.bed_name} ${patientText(b)} ${(b.patient || {}).dx || ""} ${b.label || ""}`.toLowerCase().includes(q)));
  const ids = new Set(shown.map((b) => b.bed_id));
  for (const [id, t] of tiles) t.el.hidden = !ids.has(id);
  const sorted = [...beds].sort((a, b) => (sortBy === "priority" ? priority(a) - priority(b) : 0) || bedNum(a) - bedNum(b));
  const sig = sorted.map((b) => b.bed_id).join();
  if (sig !== orderSig) {
    orderSig = sig;
    for (const b of sorted) $("grid").append(tiles.get(b.bed_id).el);
  }
  $("empty").hidden = shown.length > 0;
  for (const btn of $("filters").children) {
    const n = beds.filter(FILTERS[btn.dataset.f]).length;
    const label = btn.dataset.label || (btn.dataset.label = btn.textContent);
    if (btn.dataset.n !== String(n)) {
      btn.dataset.n = n;
      btn.innerHTML = `${esc(label)}<span class="n">${n}</span>`;
    }
  }
}
for (const btn of $("filters").children) {
  btn.onclick = () => {
    filter = btn.dataset.f;
    for (const b of $("filters").children) b.classList.toggle("on", b === btn);
    if (state) layoutGrid(state.beds);
  };
}
$("search").oninput = () => { query = $("search").value.trim(); if (state) layoutGrid(state.beds); };
$("sort").onchange = () => { sortBy = $("sort").value; orderSig = ""; if (state) layoutGrid(state.beds); };

// ---------------------------------------------------------------- KPIs
function renderKpis(beds, alerts, stats) {
  const n = (f) => beds.filter(f).length;
  const watching = n((b) => b.state === "monitoring" || b.state === "ALERT");
  const calib = n(FILTERS.calib), off = n(FILTERS.offline), checks = n(isExam);
  const open = alerts.filter((a) => a.status !== "resolved");
  const unacked = open.filter((a) => a.status === "active").length;
  const ack = stats && stats.median_ack_s != null ? fmtDur(stats.median_ack_s) : "–";
  const kpi = (cls, label, val, sub) =>
    `<div class="kpi ${cls}"><div class="kpi-label">${label}</div><div class="kpi-val">${val}</div><div class="kpi-sub">${sub}</div></div>`;
  $("kpis").innerHTML = [
    kpi("ok", "Beds monitored", `${watching}<span class="muted" style="font-size:15px"> / ${beds.length}</span>`, calib ? `${calib} calibrating` : "all calibrated"),
    kpi(`alert${open.length ? " hot" : ""}`, "Open stroke alerts", open.length, unacked ? `${unacked} awaiting acknowledgement` : "none unacknowledged"),
    kpi("", "BE-FAST checks", checks, `running now · ${stats ? stats.checks_24h : 0} done in 24 h`),
    kpi(`warn${off ? " hot" : ""}`, "Cameras offline", off, off ? "not monitoring" : "all cameras online"),
    kpi("neutral", "Median time to ack", ack, `${stats ? stats.alerts_24h : 0} alert${stats && stats.alerts_24h === 1 ? "" : "s"} in 24 h`),
  ].join("");
}

// ---------------------------------------------------------------- alerts
const RES_LABEL = { false_alarm: "False alarm", confirmed: "Confirmed, escalated", reset: "Bed reset", removed: "Bed removed" };

function renderAlerts(alerts, beds) {
  const byId = Object.fromEntries(beds.map((b) => [b.bed_id, b]));
  const open = alerts.filter((a) => a.status !== "resolved")
    .sort((a, b) => (a.status === "active" ? 0 : 1) - (b.status === "active" ? 0 : 1) || b.raised_at - a.raised_at);
  const active = open.filter((a) => a.status === "active").length;
  $("alert-count").textContent = open.length ? `${open.length} open${active ? ` · ${active} unacknowledged` : ""}` : "";

  const sig = JSON.stringify(open.map((a) => {
    const b = byId[a.bed_id] || {};
    return [a.id, a.status, a.findings.length, a.notes, a.localization && a.localization.text, Object.keys(a.checklist || {}).join(), (b.patient || {}).lkw, patientText(b), isExam(b)];
  }));
  if (sig !== alertSig) {
    alertSig = sig;
    const list = $("alert-list");
    list.replaceChildren();
    if (!open.length) list.append(el("div", "none ok", "✓ No open stroke alerts on this unit."));
    for (const a of open) list.append(alertCard(a, byId[a.bed_id]));
  }

  const closed = alerts.filter((a) => a.status === "resolved").slice(0, 8);
  const csig = closed.map((a) => a.id).join();
  if (csig !== closedSig) {
    closedSig = csig;
    const list = $("closed-list");
    list.replaceChildren();
    if (!closed.length) list.append(el("div", "none", "No closed alerts yet."));
    for (const a of closed) {
      const row = el("div", "closed-item");
      const left = el("div");
      const link = el("a", null, a.bed_name);
      link.href = `/bed/${encodeURIComponent(a.bed_id)}`;
      left.append(link, ` · ${stripTag(alertFindings(a)[0])}`);
      row.append(left, el("span", "muted", `${RES_LABEL[a.resolution] || a.resolution} ${fmtTime(a.resolved_at)}${a.resolved_by ? ` · ${a.resolved_by}` : ""}`));
      list.append(row);
    }
  }
}

function alertCard(a, bed) {
  const box = el("div", `alert-card ${a.status}`);
  const top = el("div", "alert-top");
  const title = el("a", "alert-title", `${a.bed_name} · ${a.side ? `${a.side}-sided ` : ""}stroke alert`);
  title.href = `/bed/${encodeURIComponent(a.bed_id)}`;
  const pill = a.status === "active" ? el("span", "pill alert", "Unacknowledged")
    : el("span", "pill alert-soft", `Ack ${fmtTime(a.acked_at)}${a.acked_by ? ` · ${a.acked_by}` : ""}`);
  top.append(title, pill);
  box.append(top);
  const pt = bed ? patientText(bed) : "";
  if (pt || (bed && bed.label)) box.append(el("div", "alert-pt", pt || bed.label));
  const ul = el("ul", "findings");
  for (const f of alertFindings(a)) ul.append(el("li", null, stripTag(f)));
  box.append(ul);
  if (a.localization) box.append(el("div", "alert-loc", locText(a.localization)));
  for (const n of a.notes || []) box.append(el("div", "alert-note", n));
  const meta = el("div", "alert-meta");
  meta.append(`Onset ${a.onset_time ? fmtTime(a.onset_time) : "?"} · raised ${fmtTime(a.raised_at)} (`);
  const agoSpan = el("span", null, ago(a.raised_at));
  agoSpan.dataset.ago = a.raised_at;
  meta.append(agoSpan, ")");
  box.append(meta, timersBlock(a, bed && bed.patient), el("div", "sec-label", "Code stroke checklist"), checklistBlock(a));

  const btns = el("div", "alert-btns");
  if (a.status === "active") btns.append(button("Acknowledge", "btn btn-primary", () => post(`/api/alerts/${a.id}/ack`)));
  if (bed && !isExam(bed)) btns.append(button("Run BE-FAST", "btn", () => post(`/api/beds/${encodeURIComponent(a.bed_id)}/exam`),
    "Start the voice-guided eyes / arms / legs check"));
  btns.append(
    button("False alarm", "btn", () => post(`/api/alerts/${a.id}/resolve`, { resolution: "false_alarm" }),
      "Close the alert and rebuild this bed's baseline"),
    button("Confirm stroke", "btn btn-outline-danger", () => post(`/api/alerts/${a.id}/resolve`, { resolution: "confirmed" }),
      "Mark as confirmed and escalated, close the alert and rebuild the baseline"));
  box.append(btns);
  return box;
}

function button(label, cls, onClick, title) {
  const b = el("button", cls, label);
  if (title) b.title = title;
  b.onclick = async (e) => {
    e.preventDefault();
    b.disabled = true;
    try { await onClick(); } finally { setTimeout(() => { b.disabled = false; }, 1500); }
  };
  return b;
}

// ---------------------------------------------------------------- activity log
const LOG_KINDS = { alert: ["alert", "ack", "resolve", "protocol", "reset", "patient"], exam: ["exam"] };
function renderLog(events) {
  const evs = logFilter === "all" ? events : events.filter((e) => LOG_KINDS[logFilter].includes(e.kind));
  const sig = `${logFilter}|${evs.length && evs[0].id}|${evs.length}`;
  if (sig === logSig) return;
  logSig = sig;
  const box = $("log");
  box.replaceChildren();
  if (!evs.length) box.append(el("div", "log-empty", "No events yet."));
  for (const e of evs.slice(0, 50)) box.append(eventRow(e, true));
}
$("log-filter").onchange = () => { logFilter = $("log-filter").value; logSig = ""; if (state) renderLog(state.events); };

// ---------------------------------------------------------------- sound
let audio = null, silencedUntil = 0;
const unacked = () => state && state.alerts.some((a) => a.status === "active");
$("sound").onclick = () => {
  if (audio) { audio.close(); audio = null; } else { audio = new AudioContext(); beep(); }
  renderSound();
};
$("silence").onclick = () => { silencedUntil = Date.now() + 120000; renderSound(); };
function renderSound() {
  const b = $("sound");
  b.className = `btn sound ${audio ? "on" : "off"}${!audio && unacked() ? " nag" : ""}`;
  b.textContent = audio ? "Alarm sound on" : "Enable alarm sound";
  const s = $("silence");
  s.hidden = !(audio && unacked());
  const left = (silencedUntil - Date.now()) / 1000;
  s.textContent = left > 0 ? `Silenced ${fmtDur(left)}` : "Silence 2 min";
}
function beep() {
  if (!audio) return;
  [0, 0.3, 0.6].forEach((d) => {
    const o = audio.createOscillator(), g = audio.createGain();
    o.frequency.value = 880; o.connect(g); g.connect(audio.destination);
    g.gain.setValueAtTime(0.25, audio.currentTime + d);
    o.start(audio.currentTime + d); o.stop(audio.currentTime + d + 0.18);
  });
}
setInterval(() => { if (unacked() && Date.now() > silencedUntil) beep(); }, 4000);

setInterval(() => {
  const now = serverNow();
  $("clock").textContent = fmtTimeS(now);
  $("date").textContent = fmtDate(now);
  tickTimers(document);
  if (state) renderSound();
}, 1000);

// ---------------------------------------------------------------- add camera (QR codes)
let modalSig = "";
$("add-phone").onclick = () => { $("modal").hidden = false; modalSig = ""; renderModal(); };
$("modal-close").onclick = () => { $("modal").hidden = true; };
if (location.hash === "#add-phone") $("modal").hidden = false;
$("modal").onclick = (e) => { if (e.target === $("modal")) $("modal").hidden = true; };

async function renderModal() {
  if ($("modal").hidden || !state) return;
  const tun = state.tunnel || { status: "off" };
  const sig = `${tun.status}|${tun.url}|${tun.error}`;
  if (sig === modalSig) return;
  modalSig = sig;
  const body = $("modal-body");
  body.replaceChildren();
  if (tun.status === "off") {
    body.append(el("p", null, "Phone cameras connect through a secure public link (HTTPS), which isn't running."),
      el("p", null, "Restart the server with --tunnel:"),
      el("code", "cmd", ".venv/bin/python -m strokewatch.server --camera 0 --tunnel"));
    return;
  }
  if (tun.status === "error") {
    body.append(el("p", "err", `Secure link failed: ${tun.error}`),
      el("p", "muted", "Check the server terminal. On restrictive networks, try --set tunnel.protocol=http2."));
    return;
  }
  if (tun.status !== "ready") {
    body.append(el("p", null, "Creating a secure link… this takes about 10–20 seconds."));
    return;
  }
  const r = await fetch("/api/connect");
  const { links } = await r.json();
  const qr = (target, title, hint, link) => {
    const box = el("div", "qr");
    const img = el("img");
    img.src = `/api/qr.svg?target=${target}&u=${encodeURIComponent(tun.url)}`;
    img.alt = `QR code: ${title}`;
    const a = el("a", "qr-link", "Copy link");
    a.href = link;
    a.onclick = (e) => { e.preventDefault(); navigator.clipboard?.writeText(link); a.textContent = "Copied"; };
    box.append(el("h3", null, title), img, el("p", "muted", hint), a);
    return box;
  };
  const row = el("div", "qr-row");
  row.append(
    qr("phone", "Phone as bedside camera", "Scan with the iPhone camera, open in Safari, tap Start camera. Each phone becomes a new bed.", links.phone),
    qr("dashboard", "Ward view on another device", "Opens this dashboard (view and acknowledge alerts).", links.dashboard));
  body.append(row,
    el("p", "muted", "A brand-new link can take up to a minute to work everywhere. If Safari can't find the server, wait 30 s and reload."),
    el("p", "muted", "Links carry a secret key and change every time the server restarts. Anyone with them can see the cameras: share only with the care team."));
}

// ---------------------------------------------------------------- add demo (recorded video)
$("add-demo").onclick = async () => {
  $("demo-modal").hidden = false;
  $("demo-msg").textContent = "";
  const sel = $("demo-select");
  sel.replaceChildren();
  try {
    const r = await fetch("/api/demos");
    const { demos, folder } = await r.json();
    for (const name of demos) {
      const o = el("option", null, name);
      o.value = name;
      sel.append(o);
    }
    $("demo-add").disabled = !demos.length;
    if (!demos.length) $("demo-msg").textContent = `No demo videos found in ${folder}/.`;
  } catch {
    $("demo-msg").textContent = "Could not load the demo list.";
  }
};
$("demo-close").onclick = () => { $("demo-modal").hidden = true; };
$("demo-modal").onclick = (e) => { if (e.target === $("demo-modal")) $("demo-modal").hidden = true; };
$("demo-add").onclick = async () => {
  const b = $("demo-add");
  b.disabled = true;
  $("demo-msg").textContent = "Adding…";
  try {
    const r = await fetch("/api/demos", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: $("demo-select").value }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    $("demo-msg").textContent = "";
    $("demo-modal").hidden = true;
  } catch (e) {
    $("demo-msg").textContent = `Could not add the demo: ${e.message}`;
  } finally {
    b.disabled = false;
  }
};

// ---------------------------------------------------------------- websocket
function render() {
  setServerTime(state.server_time);
  for (const bed of state.beds) renderTile(bed);
  const ids = new Set(state.beds.map((b) => b.bed_id));
  for (const [id, t] of tiles) {
    if (!ids.has(id)) { t.el.remove(); if (t.url) URL.revokeObjectURL(t.url); tiles.delete(id); history.delete(id); orderSig = ""; }
  }
  layoutGrid(state.beds);
  renderKpis(state.beds, state.alerts, state.stats);
  renderAlerts(state.alerts, state.beds);
  renderLog(state.events || []);
  renderModal();
  renderSound();
  tickTimers(document);
  const n = state.alerts.filter((a) => a.status === "active").length;
  document.title = `${n ? `(${n}) ` : ""}Azure Medical Center · Eye-CU`;
}

function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/dashboard/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { $("conn").hidden = true; };
  ws.onmessage = (e) => {
    if (typeof e.data === "string") { state = JSON.parse(e.data); render(); }
    else onThumb(e.data);
  };
  ws.onclose = () => { $("conn").hidden = false; setTimeout(connect, 1000); };
}
connect();
