// Ward dashboard: one WebSocket carries JSON state (text frames) and thumbnails (binary).
const $ = (id) => document.getElementById(id);
const tiles = new Map();      // bed_id -> {el, img, ...parts}
let state = null;             // last snapshot
let alertSig = "";            // re-render the alert list only when it actually changes
let clockSkew = 0;            // server_time - local time (s), for "x min ago" labels

const fmtTime = (epoch) => new Date(epoch * 1000).toLocaleTimeString([], { hour12: false });
const serverNow = () => Date.now() / 1000 + clockSkew;
function ago(epoch) {
  const s = Math.max(0, serverNow() - epoch);
  if (s < 60) return `${Math.round(s)} s ago`;
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  return `${Math.floor(s / 3600)} h ${Math.floor((s % 3600) / 60)} min ago`;
}
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

// ---------------------------------------------------------------- tiles
function makeTile(bed) {
  const a = el("a", "tile");
  a.href = `/bed/${encodeURIComponent(bed.bed_id)}`;
  const head = el("div", "tile-head");
  const name = el("span", "tile-name", bed.bed_name);
  const kind = el("span", `kind-badge ${bed.kind}`, { sim: "SIM", phone: "PHONE" }[bed.kind] || "");
  kind.hidden = !kind.textContent;
  const pill = el("span", "pill");
  head.append(name, kind, pill);

  const imgWrap = el("div", "tile-img");
  const img = el("img");
  img.alt = `${bed.bed_name} camera`;
  const overlay = el("div", "tile-overlay");
  imgWrap.append(img, overlay);

  const body = el("div", "tile-body");
  const status = el("div", "tile-status");
  const bar = el("div", "scorebar");
  const fill = el("div", "fill"), pre = el("div", "mark pre"), thr = el("div", "mark thr");
  bar.append(fill, pre, thr);
  const sub = el("div", "tile-sub");
  body.append(status, bar, sub);

  a.append(head, imgWrap, body);
  $("grid").append(a);
  const t = { el: a, img, pill, overlay, status, bar, fill, pre, thr, sub, name, url: null };
  tiles.set(bed.bed_id, t);
  return t;
}

const PILL = { starting: "Starting", baselining: "Baseline", monitoring: "Monitoring",
               ALERT: "ALERT", offline: "Offline" };
const stripTag = (f) => f.replace(/^Face: /, "");
// One line for an alert: "possible new LEFT facial droop" or "Arm drift: left arm … (+1 more)".
function alertHeadline(a) {
  const f = a.findings && a.findings.length ? a.findings : [`Face: possible new ${a.side} facial droop`];
  return stripTag(f[0]) + (f.length > 1 ? ` (+${f.length - 1} more)` : "");
}
// A BE-FAST check part the patient didn't follow (eyes wandered, looked the wrong way, ...).
const inconclusive = (le) => !!(le && !le.cancelled && le.inconclusive && le.inconclusive.length);
const PART_NAMES = { eyes: "eyes", arms: "arms", legs: "legs" };
const inconclusiveText = (le) =>
  `${le.inconclusive.map((k) => PART_NAMES[k] || k).join(" + ")} inconclusive (instructions not followed)`;
const SCALE = 1.5; // score bar spans 0 .. 1.5 x threshold

function renderTile(bed) {
  const t = tiles.get(bed.bed_id) || makeTile(bed);
  t.name.textContent = bed.bed_name;
  const acked = bed.alert && bed.alert.status === "acknowledged";
  const exam = bed.exam && bed.exam.active;
  t.el.className = `tile state-${bed.state}${acked ? " acked" : ""}${exam ? " exam" : ""}`;
  t.pill.textContent = acked ? "ALERT (ack)" : exam && bed.state !== "ALERT" ? "BE-FAST check"
    : PILL[bed.state] || bed.state;

  t.overlay.hidden = bed.state !== "offline";
  if (bed.state === "offline") {
    const what = bed.kind === "phone" && !bed.phone_connected ? "Phone disconnected" : "Camera offline";
    t.overlay.textContent = bed.last_frame_age == null ? `${what} · no frames yet`
      : `${what} · last frame ${ago(serverNow() - bed.last_frame_age)}`;
  }

  // status line
  if (exam) {
    t.status.textContent = `Check ${bed.exam.index}/${bed.exam.count}: ${bed.exam.label}`;
  } else if (bed.state === "ALERT" && bed.alert) {
    const onset = bed.alert.onset_time ? fmtTime(bed.alert.onset_time) : "?";
    const h = alertHeadline(bed.alert);
    t.status.textContent = `${h.charAt(0).toUpperCase()}${h.slice(1)} · onset ${onset}`;
  } else if (bed.state === "ALERT") {
    const onset = bed.onset_time ? fmtTime(bed.onset_time) : "?";
    t.status.textContent = `Possible new ${bed.side} facial droop · onset ${onset}`;
  } else if (bed.state === "baselining") {
    t.status.textContent = `Building baseline ${Math.round(bed.baseline_progress * 100)}%`;
  } else if (bed.state === "monitoring") {
    t.status.textContent = "Monitoring: no new droop";
  } else if (bed.state === "offline") {
    t.status.textContent = "Not monitoring: camera offline";
  } else {
    t.status.textContent = "Starting…";
  }

  // score bar (baseline progress while baselining)
  const thrPct = 100 / SCALE, prePct = (bed.pre_threshold / bed.threshold) * thrPct;
  t.thr.style.left = `${thrPct}%`;
  t.pre.style.left = `${prePct}%`;
  let pct, level;
  if (bed.state === "baselining" || bed.state === "starting") {
    pct = bed.baseline_progress * 100; level = "progress";
  } else {
    pct = Math.min(bed.score / (bed.threshold * SCALE), 1) * 100;
    level = bed.score >= bed.threshold ? "hot" : bed.score >= bed.pre_threshold ? "warm" : "ok";
  }
  t.fill.style.width = `${pct}%`;
  t.fill.className = `fill ${level}`;
  t.bar.classList.toggle("dim", bed.state === "offline");

  // sub line: score + why frames are being skipped
  const parts = [];
  if (bed.state === "monitoring" || bed.state === "ALERT") parts.push(`score ${bed.score.toFixed(1)} / ${bed.threshold}`);
  if (bed.state !== "offline" && bed.state !== "starting") {
    if (!bed.valid && bed.invalid_reason) parts.push(`paused: ${bed.invalid_reason}`);
    else parts.push("tracking face");
    if (bed.gaze && bed.gaze.deviation) parts.push(`gaze held ${bed.gaze.deviation}`);
    const le = bed.last_exam;
    if (!exam && le && !le.cancelled && serverNow() - le.finished_at < 1800) {
      const when = `check ${fmtTime(le.finished_at)}`;
      if (le.flagged) parts.push(`${when}: abnormal`);
      if (inconclusive(le)) parts.push(`${when}: ${inconclusiveText(le)}`);
      if (!le.flagged && !inconclusive(le)) parts.push(`${when}: normal`);
    }
  }
  if (bed.fps && bed.state !== "offline") parts.push(`${bed.fps} fps`);
  t.sub.textContent = parts.join(" · ");
  const le = bed.last_exam, recentInc = !exam && inconclusive(le) && serverNow() - le.finished_at < 1800;
  t.sub.title = recentInc ? (le.notes || []).join("\n") : "";
  t.sub.classList.toggle("warn", (!bed.valid || recentInc) && bed.state !== "offline");
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

// ---------------------------------------------------------------- alert center
const RES_LABEL = { false_alarm: "false alarm", confirmed: "confirmed, escalated", reset: "bed reset" };

function renderAlerts(alerts) {
  const open = alerts.filter((a) => a.status !== "resolved");
  const active = alerts.filter((a) => a.status === "active").length;
  $("alert-count").textContent = open.length ? `${open.length} open${active ? `, ${active} unacknowledged` : ""}` : "";

  const sig = JSON.stringify(alerts.map((a) => [a.id, a.status, (a.findings || []).length, a.notes]));
  if (sig !== alertSig) {
    alertSig = sig;
    const list = $("alert-list");
    list.replaceChildren();
    if (!alerts.length) list.append(el("p", "muted", "No alerts."));
    const order = { active: 0, acknowledged: 1, resolved: 2 };
    const sorted = [...alerts].sort((a, b) => order[a.status] - order[b.status] || b.raised_at - a.raised_at);
    for (const a of sorted.slice(0, 40)) list.append(alertItem(a));
  }
  for (const s of document.querySelectorAll("[data-ago]")) s.textContent = ago(+s.dataset.ago);
}

function alertItem(a) {
  const box = el("div", `alert-item ${a.status}`);
  const title = el("a", "alert-title");
  title.href = `/bed/${encodeURIComponent(a.bed_id)}`;
  title.textContent = `${a.bed_name}: ${alertHeadline(a)}`;
  const meta = el("div", "alert-meta");
  meta.append(`onset ${a.onset_time ? fmtTime(a.onset_time) : "?"} · raised ${fmtTime(a.raised_at)} (`);
  const agoSpan = el("span", null, ago(a.raised_at));
  agoSpan.dataset.ago = a.raised_at;
  meta.append(agoSpan, ")");
  box.append(title, meta);
  if (a.findings && a.findings.length > 1) {
    const ul = el("ul", "alert-findings");
    for (const f of a.findings) ul.append(el("li", null, stripTag(f)));
    box.append(ul);
  }
  if (a.localization) box.append(el("div", "alert-meta", `${a.localization.text} ${a.localization.caveat}`));
  for (const n of a.notes || []) box.append(el("div", "alert-note", n));

  if (a.status === "acknowledged") box.append(el("div", "alert-meta", `acknowledged ${fmtTime(a.acked_at)}`));
  if (a.status === "resolved") {
    box.append(el("div", "alert-meta", `resolved ${fmtTime(a.resolved_at)}: ${RES_LABEL[a.resolution] || a.resolution}`));
    return box;
  }
  const btns = el("div", "alert-btns");
  if (a.status === "active") btns.append(button("Acknowledge", "ack", () => post(`/api/alerts/${a.id}/ack`)));
  btns.append(
    button("False alarm", "", () => post(`/api/alerts/${a.id}/resolve`, { resolution: "false_alarm" }),
      "Close the alert and rebuild this bed's baseline"),
    button("Confirmed", "danger", () => post(`/api/alerts/${a.id}/resolve`, { resolution: "confirmed" }),
      "Mark as escalated, close the alert and rebuild this bed's baseline"));
  box.append(btns);
  return box;
}

function button(label, cls, onClick, title) {
  const b = el("button", cls, label);
  if (title) b.title = title;
  b.onclick = async () => {
    b.disabled = true;
    try { await onClick(); } finally { setTimeout(() => { b.disabled = false; }, 1500); }
  };
  return b;
}

async function post(url, body) {
  await fetch(url, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
}

// ---------------------------------------------------------------- header, sound
function renderHeader(beds) {
  const n = (s) => beds.filter((b) => b.state === s).length;
  const parts = [`${beds.length} bed${beds.length === 1 ? "" : "s"}`];
  const alert = n("ALERT"), off = n("offline"), base = n("baselining") + n("starting");
  if (alert) parts.push(`<b class="c-alert">${alert} ALERT</b>`);
  if (off) parts.push(`<b class="c-off">${off} offline</b>`);
  if (base) parts.push(`${base} building baseline`);
  parts.push(`${n("monitoring")} monitoring`);
  $("counts").innerHTML = parts.join(" · ");
}

let audio = null;
$("sound").onclick = () => {
  if (audio) { audio.close(); audio = null; } else { audio = new AudioContext(); beep(); }
  renderSoundButton();
};
function renderSoundButton() {
  const b = $("sound");
  const unacked = state && state.alerts.some((a) => a.status === "active");
  b.className = `sound ${audio ? "on" : "off"}${!audio && unacked ? " nag" : ""}`;
  b.textContent = audio ? "Alarm sound on" : "Enable alarm sound";
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
// Repeat the alarm every 4 s while any alert is unacknowledged.
setInterval(() => {
  if (state && state.alerts.some((a) => a.status === "active")) beep();
}, 4000);

setInterval(() => { $("clock").textContent = fmtTime(serverNow()); }, 1000);

// ---------------------------------------------------------------- add phone (QR codes)
let modalSig = "";
$("add-phone").onclick = () => { $("modal").hidden = false; modalSig = ""; renderModal(); };
$("modal-close").onclick = () => { $("modal").hidden = true; };
if (location.hash === "#add-phone") $("modal").hidden = false;  // direct link to the QR codes
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
    body.append(el("p", "err", `Tunnel failed: ${tun.error}`),
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
    const a = el("a", "qr-link", "copy link");
    a.href = link;
    a.onclick = (e) => { e.preventDefault(); navigator.clipboard?.writeText(link); a.textContent = "copied"; };
    box.append(el("h3", null, title), img, el("p", "muted", hint), a);
    return box;
  };
  const row = el("div", "qr-row");
  row.append(
    qr("phone", "Phone as bed camera", "Scan with the iPhone camera, open in Safari, tap Start camera. Each phone becomes a new bed.", links.phone),
    qr("dashboard", "Dashboard on another device", "Opens this dashboard (view and acknowledge alerts).", links.dashboard));
  body.append(row,
    el("p", "muted", "A brand-new link can take up to a minute to work everywhere. If Safari says it can't find the server, wait 30 s and reload."),
    el("p", "muted", "Links contain a secret key and change every time the server restarts. Anyone with them can see the cameras, so only share them with the team."));
}

// ---------------------------------------------------------------- websocket
function render() {
  clockSkew = state.server_time - Date.now() / 1000;
  for (const bed of state.beds) renderTile(bed);
  const ids = new Set(state.beds.map((b) => b.bed_id));
  for (const [id, t] of tiles) {  // phone removed
    if (!ids.has(id)) { t.el.remove(); if (t.url) URL.revokeObjectURL(t.url); tiles.delete(id); }
  }
  renderModal();
  renderAlerts(state.alerts);
  renderHeader(state.beds);
  renderSoundButton();
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
