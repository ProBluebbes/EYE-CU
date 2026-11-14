// Phone camera page: camera -> (rotate, downscale) -> JPEG over a WebSocket to StrokeWatch.
// The server sends back JSON: hello (bed + send settings), status (1/s), say (speak text;
// with a "step" during a BE-FAST check, also shown big on screen for the patient).
const $ = (id) => document.getElementById(id);
const store = {
  get: (k, d = null) => localStorage.getItem(`sw_${k}`) ?? d,
  set: (k, v) => localStorage.setItem(`sw_${k}`, v),
};
function newId() {
  if (crypto.randomUUID) return crypto.randomUUID();
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, "0")).join("");
}
// A stable id per phone, so a reload reconnects to the same bed (and keeps its baseline).
const device = store.get("device") || newId();
store.set("device", device);

let facing = store.get("facing", "user");
let rotate = +store.get("rotate", 0);
let cfg = { send_fps: 10, max_side: 640, jpeg_quality: 0.7 };
let stream = null, ws = null, running = false, busy = false;
let sendTimer = null, retryTimer = null, wakeLock = null, failures = 0;
const sentTimes = [];
const canvas = document.createElement("canvas");
const g = canvas.getContext("2d");
$("name").value = store.get("name", "");

// ---------------------------------------------------------------- ui helpers
function showMsg(text, err = false) {
  const m = $("msg");
  m.hidden = !text;
  m.textContent = text || "";
  m.className = `msg${err ? " err" : ""}`;
}
function setState(state, text) {
  $("state").className = `state ${state}`;
  $("state").textContent = text;
}
function say(text) {
  try {
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "en-US"; u.rate = 0.95;
    speechSynthesis.cancel();
    speechSynthesis.speak(u);
  } catch (e) { /* no speech support */ }
}
function renderFacing() {
  $("cam-user").classList.toggle("on", facing === "user");
  $("cam-env").classList.toggle("on", facing === "environment");
}
function applyRotation() { $("video").style.transform = `rotate(${rotate}deg)`; }
renderFacing();
applyRotation();

const STATE_TEXT = { baselining: "Building baseline", monitoring: "Monitoring",
                     ALERT: "ALERT raised on dashboard", offline: "Not receiving frames", starting: "Starting" };

// ---------------------------------------------------------------- BE-FAST check prompts
// Front camera: the screen faces the patient, so "←" is the patient's own left.
const ARROWS = { look_left: "←", look_right: "→", center: "●", arms: "↑ ↑", legs: "↑" };
let promptTimer = null;
function showPrompt(step, text) {
  clearTimeout(promptTimer);
  promptTimer = null;
  if (!step) { $("prompt").hidden = true; return; }
  $("prompt").hidden = false;
  $("prompt-arrow").textContent = (facing === "user" && ARROWS[step]) || "";
  $("prompt-text").textContent = text;
  if (step === "done") promptTimer = setTimeout(() => { $("prompt").hidden = true; promptTimer = null; }, 5000);
}

function onStatus(m) {
  const ex = m.exam || {};
  if (ex.active && $("prompt").hidden) showPrompt(ex.step, ex.say);  // e.g. after a reconnect
  else if (!ex.active && !$("prompt").hidden && !promptTimer) showPrompt(null);
  const name = m.bed_name ? `${m.bed_name}: ` : "";
  let text = STATE_TEXT[m.state] || m.state;
  if (m.state === "baselining") text += ` ${Math.round((m.baseline_progress || 0) * 100)}%`;
  setState(m.state, name + text);
  $("bar").hidden = m.state !== "baselining";
  $("bar").firstElementChild.style.width = `${(m.baseline_progress || 0) * 100}%`;
  const face = $("face");
  if (m.state === "offline" || m.state === "starting") { face.textContent = ""; return; }
  if (m.valid) { face.className = "face good"; face.textContent = "Face tracked"; }
  else { face.className = "face bad"; face.textContent = `Paused: ${m.invalid_reason || "no face"}`; }
}

// ---------------------------------------------------------------- camera
async function startCamera() {
  stopCamera();
  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: facing }, width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false,
  });
  const v = $("video");
  v.srcObject = stream;
  await v.play();
}
function stopCamera() {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
}

// Draw the current video frame rotated and scaled (longer side <= max_side) and JPEG it.
function grab() {
  const v = $("video");
  const vw = v.videoWidth, vh = v.videoHeight;
  if (!vw || !vh) return Promise.resolve(null);
  const s = Math.min(1, cfg.max_side / Math.max(vw, vh));
  const w = Math.round(vw * s), h = Math.round(vh * s);
  const swap = rotate % 180 !== 0;
  const cw = swap ? h : w, ch = swap ? w : h;
  if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.translate(cw / 2, ch / 2);
  g.rotate((rotate * Math.PI) / 180);
  g.drawImage(v, -w / 2, -h / 2, w, h);
  return new Promise((res) => canvas.toBlob(res, "image/jpeg", cfg.jpeg_quality));
}

async function sendFrame() {
  if (busy || !ws || ws.readyState !== 1 || ws.bufferedAmount > 256 * 1024) return;
  busy = true;
  try {
    const blob = await grab();
    if (blob && ws && ws.readyState === 1) {
      ws.send(await blob.arrayBuffer());
      const now = performance.now();
      sentTimes.push(now);
      while (sentTimes[0] < now - 2000) sentTimes.shift();
      $("sendinfo").textContent = `sending ${(sentTimes.length / 2).toFixed(1)} fps · ${(blob.size / 1024) | 0} KB/frame · ${canvas.width}×${canvas.height}`;
    }
  } finally { busy = false; }
}
function startSending() {
  clearInterval(sendTimer);
  sendTimer = setInterval(sendFrame, 1000 / cfg.send_fps);
}

// ---------------------------------------------------------------- connection
function connect() {
  clearTimeout(retryTimer);
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const q = new URLSearchParams({ device, name: store.get("name", "") });
  const sock = new WebSocket(`${proto}://${location.host}/api/phone/ws?${q}`);
  sock.binaryType = "arraybuffer";
  ws = sock;
  sock.onopen = () => { failures = 0; showMsg(null); setState("starting", "Connected"); };
  sock.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.type === "hello") {
      cfg = { ...cfg, ...m };
      document.title = `StrokeWatch: ${m.bed_name}`;
      startSending();
    } else if (m.type === "status") onStatus(m);
    else if (m.type === "say") { say(m.text); if (m.step !== undefined) showPrompt(m.step, m.text); }
  };
  sock.onclose = (e) => {
    if (ws !== sock) return;
    ws = null;
    if (!running) return;
    const fatal = { 4401: "Access key not accepted (the server was probably restarted). Scan the QR code on the dashboard again.",
                    4004: "This phone was removed from the dashboard. Tap Start camera to join again.",
                    4000: "This camera page was opened in another tab. Close one of them." }[e.code];
    if (fatal) { stopAll(); showMsg(fatal, true); return; }
    failures += 1;
    setState("offline", "Connection lost, reconnecting…");
    if (failures >= 5) showMsg("Still can't reach the server. If it was restarted, the link changed: scan the new QR code.", true);
    retryTimer = setTimeout(connect, 2000);
  };
}

async function requestWakeLock() {
  try { if ("wakeLock" in navigator) wakeLock = await navigator.wakeLock.request("screen"); } catch (e) { /* optional */ }
}

async function start() {
  showMsg(null);
  if (!window.isSecureContext) {
    showMsg("Open this page through the https:// link from the QR code: phone cameras need HTTPS.", true);
    return;
  }
  try { await startCamera(); } catch (e) {
    showMsg(`Camera not available (${e.name}). Allow camera access: Safari aA menu → Website Settings → Camera → Allow, then try again.`, true);
    return;
  }
  running = true;
  $("setup").hidden = true;
  $("running").hidden = false;
  setState("starting", "Connecting…");
  requestWakeLock();
  connect();
  startSending();
}

function stopAll() {
  running = false;
  clearInterval(sendTimer);
  clearTimeout(retryTimer);
  const sock = ws;
  ws = null;
  if (sock) sock.close();
  stopCamera();
  if (wakeLock) { wakeLock.release().catch(() => {}); wakeLock = null; }
  $("setup").hidden = false;
  $("running").hidden = true;
}

// ---------------------------------------------------------------- buttons
$("start").onclick = () => {
  say("StrokeWatch camera on.");  // speech must start inside the tap itself (iOS rule), before any await
  store.set("name", $("name").value.trim());
  start();
};
$("cam-user").onclick = () => { facing = "user"; store.set("facing", facing); renderFacing(); };
$("cam-env").onclick = () => { facing = "environment"; store.set("facing", facing); renderFacing(); };
$("rotate").onclick = () => { rotate = (rotate + 90) % 360; store.set("rotate", rotate); applyRotation(); };
$("flip").onclick = async () => {
  facing = facing === "user" ? "environment" : "user";
  store.set("facing", facing);
  renderFacing();
  try { await startCamera(); } catch (e) { showMsg(`Could not switch camera (${e.name}).`, true); }
};
$("voice").onclick = () => say("This is a voice test from StrokeWatch.");
$("stop").onclick = () => {
  stopAll();
  showMsg("Stopped. This bed shows as offline on the dashboard until you start again.");
};

// iOS pauses the camera when the page is hidden; resume when it's visible again.
document.addEventListener("visibilitychange", async () => {
  if (document.visibilityState !== "visible" || !running) return;
  requestWakeLock();
  const track = stream && stream.getVideoTracks()[0];
  if (!track || track.readyState === "ended" || track.muted) {
    try { await startCamera(); } catch (e) { showMsg("Camera stopped. Tap Stop, then Start camera.", true); }
  }
  if (!ws) connect();
});

// For automated testing only (no voice: speech needs a real tap).
if (new URLSearchParams(location.search).get("autostart")) start();
