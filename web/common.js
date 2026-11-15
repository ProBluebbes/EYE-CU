// Shared by the ward dashboard and the bed page: branding, staff initials, time helpers,
// the code-stroke checklist and treatment-window timers.
const HOSPITAL = "Azure Medical Center";
const UNIT = "Unit 4B · Stroke & Neuro Step-Down";

const CHECKLIST = [
  ["team", "Stroke team paged"], ["glucose", "Glucose checked"], ["nihss", "NIHSS done"],
  ["ct", "CT / CTA ordered"], ["neuro", "Neurology at bedside"], ["decision", "Treatment decision"],
];
// Reference treatment windows from last known well (hours).
const WINDOWS = [["IV thrombolysis", 4.5], ["Thrombectomy", 24]];

const KIND_BADGE = { sim: "SIM", phone: "PHONE", video: "VIDEO", demo: "DEMO", camera: "CAMERA" };

let clockSkew = 0;  // server_time - local time (s)
const serverNow = () => Date.now() / 1000 + clockSkew;
const setServerTime = (t) => { clockSkew = t - Date.now() / 1000; };

const fmtTime = (epoch) => new Date(epoch * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false });
const fmtTimeS = (epoch) => new Date(epoch * 1000).toLocaleTimeString([], { hour12: false });
const fmtDate = (epoch) => new Date(epoch * 1000).toLocaleDateString([], { weekday: "short", day: "numeric", month: "short", year: "numeric" });
function fmtDur(s) {
  s = Math.max(0, Math.floor(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`
    : `${m}:${String(sec).padStart(2, "0")}`;
}
function fmtLeft(s) {
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h} h ${m} min` : `${m} min`;
}
function ago(epoch) {
  const s = Math.max(0, serverNow() - epoch);
  if (s < 60) return `${Math.round(s)} s ago`;
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  return `${Math.floor(s / 3600)} h ${Math.floor((s % 3600) / 60)} min ago`;
}
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

// ---------------------------------------------------------------- staff initials
function staffGet() { try { return localStorage.getItem("sw_staff") || ""; } catch (e) { return ""; } }
function staffSet(v) { try { localStorage.setItem("sw_staff", v); } catch (e) { /* optional */ } }
function bindStaffInput(input) {
  input.value = staffGet();
  input.addEventListener("input", () => staffSet(input.value.trim().toUpperCase().slice(0, 12)));
  input.addEventListener("blur", () => { input.value = staffGet(); });
}

// POST with the staff initials attached (recorded in the activity log).
async function post(url, body) {
  const r = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...(body || {}), by: staffGet() || undefined }),
  });
  return r.ok;
}

// ---------------------------------------------------------------- alerts
const stripTag = (f) => f.replace(/^Face: /, "");
function alertFindings(a) {
  return a.findings && a.findings.length ? a.findings : [`Face: possible new ${a.side} facial droop`];
}
// A BE-FAST check part the patient didn't follow (eyes wandered, looked the wrong way, ...).
const inconclusive = (le) => !!(le && !le.cancelled && le.inconclusive && le.inconclusive.length);
const PART_NAMES = { eyes: "eyes", arms: "arms" };
const inconclusiveText = (le) =>
  `${le.inconclusive.map((k) => PART_NAMES[k] || k).join(" + ")} inconclusive (instructions not followed)`;
// Hedged lesion-side wording from the server (localize.py), or "".
const locText = (loc) => (loc ? `${loc.text} ${loc.caveat}` : "");

function alertHeadline(a) {
  const f = alertFindings(a);
  return stripTag(f[0]) + (f.length > 1 ? ` (+${f.length - 1} more)` : "");
}
// Last known well: entered by staff, else the camera's onset estimate, else when it was raised.
function lastKnownWell(alert, patient) {
  if (patient && patient.lkw) return { t: patient.lkw, src: "entered by staff" };
  if (alert.onset_time) return { t: alert.onset_time, src: "camera onset estimate" };
  return { t: alert.raised_at, src: "time of alert" };
}

// Timers block for an open alert: elapsed since onset + treatment windows. Returns an element
// whose [data-*] spans are refreshed by tickTimers().
function timersBlock(alert, patient) {
  const lkw = lastKnownWell(alert, patient);
  const box = el("div", "timers");
  const row = (label, cls) => {
    const r = el("div", `timer ${cls || ""}`);
    r.append(el("span", "timer-label", label));
    const v = el("span", "timer-val");
    r.append(v);
    box.append(r);
    return v;
  };
  const since = row("Since onset");
  since.dataset.since = alert.onset_time || alert.raised_at;
  for (const [name, h] of WINDOWS) {
    const v = row(`${name} window`, "window");
    v.dataset.deadline = lkw.t + h * 3600;
  }
  box.append(el("div", "timer-note", `From last known well ${fmtTime(lkw.t)} (${lkw.src}). Reference only: confirm per stroke protocol.`));
  tickTimers(box);
  return box;
}
function tickTimers(root) {
  const now = serverNow();
  for (const s of root.querySelectorAll("[data-since]")) s.textContent = fmtDur(now - +s.dataset.since);
  for (const s of root.querySelectorAll("[data-deadline]")) {
    const left = +s.dataset.deadline - now;
    s.textContent = left > 0 ? `${fmtLeft(left)} left` : "closed";
    s.parentElement.classList.toggle("closing", left > 0 && left < 3600);
    s.parentElement.classList.toggle("closed", left <= 0);
  }
  for (const s of root.querySelectorAll("[data-ago]")) s.textContent = ago(+s.dataset.ago);
}

// Code-stroke checklist for an open alert.
function checklistBlock(alert) {
  const box = el("div", "checklist");
  const done = alert.checklist || {};
  for (const [key, label] of CHECKLIST) {
    const d = done[key];
    const item = el("label", `ck${d ? " done" : ""}`);
    const cb = el("input");
    cb.type = "checkbox";
    cb.checked = !!d;
    cb.onchange = async () => {
      cb.disabled = true;
      await post(`/api/alerts/${alert.id}/step`, { step: key, done: cb.checked });
    };
    const txt = el("span", "ck-label", label);
    item.append(cb, txt);
    if (d) item.append(el("span", "ck-meta", `${fmtTime(d.t)}${d.by ? ` · ${d.by}` : ""}`));
    box.append(item);
  }
  return box;
}

const LEVEL_ICON = { alert: "●", warn: "▲", ok: "✓", info: "•" };
function eventRow(e, withBed) {
  const r = el("div", `ev ev-${e.level}`);
  r.append(el("span", "ev-time", fmtTimeS(e.t)), el("span", "ev-icon", LEVEL_ICON[e.level] || "•"));
  const body = el("span", "ev-text");
  if (withBed && e.bed_name) {
    const b = el("a", "ev-bed", e.bed_name);
    b.href = `/bed/${encodeURIComponent(e.bed_id)}`;
    body.append(b, " ");
  }
  body.append(e.text);
  if (e.by) body.append(el("span", "ev-by", ` · ${e.by}`));
  r.append(body);
  return r;
}

// Brand bar shared by both pages.
function brandBar(subtitle) {
  const bar = el("div", "brandbar");
  const logo = el("div", "logo");
  logo.innerHTML = `<svg viewBox="0 0 32 32" width="30" height="30" aria-hidden="true"><rect width="32" height="32" rx="7" fill="#2b8ce6"/><path d="M13 7h6v6h6v6h-6v6h-6v-6H7v-6h6z" fill="#fff"/></svg>`;
  const names = el("div", "brand-names");
  names.append(el("div", "brand-hosp", HOSPITAL), el("div", "brand-app", subtitle || "Eye-CU · Continuous Neuro Monitoring"));
  logo.append(names);
  bar.append(logo);
  return bar;
}
