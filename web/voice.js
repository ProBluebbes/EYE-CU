// Shared voice for the phone and bed pages. Plays the recorded clip for a sentence
// (web/voice/*.mp3 + manifest.json, made by scripts/make_voice.py); sentences without a clip
// use the nicest built-in browser voice. Browsers only allow sound after the user has tapped or
// clicked the page: the first tap/click/key anywhere unlocks it (Voice.unlocked).
const Voice = (() => {
  const audio = new Audio();
  audio.preload = "auto";
  let lines = {};        // sentence -> { file, seconds }
  let unlocked = false;
  let seq = 0;           // newest say() wins
  const listeners = [];
  const notify = () => listeners.forEach((f) => f(unlocked));

  fetch("/static/voice/manifest.json", { cache: "no-cache" })
    .then((r) => (r.ok ? r.json() : {}))
    .then((m) => {
      lines = m.lines || {};
      // warm the cache so the first prompt starts without a download (~1 MB in all)
      for (const c of Object.values(lines)) fetch(`/static/voice/${c.file}`).catch(() => {});
    })
    .catch(() => {});

  // 0.1 s of silence, played inside the first tap so later prompts are allowed to play.
  const SILENT = (() => {
    const n = 800, b = new DataView(new ArrayBuffer(44 + n));
    const str = (o, s) => [...s].forEach((ch, i) => b.setUint8(o + i, ch.charCodeAt(0)));
    str(0, "RIFF"); b.setUint32(4, 36 + n, true); str(8, "WAVEfmt ");
    b.setUint32(16, 16, true); b.setUint16(20, 1, true); b.setUint16(22, 1, true);
    b.setUint32(24, 8000, true); b.setUint32(28, 8000, true); b.setUint16(32, 1, true);
    b.setUint16(34, 8, true); str(36, "data"); b.setUint32(40, n, true);
    for (let i = 0; i < n; i++) b.setUint8(44 + i, 128);
    return URL.createObjectURL(new Blob([b], { type: "audio/wav" }));
  })();

  function unlock() {
    if (unlocked) return;
    unlocked = true;
    if (audio.paused) {
      audio.src = SILENT;
      audio.play().catch((e) => { if (e.name === "NotAllowedError") { unlocked = false; notify(); } });
    }
    try { speechSynthesis.speak(new SpeechSynthesisUtterance("")); } catch (e) { /* no speech */ }
    notify();
  }
  // Events that count as a user gesture for media playback (pointerdown doesn't, for touch).
  for (const ev of ["pointerup", "touchend", "click", "keydown"]) {
    document.addEventListener(ev, unlock, { capture: true });
  }

  function bestVoice() {
    const vs = speechSynthesis.getVoices().filter((v) => /^en([-_]|$)/i.test(v.lang));
    const novelty = /Albert|Bad News|Bahh|Bells|Boing|Bubbles|Cellos|Wobble|Jester|Organ|Trinoids|Whisper|Zarvox|Superstar|Junior|Ralph|Fred|Grandpa|Grandma|Rocko|Shelley|Eddy|Reed|Sandy|Flo/;
    const score = (v) => (/premium/i.test(v.name) ? 40 : /enhanced|natural|neural/i.test(v.name) ? 30 : 0)
      + (/google/i.test(v.name) ? 10 : 0) + (/en-US/i.test(v.lang) ? 3 : 0)
      + (/Ava|Samantha|Zoe|Allison|Susan|Serena|Karen|Moira/.test(v.name) ? 5 : 0)
      - (novelty.test(v.name) ? 100 : 0);
    return vs.sort((a, b) => score(b) - score(a))[0] || null;
  }

  function speak(text) {  // fallback: built-in browser voice, a little slower and calmer
    try {
      const u = new SpeechSynthesisUtterance(text);
      const v = bestVoice();
      if (v) u.voice = v;
      u.lang = (v && v.lang) || "en-US";
      u.rate = 0.9;
      speechSynthesis.cancel();
      speechSynthesis.speak(u);
    } catch (e) { /* no speech support */ }
  }

  function stop() {
    seq++;
    audio.pause();
    try { speechSynthesis.cancel(); } catch (e) { /* no speech support */ }
  }

  function say(text) {
    if (!text) return;
    stop();
    const id = seq, c = lines[text];
    if (!c) { speak(text); return; }
    audio.src = `/static/voice/${c.file}`;
    audio.play().catch((e) => {
      if (id !== seq || e.name === "AbortError") return;  // replaced by a newer prompt
      if (e.name === "NotAllowedError") { unlocked = false; notify(); return; }
      speak(text);  // clip missing or unplayable
    });
  }

  return {
    say, stop,
    get unlocked() { return unlocked; },
    onUnlockChange(f) { listeners.push(f); },
  };
})();
