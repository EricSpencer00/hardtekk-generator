/* remix-app.js — UI for remix.html. DSP runs in remix-worker.js.
 *
 * The mixer plays the six buses live through WebAudio (level + 3-band EQ per
 * bus, a light preview master). "render wav" sends the same levels and EQ to
 * the worker, which runs the real master chain. The kick panel re-renders only
 * the kick bus.
 */
const $ = id => document.getElementById(id);
const SR = 44100;
const SAMPLES = { kick: "kick-hardtekk_C.wav", snare: "snare-hardtekk-distorted-punch_D#_major.wav",
                  hat: "phonk-closed-hi-hats-thin_C_minor.wav", bell: "hardtekk-bell-shot.wav" };
const LAYERS = [["kick", "kick"], ["bass808", "808"], ["bed", "synths"], ["vocals", "vocals"], ["perc", "drums"], ["fx", "bells"], ["sfx", "effects"]];
const worker = new Worker("remix-worker.js");
let files = [], ctx = null, buffers = {}, chain = {}, master = null, sources = [], playing = false, startAt = 0, offset = 0, meta = null;

/* ------------------------------------------------------------ log + progress */
function log(msg, err = false) { const d = document.createElement("div"); d.textContent = msg; if (err) d.className = "err"; $("log").appendChild(d); $("log").scrollTop = 1e9; }
function prog(p) { $("prog").style.width = `${Math.round(p * 100)}%`; }
const busy = on => { for (const id of ["go", "k-apply", "render", "rebuild", "shuffle"]) $(id).disabled = on; };

/* ------------------------------------------------------------ files in */
function setFiles(list) {
  const got = [...list].filter(f => /audio|octet/.test(f.type) || /\.(wav|mp3|m4a|flac|ogg|aiff?)$/i.test(f.name));
  if (!got.length) return;
  files = got;
  $("files").textContent = files.map(f => f.name).join("  ·  ");
  $("go").disabled = false; $("go").hidden = false;
  if (meta) $("go").click();                          // a song is already loaded: replacing it starts the build
}
$("drop").addEventListener("click", () => $("file").click());
$("file").addEventListener("change", e => { setFiles(e.target.files); e.target.value = ""; });
// drop anywhere on the page
let dragDepth = 0;
addEventListener("dragenter", e => { e.preventDefault(); dragDepth++; document.body.classList.add("dragging"); });
addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; document.body.classList.remove("dragging"); } });
addEventListener("dragover", e => e.preventDefault());
addEventListener("drop", e => { e.preventDefault(); dragDepth = 0; document.body.classList.remove("dragging"); setFiles(e.dataTransfer.files); });

async function decode(ac, buf) {
  const a = await ac.decodeAudioData(buf);
  const L = a.getChannelData(0).slice(), R = a.numberOfChannels > 1 ? a.getChannelData(1).slice() : L.slice();
  return [L, R];
}
const normPeak = y => { let m = 0; for (const c of y) for (let i = 0; i < c.length; i++) m = Math.max(m, Math.abs(c[i])); return y.map(c => c.map(v => v / (m || 1))); };
function stemRole(name) { const n = name.toLowerCase(); return ["vocals", "drums", "bass", "other"].find(r => n.includes(r)); }

/* ------------------------------------------------------------ build */
function fxOpts() { return { risers: $("fx-risers").checked, impacts: $("fx-impacts").checked, echo: $("fx-echo").checked, tapeStop: $("fx-tape").checked, fadeInBars: +$("fx-fin").value, fadeOutBars: +$("fx-fout").value }; }
function opts() { return { semis: +$("semis").value, pattern: $("pattern").value, h16: $("h16").checked, ...kickOpts(), ...fxOpts() }; }
$("go").addEventListener("click", async () => {
  stop(); busy(true); $("log").textContent = ""; prog(0); arrEdited = false; meta = null;
  try {
    const ac = new OfflineAudioContext(2, 1, SR);   // decode at 44.1 kHz whatever the device runs at
    const samples = {};
    for (const [k, f] of Object.entries(SAMPLES)) {
      const r = await fetch("samples/" + encodeURIComponent(f)); if (!r.ok) throw new Error(`sample ${f}: HTTP ${r.status}`);
      samples[k] = normPeak(await decode(ac, await r.arrayBuffer()));
    }
    const ks = await kickSample(); samples.kick = ks.data; samples._kickId = ks.id;
    const roles = files.map(f => stemRole(f.name));
    const msg = { type: "build", samples, opts: opts() }, transfer = [];
    if (files.length >= 4 && ["vocals", "drums", "bass", "other"].every(r => roles.includes(r))) {
      msg.stems = {};
      for (const r of ["vocals", "drums", "bass", "other"]) { msg.stems[r] = await decode(ac, await files[roles.indexOf(r)].arrayBuffer()); transfer.push(...msg.stems[r].map(c => c.buffer)); }
      log("using the 4 stems");
    } else {
      msg.mix = await decode(ac, await files[0].arrayBuffer()); transfer.push(...msg.mix.map(c => c.buffer));
      log(`${files[0].name}: ${(msg.mix[0].length / SR).toFixed(1)} s`);
    }
    worker.postMessage(msg, transfer);
  } catch (e) { log(e.message, true); busy(false); }
});
/* Small changes reuse the worker's cache: no decode, separation or analysis.
 * A groove or hats change only redoes the drums; a speed seen before is cached. */
let pendingResume = null, queued = null, arrEdited = false, timer = 0;
const displayName = () => files.length >= 4 ? files[0].name.replace(/[-_ ]*(vocals|drums|bass|other)(\.[^.]+)$/i, "$2") + " (stems)" : files[0]?.name || "";
function update(o) {
  if (!meta) return;
  if (busyNow()) { queued = o; return; }
  pendingResume = playing ? position() : null; stop(false);
  busy(true); prog(0); worker.postMessage({ type: "rebuild", opts: o });
}
const busyNow = () => $("render").disabled;
const currentOpts = () => ({ ...opts(), ...(arrEdited ? { arrangement: $("arr").value.trim() } : {}) });
for (const id of ["semis", "pattern", "h16"]) $(id).addEventListener("change", () => { clearTimeout(timer); timer = setTimeout(() => update(currentOpts()), 120); });
$("arr").addEventListener("input", () => arrEdited = true);
$("rebuild").addEventListener("click", () => { arrEdited = true; update(currentOpts()); });

const STAGE = { separating: [0, 0.45], analysing: [0.45, 0.6], building: [0.6, 1], mastering: [0, 1] };
worker.onmessage = e => {
  const m = e.data;
  if (m.type === "log") log(m.msg);
  else if (m.type === "shuffled") { $("arr").value = m.arrangement; arrEdited = true; log(`shuffle #${shuffleSeed}: ${m.arrangement}`); busy(false); update(currentOpts()); }
  else if (m.type === "progress") { const [a, b] = STAGE[m.stage] || [0, 1]; prog(a + (b - a) * m.p); }
  else if (m.type === "error") { log(m.msg, true); busy(false); }
  else if (m.type === "built") {
    const resume = pendingResume; pendingResume = null;
    meta = m.meta; loadLayers(m.layers); showMeta(); busy(false); prog(1);
    EDITOR.load(meta); $("editor").hidden = $("fxpanel").hidden = false;
    $("drop").classList.add("loaded"); $("droptitle").textContent = displayName(); $("files").textContent = ""; $("go").hidden = true;
    if (resume !== null) play(resume);
    if (queued) { const q = queued; queued = null; update(q); }
  }
  else if (m.type === "kick") { const was = playing, at = position(); stop(); buffers.kick = toBuffer(m.layer); if (was) play(at); busy(false); log("kick re-rendered"); }
  else if (m.type === "wav") {
    const url = URL.createObjectURL(new Blob([m.wav], { type: "audio/wav" }));
    const base = (files[0]?.name || "remix").replace(/\.[^.]+$/, "").replace(/[-_ ]*(vocals|drums|bass|other)$/i, "");
    $("dl").href = url; $("dl").download = `${base} (hardtekk remix).wav`; $("dl").hidden = false;
    $("final").src = url; $("final").hidden = false; busy(false); prog(1); log(`rendered: ${m.lufs.toFixed(1)} LUFS`);
  }
};

/* ------------------------------------------------------------ live mixer */
function ensureCtx() {
  if (ctx) return;
  ctx = new AudioContext({ sampleRate: SR });
  const comp = ctx.createDynamicsCompressor(); comp.threshold.value = -18; comp.ratio.value = 4; comp.attack.value = 0.003; comp.release.value = 0.1; comp.knee.value = 6;
  const shaper = ctx.createWaveShaper(), curve = new Float32Array(2048);
  for (let i = 0; i < curve.length; i++) { const x = i / 1023.5 - 1; curve[i] = Math.tanh(1.6 * x) / Math.tanh(1.6); }
  shaper.curve = curve; shaper.oversample = "4x";
  const bus = ctx.createGain(), out = ctx.createGain(); out.gain.value = 0.9;
  master = { bus, comp, shaper, out };
  routeMaster();
  out.connect(ctx.destination);
  for (const [k] of LAYERS) {
    const lo = ctx.createBiquadFilter(); lo.type = "lowshelf"; lo.frequency.value = 120;
    const mid = ctx.createBiquadFilter(); mid.type = "peaking"; mid.frequency.value = 1000; mid.Q.value = 0.8;
    const hi = ctx.createBiquadFilter(); hi.type = "highshelf"; hi.frequency.value = 6000;
    const g = ctx.createGain(); lo.connect(mid).connect(hi).connect(g).connect(bus);
    chain[k] = { lo, mid, hi, g, input: lo };
  }
}
function routeMaster() {
  const { bus, comp, shaper, out } = master; bus.disconnect(); comp.disconnect(); shaper.disconnect();
  if ($("pmaster").checked) { bus.connect(comp).connect(shaper).connect(out); bus.gain.value = 1.6; } else { bus.connect(out); bus.gain.value = 0.7; }
}
$("pmaster").addEventListener("change", () => master && routeMaster());
function toBuffer(L) { const b = ctx.createBuffer(2, L[0].length, SR); b.copyToChannel(L[0], 0); b.copyToChannel(L[1], 1); return b; }
function loadLayers(layers) { ensureCtx(); buffers = {}; for (const [k] of LAYERS) if (layers[k]) buffers[k] = toBuffer(layers[k]); buildStrips(); applyStrips(); offset = 0; $("player").hidden = $("kickpanel").hidden = $("outpanel").hidden = false; }
const duration = () => buffers.kick ? buffers.kick.duration : 0;
const position = () => playing ? Math.min(duration(), offset + ctx.currentTime - startAt) : offset;
function play(at = position()) {
  ensureCtx(); stop(false); ctx.resume(); offset = at; startAt = ctx.currentTime + 0.05;
  sources = Object.entries(buffers).map(([k, b]) => { const s = ctx.createBufferSource(); s.buffer = b; s.connect(chain[k].input); s.start(startAt, at); return s; });
  playing = true; $("play").textContent = "pause";
}
function stop(reset = true) { for (const s of sources) try { s.stop(); } catch {} sources = []; if (playing && !reset) offset = position(); playing = false; $("play").textContent = "play"; if (reset) offset = 0; }
$("play").addEventListener("click", () => { if (playing) { const at = position(); stop(false); offset = at; } else play(); });
$("stop").addEventListener("click", () => stop(true));
$("seek").addEventListener("input", () => { const at = $("seek").value / 1000 * duration(); if (playing) play(at); else offset = at; });
setInterval(() => {
  if (!ctx || !duration()) return;
  const p = position(); $("time").textContent = `${Math.floor(p / 60)}:${String(Math.floor(p % 60)).padStart(2, "0")}`;
  if (playing && document.activeElement !== $("seek")) $("seek").value = Math.round(p / duration() * 1000);
  EDITOR.playhead(p);
  if (playing && p >= duration()) stop(true);
}, 100);

/* ------------------------------------------------------------ strips */
const mixState = { gains: {}, eq: {}, mute: {} };
function buildStrips() {
  $("strips").innerHTML = LAYERS.map(([k, name]) => `
    <div class="strip" data-k="${k}">
      <h3>${name}<input type="checkbox" data-mute="${k}" title="mute"></h3>
      <label>level <input type="range" data-p="gain" min="-24" max="12" step="0.5" value="${mixState.gains[k] ?? 0}"><output></output></label>
      <label>low <input type="range" data-p="low" min="-12" max="12" step="0.5" value="${mixState.eq[k]?.low ?? 0}"><output></output></label>
      <label>mid <input type="range" data-p="mid" min="-12" max="12" step="0.5" value="${mixState.eq[k]?.mid ?? 0}"><output></output></label>
      <label>high <input type="range" data-p="high" min="-12" max="12" step="0.5" value="${mixState.eq[k]?.high ?? 0}"><output></output></label>
    </div>`).join("");
  $("strips").querySelectorAll("input").forEach(el => el.addEventListener("input", applyStrips));
}
function applyStrips() {
  for (const el of $("strips").querySelectorAll(".strip")) {
    const k = el.dataset.k, v = p => +el.querySelector(`[data-p=${p}]`).value;
    mixState.gains[k] = v("gain"); mixState.eq[k] = { low: v("low"), mid: v("mid"), high: v("high") };
    mixState.mute[k] = el.querySelector("[data-mute]").checked;
    el.querySelectorAll("label").forEach(l => l.querySelector("output").textContent = (+l.querySelector("input").value).toFixed(1));
    const c = chain[k]; if (!c) continue;
    c.g.gain.value = mixState.mute[k] ? 0 : Math.pow(10, mixState.gains[k] / 20);
    c.lo.gain.value = mixState.eq[k].low; c.mid.gain.value = mixState.eq[k].mid; c.hi.gain.value = mixState.eq[k].high;
  }
}
function showMeta() {
  $("info").textContent = `${meta.bpm.toFixed(1)} BPM · ${meta.key} · ${meta.bars} bars · ${meta.kicks} kicks`;
  $("arr").value = meta.arrangement;
  $("sections").innerHTML = meta.blocks.map(b => `<span class="k-${b.kind}" style="flex:${b.n}" title="${b.name} ← bars ${b.src}-${b.src + b.n - 1}">${b.name}</span>`).join("");
}

/* ------------------------------------------------------------ kick + render */
function kickOpts() { return { kickTune: +$("k-tune").value, tailLp: +$("k-tail").value, buzz: +$("k-buzz").value, kickDrive: +$("k-drive").value }; }
for (const id of ["k-tune", "k-tail", "k-buzz", "k-drive", "lufs"]) { const el = $(id), o = $(id + "-v"); el.addEventListener("input", () => o.textContent = el.value); }
async function kickSample() {
  const f = $("k-sample").value, r = await fetch("samples/" + f.split("/").map(encodeURIComponent).join("/"));
  return { id: f, data: normPeak(await decode(new OfflineAudioContext(2, 1, SR), await r.arrayBuffer())) };
}
$("k-apply").addEventListener("click", async () => { busy(true); log("re-rendering the kick ..."); worker.postMessage({ type: "kick", opts: kickOpts(), sample: await kickSample() }); });
$("k-sample").addEventListener("change", () => meta && $("k-apply").click());
$("render").addEventListener("click", () => {
  busy(true); prog(0); log("rendering the master ...");
  const gains = {}; for (const [k] of LAYERS) gains[k] = mixState.mute[k] ? -120 : mixState.gains[k];
  worker.postMessage({ type: "master", mix: { gains, eq: mixState.eq, targetLufs: +$("lufs").value } });
});

EDITOR.init({
  onApply: score => { pendingResume = playing ? position() : null; stop(false); busy(true); prog(0); worker.postMessage({ type: "score", score }); },
  name: () => displayName().replace(/\.[^.]+$/, "").replace(/ \(stems\)$/, ""),
  log,
});

// effects: a toggle re-renders only the drum/FX pass (the song is cached)
for (const id of ["fx-risers", "fx-impacts", "fx-echo", "fx-tape", "fx-fin", "fx-fout"]) {
  const el = $(id), out = $(id + "-v");
  el.addEventListener("input", () => { if (out) out.textContent = el.value; });
  el.addEventListener("change", () => { if (!meta || busyNow()) return; pendingResume = playing ? position() : null; stop(false); busy(true); worker.postMessage({ type: "score", score: meta.score, opts: fxOpts() }); });
}
// shuffle: a new arrangement dealt from the song's blocks; each number is reproducible
let shuffleSeed = 0;
$("shuffle").addEventListener("click", () => { if (!meta) return; busy(true); worker.postMessage({ type: "shuffle", seed: ++shuffleSeed }); });
