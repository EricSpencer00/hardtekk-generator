/* tools/remix_node.js — run the remix engine outside the browser.
 *
 *   node tools/remix_node.js out.wav --stems DIR      # DIR has vocals/drums/bass/other.wav (Demucs)
 *   node tools/remix_node.js out.wav --mix song.wav    # in-engine separation
 *   node tools/remix_node.js out.wav --mix song.wav --truth DIR   # ... and measure what of DIR's drums+bass
 *                                                                  # and vocals reaches the chopped vocal
 *   --set=mode=continuous, --set=bpm=180, ... set any engine option
 *
 * Writes out.wav plus out.json (analysis, arrangement, score, chop placements, timings) and, with
 * --layers, one WAV per mix bus next to it.
 */
const fs = require("fs"), path = require("path");
const R = require("../remix-engine.js");
R.LOG.fn = m => console.log(m);

function readWav(file) {
  const b = fs.readFileSync(file), v = new DataView(b.buffer, b.byteOffset, b.byteLength);
  let p = 12, fmt = null, data = null;
  while (p + 8 <= b.length) {
    const id = b.toString("ascii", p, p + 4), sz = v.getUint32(p + 4, true);
    if (id === "fmt ") fmt = { tag: v.getUint16(p + 8, true), ch: v.getUint16(p + 10, true), sr: v.getUint32(p + 12, true), bits: v.getUint16(p + 22, true) };
    if (id === "data") data = [p + 8, sz];
    p += 8 + sz + (sz & 1);
  }
  if (!fmt || !data) throw new Error("not a wav: " + file);
  const tag = fmt.tag === 0xfffe ? (fmt.bits === 32 ? 3 : 1) : fmt.tag;
  const bps = fmt.bits / 8, n = Math.floor(data[1] / (bps * fmt.ch)), out = [new Float32Array(n), new Float32Array(n)];
  for (let i = 0; i < n; i++) for (let c = 0; c < fmt.ch; c++) {
    const o = data[0] + (i * fmt.ch + c) * bps; let s;
    if (tag === 3) s = bps === 4 ? v.getFloat32(o, true) : v.getFloat64(o, true);
    else if (bps === 2) s = v.getInt16(o, true) / 32768;
    else if (bps === 3) { const x = b[o] | (b[o + 1] << 8) | (b[o + 2] << 16); s = (x & 0x800000 ? x - 0x1000000 : x) / 8388608; }
    else s = v.getInt32(o, true) / 2147483648;
    out[c][i] = s;
  }
  if (fmt.ch === 1) out[1].set(out[0]);
  return fmt.sr === R.SR ? out : out.map(c => R.resample(c, fmt.sr, R.SR));
}
function loadSamples(dir, kickFile) {
  const f = n => readWav(path.join(dir, n));
  const norm = y => { let m = 0; for (const c of y) for (const v of c) m = Math.max(m, Math.abs(v)); return y.map(c => c.map(v => v / (m || 1))); };
  return { kick: norm(kickFile ? readWav(kickFile) : f("kick-hardtekk_C.wav")), snare: norm(f("snare-hardtekk-distorted-punch_D#_major.wav")),
           hat: norm(f("phonk-closed-hi-hats-thin_C_minor.wav")), bell: norm(f("hardtekk-bell-shot.wav")) };
}

(async () => {
  const args = process.argv.slice(2), out = args[0], get = k => { const i = args.indexOf(k); return i > 0 ? args[i + 1] : null; };
  const t0 = Date.now(), lap = m => console.log(`[${((Date.now() - t0) / 1000).toFixed(1)}s] ${m}`), times = {};
  const stage = (k, t) => { times[k] = (Date.now() - t) / 1000; };
  const stems4 = d => Object.fromEntries(["vocals", "drums", "bass", "other"].map(k => [k, readWav(path.join(d, k + ".wav"))]));
  let stems, ts = Date.now();
  if (get("--stems")) { stems = stems4(get("--stems")); lap("stems loaded"); }
  else {
    const mix = readWav(get("--mix")), probes = {}; lap("mix loaded");
    if (get("--truth")) { const t = stems4(get("--truth")); probes.db = t.drums.map((c, k) => c.map((v, i) => v + t.bass[k][i])); probes.v = t.vocals; }
    ts = Date.now(); stems = await R.separate(mix, undefined, probes); stage("separate", ts); lap("separated");
  }
  const samples = loadSamples(path.join(__dirname, "..", "samples"), get("--kick"));
  ts = Date.now(); const an = await R.analyze(stems); stage("analyse", ts);
  lap(`analysed: ${an.bpm.toFixed(3)} BPM, downbeat ${an.b0.toFixed(3)} s, key ${R.NAMES[an.key.tonic]} ${an.key.mode}, ${an.nBars} bars`);
  const opts = {}; if (get("--arrangement")) opts.arrangement = get("--arrangement");
  if (get("--shuffle")) { opts.arrangement = R.shuffleArrangement(an, +get("--shuffle")); lap("shuffled: " + opts.arrangement); }
  for (const f of ["risers", "impacts", "echo", "tapeStop"]) if (args.includes("--no-" + f)) opts[f] = false;
  for (const a of args) { const m = /^--set=(\w+)=(.+)$/.exec(a); if (m) opts[m[1]] = isNaN(+m[2]) ? m[2] : +m[2]; }
  ts = Date.now(); const b = await R.build(stems, samples, an, opts); stage("build", ts); lap(`built: ${b.meta.N_BARS} bars, ${b.meta.hits.length} kicks, ${b.meta.key}`);
  ts = Date.now(); const m = await R.master(b.layers, b.meta, b.masks); stage("master", ts); lap(`mastered: ${m.lufs.toFixed(2)} LUFS`);
  fs.writeFileSync(out, Buffer.from(R.encodeWav(m.out)));
  const NAMES = R.NAMES;
  fs.writeFileSync(out.replace(/\.wav$/, ".json"), JSON.stringify({
    analysis: { bpm: an.bpm, b0: an.b0, nBars: an.nBars, key: an.key, chords: an.chords.map(c => NAMES[c.root] + (c.minor ? "m" : "")), bassPc: an.bassPc, ev: Object.fromEntries(Object.entries(an.ev).map(([k, v]) => [k, Array.from(v)])), warp: an.warp, semis: opts.semis ?? R.DEFAULTS.semis },
    mode: b.meta.mode || "continuous", blocks: b.meta.blocks, OB0: b.meta.OB0, BEAT: b.meta.BEAT, key: b.meta.key,
    hits: b.meta.hits.map(h => [h.t, NAMES[h.pc], h.long, h.kind]), score: b.meta.score, fills: b.meta.plan.map(p => !!p.fill),
    chop: b.meta.chop, times, lufs: m.lufs,
  }));
  fs.writeFileSync(out.replace(/\.wav$/, ".mid"), Buffer.from(R.encodeMidi(b.meta)));
  if (args.includes("--layers")) for (const [k, L] of Object.entries(b.layers)) fs.writeFileSync(out.replace(/\.wav$/, `_${k}.wav`), Buffer.from(R.encodeWav(L)));
  lap("written " + out);
})().catch(e => { console.error(e); process.exit(1); });
