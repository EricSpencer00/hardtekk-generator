/* tools/stretch_check.js — objective checks of the engine's time stretcher (WSOLA, onsets locked).
 *
 *   node tools/stretch_check.js [vocals.wav t0 t1]     # optional: a sung note from a real vocal stem
 *
 * For each ratio r prints: round-trip (r then 1/r) log-spectral distance on a tone with vibrato and on
 * drum-like bursts; transient timing error and smear (10-90% energy rise) of the bursts, with and without
 * onset locking; pitch drift of the stretched tone (and the sung note) in cents.
 */
const fs = require("fs");
const R = require("../remix-engine.js"), SR = R.SR;

const tone = (sec, f0) => { const n = Math.round(sec * SR), y = new Float32Array(n); let ph = 0; for (let i = 0; i < n; i++) { const t = i / SR, f = f0 * Math.pow(2, 0.2 / 12 * Math.sin(2 * Math.PI * 5.5 * t)); ph += 2 * Math.PI * f / SR; for (let h = 1; h <= 8; h++) y[i] += Math.sin(h * ph) / h * 0.15; } return y; };
const bursts = sec => { const n = Math.round(sec * SR), y = new Float32Array(n), on = []; let s = 7; const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
  for (let t = 0.1; t < sec - 0.2; t += 0.21 + 0.1 * rnd()) { const i0 = Math.round(t * SR); on.push(i0); for (let i = 0; i < 0.06 * SR; i++) y[i0 + i] += (rnd() * 2 - 1) * Math.exp(-i / (0.008 * SR)) * 0.8; } return { y, on }; };
function lsd(a, b) {                                   // mean log-spectral distance (dB) over frames with energy
  const N = 2048, H = 512, w = Float64Array.from({ length: N }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N)), spec = x => { const out = []; for (let s = 0; s + N <= x.length; s += H) { const re = new Float64Array(N), im = new Float64Array(N); for (let i = 0; i < N; i++) re[i] = x[s + i] * w[i]; R.fft(re, im); out.push(Array.from({ length: N / 2 }, (_, k) => re[k] * re[k] + im[k] * im[k])); } return out; };
  const A = spec(a), B = spec(b), n = Math.min(A.length, B.length); let d = 0, m = 0;
  for (let f = 0; f < n; f++) { const e = A[f].reduce((x, y) => x + y, 0); if (e < 1e-3) continue; let s = 0; for (let k = 0; k < N / 2; k++) s += (10 * Math.log10((A[f][k] + 1e-9) / (B[f][k] + 1e-9))) ** 2; d += Math.sqrt(s / (N / 2)); m++; }
  return d / (m || 1);
}
function rise(y, c) {                                  // onset (steepest rise of a 1 ms envelope within 25 ms of sample c) and its 10-90% energy rise time
  const a = Math.max(45, c - Math.round(0.025 * SR)), e = Math.min(y.length, c + Math.round(0.025 * SR)), en = i => { let s = 0; for (let j = i - 44; j < i; j++) s += y[j] * y[j]; return s; };
  let bi = a, bd = -1; for (let i = a + 1; i < e; i++) { const d = en(i) - en(i - 1); if (d > bd) { bd = d; bi = i; } }
  const s0 = bi - Math.round(0.005 * SR), s1 = Math.min(y.length, bi + Math.round(0.045 * SR)), cum = []; let s = 0;
  for (let i = s0; i < s1; i++) { s += y[i] * y[i]; cum.push(s); }
  return { on: bi / SR, smear: (cum.findIndex(v => v >= 0.9 * s) - cum.findIndex(v => v >= 0.1 * s)) / SR };
}
const f0 = y => { const tr = R.yinTrack(y, SR, 60, 1000, 1024, 128), f = []; for (let i = 0; i < tr.t.length; i++) if (tr.voiced[i]) f.push(Math.log2(tr.f0[i])); f.sort((a, b) => a - b); return Math.pow(2, f[f.length >> 1]); };
const st = x => [x, Float32Array.from(x)];

const args = process.argv.slice(2);
let sung = null;
if (args[0]) { const b = fs.readFileSync(args[0]), v = new DataView(b.buffer, b.byteOffset); let p = 12; while (b.toString("ascii", p, p + 4) !== "data") p += 8 + v.getUint32(p + 4, true); const ch = v.getUint16(22, true), a = Math.round(+args[1] * SR), e = Math.round(+args[2] * SR); sung = new Float32Array(e - a); for (let i = 0; i < sung.length; i++) { let s = 0; for (let c = 0; c < ch; c++) s += v.getInt16(p + 8 + ((a + i) * ch + c) * 2, true) / 32768; sung[i] = s / ch; } }
const T = tone(3, 220), B = bursts(4), fT = f0(T), fS = sung && f0(sung);
console.log("ratio  roundtrip-LSD dB (tone / bursts)  onset error ms median/max (locked | not)  smear ms (orig / locked / not)  pitch drift cents (tone / sung)");
for (const r of [0.75, 0.88, 1.12, 1.33]) {
  const lsdT = lsd(T, R.stretch(R.stretch(st(T), r)[0] && R.stretch(st(T), r), 1 / r)[0]);
  const on = B.on, back = (y, locks) => R.stretch(y, 1 / r, locks.map(c => Math.round(c * r)))[0];
  const yl = R.stretch(st(B.y), r, on), yn = R.stretch(st(B.y), r, []);
  const lsdB = lsd(B.y, back(yl, on));
  const meas = y => on.map(c => { const q = rise(y[0], Math.round(c * r)), o = rise(B.y, c); return { err: Math.abs(q.on - (o.on * r)) * 1000, smear: q.smear * 1000, o: o.smear * 1000 }; });
  const L = meas(yl), U = meas(yn), med = a => a.slice().sort((x, y) => x - y)[a.length >> 1], mx = a => Math.max(...a);
  const cents = y => 1200 * Math.log2(f0(y) / fT), cs = fS ? (1200 * Math.log2(f0(R.stretch(st(sung), r)[0]) / fS)).toFixed(1) : "-";
  console.log(`${r.toFixed(2)}   ${lsdT.toFixed(2)} / ${lsdB.toFixed(2)}                      ${med(L.map(x => x.err)).toFixed(2)}/${mx(L.map(x => x.err)).toFixed(2)} | ${med(U.map(x => x.err)).toFixed(2)}/${mx(U.map(x => x.err)).toFixed(2)}             ${med(L.map(x => x.o)).toFixed(1)} / ${med(L.map(x => x.smear)).toFixed(1)} / ${med(U.map(x => x.smear)).toFixed(1)}                ${cents(R.stretch(st(T), r)[0]).toFixed(1)} / ${cs}`);
}
