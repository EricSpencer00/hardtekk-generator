/* remix-engine.js — the v10 hardtekk remix pipeline, in plain JS.
 *
 * Port of the Python pipeline that produced "X with the boot up" v10 B:
 *   separate -> speed the whole song up (+4 st) -> beat grid -> per-beat map of
 *   what plays where -> chords and key -> arrangement from the song's own
 *   4-bar blocks -> hardtekk kit (sustained kick tails carry the bassline, in
 *   key) -> mix buses. The master chain lives here too, for the final render.
 *
 * Runs in a Web Worker (remix-worker.js) and in Node (tools/remix_node.js).
 * Everything is offline DSP on Float32Arrays; stereo is [L, R].
 */
(function (root) {
"use strict";
const SR = 44100;
const NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const LOG = { fn: () => {} };
const log = (...a) => LOG.fn(a.join(" "));

/* ------------------------------------------------------------ basics */
const zeros = n => new Float32Array(Math.max(0, Math.floor(n)));
const st2 = n => [zeros(n), zeros(n)];
function mono(x) { const o = zeros(x[0].length); for (let i = 0; i < o.length; i++) o[i] = 0.5 * (x[0][i] + x[1][i]); return o; }
function rmsDb(a, i0 = 0, i1 = a.length) {
  let s = 0; for (let i = i0; i < i1; i++) s += a[i] * a[i];
  return 10 * Math.log10(s / Math.max(1, i1 - i0) + 1e-24);
}
function peakAbs(a) { let m = 0; for (let i = 0; i < a.length; i++) { const v = Math.abs(a[i]); if (v > m) m = v; } return m; }
const hzToMidi = f => 69 + 12 * Math.log2(f / 440);
const midiToHz = m => 440 * Math.pow(2, (m - 69) / 12);

/* ------------------------------------------------------------ filters */
function rbj(type, f, q, sr = SR, gainDb = 0) {
  f = Math.min(Math.max(f, 1), sr * 0.45);
  const w = 2 * Math.PI * f / sr, cw = Math.cos(w), sw = Math.sin(w), al = sw / (2 * q), A = Math.pow(10, gainDb / 40);
  let b0, b1, b2, a0, a1, a2;
  if (type === "low") { b0 = (1 - cw) / 2; b1 = 1 - cw; b2 = b0; a0 = 1 + al; a1 = -2 * cw; a2 = 1 - al; }
  else if (type === "high") { b0 = (1 + cw) / 2; b1 = -(1 + cw); b2 = b0; a0 = 1 + al; a1 = -2 * cw; a2 = 1 - al; }
  else if (type === "peak") { b0 = 1 + al * A; b1 = -2 * cw; b2 = 1 - al * A; a0 = 1 + al / A; a1 = -2 * cw; a2 = 1 - al / A; }
  else if (type === "highshelf") {
    const s = 2 * Math.sqrt(A) * al;
    b0 = A * ((A + 1) + (A - 1) * cw + s); b1 = -2 * A * ((A - 1) + (A + 1) * cw); b2 = A * ((A + 1) + (A - 1) * cw - s);
    a0 = (A + 1) - (A - 1) * cw + s; a1 = 2 * ((A - 1) - (A + 1) * cw); a2 = (A + 1) - (A - 1) * cw - s;
  } else if (type === "lowshelf") {
    const s = 2 * Math.sqrt(A) * al;
    b0 = A * ((A + 1) - (A - 1) * cw + s); b1 = 2 * A * ((A - 1) - (A + 1) * cw); b2 = A * ((A + 1) - (A - 1) * cw - s);
    a0 = (A + 1) + (A - 1) * cw + s; a1 = -2 * ((A - 1) + (A + 1) * cw); a2 = (A + 1) + (A - 1) * cw - s;
  } else throw new Error("filter type " + type);
  return { b0: b0 / a0, b1: b1 / a0, b2: b2 / a0, a1: a1 / a0, a2: a2 / a0 };
}
function biquad(x, c) {               // in place, direct form I in float64
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let i = 0; i < x.length; i++) {
    const xn = x[i], yn = c.b0 * xn + c.b1 * x1 + c.b2 * x2 - c.a1 * y1 - c.a2 * y2;
    x2 = x1; x1 = xn; y2 = y1; y1 = yn; x[i] = yn;
  }
  return x;
}
const BUTTER_Q = { 2: [Math.SQRT1_2], 4: [0.5411961, 1.3065630] };
/** Butterworth low/high-pass of `order` (2 or 4), causal, like scipy sosfilt(butter(...)). */
function filt(x, type, f, order = 4) {
  const y = Float32Array.from(x);
  for (const q of BUTTER_Q[order]) biquad(y, rbj(type, f, q));
  return y;
}
function filtfilt(x, type, f, order = 4) {
  const y = filt(x, type, f, order); y.reverse();
  const z = filt(y, type, f, order); z.reverse(); return z;
}
const band = (x, lo, hi, order = 4) => filt(filt(x, "high", lo, order), "low", hi, order);
const filt2 = (s, type, f, order = 4) => [filt(s[0], type, f, order), filt(s[1], type, f, order)];

/* ------------------------------------------------------------ FFT */
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit;
    if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang), h = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < h; k++) {
        const a = i + k, b = a + h, tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
      }
    }
  }
}
function ifft(re, im) { for (let i = 0; i < im.length; i++) im[i] = -im[i]; fft(re, im); const n = re.length; for (let i = 0; i < n; i++) { re[i] /= n; im[i] = -im[i] / n; } }
const hann = n => { const w = new Float64Array(n); for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / n); return w; };

/* ------------------------------------------------------------ resampling */
// Windowed-sinc with a lookup table. ratio = in_rate/out_rate.
const SINC_ZC = 16, SINC_RES = 512;
const SINC_TAB = (() => {
  const n = SINC_ZC * SINC_RES + 1, t = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = i / SINC_RES, w = 0.42 + 0.5 * Math.cos(Math.PI * x / SINC_ZC) + 0.08 * Math.cos(2 * Math.PI * x / SINC_ZC);
    t[i] = (x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x)) * w;
  }
  return t;
})();
/** x read at fractional position t (samples), band-limited to `cut` of Nyquist. */
function sincAt(x, t, cut) {
  const half = SINC_ZC / cut, a = Math.max(0, Math.ceil(t - half)), b = Math.min(x.length - 1, Math.floor(t + half));
  let s = 0;
  for (let j = a; j <= b; j++) {
    const d = Math.abs(t - j) * cut * SINC_RES, k = d | 0;
    if (k >= SINC_TAB.length - 1) continue;
    const fr = d - k; s += x[j] * (SINC_TAB[k] + (SINC_TAB[k + 1] - SINC_TAB[k]) * fr);
  }
  return s * cut;
}
function resample(x, fromRate, toRate) {
  const ratio = fromRate / toRate, cut = Math.min(1, 1 / ratio) * 0.97;
  const n = Math.floor(x.length / ratio), y = zeros(n);
  for (let i = 0; i < n; i++) y[i] = sincAt(x, i * ratio, cut);
  return y;
}
/** Varispeed onto a rigid grid: source time src[k] (s) lands at dst[k]; linear in between and beyond. */
function warpTime(x, src, dst) {
  const K = src.length, edge = (src[K - 1] - src[0]) / (dst[K - 1] - dst[0]);
  const n = Math.max(1, Math.floor((dst[K - 1] + (x.length / SR - src[K - 1]) / edge) * SR)), y = zeros(n);
  let k = 0;
  for (let i = 0; i < n; i++) {
    const u = i / SR; while (k < K - 2 && u >= dst[k + 1]) k++;
    const r = u < dst[0] || u >= dst[K - 1] ? edge : (src[k + 1] - src[k]) / (dst[k + 1] - dst[k]);
    const t = u < dst[0] ? src[0] + (u - dst[0]) * r : u >= dst[K - 1] ? src[K - 1] + (u - dst[K - 1]) * r : src[k] + (u - dst[k]) * r;
    y[i] = t >= 0 ? sincAt(x, t * SR, Math.min(1, 1 / r) * 0.97) : 0;
  }
  return y;
}
/** Varispeed: pitch and length change together, like a sampler. */
const repitch = (x, semis) => Math.abs(semis) < 1e-4 ? Float32Array.from(x) : resample(x, SR * Math.pow(2, semis / 12), SR);
const repitch2 = (s, semis) => [repitch(s[0], semis), repitch(s[1], semis)];

/* ------------------------------------------------------------ time stretch (pitch kept) */
/** WSOLA: stereo x stretched to `alpha` x its length, pitch unchanged. Hann frames of 35 ms at 50% overlap;
 *  each reads the input near its ideal place, moved by up to 11.6 ms so it continues the previous frame's
 *  waveform. Onsets (input samples) stay whole and exactly on time: the frame nearest each one reads it
 *  at onset x alpha, frames still holding it continue that read, earlier frames stop at it and later
 *  frames start there. The result's .plan replays the same reads on another signal of the same length. */
function stretch(x, alpha, onsets = [], plan = null) {
  const N = 1536, HS = N / 2, L = x[0].length, w = hann(N);
  if (!plan) {
    const Lo = Math.max(1, Math.round(L * alpha)), K = Math.ceil(Lo / HS) + 1, D = 4, TOL = 128, M = HS / D;
    const m = mono(x), nd = Math.ceil(L / D), d = new Float32Array(nd);
    for (let i = 0; i < L; i++) d[(i / D) | 0] += m[i];
    const at = i => i >= 0 && i < nd ? d[i] : 0, xm = i => i >= 0 && i < L ? m[i] : 0;
    const lock = new Map();
    for (const c of onsets) { const k = Math.round(c * alpha / HS); if (k >= 0 && k < K && !lock.has(k)) lock.set(k, c); }
    const p = new Int32Array(K), lo = new Int32Array(K), hi = new Int32Array(K).fill(L), keys = [...lock.keys()].sort((a, b) => a - b);
    let held = -1;                                                        // the last locked onset
    for (let k = 0; k < K; k++) {
      const o = k * HS, tau = Math.round(o / alpha);
      if (lock.has(k)) { held = lock.get(k); p[k] = Math.round(held + o - held * alpha); continue; }
      if (k === 0) { p[k] = tau; continue; }
      if (held >= 0 && p[k - 1] + HS - N / 2 < held) { p[k] = p[k - 1] + HS; continue; }
      const ref = Math.floor((p[k - 1] + HS - N / 2) / D), c0 = Math.floor((tau - N / 2) / D);
      let best = 0, bv = -Infinity;
      for (let dl = -TOL; dl <= TOL; dl++) {
        let c = 0, e = 1e-9; for (let j = 0; j < M; j++) { const v = at(c0 + dl + j); c += v * at(ref + j); e += v * v; }
        if (c / Math.sqrt(e) > bv) { bv = c / Math.sqrt(e); best = dl; }
      }
      let pk = tau + best * D; bv = -Infinity;                            // refine to the sample
      for (let f = -D; f <= D; f++) {
        let c = 0, e = 1e-9; for (let j = 0; j < HS; j += 2) { const v = xm(tau + best * D + f - N / 2 + j); c += v * xm(p[k - 1] + HS - N / 2 + j); e += v * v; }
        if (c / Math.sqrt(e) > bv) { bv = c / Math.sqrt(e); pk = tau + best * D + f; }
      }
      p[k] = pk;
    }
    for (let k = 0, j = 0; k < K; k++) {                                  // reads never cross an onset that is not theirs
      while (j < keys.length && keys[j] < k) j++;
      if (j > 0) lo[k] = lock.get(keys[j - 1]);
      const nx = keys[j] === k ? j + 1 : j; if (nx < keys.length) hi[k] = lock.get(keys[nx]);
    }
    plan = { p, lo, hi, K, Lo };
  }
  const { p, lo, hi, K, Lo } = plan, y = st2(Lo), ws = new Float32Array(Lo);
  for (let k = 0; k < K; k++) for (let j = 0; j < N; j++) {
    const oi = k * HS - N / 2 + j, ii = p[k] - N / 2 + j;
    if (oi < 0 || oi >= Lo || ii < lo[k] || ii >= hi[k] || ii < 0 || ii >= L) continue;
    y[0][oi] += x[0][ii] * w[j]; y[1][oi] += x[1][ii] * w[j]; ws[oi] += w[j];
  }
  for (let i = 0; i < Lo; i++) { const g = 1 / Math.max(ws[i], 0.1); y[0][i] *= g; y[1][i] *= g; }
  y.plan = plan;
  return y;
}

/* ------------------------------------------------------------ separation (no stems given) */
/** HPSS + centre extraction. Returns stems like Demucs would, approximately:
 *  drums = percussive, bass = harmonic below 120 Hz, vocals = centred harmonic
 *  150 Hz - 6 kHz, other = the rest of the harmonic part. `probes` ({name: stereo}, for measuring leakage) get
 *  the same masks: probe[name].vocals is what of that signal ends up in the vocal stem. */
async function separate(mix, progress = () => {}, probes = {}) {
  const N = 4096, HOP = 1024, B = N / 2 + 1, w = hann(N), L = mix[0].length;
  const F = Math.ceil(L / HOP) + 1;
  const mag = new Float32Array(F * B);
  const re = new Float64Array(N), im = new Float64Array(N);
  for (let f = 0; f < F; f++) {
    const s = f * HOP - N / 2;
    for (let i = 0; i < N; i++) { const j = s + i; re[i] = j >= 0 && j < L ? 0.5 * (mix[0][j] + mix[1][j]) * w[i] : 0; im[i] = 0; }
    fft(re, im);
    for (let k = 0; k < B; k++) mag[f * B + k] = Math.hypot(re[k], im[k]);
    if (f % 400 === 0) { progress(0.25 * f / F); await tick(); }
  }
  const out = { h: st2(L), v: st2(L) }, norm = new Float64Array(L + N);
  const reL = new Float64Array(N), imL = new Float64Array(N), reR = new Float64Array(N), imR = new Float64Array(N);
  const tmp = new Float32Array(17), H = new Float32Array(B), P = new Float32Array(B);
  const kLo = Math.round(150 * N / SR), kHi = Math.round(6000 * N / SR);
  const PR = Object.entries(probes).map(([name, x]) => ({ name, x, h: st2(L), v: st2(L), mh: new Float64Array(B), mv: new Float64Array(B) }));
  for (let f = 0; f < F; f++) {
    for (let k = 0; k < B; k++) {                       // harmonic: median across time
      let n = 0; for (let d = -8; d <= 8; d++) { const g = f + d; if (g >= 0 && g < F) tmp[n++] = mag[g * B + k]; }
      H[k] = median(tmp, n);
    }
    for (let k = 0; k < B; k++) {                       // percussive: median across frequency
      let n = 0; for (let d = -8; d <= 8; d++) { const q = k + d; if (q >= 0 && q < B) tmp[n++] = mag[f * B + q]; }
      P[k] = median(tmp, n);
    }
    const s = f * HOP - N / 2;
    for (let i = 0; i < N; i++) { const j = s + i, ok = j >= 0 && j < L; reL[i] = ok ? mix[0][j] * w[i] : 0; reR[i] = ok ? mix[1][j] * w[i] : 0; imL[i] = 0; imR[i] = 0; }
    fft(reL, imL); fft(reR, imR);
    const vre = [new Float64Array(N), new Float64Array(N)], vim = [new Float64Array(N), new Float64Array(N)];
    for (let k = 0; k < B; k++) {
      const h2 = H[k] * H[k], p2 = P[k] * P[k], mh = h2 / (h2 + p2 + 1e-18);
      const lr = reL[k] * mh, li = imL[k] * mh, rr = reR[k] * mh, ri = imR[k] * mh;
      // centre mask on the harmonic part: how much of it is the same in both channels
      let mc = 0;
      if (k >= kLo && k <= kHi) {
        const mr = 0.5 * (lr + rr), mi = 0.5 * (li + ri), sr_ = 0.5 * (lr - rr), si = 0.5 * (li - ri);
        const m2 = mr * mr + mi * mi, s2 = sr_ * sr_ + si * si; mc = m2 / (m2 + 4 * s2 + 1e-18); mc = mc * mc;
      }
      reL[k] = lr; imL[k] = li; reR[k] = rr; imR[k] = ri;
      vre[0][k] = lr * mc; vim[0][k] = li * mc; vre[1][k] = rr * mc; vim[1][k] = ri * mc;
      for (const pr of PR) { pr.mh[k] = mh; pr.mv[k] = mh * mc; }
    }
    for (const pr of PR) for (let c = 0; c < 2; c++) {                  // the same masks on each probe
      const r = new Float64Array(N), i_ = new Float64Array(N);
      for (let i = 0; i < N; i++) { const j = s + i; r[i] = j >= 0 && j < L ? pr.x[c][j] * w[i] : 0; }
      fft(r, i_);
      for (const [m, dst] of [[pr.mh, pr.h], [pr.mv, pr.v]]) {
        const a = new Float64Array(N), b = new Float64Array(N);
        for (let k = 0; k < B; k++) { a[k] = r[k] * m[k]; b[k] = i_[k] * m[k]; }
        mirror(a, b, N); ifft(a, b);
        for (let i = 0; i < N; i++) { const j = s + i; if (j >= 0 && j < L) dst[c][j] += a[i] * w[i]; }
      }
    }
    for (const [r, i_] of [[reL, imL], [reR, imR], [vre[0], vim[0]], [vre[1], vim[1]]]) mirror(r, i_, N);
    ifft(reL, imL); ifft(reR, imR); ifft(vre[0], vim[0]); ifft(vre[1], vim[1]);
    for (let i = 0; i < N; i++) {
      const j = s + i; if (j < 0 || j >= L) continue;
      out.h[0][j] += reL[i] * w[i]; out.h[1][j] += reR[i] * w[i];
      out.v[0][j] += vre[0][i] * w[i]; out.v[1][j] += vre[1][i] * w[i];
      norm[j] += w[i] * w[i];
    }
    if (f % 200 === 0) { progress(0.25 + 0.75 * f / F); await tick(); }
  }
  for (let j = 0; j < L; j++) { const g = 1 / Math.max(norm[j], 1e-3); for (const s of [out.h, out.v, ...PR.flatMap(p => [p.h, p.v])]) { s[0][j] *= g; s[1][j] *= g; } }
  const probe = Object.fromEntries(PR.map(p => [p.name, { vocals: p.v, other: p.h.map((c, k) => { const lo = filtfilt(c, "low", 120, 4); return Float32Array.from(c, (x, j) => x - lo[j] - p.v[k][j]); }) }]));
  // zero-phase split: subtracting a phase-shifted low-pass would double the bass instead of removing it
  const drums = [zeros(L), zeros(L)], bass = [filtfilt(out.h[0], "low", 120, 4), filtfilt(out.h[1], "low", 120, 4)];
  const other = [zeros(L), zeros(L)];
  for (let c = 0; c < 2; c++) for (let j = 0; j < L; j++) {
    drums[c][j] = mix[c][j] - out.h[c][j];
    other[c][j] = out.h[c][j] - bass[c][j] - out.v[c][j];
  }
  return { drums, bass, other, vocals: out.v, inEngine: true, ...(PR.length ? { probe } : {}) };
}
/** Stems from a neural separator: htdemucs (drums, bass, other, vocals) run on 7.8 s chunks overlapped by a
 *  quarter and crossfaded. The graph has no FFT: the engine makes htdemucs's spectrogram of each chunk (htSpec),
 *  `run(x, mag)` (the caller owns the runtime) returns the model's frequency branch (complex spectrogram per
 *  source) and time branch, and the stem is time + inverse spectrogram (htISpec). x: Float32Array [2 x N];
 *  mag: [4 x 2048 x 336] (L re, L im, R re, R im); freq: [4 x 4 x 2048 x 336]; time: [4 x 2 x N]. */
async function separateModel(mix, run, progress = () => {}, N = 343980) {
  const L = mix[0].length, ov = N >> 2, stride = N - ov, n = Math.max(1, Math.ceil(L / stride));
  const out = Array.from({ length: 4 }, () => st2(L)), ws = new Float32Array(L), w = new Float32Array(N);
  for (let i = 0; i < N; i++) w[i] = Math.min(1, i / (ov - 1), (N - 1 - i) / (ov - 1));
  for (let c = 0; c < n; c++) {
    const a = c * stride, e = Math.min(a + N, L), x = new Float32Array(2 * N), mag = new Float32Array(4 * HT.F * HT.T);
    for (let ch = 0; ch < 2; ch++) { x.set(mix[ch].subarray(a, e), ch * N); htSpec(x.subarray(ch * N, (ch + 1) * N), mag, ch); }
    const { freq, time } = await run(x, mag);
    for (let k = 0; k < 4; k++) for (let ch = 0; ch < 2; ch++) {
      const y = htISpec(freq, k, ch, N), o = out[k][ch], tb = (2 * k + ch) * N;
      for (let i = 0; i < e - a; i++) o[a + i] += (y[i] + time[tb + i]) * w[i];
    }
    for (let i = 0; i < e - a; i++) ws[a + i] += w[i];
    progress((c + 1) / n); await tick();
  }
  for (const st of out) for (const ch of st) for (let i = 0; i < L; i++) ch[i] /= Math.max(ws[i], 1e-8);
  return { drums: out[0], bass: out[1], other: out[2], vocals: out[3] };
}
/* htdemucs's spectrogram: 4096-point periodic Hann, hop 1024, scaled by 1/64 (torch "normalized"), the chunk
 * reflect-padded by 1536 (+ the rest of the last hop) and centred; frames 2..337, bins 0..2047. */
const HT = { NF: 4096, H: 1024, F: 2048, T: 336, pad: 1536, w: hann(4096) };
function htSpec(x, mag, ch) {
  const { NF, H, F, T, pad, w } = HT, L = x.length, re = new Float64Array(NF), im = new Float64Array(NF);
  const at = i => { while (i < 0 || i >= L) i = i < 0 ? -i : 2 * (L - 1) - i; return x[i]; };
  for (let t = 0; t < T; t++) {
    const s0 = (t + 2) * H - NF / 2 - pad;                              // first source sample of the frame
    for (let n = 0; n < NF; n++) { re[n] = at(s0 + n) * w[n] / 64; im[n] = 0; }
    fft(re, im);
    for (let f = 0; f < F; f++) { mag[((2 * ch) * F + f) * T + t] = re[f]; mag[((2 * ch + 1) * F + f) * T + t] = im[f]; }
  }
}
/** Inverse of htSpec for source k, channel ch of the model's frequency branch: overlap-add of the frames,
 *  divided by the summed squared window, as torch.istft. Returns the chunk's N samples. */
function htISpec(freq, k, ch, N) {
  const { NF, H, F, T, pad, w } = HT, re = new Float64Array(NF), im = new Float64Array(NF), o = (T + 3) * H + NF;
  const y = new Float64Array(o), den = new Float64Array(o), b = k * 4 + 2 * ch;
  for (let t = 0; t < T; t++) {
    re.fill(0); im.fill(0);
    for (let f = 0; f < F; f++) { re[f] = freq[(b * F + f) * T + t] * 64; im[f] = freq[((b + 1) * F + f) * T + t] * 64; }
    for (let f = 1; f < F; f++) { re[NF - f] = re[f]; im[NF - f] = -im[f]; }
    ifft(re, im);
    for (let n = 0; n < NF; n++) y[t * H + n] += re[n] * w[n];
  }
  for (let t = -2; t < T + 2; t++) for (let n = 0; n < NF; n++) { const j = t * H + n; if (j >= 0 && j < o) den[j] += w[n] * w[n]; }
  const out = new Float32Array(N), s0 = NF / 2 + pad - 2 * H;           // chunk sample 0 in frame-0-of-T time
  for (let i = 0; i < N; i++) { const j = i + s0; out[i] = den[j] > 1e-8 ? y[j] / den[j] : 0; }
  return out;
}
function median(a, n) {
  const b = a.subarray(0, n); for (let i = 1; i < n; i++) { const v = b[i]; let j = i - 1; while (j >= 0 && b[j] > v) { b[j + 1] = b[j]; j--; } b[j + 1] = v; }
  return b[n >> 1];
}
function mirror(re, im, N) { for (let k = 1; k < N / 2; k++) { re[N - k] = re[k]; im[N - k] = -im[k]; } }
const tick = () => new Promise(r => setTimeout(r, 0));

/* ------------------------------------------------------------ pitch tracking (YIN) */
/** f0 track of a low voice. Returns {t, f0, rms, voiced} at `hop` on an 11025 Hz copy. */
function yinTrack(x, sr, fmin = 28, fmax = 120, frame = 1024, hop = 128) {
  const ds = 11025, y = sr === ds ? x : resample(x, sr, ds);
  const nF = Math.max(0, Math.floor((y.length - frame) / hop));
  const t = new Float32Array(nF), f0 = new Float32Array(nF), rms = new Float32Array(nF), voiced = new Uint8Array(nF);
  const W = frame, tmin = Math.floor(ds / fmax), tmax = Math.min(W - 1, Math.ceil(ds / fmin));
  const Nf = 2 * W, re = new Float64Array(Nf), im = new Float64Array(Nf), d = new Float64Array(tmax + 2);
  let rmax = 0;
  for (let f = 0; f < nF; f++) {
    const s = f * hop; let e = 0;
    for (let i = 0; i < Nf; i++) { re[i] = i < W ? y[s + i] : 0; im[i] = 0; }
    for (let i = 0; i < W; i++) e += y[s + i] * y[s + i];
    rms[f] = Math.sqrt(e / W); t[f] = (s + W / 2) / ds; if (rms[f] > rmax) rmax = rms[f];
    fft(re, im); for (let i = 0; i < Nf; i++) { re[i] = re[i] * re[i] + im[i] * im[i]; im[i] = 0; } ifft(re, im);
    // d(tau) = e0 + e_tau - 2 r(tau) over the first W - tau samples
    const cs = new Float64Array(W + 1); for (let i = 0; i < W; i++) cs[i + 1] = cs[i] + y[s + i] * y[s + i];
    let run = 0; d[0] = 1;
    for (let tau = 1; tau <= tmax; tau++) {
      const n = W - tau, e0 = cs[n], e1 = cs[W] - cs[tau];
      // autocorrelation from the FFT covers the full window; rescale by overlap
      const dv = e0 + e1 - 2 * re[tau]; run += dv; d[tau] = run > 0 ? dv * tau / run : 1;
    }
    let best = -1;
    for (let tau = tmin; tau < tmax; tau++) if (d[tau] < 0.2 && d[tau] < d[tau + 1]) { best = tau; break; }
    if (best < 0) { let m = 1e9; for (let tau = tmin; tau < tmax; tau++) if (d[tau] < m) { m = d[tau]; best = tau; } if (m > 0.45) best = -1; }
    if (best > 0) {
      const a = d[best - 1], b = d[best], c = d[best + 1], den = a - 2 * b + c;
      const tb = best + (den !== 0 ? 0.5 * (a - c) / den : 0);
      f0[f] = ds / tb; voiced[f] = 1;
    }
  }
  for (let f = 0; f < nF; f++) if (rms[f] < rmax * 0.05) voiced[f] = 0;
  return { t, f0, rms, voiced };
}
/** Loudness-weighted pitch class in [t0, t1) of a track, counting the notes it settles on:
 *  frames in the middle of a slide (pitch moving > 0.12 st per frame) count 10%. Null if nothing voiced. */
function trackPc(tr, t0, t1, shift = 0) {
  const h = new Float64Array(12); let n = 0;
  for (let i = 0; i < tr.t.length; i++) {
    if (tr.t[i] < t0 || tr.t[i] >= t1 || !tr.voiced[i]) continue;
    const m = hzToMidi(tr.f0[i]), pc = ((Math.round(m) % 12) + 12) % 12;
    const prev = i > 0 && tr.voiced[i - 1] ? Math.abs(m - hzToMidi(tr.f0[i - 1])) : 0, next = i + 1 < tr.t.length && tr.voiced[i + 1] ? Math.abs(hzToMidi(tr.f0[i + 1]) - m) : 0;
    h[pc] += tr.rms[i] * (Math.max(prev, next) > 0.12 ? 0.1 : 1); n++;
  }
  if (n < 4) return null;
  let b = 0; for (let i = 1; i < 12; i++) if (h[i] > h[b]) b = i;
  return (b + shift + 120) % 12;
}

/** The bassline as notes: settled frames only (slides and blips < 40 ms dropped), snapped to the
 *  song's semitones. Gaps under 150 ms hold the previous note. Kick and dark 808 both read this. */
function noteTimeline(tr, cents = 0) {
  const notes = [], dt = tr.t.length > 1 ? tr.t[1] - tr.t[0] : 0.006;
  let cur = null;
  for (let i = 0; i < tr.t.length; i++) {
    const steady = tr.voiced[i] && (i === 0 || !tr.voiced[i - 1] || Math.abs(hzToMidi(tr.f0[i]) - hzToMidi(tr.f0[i - 1])) < 0.12)
                   && (i + 1 >= tr.t.length || !tr.voiced[i + 1] || Math.abs(hzToMidi(tr.f0[i + 1]) - hzToMidi(tr.f0[i])) < 0.12);
    if (!steady) continue;
    const m = Math.round(hzToMidi(tr.f0[i]) - cents / 100);
    if (cur && cur.m === m && tr.t[i] - cur.e < 0.15) cur.e = tr.t[i] + dt;
    else { if (cur) notes.push(cur); cur = { m, s: tr.t[i], e: tr.t[i] + dt }; }
  }
  if (cur) notes.push(cur);
  const kept = notes.filter(n => n.e - n.s >= 0.04);
  for (let i = 0; i + 1 < kept.length; i++) if (kept[i + 1].s - kept[i].e < 0.15) kept[i].e = kept[i + 1].s;   // hold through short gaps
  return kept;
}
/** Pitch class of the timeline note covering most of [t0, t1); null if none. */
function noteAt(notes, t0, t1) {
  let best = null, bo = 0;
  for (const n of notes) { if (n.e <= t0) continue; if (n.s >= t1) break; const o = Math.min(n.e, t1) - Math.max(n.s, t0); if (o > bo) { bo = o; best = n; } }
  return best ? ((best.m % 12) + 12) % 12 : null;
}

/* ------------------------------------------------------------ analysis */
function onsetFlux(x, hop = 128, n = 1024) {
  const re = new Float64Array(n), im = new Float64Array(n), w = hann(n), F = Math.floor((x.length - n) / hop);
  const prev = new Float64Array(n / 2 + 1), o = new Float32Array(Math.max(F, 0));
  for (let f = 0; f < F; f++) {
    for (let i = 0; i < n; i++) { re[i] = x[f * hop + i] * w[i]; im[i] = 0; }
    fft(re, im); let s = 0;
    for (let k = 0; k <= n / 2; k++) { const m = Math.log1p(100 * Math.hypot(re[k], im[k])); const d = m - prev[k]; if (d > 0) s += d; prev[k] = m; }
    o[f] = s;
  }
  return { o, hop, n };
}
/** Onsets: each spectral-flux peak moved to the steepest rise of the envelope in the 40 ms before it.
 *  Returns times (s), strengths, and the flux itself. */
function onsetTimes(x, minGap = 0.05, thr = 0.2) {
  const { o } = onsetFlux(x, 128, 1024), pk = peaks(o, Math.round(minGap * SR / 128), thr);
  const env = smooth(Float32Array.from(filt(x, "high", 60, 2), Math.abs), Math.round(0.002 * SR));
  const t = pk.map(i => {
    const c = i * 128 + 512, a = Math.max(1, c - Math.round(0.04 * SR)), e = Math.min(env.length - 1, c + Math.round(0.01 * SR));
    let bi = a, bd = -1; for (let j = a; j < e; j++) { const d = env[j] - env[j - 1]; if (d > bd) { bd = d; bi = j; } }
    return bi / SR;
  });
  return { t, s: pk.map(i => o[i]), o };
}
function peaks(o, minDist, thrRel = 0.3) {
  let mx = 0; for (let i = 0; i < o.length; i++) if (o[i] > mx) mx = o[i];
  const out = [];
  for (let i = 1; i < o.length - 1; i++) if (o[i] > o[i - 1] && o[i] >= o[i + 1] && o[i] > thrRel * mx) {
    if (out.length && i - out[out.length - 1] < minDist) { if (o[i] > o[out[out.length - 1]]) out[out.length - 1] = i; }
    else out.push(i);
  }
  return out;
}
function circFit(times, weights, period) {
  let c = 0, s = 0, W = 0;
  for (let i = 0; i < times.length; i++) { const a = 2 * Math.PI * times[i] / period, w = weights ? weights[i] : 1; c += w * Math.cos(a); s += w * Math.sin(a); W += w; }
  const R = Math.hypot(c, s) / (W || 1); let ph = Math.atan2(s, c); if (ph < 0) ph += 2 * Math.PI;
  return { R, phase: ph / (2 * Math.PI) * period };
}
/** Tempo and beat phase from the drums: kick onsets for tempo, all drum onsets for phase. */
function beatGrid(drums) {
  // tempo: where all the drum onsets cluster on the beat and the 8th, 100-200 BPM (the kick alone, often
  // syncopated, locked to 4:3 and 5:4 of the tempo on 3 of 10 test songs)
  const { t: ft, s: fw, o } = onsetTimes(drums, 0.05, 0.2);
  let best = { R: -1 };
  for (let bpm = 100; bpm < 200; bpm += 0.02) { const P = 60 / bpm, r = (circFit(ft, fw, P).R + circFit(ft, fw, P / 2).R) / 2; if (r > best.R) best = { R: r, bpm }; }
  const bpm = best.bpm;
  let per = 60 / bpm, ph = circFit(ft, fw, per).phase;
  // least-squares period and phase on the strong onsets near the grid
  const fwMed = Float32Array.from(fw).sort()[fw.length >> 1] || 0;
  for (let it = 0; it < 3; it++) {
    let n = 0, sk = 0, st_ = 0, skk = 0, skt = 0;
    ft.forEach((t, j) => { if (fw[j] < fwMed) return; const k = Math.round((t - ph) / per), r = t - (ph + k * per); if (Math.abs(r) > 0.03) return; n++; sk += k; st_ += t; skk += k * k; skt += k * t; });
    if (n < 16) break;
    per = (n * skt - sk * st_) / (n * skk - sk * sk); ph = (st_ - per * sk) / n; ph = ((ph % per) + per) % per;
  }
  // the song's own beats where its tempo moves (live drummer, rubato, tempo automation): dynamic
  // programming over the onset envelope (Ellis 2007), smoothed over +-4 beats against frame jitter
  const P = per * SR / 128, sd = Math.sqrt(o.reduce((s, v) => s + v * v, 0) / o.length) || 1, score = new Float64Array(o.length), back = new Int32Array(o.length).fill(-1);
  for (let t = 0; t < o.length; t++) {
    let bs = 0, bi = -1;
    for (let u = t - Math.round(2 * P); u <= t - Math.round(P / 2); u++) { if (u < 0) continue; const v = score[u] - 100 * Math.log((t - u) / P) ** 2; if (v > bs) { bs = v; bi = u; } }
    score[t] = o[t] / sd + bs; back[t] = bi;
  }
  let end = o.length - 1; for (let t = Math.max(0, o.length - Math.round(P)); t < o.length; t++) if (score[t] > score[end]) end = t;
  const raw = []; for (let t = end; t >= 0; t = back[t]) raw.unshift((t * 128 + 512) / SR);
  const beats = raw.map((_, i) => {
    let n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (let j = Math.max(0, i - 4); j <= Math.min(raw.length - 1, i + 4); j++) { n++; sx += j; sy += raw[j]; sxx += j * j; sxy += j * raw[j]; }
    const m = (n * sxy - sx * sy) / (n * sxx - sx * sx || 1); return (sy - m * sx) / n + m * i;
  });
  // evidence: how tightly the drum onsets sit on each grid (1 = every onset on a beat)
  // (beats and 8ths: hats on the off-beats count as evidence too)
  const fit = phase => { let W = 0; const z = [0, 0, 0, 0]; ft.forEach((t, j) => { const p = phase(t); if (isNaN(p)) return; W += fw[j]; for (const m of [1, 2]) { z[2 * m - 2] += fw[j] * Math.cos(2 * Math.PI * m * p); z[2 * m - 1] += fw[j] * Math.sin(2 * Math.PI * m * p); } }); return (Math.hypot(z[0], z[1]) + Math.hypot(z[2], z[3])) / (2 * W || 1); };
  const Rrigid = fit(t => (t - ph) / per);
  let k = 0; const Rbeats = fit(t => { while (k > 0 && beats[k] > t) k--; while (k + 1 < beats.length && beats[k + 1] <= t) k++; return k + 1 < beats.length && t >= beats[0] ? (t - beats[k]) / (beats[k + 1] - beats[k]) : NaN; });
  return { bpm: 60 / per, b0: ph, R: best.R, beats, Rrigid, Rbeats, nOnsets: ft.length, on: { t: ft, s: fw } };
}
/** Per-beat RMS (dB) of a signal. */
function perBeat(x, b0, beat, nb) {
  const out = new Float32Array(nb);
  for (let i = 0; i < nb; i++) { const a = Math.round((b0 + i * beat) * SR), e = Math.round((b0 + (i + 1) * beat) * SR); out[i] = rmsDb(x, Math.max(0, a), Math.min(x.length, e)); }
  return out;
}
function lane(db, drop) {
  const s = Float32Array.from(db).sort(); const ref = s[Math.floor(s.length * 0.9)];
  return Uint8Array.from(db, v => v > ref - drop ? 1 : 0);
}
/** Chroma of a mono signal per window [t0,t1): 12 bins, unit norm. */
function chromaWindows(x, wins, fmin = 32, fmax = 4200) {
  const N = 16384, re = new Float64Array(N), im = new Float64Array(N), w = hann(N);
  const kb = []; for (let k = 1; k < N / 2; k++) { const f = k * SR / N; if (f >= fmin && f <= fmax) kb.push([k, ((Math.round(hzToMidi(f)) % 12) + 12) % 12]); }
  return wins.map(([t0, t1]) => {
    const c = new Float64Array(12); const a = Math.round(t0 * SR), e = Math.round(t1 * SR);
    for (let s = a; s + N <= Math.max(e, a + N); s += N / 2) {
      for (let i = 0; i < N; i++) { const j = s + i; re[i] = j >= 0 && j < x.length ? x[j] * w[i] : 0; im[i] = 0; }
      fft(re, im);
      for (const [k, pc] of kb) c[pc] += re[k] * re[k] + im[k] * im[k];
    }
    const nrm = Math.hypot(...c) || 1; return Array.from(c, v => v / nrm);
  });
}
const KS_MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const KS_MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];
function corr(a, b) { const ma = a.reduce((s, v) => s + v, 0) / 12, mb = b.reduce((s, v) => s + v, 0) / 12; let n = 0, da = 0, db = 0; for (let i = 0; i < 12; i++) { n += (a[i] - ma) * (b[i] - mb); da += (a[i] - ma) ** 2; db += (b[i] - mb) ** 2; } return n / Math.sqrt(da * db + 1e-12); }
function detectKey(chroma) {
  let best = { r: -2 };
  for (let t = 0; t < 12; t++) for (const [prof, mode] of [[KS_MAJ, "major"], [KS_MIN, "minor"]]) {
    const rot = prof.map((_, i) => prof[(i - t + 12) % 12]); const r = corr(rot, chroma); if (r > best.r) best = { r, tonic: t, mode };
  }
  return best;
}
function triadRoot(c) {
  let best = { s: -1 };
  for (let r = 0; r < 12; r++) for (const [q, iv] of [["", [0, 4, 7]], ["m", [0, 3, 7]]]) {
    let s = 0.3 * c[r]; for (const i of iv) s += c[(r + i) % 12];
    if (s > best.s) best = { s, root: r, q };
  }
  return best;
}

/** Everything about the source we need: grid, map of parts per beat, chords, key, the 808's notes.
 *  `src` keeps the song's own beats in its own time for chop mode; continuous mode reads the stems
 *  warped onto a rigid grid (`stemsW`) when the song's tempo moves. */
async function analyze(stems, progress = () => {}) {
  let dm = mono(stems.drums), grid = beatGrid(dm), warp = null;
  const src = { beats: grid.beats, per: 60 / grid.bpm, ph: grid.b0, on: grid.on, moving: false, down: 0,
                onK: onsetTimes(filtfilt(dm, "low", 150), 0.08, 0.2), onM: onsetTimes(filtfilt(dm, "low", 5000), 0.05, 0.2) };
  // A song whose tempo moves is warped onto a rigid grid (the stems are replaced in place), but only when its
  // tracked beats stray from one fixed tempo by more than a flam (10 ms, p90) AND explain the drum onsets better
  // than one tempo by more than chance (2 / sqrt(onsets): two standard errors of the phase concentration).
  const B = grid.beats, K = B.length, perW = K > 8 ? (B[K - 1] - B[0]) / (K - 1) : 0;
  const stray = K > 8 ? Float32Array.from(B, (t, k) => Math.abs(t - (B[0] + k * perW))).sort()[Math.floor(0.9 * K)] : 0;
  log(`  beats: one tempo explains the drum onsets ${grid.Rrigid.toFixed(2)}, the tracked beats ${grid.Rbeats.toFixed(2)}; tracked beats stray ${Math.round(stray * 1000)} ms (p90) from one tempo; ${grid.nOnsets} onsets`);
  if (stray > 0.010 && grid.Rbeats - grid.Rrigid > 2 / Math.sqrt(grid.nOnsets || 1)) {
    const dst = Array.from(B, (_, k) => B[0] + k * perW); warp = { src: B, dst }; src.moving = true;
    stems = Object.fromEntries(["vocals", "drums", "bass", "other"].map(k => [k, stems[k].map(c => warpTime(c, B, dst))]));
    dm = mono(stems.drums); grid = { ...grid, bpm: 60 / perW, b0: B[0] % perW };
    log(`  warped onto a rigid ${(60 / perW).toFixed(2)} BPM grid`);
  }
  const bm = mono(stems.bass), om = mono(stems.other), vm = mono(stems.vocals);
  progress(0.2); await tick();
  const beat = 60 / grid.bpm, dur = dm.length / SR;
  let nb = Math.floor((dur - grid.b0) / beat);
  const kickL = filtfilt(dm, "low", 120), snL = band(dm, 1000, 6000);
  const lanesDb = { kick: perBeat(kickL, grid.b0, beat, nb), snare: perBeat(snL, grid.b0, beat, nb),
                    bass: perBeat(bm, grid.b0, beat, nb), other: perBeat(om, grid.b0, beat, nb), vocals: perBeat(vm, grid.b0, beat, nb) };
  const ev = { kick: lane(lanesDb.kick, 12), snare: lane(lanesDb.snare, 15), bass: lane(lanesDb.bass, 12), other: lane(lanesDb.other, 12), vocals: lane(lanesDb.vocals, 14) };
  // downbeat: which beat of 4 starts the bars, judged by where parts come in and drop out
  const nov = new Float32Array(nb);
  for (const k of ["kick", "bass", "vocals", "other"]) for (let i = 4; i < nb - 4; i++) {
    let a = 0, b = 0; for (let j = 0; j < 4; j++) { a += lanesDb[k][i - 1 - j]; b += lanesDb[k][i + j]; }
    nov[i] += Math.abs(b - a) / 4;
  }
  // and the backbeat: kick on 1 and 3, snare on 2 and 4. Per bar, (kick - snare) on 1+3 minus on 2+4; when its
  // mean is more than 3 standard errors from zero it fixes which pair of beats can be beat 1 (half-time
  // grooves, snare on 3 only, average out to zero here and leave the choice to the section changes)
  const bb = []; for (let i = 0; i + 3 < nb; i += 4) { const d = j => lanesDb.kick[i + j] - lanesDb.snare[i + j]; bb.push((d(0) + d(2) - d(1) - d(3)) / 2); }
  const bm_ = bb.reduce((a, v) => a + v, 0) / (bb.length || 1), bse = Math.sqrt(bb.reduce((a, v) => a + (v - bm_) ** 2, 0) / Math.max(1, bb.length - 1) / (bb.length || 1)) || 1;
  const parity = Math.abs(bm_ / bse) > 3 ? (bm_ > 0 ? 0 : 1) : null;
  let off = 0, bs = -1;
  for (let o = 0; o < 4; o++) { if (parity !== null && o % 2 !== parity) continue; let s = 0; for (let i = o; i < nb; i += 4) s += nov[i] * ((i - o) % 16 === 0 ? 2 : 1); if (s > bs) { bs = s; off = o; } }
  log(`  downbeat: backbeat evidence ${(bm_ / bse).toFixed(1)} SE${parity === null ? " (weak: section changes decide)" : ""}, bars start on beat ${off + 1} of the grid`);
  const b0 = grid.b0 + off * beat; nb -= off;
  src.down = src.moving ? ((Math.round((b0 - src.beats[0]) / beat) % 4) + 4) % 4 : off;   // beat 1 on src's beat numbering
  for (const k in ev) ev[k] = ev[k].slice(off);
  const full = Float32Array.from(dm, (v, i) => v + bm[i] + om[i] + vm[i]), fullDb = perBeat(full, grid.b0 + off * beat, beat, nb);
  const nBars = Math.floor(nb / 4);
  progress(0.35); await tick();
  // chords per half bar from the synths + 808
  const tonal = Float32Array.from(om, (v, i) => v + 0.6 * bm[i]);
  const wins = []; for (let hb = 0; hb < nBars * 2; hb++) wins.push([b0 + hb * 2 * beat, b0 + (hb + 1) * 2 * beat]);
  const chroma = chromaWindows(tonal, wins);
  const chords = chroma.map(c => { const t = triadRoot(c); return { root: t.root, minor: t.q === "m", chroma: c }; });
  const total = new Array(12).fill(0); for (const c of chroma) c.forEach((v, i) => total[i] += v);
  const key = detectKey(total);
  progress(0.6); await tick();
  const bassTrack = yinTrack(bm, SR); progress(0.9); await tick();
  // the 808's own tuning (cents off concert pitch): kicks are tuned to it, not to A=440
  let cs = 0, cc = 0;
  for (let i = 0; i < bassTrack.t.length; i++) if (bassTrack.voiced[i]) { const m = hzToMidi(bassTrack.f0[i]), d = m - Math.round(m), w = bassTrack.rms[i]; cs += w * Math.sin(2 * Math.PI * d); cc += w * Math.cos(2 * Math.PI * d); }
  const tuneCents = Math.round(Math.atan2(cs, cc) / (2 * Math.PI) * 100);
  const bassPc = []; for (let hb = 0; hb < nBars * 2; hb++) bassPc.push(trackPc(bassTrack, b0 + hb * 2 * beat + 0.05, b0 + (hb + 1) * 2 * beat));
  const barDb = Array.from({ length: nBars }, (_, b) => 10 * Math.log10(fullDb.slice(4 * b, 4 * b + 4).reduce((x, v) => x + Math.pow(10, v / 10), 0) / 4));
  return { bpm: grid.bpm, b0, beat, nBars, gridR: grid.R, ev, chords, key, bassPc, tuneCents, warp, barDb, src, stemsW: warp ? stems : null };
}

/* ------------------------------------------------------------ chop mode: the vocal as phrases */
const avg = a => a.reduce((s, v) => s + v, 0) / (a.length || 1);
const sdev = a => { const m = avg(a); return Math.sqrt(avg(a.map(v => (v - m) ** 2))); };
const pcOf = m => ((Math.round(m) % 12) + 12) % 12;
/** The vocal cut into phrases, in the song's own time, with the evidence chop mode decides on: each phrase's
 *  first strong onset and where it sits against the song's beat (on it, or a pickup), the song's beat length
 *  there, how sure the vocal is (voiced share), its melody and rhythm per beat (for the hook) and the
 *  backing's chroma per beat (for the roots). Each decision has a fallback; no threshold is absolute. */
async function analyzePhrases(stems, an, progress = () => {}) {
  const src = an.src, P0 = src.per, v = stems.inEngine ? filt(mono(stems.vocals), "high", 300, 4) : filt(mono(stems.vocals), "high", 120, 2), L = v.length;
  const HOP = 256, fs = SR / HOP, F = Math.floor(L / HOP), db = new Float32Array(F);
  for (let f = 0; f < F; f++) db[f] = rmsDb(v, Math.max(0, f * HOP - 512), Math.min(L, f * HOP + 512));
  // where it sings: within 18 dB of the loudest moment in the surrounding 2 s and 6 dB over the floor;
  // gaps shorter than a quarter beat are breaths inside a line
  const floor = Float32Array.from(db).sort()[Math.floor(0.1 * F)], top = minFilter(Float32Array.from(db, x => -x), Math.round(2 * fs));
  const utt = [];
  for (let f = 0, s = -1; f <= F; f++) {
    const on = f < F && db[f] > -top[f] - 18 && db[f] > floor + 6;
    if (on && s < 0) s = f;
    if (!on && s >= 0) { const u = utt[utt.length - 1], a = s / fs, b = f / fs; if (u && a - u[1] < Math.max(0.12, 0.25 * P0)) u[1] = b; else utt.push([a, b]); s = -1; }
  }
  // phrases: utterances less than 3/4 beat apart, up to 4 bars; a longer line splits at its quietest point
  const ph = [];
  for (const [a, b] of utt) {
    if (b - a < 0.12) continue;
    const p = ph[ph.length - 1];
    if (p && a - p.t1 < 0.75 * P0 && b - p.t0 <= 16 * P0) p.t1 = b; else ph.push({ t0: a, t1: b });
  }
  for (let i = 0; i < ph.length; i++) {
    const p = ph[i]; if (p.t1 - p.t0 <= 16 * P0) continue;
    let bf = Math.round((p.t0 + 4 * P0) * fs); for (let f = bf; f < (p.t1 - 4 * P0) * fs; f++) if (db[f] < db[bf]) bf = f;
    ph.splice(i + 1, 0, { t0: bf / fs, t1: p.t1 }); p.t1 = bf / fs; i--;
  }
  progress(0.2); await tick();
  const von = onsetTimes(v, 0.05, 0.1), vt = yinTrack(v, SR, 70, 900, 512, 128);
  const bm = mono(stems.bass), om = mono(stems.other), dm = mono(stems.drums), tonal = Float32Array.from(om, (x, i) => x + 0.6 * bm[i]);
  const full = Float32Array.from(om, (x, i) => x + bm[i] + dm[i] + v[i]);
  const notes = noteTimeline(yinTrack(bm, SR), an.tuneCents || 0);
  progress(0.5); await tick();
  const beatPos = t => {                                                // fractional index of t on the song's own beats
    const B = src.beats; if (!src.moving) return (t - src.ph) / P0;
    if (t <= B[0]) return (t - B[0]) / P0; let lo = 0, hi = B.length - 1; if (t >= B[hi]) return hi + (t - B[hi]) / P0;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (B[m] <= t) lo = m; else hi = m; }
    return lo + (t - B[lo]) / (B[hi] - B[lo]);
  };
  const localP = (a, b) => {                                            // the song's beat length over [a, b)
    const B = src.moving ? src.beats.filter(t => t >= a && t < b) : []; if (B.length < 4) return P0;
    let sx = 0, sy = 0, sxx = 0, sxy = 0, n = B.length; B.forEach((t, i) => { sx += i; sy += t; sxx += i * i; sxy += i * t; });
    return (n * sxy - sx * sy) / (n * sxx - sx * sx);
  };
  // the song's beat, windows of +-8 beats every 4 beats: a straight grid fitted to the drum onsets there. Period
  // (within 1.5% of the song's) and the 16th grid's phase from where all onsets cluster on beat, 8th and 16th (ok
  // beyond 2 standard errors); which 16th is the beat from where the kick's (and, at half weight, the snare band's)
  // onsets fall, chosen for the whole song at once (Viterbi: a jump of n 16ths between windows costs 0.3 n of hit
  // share) so a syncopated kick does not flip single windows onto the "and"
  const near = (o, a, b) => { const T = [], Wt = []; o.t.forEach((t, k) => { if (t >= a && t < b) { T.push(t); Wt.push(o.s[k]); } }); return [T, Wt]; };
  const fit16 = (a, b) => {
    const Pc = localP(a, b), [T, Wt] = near(src.on, a, b);
    if (T.length < 8) return { ok: false, P: Pc };
    let g = { s: -1 };
    for (let f = -0.015; f <= 0.0151; f += 0.0005) {
      const P = Pc * (1 + f), r4 = circFit(T, Wt, P / 4), sc = circFit(T, Wt, P).R + circFit(T, Wt, P / 2).R + r4.R;
      if (sc > g.s) g = { s: sc, P, ph16: r4.phase, R: r4.R };
    }
    if (g.R <= 2 / Math.sqrt(T.length)) return { ok: false, P: Pc };
    const [K, Kw] = near(src.onK, a, b), [M, Mw] = near(src.onM, a, b), hit = new Float64Array(4);
    for (const [U, Uw, w] of [[K, Kw, 1], [M, Mw, 0.5]]) U.forEach((t, k) => { const x = (((t - g.ph16) / g.P) % 1 + 1) % 1 * 4, j = Math.round(x) % 4; if (Math.abs(x - Math.round(x)) < 0.25) hit[j] += w * Uw[k]; });
    const tot = hit.reduce((x, y) => x + y, 0) || 1;
    return { ok: true, P: g.P, ph16: g.ph16, R: g.R, hit: Array.from(hit, h => h / tot) };
  };
  const W_ = []; for (let c = Math.max(0, src.ph); c < L / SR; c += 4 * P0) { const g = fit16(c - 8 * P0, c + 8 * P0); if (g.ok) W_.push({ c, ...g }); }
  const at = (ph, P, t) => ph + Math.round((t - ph) / P) * P;            // the grid's beat nearest t
  const cand = (w, j) => at(w.ph16 + j * w.P / 4, w.P, w.c), d16 = (x, y, P) => Math.round(4 * (((x - y) / P % 1 + 1.5) % 1 - 0.5));
  const vit = W_.map(() => ({ v: new Float64Array(4), from: new Int8Array(4) }));
  W_.forEach((w, i) => {
    for (let j = 0; j < 4; j++) {
      let best = 0, bj = 0;
      if (i > 0) { best = -Infinity; const u = W_[i - 1]; for (let k = 0; k < 4; k++) { const v = vit[i - 1].v[k] - 0.3 * Math.abs(d16(cand(w, j), at(cand(u, k), u.P, w.c), w.P)); if (v > best) { best = v; bj = k; } } }
      vit[i].v[j] = best + w.hit[j]; vit[i].from[j] = bj;
    }
  });
  for (let i = W_.length - 1, j = W_.length ? vit[i].v.indexOf(Math.max(...vit[i].v)) : 0; i >= 0; j = vit[i].from[j], i--) W_[i].beat = cand(W_[i], j);
  // the grid around a phrase: its own fit for period and 16th phase, the beat at the 16th nearest the song's beat there
  const fitGrid = (a, b) => {
    const g = fit16(a, b); if (!g.ok || !W_.length) return { ok: false, P: g.P };
    const mid = (a + b) / 2, w = W_.reduce((x, y) => Math.abs(y.c - mid) < Math.abs(x.c - mid) ? y : x);
    return { ok: true, P: g.P, ph: at(g.ph16, g.P / 4, at(w.beat, w.P, mid)), R: g.R };
  };
  const beatAt = k => { const B = src.beats; return !src.moving ? src.ph + k * P0 : k <= 0 ? B[0] + k * P0 : k >= B.length - 1 ? B[B.length - 1] + (k - B.length + 1) * P0 : B[k]; };
  let snapped = 0;
  const all = ph.map((p, i) => {
    let smax = 0; von.t.forEach((t, k) => { if (t >= p.t0 - 0.06 && t < p.t1) smax = Math.max(smax, von.s[k]); });
    const k0 = von.t.findIndex((t, k) => t >= p.t0 - 0.06 && t < p.t0 + Math.min(0.3, 0.6 * P0) && von.s[k] >= 0.3 * smax);
    const on = k0 >= 0 ? von.t[k0] : p.t0, g = fitGrid(on - 8 * P0, p.t1 + 8 * P0), P = g.P;
    // the bar the phrase starts in (a start in its last beat is a pickup into the next bar), its bar line on the
    // local grid; the phrase keeps its place in the bar. No grid: the onset itself becomes the bar line.
    let bar0 = on, q = 0;
    if (g.ok) {
      const k = beatPos(on); let kb = src.down + 4 * Math.floor((k - src.down) / 4); if (k - kb >= 3) kb += 4;
      bar0 = g.ph + Math.round((beatAt(kb) - g.ph) / P) * P; q = (on - bar0) / P;
    } else snapped++;
    const gridR = g.R ?? 0, gridOk = g.ok;
    let nv = 0, na = 0; for (let k = 0; k < vt.t.length; k++) if (vt.t[k] >= p.t0 && vt.t[k] < p.t1) { na++; nv += vt.voiced[k]; }
    const kb = Math.max(1, Math.ceil((p.t1 - bar0) / P - 0.1));
    return { i, t0: p.t0, t1: p.t1, s0: Math.min(p.t0, on) - 0.03, on, q, P, bar0, kb, gridR, gridOk, vf: na ? nv / na : 0, loud: rmsDb(full, Math.round(on * SR), Math.round(p.t1 * SR)) };
  });
  progress(0.6); await tick();
  // usable: voiced at least half as often as the song's typical phrase, longer than half a beat
  const vfMed = all.map(p => p.vf).sort((a, b) => a - b)[all.length >> 1] || 0;
  let list = all.filter(p => p.vf >= 0.5 * vfMed && p.t1 - p.on >= 0.5 * P0), how = "repetition", conf = 0;
  if (list.length < 2) {                                                // too few phrases: the vocal's loudest 2-bar chops
    const C = []; for (let t = src.ph; t + 8 * P0 < L / SR; t += 8 * P0) C.push({ t, e: rmsDb(v, Math.round(t * SR), Math.round((t + 8 * P0) * SR)) });
    list = C.sort((a, b) => b.e - a.e).slice(0, 4).sort((a, b) => a.t - b.t).map((c, k) => ({ i: all.length + k, t0: c.t, t1: c.t + 8 * P0, s0: c.t - 0.03, on: c.t, q: 0, P: P0, bar0: c.t, kb: 8, vf: 1, loud: c.e, gridOk: false, chop: true }));
    all.push(...list); how = "loop";
  }
  // per beat from the bar line: the vocal's melody (chroma) and rhythm (16th energies), the backing's chroma
  const wins = []; list.forEach(p => { for (let j = 0; j < p.kb; j++) wins.push([p.bar0 + j * p.P, p.bar0 + (j + 1) * p.P]); });
  const vc = chromaWindows(v, wins, 100, 2000), tc = chromaWindows(tonal, wins);
  let w0 = 0;
  for (const p of list) {
    p.vc = vc.slice(w0, w0 + p.kb); p.tc = tc.slice(w0, w0 + p.kb); w0 += p.kb;
    p.rh = []; for (let j = 0; j < 4 * p.kb; j++) { const a = Math.round((p.bar0 + j * p.P / 4) * fs), e = Math.round((p.bar0 + (j + 1) * p.P / 4) * fs); let s = 0; for (let f = Math.max(0, a); f < Math.min(F, e); f++) s += Math.pow(10, db[f] / 10); p.rh.push(s / Math.max(1, e - a)); }
    const m = avg(p.rh), sd = sdev(p.rh) || 1; p.rh = p.rh.map(x => (x - m) / sd);
    const c = new Array(12).fill(0); p.vc.concat(p.tc).forEach(x => x.forEach((y, k) => c[k] += y));
    const mx = Math.max(...c); p.set = c.map((y, k) => y >= 0.5 * mx ? k : -1).filter(k => k >= 0); p.key = detectKey(c);
  }
  progress(0.8); await tick();
  // the hook: phrases that repeat (melody + rhythm at the best beat lag, so lines cut at different places still
  // match), most often, loudest (mix power)
  const sim = (a, b) => {
    let best = 0;
    for (let lag = -b.kb + 1; lag < a.kb; lag++) {
      let s = 0, r = 0, n = 0;
      for (let j = Math.max(0, lag); j < Math.min(a.kb, b.kb + lag); j++) { const k = j - lag; s += a.vc[j].reduce((x, y, c) => x + y * b.vc[k][c], 0); n++; for (let u = 0; u < 4; u++) r += a.rh[4 * j + u] * b.rh[4 * k + u]; }
      if (n >= Math.min(4, a.kb, b.kb)) best = Math.max(best, (0.6 * s / n + 0.4 * Math.max(0, r / (4 * n))) * Math.sqrt(n / Math.max(a.kb, b.kb)));
    }
    return best;
  };
  const S = list.map(a => list.map(b => a === b ? 0 : sim(a, b))), off = S.flatMap((r, i) => r.filter((_, j) => j !== i));
  const thr = avg(off) + 1.5 * sdev(off), lmax = Math.max(...list.map(p => p.loud));
  list.forEach((p, i) => { p.rep = Math.max(0, ...S[i]); p.nrep = S[i].filter(x => x >= thr).length; p.score = (p.rep + 0.25 * Math.min(p.nrep, 4)) * Math.pow(10, (p.loud - lmax) / 10); });
  const sc = list.map(p => p.score), cut = avg(sc) + 0.5 * sdev(sc);
  list.forEach(p => p.hook = p.score >= cut);
  const hs = sc.filter(x => x >= cut), rs = sc.filter(x => x < cut);
  conf = rs.length ? (avg(hs) - avg(rs)) / (sdev(sc) || 1) : 0;
  if (how !== "loop" && (conf < 1 || !list.some(p => p.hook && p.nrep > 0))) {   // weak evidence: the loudest repeated phrase and its repeats
    const rep = list.filter(p => p.nrep > 0), h = (rep.length ? rep : list).reduce((a, b) => b.loud > a.loud ? b : a);
    const hi = list.indexOf(h); list.forEach((p, i) => p.hook = p === h || S[hi][i] >= thr);
    how = rep.length ? "loudest repeated" : "loudest";
  }
  if (how === "loop") list.forEach(p => p.hook = true);
  // hook runs (consecutive hook phrases): the first drop takes the strongest (mean score); later drops take
  // runs that repeat it with at least half its material, strongest first, else the strongest again
  let hookRuns = []; list.forEach((p, i) => { if (!p.hook) return; const r = hookRuns[hookRuns.length - 1]; if (r && list[i - 1] === r[r.length - 1]) r.push(p); else hookRuns.push([p]); });
  const dur = r => r.reduce((s, p) => s + p.t1 - p.on, 0), rs_ = r => avg(r.map(p => p.score));
  if (hookRuns.length) {
    const top = hookRuns.reduce((a, b) => rs_(b) > rs_(a) ? b : a), ti = top.map(p => list.indexOf(p));
    hookRuns = [top, ...hookRuns.filter(r => r !== top && dur(r) >= 0.5 * dur(top) && r.some(p => ti.some(i => S[i][list.indexOf(p)] >= thr))).sort((a, b) => rs_(b) - rs_(a))];
  }
  log(`  phrases: ${all.length} found, ${list.length} usable (voiced share >= ${(0.5 * vfMed).toFixed(2)});${snapped ? ` ${snapped} with no clear beat around them start on their bar line;` : ""} hook by ${how} (evidence ${conf.toFixed(1)} SD): ${hookRuns.map(r => r.map(p => p.on.toFixed(1)).join("+")).join(", ")} s`);
  progress(1);
  return { all, list, hookRuns: hookRuns.map(r => r.map(p => p.i)), hook: { how, conf }, notes, P0 };
}
/** The bass root for the source span [a, b) of a phrase: the song's bass note sounding longest there (>= 35% of
 *  the span), else the backing's triad root, else the tonic of the phrase's best-fit key, else the span's
 *  strongest pitch class. A root outside the span's own pitch set (vocal + backing chroma >= half its peak
 *  there; the phrase's when the span is too short to tell) is not taken. */
function rootAt(p, notes, a, b) {
  a = Math.max(a, p.bar0); b = Math.min(b, p.t1); if (b - a < 0.25 * p.P) { a = p.bar0; b = p.t1; }
  const c = new Array(12).fill(0), v = new Array(12).fill(0);
  for (let j = Math.max(0, Math.floor((a - p.bar0) / p.P)); j < Math.min(p.kb, Math.ceil((b - p.bar0) / p.P)); j++) p.tc[j].forEach((x, k) => { c[k] += x; v[k] += x + p.vc[j][k]; });
  const mx = Math.max(...v), set = mx > 0 ? v.map((x, k) => x >= 0.5 * mx ? k : -1).filter(k => k >= 0) : p.set;
  const cov = new Float64Array(12);
  for (const n of notes) { if (n.e <= a) continue; if (n.s >= b) break; cov[pcOf(n.m)] += Math.min(n.e, b) - Math.max(n.s, a); }
  let pc = 0; for (let k = 1; k < 12; k++) if (cov[k] > cov[pc]) pc = k;
  if (cov[pc] >= 0.35 * (b - a) && set.includes(pc)) return { pc, how: "bass", conf: cov[pc] / (b - a) };
  const t = triadRoot(c);
  if (c.some(x => x > 0) && set.includes(t.root)) return { pc: t.root, how: "chord", conf: t.s / (Math.hypot(...c) || 1) };
  if (set.includes(p.key.tonic)) return { pc: p.key.tonic, how: "key", conf: p.key.r };
  return mx > 0 ? { pc: v.indexOf(mx), how: "strongest", conf: mx / v.reduce((x, y) => x + y, 0) } : { pc: p.key.tonic, how: "key", conf: p.key.r };
}

/* ------------------------------------------------------------ arrangement */
/** Sections in the song's own order, cut on its 8-bar phrase lines: [{name, src, n, kind}].
 *  Phrase lines: the bar phase (0-7) where loudness and chords change most. The hook: phrases whose chords
 *  repeat elsewhere in the song and that are loud (repetition x loudness above the song's mean)
 *  become the drops; the 4 bars before each drop build into it; quiet phrases are breaks, the rest verses;
 *  first phrase intro, last outro. The song is never re-ordered, so its lines stay whole sentences. */
function autoArrange(an) {
  const nB = an.nBars, top = Math.max(...an.barDb), act = an.barDb.map(v => Math.pow(10, (v - top) / 20));   // bar loudness, 1 = loudest
  const chroma = (b0, b1) => { const c = new Array(12).fill(0); for (let h = 2 * b0; h < 2 * b1 && h < an.chords.length; h++) an.chords[h].chroma.forEach((v, i) => c[i] += v); const n = Math.hypot(...c) || 1; return c.map(v => v / n); };
  const cos = (x, y) => x.reduce((s, v, i) => s + v * y[i], 0);
  const mean = (a, i0, i1) => { let s = 0, n = 0; for (let i = Math.max(0, i0); i < Math.min(a.length, i1); i++) { s += a[i]; n++; } return n ? s / n : 0; };
  const nov = act.map((_, b) => b < 4 || b + 4 > nB ? 0 : Math.abs(mean(act, b, b + 4) - mean(act, b - 4, b)) + (1 - cos(chroma(b - 4, b), chroma(b, b + 4))));
  let ph = 0, bs = -1; for (let o = 0; o < 8; o++) { let s = 0; for (let b = o; b < nB; b += 8) s += nov[b]; if (s > bs) { bs = s; ph = o; } }
  const P = []; if (ph >= 4) P.push([0, ph]);                          // a pickup of 4+ bars is its own phrase
  for (let b = ph >= 4 ? ph : 0; b < nB; b = P[P.length - 1][1]) { const e = b === 0 && ph ? ph + 8 : b + 8; P.push([b, Math.min(nB, e)]); }
  if (P.length > 1 && P[P.length - 1][1] - P[P.length - 1][0] < 4) P[P.length - 2][1] = P.pop()[1];
  const C = P.map(([a, e]) => chroma(a, e)), A = P.map(([a, e]) => mean(act, a, e));
  const rep = C.map((c, i) => Math.max(0, ...C.map((d, j) => j === i ? 0 : cos(c, d))));
  const score = rep.map((r, i) => r * A[i]), avg = score.reduce((x, y) => x + y, 0) / score.length;
  const hook = score.map(v => v > avg); if (!hook.some(Boolean)) hook[score.indexOf(Math.max(...score))] = true;
  const quiet = A.reduce((x, y) => x + y, 0) / A.length;
  const S = [], add = (kind, src, n) => { if (n <= 0) return; const last = S[S.length - 1]; if (last && last.kind === kind && last.src + last.n === src && kind !== "build") last.n += n; else S.push({ kind, src, n }); };
  P.forEach(([a, e], i) => {
    if (i === 0 && P.length > 2) return add("intro", a, e - a);
    if (i === P.length - 1 && P.length > 2 && !hook[i]) return add("outro", a, e - a);
    if (hook[i]) return add("drop", a, e - a);
    const k = A[i] < 0.75 * quiet ? "break" : "verse";
    if (hook[i + 1] && e - a >= 8) { add(k, a, e - a - 4); add("build", e - 4, 4); } else if (hook[i + 1]) add("build", a, e - a); else add(k, a, e - a);
  });
  const count = {}; S.forEach(x => { count[x.kind] = (count[x.kind] || 0) + 1; x.name = x.kind + count[x.kind]; });
  log(`  phrases from bar ${ph}: ${P.map(([a, e], i) => `${a}-${e - 1}${hook[i] ? "*" : ""}`).join(" ")}  (* = hook)`);
  return { sections: S, teaser: P[hook.indexOf(true)][0] };
}
/** A new arrangement dealt from the song's own 4-bar blocks, the way a remixer re-cuts a track.
 *  Rules: intro first, a build before each drop, the hook comes back over the drums, outro last.
 *  The same seed always gives the same arrangement. */
function shuffleArrangement(an, seed = 1) {
  const rnd = mulberry(seed * 7919 + 13), pick = a => a[Math.floor(rnd() * a.length)];
  const G = Math.floor(an.nBars / 4), sum = (lane, g) => { let s = 0; for (let i = g * 16; i < g * 16 + 16 && i < lane.length; i++) s += lane[i]; return s; };
  const drum = [], voc = [];
  for (let g = 0; g < G; g++) { drum.push(sum(an.ev.kick, g) >= 12); voc.push(sum(an.ev.vocals, g) >= 8); }
  const runs = []; for (let g = 0; g < G;) { if (!drum[g]) { g++; continue; } let e = g; while (e < G && drum[e]) e++; runs.push([g, e]); g = e; }
  const pre = runs.map(([a]) => a - 1).filter(g => g >= 0 && !drum[g]);
  const T = pre.filter(g => voc[g]).pop() ?? pre.pop() ?? Math.min(G - 1, 2);
  const lastV = [...Array(G).keys()].filter(g => voc[g] && g + 1 < G).pop() ?? G - 2;
  const used = new Set([0, 1, lastV - 1, lastV]);
  const S = [], add = (kind, g, n) => { if (g == null) return; S.push({ kind, src: g * 4, n: n * 4 }); };
  const dropRun = k => {                                                // k contiguous drum groups, any run long enough
    const ok = runs.filter(([a, b]) => b - a >= k); if (!ok.length) return null;
    const [a, b] = pick(ok), st = a + Math.floor(rnd() * (b - a - k + 1)); return st;
  };
  const freePair = () => { const c = []; for (let g = 2; g + 1 < G; g++) if (!drum[g] && !drum[g + 1] && voc[g] && !used.has(g) && !used.has(g + 1) && g !== T && g + 1 !== T) c.push(g); const g = c.length ? pick(c) : null; if (g != null) { used.add(g); used.add(g + 1); } return g; };
  const build = () => { const c = pre.filter(g => !used.has(g)); const g = c.length ? pick(c) : T; used.add(g); return g; };
  const drop = k => { let st = dropRun(k); if (st == null) { st = dropRun(2) ?? runs[0]?.[0] ?? 0; k = Math.min(k, 2); } add("drop", st, k); };
  add("intro", 0, Math.min(2, G));
  const template = Math.floor(rnd() * 3);
  if (template === 0) {          // tease early: hook build straight into a long drop
    add("build", T, 1); drop(4); add("verse", freePair(), 2); add("break", freePair(), 2);
    add("build", build(), 1); drop(3); add("drop", T, 1); add("verse", freePair(), 2); add("build", T, 1); drop(4);
  } else if (template === 1) {   // short drop, then a long ride with the hook in the middle
    drop(2); add("verse", freePair(), 2); add("build", T, 1); drop(4); add("break", freePair(), 2);
    add("build", build(), 1); drop(2); add("drop", T, 1); drop(1);
  } else {                       // the original shape, re-dealt
    add("verse", freePair(), 2); add("build", T, 1); drop(4); add("break", freePair(), 2);
    add("build", build(), 1); drop(3); add("drop", T, 1); add("verse", freePair(), 2); add("build", build(), 1); drop(3);
  }
  add("outro", Math.max(0, lastV - 1), 2);
  return S.map(x => `${x.kind}:${x.src}:${x.n}`).join(", ");
}
/** "intro:0:8, drop:16:16, ..." (kind:src:bars) or "intro:8, drop:16, ..." (chop mode: kind:bars) -> sections. */
function parseArrangement(txt) {
  const S = []; const count = {};
  for (const part of txt.split(/[,\n]+/).map(s => s.trim()).filter(Boolean)) {
    const f = part.split(":"), kind = f[0]; count[kind] = (count[kind] || 0) + 1;
    S.push({ name: kind + count[kind], kind, src: f.length > 2 ? +f[1] : 0, n: +f[f.length - 1] });
  }
  return S;
}

/* ------------------------------------------------------------ samples */
/** Keep the thump, splice out the sharp middle of the kick's pitch glide, go straight to its steady tail. */
function flattenGlide(y, attackEnd = 0.040, steadyFrom = 0.135) {
  const lo = filtfilt(mono(y), "low", 150), up = [];
  for (let i = 0; i < lo.length - 1; i++) if (lo[i] < 0 && lo[i + 1] >= 0) up.push(i);
  if (up.length < 4) return y;
  const near = t => up.reduce((b, i) => Math.abs(i - t * SR) < Math.abs(b - t * SR) ? i : b, up[0]);
  const a = near(attackEnd), b = near(steadyFrom), xf = Math.round(0.003 * SR);
  return y.map(ch => {
    const head = ch.slice(0, a + xf), tail = ch.slice(b);
    for (let i = 0; i < xf; i++) { const w = i / xf; head[a + i] = head[a + i] * (1 - w) + tail[i] * w; }
    const o = zeros(head.length + tail.length - xf); o.set(head); o.set(tail.subarray(xf), head.length); return o;
  });
}
function tailF0(y) {
  const s = mono(y).subarray(Math.round(0.05 * SR)); const N = 1 << 18;
  const re = new Float64Array(N), im = new Float64Array(N);
  const n = Math.min(s.length, N); for (let i = 0; i < n; i++) re[i] = s[i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / n));
  fft(re, im); let best = 0, bk = 0;
  for (let k = Math.ceil(28 * N / SR); k < 80 * N / SR; k++) { const m = re[k] * re[k] + im[k] * im[k]; if (m > best) { best = m; bk = k; } }
  return bk * SR / N;
}
function tuneTo(y, target) {
  let semis = 12 * Math.log2(target / tailF0(y)), out = y;
  for (let k = 0; k < 5; k++) { out = repitch2(y, semis); const err = 12 * Math.log2(tailF0(out) / target); if (Math.abs(err) < 0.08) break; semis -= err; }
  return out;
}
function fadeOut(y, s = 0.006) { const n = y[0].length, k = Math.min(n, Math.max(1, Math.round(s * SR))); for (const c of y) for (let i = 0; i < k; i++) c[n - k + i] *= Math.cos(i / k * Math.PI / 2) ** 2; return y; }
function fadeIn(y, s) { const k = Math.min(y[0].length, Math.max(1, Math.round(s * SR))); for (const c of y) for (let i = 0; i < k; i++) c[i] *= Math.sin(i / k * Math.PI / 2) ** 2; return y; }
const normPeak = y => { const m = Math.max(peakAbs(y[0]), peakAbs(y[1])) || 1; return y.map(c => Float32Array.from(c, v => v / m)); };
const slice2 = (y, a, b) => [y[0].slice(a, b), y[1].slice(a, b)];

/** Pitch-class energy of a sample (FFT over [a, b) s, lo..hi Hz). */
function pcProfile(y, a, b, lo, hi) {
  const s = mono(y).subarray(Math.round(a * SR), Math.round(b * SR)), N = 1 << 17;
  const re = new Float64Array(N), im = new Float64Array(N), n = Math.min(s.length, N);
  for (let i = 0; i < n; i++) re[i] = s[i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / n));
  fft(re, im); const pc = new Float64Array(12); let tot = 0;
  for (let k = 1; k < N / 2; k++) { const f = k * SR / N; if (f < lo || f > hi) continue; const e = re[k] * re[k] + im[k] * im[k]; pc[((Math.round(hzToMidi(f)) % 12) + 12) % 12] += e; tot += e; }
  return Array.from(pc, v => v / (tot || 1));
}
/** Smallest transposition (|t| <= 3) that puts most of a sound's energy on scale notes. */
function bestKeyShift(prof, scale) {
  let best = 0, bs = -1e9;
  for (let t = -3; t <= 3; t++) { let s_ = 0; for (let pc = 0; pc < 12; pc++) s_ += prof[pc] * (scale.includes((pc + t + 12) % 12) ? 1 : -1); s_ -= 0.03 * Math.abs(t); if (s_ > bs) { bs = s_; best = t; } }
  return best;
}

/** Where a kick sample's pitch settles, its tail pitch, and whether the drop before it is fast and wide. */
const _settle = new WeakMap();
function kickSettle(y) {
  if (_settle.has(y)) return _settle.get(y);
  const tr = yinTrack(mono(y), SR, 28, 300, 512, 32), tail = tailF0(y);
  let settle = 0.02, top = 0;
  for (let i = 0; i < tr.t.length; i++) if (tr.voiced[i] && tr.t[i] < 0.03) top = Math.max(top, tr.f0[i]);
  for (let i = 0; i < tr.t.length; i++) if (tr.voiced[i] && tr.t[i] > 0.01 && Math.abs(tr.f0[i] / tail - 1) < 0.05) { settle = tr.t[i]; break; }
  const v = { tail, settle: Math.min(settle, 0.2), fast: top > 2.2 * tail && settle < 0.2 };
  _settle.set(y, v); return v;
}

/* ------------------------------------------------------------ defaults */
const DEFAULTS = {
  mode: "chop",             // "chop": the vocal's phrases on a fixed groove | "continuous": the whole song sped up under new drums
  bpm: "auto",              // chop mode's tempo: "auto" (tempoFit) or a number
  semis: 4,                 // pitch up (chop) / speed up (continuous: +4 st = 140 -> 176 BPM, v10 B)
  pattern: "alt",           // drops: four on the floor, then x x xxx (2-bar cycle)
  h16: true,                // 16th hats in the drops
  tailLp: 900, buzz: 0.05,  // kick tone: tail low-pass, how much buzz survives the attack
  kickDrive: 2.0,           // tanh drive on the generated tail (harmonics of a tuned note stay in key)
  matchEq: true,            // EQ the master toward the hardtekk reference balance
  kickTune: 0,              // manual offset in semitones on every kick note
  targetLufs: -5.5,
  arrangement: "",          // empty = automatic
  fadeInBars: 2, fadeOutBars: 8,
  risers: true, impacts: true, echo: true, tapeStop: true,
  kickTone: "dark",         // "punch" | "dark": softer click, deeper pitch drop, sub-heavy tail
  bassVoice: "dark",        // "song" | "dark": outside the drops, the song's 808 or a dark sub resynthesised from it
  bassLo: 29, bassHi: 58,   // every bass note (kick tails, dark 808) is folded into this one-octave window, Hz
  bellTune: false,          // false: the bell plays at its own pitch
  kickClick: "sample",      // "sample" | "synth": the kick's attack from the sample or generated (no sample pitch)
};

/* ------------------------------------------------------------ build the remix */
/** Drum plan per output bar. Only the drops depend on the groove. */
function planBars(barInfo, pattern, h16) {
  return barInfo.map(bi => {
    const k = bi.kind, i = bi.idx, n = bi.n, p = { kick: [], lp: null, snare: false, hats: false, h16: false, roll: null, riser: false };
    const E = [0, 1, 2, 2.5, 3], F4 = [0, 1, 2, 3];
    if (k === "drop") {
      const tekk = { tekk: true, alt: i % 2 === 1, smooth: i % 4 === 3 }[pattern];
      Object.assign(p, { kick: tekk ? E : F4, snare: true, hats: true, h16 });
      if ((i + 1) % 8 === 0 && i + 1 < n) p.kick = [0, 1, 2, 2.5, 3, 3.5];
    } else if (k === "verse") { Object.assign(p, { kick: F4, hats: true }); if (i === n - 1) Object.assign(p, { kick: [0, 1, 2], roll: "short" }); }
    else if (k === "intro") { if (i >= 4) Object.assign(p, { kick: F4, lp: [180, 400, 1200, null][i - 4], hats: i >= 6 }); if (i === 7) Object.assign(p, { kick: [0, 1, 2], roll: "full", riser: true }); }
    else if (k === "build") Object.assign(p, { kick: [[0, 1, 2, 3], [0, 1, 2, 3], [0, .5, 1, 1.5, 2, 2.5, 3, 3.5], [0, .5, 1, 1.5, 2]][Math.min(i, 3)], lp: [250, 700, null, null][Math.min(i, 3)], hats: i >= 1, riser: i === n - 1, roll: i === n - 1 ? "full" : null });
    else if (k === "break") { if (i === n - 1) p.roll = "short"; }
    else if (k === "outro") { if (i < 4) Object.assign(p, { kick: F4, hats: true, lp: [null, null, 1500, 400][i] }); }
    return p;
  });
}

/** Stage 1, per speed: the song's tonal stems sped up as one piece. The costly part. */
async function stageSpeed(stems, semis, cache, progress) {
  cache.speed = cache.speed || new Map();
  if (cache.speed.has(semis)) { const v = cache.speed.get(semis); cache.speed.delete(semis); cache.speed.set(semis, v); return v; }
  const SRC = {};
  let k = 0;
  for (const name of ["vocals", "other", "bass"]) { SRC[name] = repitch2(stems[name], semis); progress(0.05 + 0.25 * ++k / 3); await tick(); }
  const m = mono(SRC.vocals), w = Math.round(0.02 * SR), vEnv = zeros(m.length); let acc = 0;
  for (let i = 0; i < m.length; i++) { acc += Math.abs(m[i]); if (i >= w) acc -= Math.abs(m[i - w]); vEnv[i] = acc / w; }
  const { o: fl } = onsetFlux(m, 128, 1024), vOn = peaks(fl, Math.round(0.05 * SR / 128), 0.15).map(i => (i * 128 + 512) / SR);
  const v = { SRC, vEnv, vOn };
  cache.speed.set(semis, v);
  while (cache.speed.size > 2) cache.speed.delete(cache.speed.keys().next().value);   // ~150 MB each
  return v;
}

/** Stage 2, per speed + arrangement: splice, teaser, stutter, everything that does not depend on the groove. */
async function stageArrange(an, sp, o, sections, teaserSrc, cache, progress) {
  const key = `${o.semis}|${sections.map(s => `${s.kind}:${s.src}:${s.n}`).join(",")}`;
  cache.arrange = cache.arrange || new Map();
  if (cache.arrange.has(key)) return cache.arrange.get(key);
  const RATE = Math.pow(2, o.semis / 12), BEAT = an.beat / RATE, BAR = 4 * BEAT, EIGHTH = BEAT / 2, SB0 = an.b0 / RATE, OB0 = 0.05;
  const blocks = []; let ob = 0;
  for (const s of sections) { const n = Math.min(s.n, Math.max(0, an.nBars - s.src)); if (n > 0) { blocks.push({ ...s, n, out: ob }); ob += n; } }
  const N_BARS = ob, N = Math.round((OB0 + N_BARS * BAR + 3) * SR);
  const ot = (bar, b = 0) => OB0 + bar * BAR + b * BEAT, stt = (bar, b = 0) => SB0 + bar * BAR + b * BEAT;
  const barInfo = []; for (const b of blocks) for (let i = 0; i < b.n; i++) barInfo.push({ block: b.name, kind: b.kind, src: b.src + i, idx: i, n: b.n });
  const { SRC, vEnv, vOn } = sp;
  // seams where the vocal is quietest near each boundary
  const ve = t => vEnv[Math.min(vEnv.length - 1, Math.max(0, Math.round(t * SR)))];
  blocks.forEach((b, i) => {
    if (i === 0) { b.d0 = 0; return; }
    const p = blocks[i - 1];
    if (p.kind === "build") { b.d0 = p.d1 = 0; return; }
    let best = 0, bc = 1e9;
    for (let d = -0.6 * BEAT; d < 0.02; d += 0.005) { const c = Math.max(ve(stt(p.src + p.n) + d), ve(stt(b.src) + d)); if (c < bc) { bc = c; best = d; } }
    b.d0 = p.d1 = best;
  });
  if (blocks.length) blocks[blocks.length - 1].d1 = 0;
  const XF = 2 * Math.round(0.006 * SR), XH = XF / 2;
  const TR = { vocals: st2(N), other: st2(N), bass: st2(N) };
  for (const b of blocks) {
    const a = stt(b.src) + b.d0, e = stt(b.src + b.n) + b.d1, o0 = ot(b.out) + b.d0;
    const i0 = Math.round(a * SR) - XH, i1 = Math.round(e * SR) + XH, dst = Math.round(o0 * SR) - XH;
    for (const k in TR) for (let c = 0; c < 2; c++) {
      const src = SRC[k][c], out = TR[k][c];
      for (let i = Math.max(0, i0); i < i1 && i < src.length; i++) {
        const j = dst + (i - i0); if (j < 0 || j >= N) continue;
        const r = i - i0, wv = r < XF ? Math.sin(r / XF * Math.PI / 2) ** 2 : (i1 - i < XF ? Math.cos((XF - (i1 - i)) / XF * Math.PI / 2) ** 2 : 1);
        out[j] += src[i] * wv;
      }
    }
  }
  const vox = TR.vocals, other = TR.other, bass = TR.bass;
  progress(0.45); await tick();
  // intro: tease the pre-drop hook, band-limited, over bars 4-7
  if (barInfo.length > 8 && barInfo[0].kind === "intro") {
    const a = Math.round(ot(0) * SR), e = Math.round(ot(8) * SR);
    for (const c of vox) for (let i = a; i < e && i < N; i++) c[i] *= 0.35;
    const h0 = Math.round((stt(teaserSrc) - 0.25 * BEAT) * SR), h1 = Math.round(stt(teaserSrc + 4) * SR);
    const hook = fadeIn(fadeOut(slice2(SRC.vocals, Math.max(0, h0), h1).map(c => band(c, 350, 3500, 2)), 0.03), 0.02);
    place(vox, hook, ot(4) - 0.25 * BEAT, 0.8);
  }
  // vocal stutter into each drop that follows a build
  for (const b of blocks.filter(b => b.kind === "build")) {
    const last = b.out + b.n - 1, ts = stt(b.src + b.n - 1);
    const c0 = vOn.reduce((p, v) => Math.abs(v - ts) < Math.abs(p - ts) ? v : p, vOn[0] ?? ts);
    const chop = fadeIn(fadeOut(slice2(SRC.vocals, Math.round(c0 * SR), Math.round((c0 + EIGHTH) * SR)), 0.01), 0.003);
    const a = Math.round(ot(last, 2) * SR), e = Math.round(ot(last + 1) * SR);
    for (const c of vox) for (let i = a; i < e && i < N; i++) c[i] *= Math.sqrt(1 - (i - a) / (e - a));
    for (let q = 0; q < 4; q++) place(vox, chop, ot(last, 2 + q * 0.5), 0.55 + 0.1 * q);
  }
  // masks: which bars carry drums don't depend on the groove
  const shape = planBars(barInfo, "alt", false);
  const mask = pred => { const m = zeros(N); barInfo.forEach((bi, obi) => { if (pred(bi, obi)) m.fill(1, Math.round((ot(obi) - 0.005) * SR), Math.min(N, Math.round((ot(obi + 1) - 0.005) * SR))); }); return smooth(m, Math.round(0.012 * SR)); };
  const drumM = mask((bi, obi) => ["drop", "verse"].includes(bi.kind) || (["build", "intro", "outro"].includes(bi.kind) && shape[obi].kick.length));
  const dropM = mask(bi => bi.kind === "drop");
  // the 808 as it plays in the remix (after splicing), for the intro kicks
  const out808 = yinTrack(mono(bass), SR, 28, 120, 1024, 64);
  progress(0.6); await tick();
  // bed before ducking; 808 only outside the drops
  const bedPre = st2(N), bass808 = st2(N);
  for (let c = 0; c < 2; c++) {
    const hp = filt(filt(other[c], "high", 120, 4), "high", 120, 4);       // 8th order: the kick alone owns the low end
    for (let i = 0; i < N; i++) { const d = drumM[i]; bedPre[c][i] = other[c][i] * (1 - d) + hp[i] * d; bass808[c][i] = bass[c][i] * (1 - dropM[i]) * 0.85; }
  }
  const sweeps = [[0, Math.round(ot(4) * SR), 350, 18000, 1.7, "lp"]];
  shape.forEach((p, obi) => { if (p.riser) sweeps.push([Math.round(ot(obi) * SR), Math.round(ot(obi + 1) * SR), 30, 800, 1.5, "hp"]); });
  if (N_BARS > 4) sweeps.push([Math.round(ot(N_BARS - 4) * SR), Math.round(ot(N_BARS) * SR), 18000, 300, 0.8, "lp"]);
  for (const [a, e, f0, f1, pw, kind] of sweeps) for (const c of bass808) svf(c, a, e, i => f0 * Math.pow(f1 / f0, Math.pow((i - a) / (e - a), pw)), kind);
  const { voxComp, act } = vocalBus(vox, dropM), bedPres = bedPre.map(c => band(c, 900, 4200));
  progress(0.75); await tick();
  const v = { key, blocks, barInfo, N, N_BARS, BEAT, BAR, OB0, ot, out808, drumM, dropM, bedPre, bedPres, bass808, voxComp, act, sweeps };
  cache.arrange.set(key, v);
  while (cache.arrange.size > 2) cache.arrange.delete(cache.arrange.keys().next().value);
  return v;
}

/** Vocals: high-pass + light compression; their 1-4 kHz activity keys the bed's and the kick's presence dip. */
function vocalBus(vox, dropM) {
  const lift = Math.pow(10, 4.5 / 20) - 1;
  const voxComp = vox.map(c => compress(filt(c, "high", 90, 2), -20, 2.5, 5, 80));
  const m = mono(voxComp); for (let i = 0; i < m.length; i++) m[i] *= 1 + dropM[i] * lift;
  const p = band(m, 900, 4200), e = smooth(Float32Array.from(p, Math.abs), Math.round(0.04 * SR));
  const nz = Float32Array.from(e.filter(v => v > 1e-4)).sort(), ref = nz.length ? nz[Math.floor(nz.length * 0.7)] : 1;
  return { voxComp, act: smooth(Float32Array.from(e, v => Math.min(1, v / (ref + 1e-9))), Math.round(0.06 * SR)) };
}

/** Build the remix: chop mode (default) or the continuous mode that keeps the whole song running. */
async function build(stems, samples, an, opts = {}, progress = () => {}, cache = {}) {
  const o = { ...DEFAULTS, ...opts };
  return o.mode === "continuous" ? buildContinuous(an.stemsW || stems, samples, an, o, progress, cache) : buildChop(stems, samples, an, o, progress, cache);
}

/* ------------------------------------------------------------ chop mode: build */
const K4 = "x...x...x...x...", OFF = "..x...x...x...x.", BB = "....x.......x...", S8 = "x.x.x.x.x.x.x.x.", S16 = "xxxxxxxxxxxxxxxx";
/** Section templates: bar ranges (negative = counted from the section's end) with one 16-step pattern per lane.
 *  Drops: tonal kick on every beat, off-beat bass, clap on 2 and 4, off-beat hat; 16th hats join after 8 bars.
 *  Builds: backbeat, then a snare roll that doubles, kick in 8ths, a bar of roll alone. Breaks have no kick.
 *  Fill bars may change elements off the 8-bar lines. */
const TEMPLATES = {
  intro: [[0, 8, { kick: K4, hat: OFF }]],
  build: [[0, 4, { kick: K4, hat: OFF, snare: BB }], [4, -2, { kick: K4, hat: OFF, roll: S8, fill: 1 }], [-2, -1, { kick: S8, hat: OFF, roll: S16, fill: 1 }], [-1, 0, { roll: S16, fill: 1 }]],
  drop: [[0, 8, { kick: K4, bass: OFF, snare: BB, hat: OFF }], [8, 16, { kick: K4, bass: OFF, snare: BB, hat: OFF, h16: 1 }]],
  break: [[0, -1, {}], [-1, 0, { roll: "............xxxx", fill: 1 }]],
  outro: [[0, 8, { kick: K4, hat: OFF }]],
};
const FORM = { full: "intro:8, build:8, drop:16, break:8, build:8, drop:16, outro:8", short: "intro:8, build:8, drop:16, outro:8" };

/** Chop mode: a fixed-tempo groove from the templates, with the song's vocal phrases cut out, each
 *  stretched (pitch kept) by its own tempo onto whole beats and started on a bar line. The song's drums and
 *  bass are not used; its backing plays (high-passed, ducked) in breaks and builds only. */
/** Chop mode's tempo and pitch shift. One song beat becomes m remix beats (1/2, 1 or 2); the pitch shift's
 *  resample already speeds the vocal up by R, so the stretch left for WSOLA is m x (60 / bpm) x R / (song beat).
 *  With bpm "auto": the tempo in 150-200 BPM and the shift within 2 st of the asked one that bring the median
 *  phrase's stretch nearest 1, where a semitone away from the asked shift weighs as 3% stretch and the
 *  distance from 175 BPM as a twentieth of its log. With a number: that tempo, the asked shift. */
function tempoFit(Ps, P0, bpm, semis) {
  const Pm = Ps.slice().sort((a, b) => a - b)[Ps.length >> 1] || P0, auto = bpm === "auto" || !(+bpm > 0);
  let best = null;
  for (let s = semis - (auto ? 2 : 0); s <= semis + (auto ? 2 : 0); s++) for (const m of [0.5, 1, 2]) {
    const R = Math.pow(2, s / 12), b = auto ? Math.min(200, Math.max(150, 60 * m * R / Pm)) : +bpm, a = m * 60 / b * R / Pm;
    const cost = Math.abs(Math.log(a)) + 0.03 * Math.abs(s - semis) + 0.05 * Math.abs(Math.log(b / 175));
    if (!best || cost < best.cost) best = { bpm: b, semis: s, m, cost };
  }
  return best;
}

async function buildChop(stems, samples, an, o, progress, cache) {
  const ph = cache.phrases || (cache.phrases = await analyzePhrases(stems, an, p => progress(0.4 * p)));
  const fit = tempoFit(ph.list.map(p => p.P), ph.P0, o.bpm, o.semis), m = fit.m, asked = { bpm: o.bpm, semis: o.semis };
  o = { ...o, bpm: fit.bpm, semis: fit.semis };
  const T = 60 / o.bpm, BEAT = T, BAR = 4 * T, R = Math.pow(2, o.semis / 12), OB0 = 0.05;
  const use = ph.list.map(p => ({ ...p, alpha: m * T * R / p.P })).filter(p => Math.abs(Math.log(p.alpha)) <= Math.log(1.42));
  const out = (p, t) => (t - p.bar0) * p.alpha / R;                    // source time -> remix seconds after the phrase's bar line
  for (const p of use) p.bars = Math.max(1, Math.ceil((out(p, p.t1) - 0.25 * BEAT) / BAR));
  const byId = new Map(use.map(p => [p.i, p])), verse = use.filter(p => !p.hook);
  const runs = ph.hookRuns.map(r => r.map(i => byId.get(i)).filter(Boolean)).filter(r => r.length);
  if (!runs.length && use.length) runs.push([use.reduce((a, b) => b.loud > a.loud ? b : a)]);
  const sections = parseArrangement(o.arrangement || (verse.length >= 2 && use.length >= 4 ? FORM.full : FORM.short));
  let ob = 0; const blocks = sections.map(s => { const b = { ...s, src: 0, out: ob }; ob += s.n; return b; });
  const N_BARS = ob, N = Math.round((OB0 + N_BARS * BAR + 3) * SR), ot = (bar, b = 0) => OB0 + bar * BAR + b * BEAT;
  const barInfo = []; for (const b of blocks) for (let i = 0; i < b.n; i++) barInfo.push({ block: b.name, kind: b.kind, idx: i, n: b.n });
  // phrases onto bar lines, whole, in the song's order: drops loop their hook run, other sections take the
  // next verse phrases; a pickup may reach back into the previous bar when nothing sounds there
  const placed = []; let vi = 0, di = 0, lastEnd = -Infinity;
  const tryPut = (p, bar, end, b) => {
    let at = bar; while (at < end && ot(at) + out(p, p.s0) < lastEnd - 0.05) at++;
    if (at + p.bars > end) return -1;
    placed.push({ p, bar: at, block: b }); lastEnd = ot(at) + out(p, p.t1); return at + p.bars;
  };
  for (const b of blocks) {
    const end = b.out + b.n - (b.kind === "build" ? 2 : 0);               // a build's last 2 bars: the hook stutter
    let bar = b.out;
    if (b.kind === "drop") {
      const run = runs[Math.min(di++, runs.length - 1)] || [];
      for (let k = 0, miss = 0; run.length && bar < end && miss < run.length; k++) { const nx = tryPut(run[k % run.length], bar, end, b); if (nx < 0) miss++; else { bar = nx; miss = 0; } }
    } else for (let n = 0; n < verse.length && bar < end; n++) {
      const p = verse[vi % verse.length], nx = tryPut(p, bar, end, b);
      if (nx >= 0) { bar = nx; vi++; } else if (p.bars > end - b.out) vi++; else break;
    }
  }
  progress(0.5); await tick();
  // the bass root of every bar, from the source span its phrase plays there (sounding pitch: + the shift)
  const roots = new Array(N_BARS).fill(null);
  for (const pl of placed) for (let k = 0; k < pl.p.bars; k++) {
    const p = pl.p, a = p.bar0 + k * BAR * R / p.alpha, r = rootAt(p, ph.notes, a, a + BAR * R / p.alpha);
    roots[pl.bar + k] = { ...r, pc: (r.pc + o.semis) % 12, src: [a, a + BAR * R / p.alpha], phrase: p.i };
  }
  blocks.forEach((b, i) => { const nx = blocks[i + 1]; if (b.kind === "build" && nx && roots[nx.out]) for (let k = b.out + b.n - 2; k < b.out + b.n; k++) roots[k] = roots[k] || { ...roots[nx.out], how: "next drop" }; });
  const keyPc = (an.key.tonic + o.semis) % 12;
  for (let k = 0, last = roots.find(Boolean) || { pc: keyPc, how: "song key" }; k < N_BARS; k++) roots[k] = roots[k] ? (last = roots[k]) : { ...last, carried: true };

  const ck = `${o.semis}|${o.bpm}|${blocks.map(b => b.kind + b.n).join(",")}`;
  if (!cache.chop || cache.chop.key !== ck) cache.chop = { key: ck, ...(await renderChops(stems, ph, placed, blocks, { T, BAR, R, m, N, ot, out }, o, progress)) };
  const { vox, bed, leak } = cache.chop;

  // the groove: the templates as a score; kick and bass play each bar's root
  const score = { kick: [], bass: [], snare: [], hat: [], bell: [] }, plan = [];
  const TEKK = "x...x...x.x.x...", FILL = "x...x...x.x.x.x.";
  for (const b of blocks) {
    const rows = (TEMPLATES[b.kind] || []).map(([a, e, pat]) => [a < 0 ? b.n + a : a, e <= 0 ? b.n + e : e, pat]);
    const r0 = Math.min(b.n, ...rows.filter(r => r[2].roll).map(r => r[0]));
    for (let i = 0; i < b.n; i++) {
      const bar = b.out + i, pc = roots[bar].pc, lane = {};
      for (const [a, e, pat] of rows) if (i >= a && i < e) Object.assign(lane, pat);
      if (b.kind === "drop" && lane.kick) {
        if (o.pattern === "tekk" || (o.pattern === "alt" && i % 2)) lane.kick = TEKK;
        if (i % 8 === 7 && i + 1 < b.n) { lane.kick = FILL; lane.fill = 1; }
      }
      const x = (k, s) => !!lane[k] && lane[k][s] === "x";
      for (let s = 0; s < 16; s++) {
        const st = bar * 16 + s;
        if (x("kick", s)) score.kick.push({ s: st, pc });
        if (x("bass", s) && !x("kick", s)) score.bass.push({ s: st, pc });
        if (x("snare", s)) score.snare.push({ s: st, g: 0.22 });
        if (x("roll", s)) score.snare.push({ s: st, g: 0.06 + 0.24 * Math.pow(((i - r0) * 16 + s) / ((b.n - r0) * 16), 1.5) });
        if (x("hat", s)) score.hat.push({ s: st, g: 0.34 });
        if (lane.h16 && o.h16 && s % 2) score.hat.push({ s: st, g: 0.13 });
      }
      const lp = b.kind === "intro" ? 200 * Math.pow(2, 0.6 * i) : b.kind === "outro" && i >= b.n - 4 ? 3000 * Math.pow(2, -0.8 * (i - b.n + 5)) : null;
      plan.push({ lp, fill: !!lane.fill, kind: b.kind });
    }
  }
  for (const b of blocks.filter((b, i) => b.kind === "drop" && i > 0 && blocks[i - 1].kind !== "drop")) for (const s of [4, 12]) score.bell.push({ s: (b.out - 2) * 16 + s, g: 0.5, semis: 0 });
  const mask = pred => { const mk = zeros(N); barInfo.forEach((bi, k) => { if (pred(bi, k)) mk.fill(1, Math.round((ot(k) - 0.005) * SR), Math.min(N, Math.round((ot(k + 1) - 0.005) * SR))); }); return smooth(mk, Math.round(0.012 * SR)); };
  const kickBars = new Set(score.kick.map(e => Math.floor(e.s / 16)));
  const drumM = mask((_, k) => kickBars.has(k)), dropM = mask(bi => bi.kind === "drop");
  const sweeps = blocks.filter(b => b.kind === "build").map(b => [Math.round(ot(b.out + b.n - 2) * SR), Math.round(ot(b.out + b.n) * SR), 30, 800, 1.5, "hp"]);
  const { voxComp, act } = vocalBus(vox, dropM);
  const A = { blocks, barInfo, N, N_BARS, BEAT, BAR, OB0, ot, drumM, dropM, bedPre: bed, bedPres: bed.map(c => band(c, 900, 4200)), voxComp, act, sweeps };
  const chop = { bpm: o.bpm, semis: o.semis, m, asked, hook: ph.hook, leak, inEngine: !!stems.inEngine,
    phrases: ph.all.map(p => ({ i: p.i, t0: p.t0, t1: p.t1, on: p.on, q: p.q, P: p.P, vf: p.vf, gridOk: p.gridOk, used: byId.has(p.i), hook: !!byId.get(p.i)?.hook, alpha: byId.get(p.i)?.alpha, rep: p.rep, nrep: p.nrep, loud: p.loud, score: p.score })),
    placed: placed.map(pl => ({ i: pl.p.i, bar: pl.bar, bars: pl.p.bars, kind: pl.block.kind, alpha: pl.p.alpha, bar0: pl.p.bar0, on: pl.p.on, s0: pl.p.s0, t1: pl.p.t1, q: pl.p.q, hook: pl.p.hook })),
    roots: roots.map(r => ({ pc: r.pc, how: r.how, src: r.src, phrase: r.phrase, carried: !!r.carried })) };
  log(`  chop: ${o.bpm.toFixed(1)} BPM, ${o.semis >= 0 ? "+" : ""}${o.semis} st, 1 song beat = ${m} beat(s), stretch ${use.length ? use.map(p => p.alpha).sort((a, b) => a - b).filter((_, k, a) => k === 0 || k === a.length - 1).map(v => v.toFixed(2)).join("-") : "-"}, ${placed.length} phrases placed; ` + blocks.map(b => `${b.name}:${b.n}`).join(" "));
  const ctx = { mode: "chop", tuneCents: an.tuneCents || 0, A, plan, scales: [], key: `${NAMES[keyPc]} ${an.key.mode}`, scale: [], arrangement: blocks.map(b => `${b.kind}:${b.n}`).join(", "), o, chop };
  progress(0.9); await tick();
  const r = mixScore(ctx, score, samples, o);
  progress(1);
  return r;
}

/** The vocal (and, in breaks and builds, the backing) of every placed phrase: varispeed by the pitch shift,
 *  stretch to its beats with its onsets locked, start on its bar line. The backing, and the vocal when the
 *  engine separated it, are high-passed at 300 Hz: that is where the song's kick and 808 leak through the
 *  in-engine separator (3-6x less leak for 2-12 points of the vocal's energy on the test songs). A build's
 *  last two bars stutter the first syllable of the hook that follows. */
async function renderChops(stems, ph, placed, blocks, g, o, progress) {
  const { T, BAR, R, m, N, ot, out } = g, vox = st2(N), bed = st2(N);
  const leak = stems.probe ? { vox: [0, 0], db: [0, 0], v: [0, 0], bed: 0, bedDb: 0 } : null;
  const E = y => { let s = 0; for (const c of y) for (let i = 0; i < c.length; i++) s += c[i] * c[i]; return s; };
  const hp = (y, on = true) => on ? y.map(c => filt(c, "high", 300, 4)) : y;
  const seg = (x, a, e) => fadeOut(fadeIn(slice2(x, a, e), 0.005), 0.04);
  const next = new Map(ph.all.map((p, k) => [p.i, ph.all.slice(k + 1).find(q => q.s0 > p.on)?.s0 ?? Infinity]));
  for (const [k, pl] of placed.entries()) {
    const p = pl.p, a = Math.max(0, Math.round(p.s0 * SR)), e = Math.min(stems.vocals[0].length, Math.round(Math.min(p.t1 + 0.12, next.get(p.i)) * SR));
    if (e - a < 256) continue;
    const y = repitch2(hp(seg(stems.vocals, a, e), stems.inEngine), o.semis), on = onsetTimes(mono(y), 0.08, 0.3);
    const locks = on.t.map((t, j) => [t, on.s[j]]).sort((u, w) => w[1] - u[1]).map(([t]) => Math.round(t * SR));
    place(vox, stretch(y, p.alpha, locks), ot(pl.bar) + out(p, p.s0));
    if (leak) for (const [n, x] of [["vox", stems.vocals], ["db", stems.probe.db.vocals], ["v", stems.probe.v.vocals]]) { const s_ = seg(x, a, e); leak[n][0] += E(s_); leak[n][1] += E(hp(s_)); }
    if (["break", "build"].includes(pl.block.kind)) {                    // the backing under this phrase's bars
      const b0 = Math.round(p.bar0 * SR), b1 = Math.min(stems.other[0].length, Math.round((p.bar0 + pl.p.bars * BAR * R / p.alpha) * SR));
      if (b1 - Math.max(0, b0) > 256) {
        const raw = hp(fadeOut(fadeIn(slice2(stems.other, Math.max(0, b0), b1), 0.01), 0.01));
        place(bed, stretch(repitch2(raw, o.semis), p.alpha), ot(pl.bar) + Math.max(0, -b0) / SR * p.alpha / R);
        if (leak) { leak.bed += E(raw); leak.bedDb += E(hp(slice2(stems.probe.db.other, Math.max(0, b0), b1))); }
      }
    }
    if (k % 4 === 0) { progress(0.5 + 0.35 * k / placed.length); await tick(); }
  }
  // the hook stutter: its first 8th, in 8ths then 16ths, up to the drop's pickup
  blocks.forEach((b, i) => {
    if (b.kind !== "build") return;
    const first = placed.find(pl => pl.block === blocks[i + 1]), p = first ? first.p : placed.find(pl => pl.p.hook)?.p;
    if (!p) return;
    const a = Math.round(p.on * SR), e = Math.round((p.on + p.P / (2 * m)) * SR);
    const chop = stretch(repitch2(slice2(stems.vocals, a, e), o.semis), p.alpha), stop = first ? ot(first.bar) + out(p, p.s0) - 0.01 : Infinity;
    for (let s = 0; s < 24; s++) {
      const bar = b.out + b.n - 2 + (s < 8 ? 0 : 1), q = s < 8 ? s / 2 : (s - 8) / 4, len = s < 8 ? T / 2 : T / 4, t = ot(bar, q);
      const n = Math.round(Math.min(len, stop - t) * SR); if (n < 64) break;
      place(vox, fadeOut(fadeIn(slice2(chop, 0, n), 0.002), 0.008), t, 0.5 + 0.35 * s / 23);
    }
  });
  return { vox, bed, leak };
}

/** Continuous mode: the whole song sped up on its own (detected) beat grid, the new drums on top. */
async function buildContinuous(stems, samples, an, o, progress, cache) {
  const keyTonic = (an.key.tonic + o.semis) % 12;
  const SCALE = (an.key.mode === "minor" ? [0, 2, 3, 5, 7, 8, 10] : [0, 2, 4, 5, 7, 9, 11]).map(i => (keyTonic + i) % 12);
  const inScale = pc => SCALE.includes(((pc % 12) + 12) % 12);
  const auto = autoArrange(an);
  const sections = o.arrangement ? parseArrangement(o.arrangement) : auto.sections;
  progress(0.02); await tick();
  const sp = await stageSpeed(stems, o.semis, cache, progress);
  const A = await stageArrange(an, sp, o, sections, auto.teaser, cache, progress);
  const { blocks, barInfo, N, N_BARS, BEAT, BAR, OB0, ot, out808, drumM, dropM } = A;
  log(`  arrangement: ${blocks.map(b => `${b.name}<${b.src}-${b.src + b.n - 1}`).join(" ")}`);

  const plan = planBars(barInfo, o.pattern, o.h16);
  // key-constrained kick notes: synth chroma + the 808's note, scale notes only
  const ev = an.ev;
  const drumRegion = src => { const g = Math.floor(src / 4) * 4; let s = 0; for (let i = g * 4; i < (g + 4) * 4 && i < ev.kick.length; i++) s += ev.kick[i]; return s >= 8; };
  const on = (lane, src, b) => !!ev[lane][src * 4 + Math.floor(b)];
  // each section's own key: verses can use different notes than the drops
  const rot = c => c.map((_, i) => c[((i - o.semis) % 12 + 12) % 12]);
  const hbOf = src => Math.min(an.chords.length - 1, src * 2);
  for (const blk of blocks) {
    const c = new Array(12).fill(0);
    for (let src = blk.src; src < blk.src + blk.n; src++) for (const hb of [hbOf(src), hbOf(src) + 1]) {
      const ch = an.chords[Math.min(hb, an.chords.length - 1)]; ch.chroma.forEach((v, i) => c[i] += v);
      if (an.bassPc[hb] != null) c[an.bassPc[hb]] += 0.3;
    }
    const k = detectKey(c), t = (k.tonic + o.semis) % 12;
    blk.scale = (k.mode === "minor" ? [0, 2, 3, 5, 7, 8, 10] : [0, 2, 4, 5, 7, 9, 11]).map(i => (t + i) % 12);
    blk.key = `${NAMES[t]} ${k.mode}`;
  }
  const scaleOf = obi => blocks.find(bk => bk.name === barInfo[obi].block).scale;
  log("  section keys: " + blocks.map(bk => `${bk.name} ${bk.key}`).join(", "));
  /** One kick note per bar: both halves' synth chroma + the 808, from the section's own scale. */
  const keyRoot = obi => {
    const src = barInfo[obi].src, sc = scaleOf(obi), hb = hbOf(src);
    const c = rot(an.chords[hb].chroma).map((v, i) => v + rot(an.chords[Math.min(hb + 1, an.chords.length - 1)].chroma)[i]);
    const bps = [an.bassPc[hb], an.bassPc[hb + 1]].filter(v => v != null).map(v => (v + o.semis) % 12);
    let best = sc[0], bs = -1e9;
    for (const pc of sc) { const s = c[pc] + 0.5 * c[(pc + 7) % 12] + 0.35 * c[(pc + 3) % 12] + 0.35 * c[(pc + 4) % 12] + 0.4 * bps.filter(v => v === pc).length; if (s > bs) { bs = s; best = pc; } }
    return best;
  };
  const barRoot = barInfo.map((_, obi) => keyRoot(obi));
  /** The kick plays the song's own 808 note at that moment: the kick tail is the bassline in the
   *  drops, so it must be the song's bassline, not a guess. Falls back to the last 808 note heard,
   *  then to the bar's root. */
  let last808 = null;
  const bassNotes = A.bassNotes || (A.bassNotes = noteTimeline(out808, an.tuneCents || 0));
  const kickNote = (obi, b, dur) => {
    const t0 = ot(obi, b), len = barInfo[obi].kind === "drop" ? Math.min(0.3, (dur || 0.5) * BEAT) : Math.min(0.15, (dur || 0.5) * BEAT);
    const heard = barInfo[obi].kind === "drop" ? Math.min(0.3, (dur || 0.5) * BEAT) : 0.12;   // how long this kick is heard
    const pc = noteAt(bassNotes, t0 + 0.02, t0 + heard);               // the bass note while this kick sounds
    if (pc !== null) { last808 = pc; return pc; }
    return last808 ?? barRoot[obi];
  };
  const kickAllowed = (obi, b) => { const bi = barInfo[obi]; return !drumRegion(bi.src) || on("kick", bi.src, b); };
  const fullStop = (obi, b) => { const bi = barInfo[obi]; return drumRegion(bi.src) && !on("bass", bi.src, b) && !on("other", bi.src, b); };
  const hitLen = (obi, b) => { const ks = plan[obi].kick.filter(x => x > b); return ks.length ? ks[0] - b : 1; };
  const hits = [];
  plan.forEach((p, obi) => p.kick.forEach(b => { if (kickAllowed(obi, b)) hits.push({ t: ot(obi, b), pc: kickNote(obi, b, hitLen(obi, b)), scale: scaleOf(obi), lp: p.lp, long: barInfo[obi].kind === "drop", kind: barInfo[obi].kind, bar: obi, beat: b }); }));
  progress(0.8); await tick();

  // the drums as an editable score: 16th steps from bar 0 of the remix
  const step = (obi, b) => Math.round(obi * 16 + b * 4);
  const score = { kick: hits.map(h => ({ s: step(h.bar, h.beat), pc: h.pc })), snare: [], hat: [], bell: [] };
  plan.forEach((p, obi) => {
    for (let b = 0; b < 4; b++) {
      if (fullStop(obi, b)) continue;
      if (p.hats) score.hat.push({ s: step(obi, b + 0.5), g: 0.34 });
      if (p.h16) for (const q of [0.25, 0.75]) score.hat.push({ s: step(obi, b + q), g: 0.13 });
      if (p.snare && (b === 1 || b === 3)) score.snare.push({ s: step(obi, b), g: 0.22 });
    }
    if (p.roll === "full") {
      for (let k = 0; k < 8; k += 2) score.snare.push({ s: obi * 16 + k, g: 0.06 + 0.012 * k });
      for (let k = 8; k < 16; k++) score.snare.push({ s: obi * 16 + k, g: 0.12 + 0.018 * (k - 8) });
    } else if (p.roll === "short") for (let k = 12; k < 16; k++) score.snare.push({ s: obi * 16 + k, g: 0.10 + 0.03 * (k - 12) });
  });
  // double bell: _x_x two bars before each drop, transposed (at most 3 st) to fit that bar's key
  const bellProf = pcProfile(samples.bell.map(c => filt(c, "high", 150, 2)), 0.02, 1.5, 150, 6000);
  for (const b of blocks.filter((b, i) => b.kind === "drop" && (i === 0 || blocks[i - 1].kind !== "drop"))) if (b.out >= 2) {
    const semis = o.bellTune ? bestKeyShift(bellProf, scaleOf(b.out - 2)) : 0;
    for (const bt of [1, 3]) score.bell.push({ s: step(b.out - 2, bt), g: 0.5, semis });
  }
  const arrangement = blocks.map(b => `${b.kind}:${b.src}:${b.n}`).join(", ");
  log(`  the song's 808 sits ${an.tuneCents >= 0 ? "+" : ""}${an.tuneCents} cents from A=440; kicks follow it`);
  const ctx = { tuneCents: an.tuneCents || 0, A, plan, scales: barInfo.map((_, obi) => scaleOf(obi)), key: `${NAMES[keyTonic]} ${an.key.mode}`, scale: SCALE.map(p => NAMES[p]), arrangement, o };
  progress(0.85); await tick();
  const r = mixScore(ctx, score, samples, o);
  progress(1);
  return r;
}

/** Render the drums from a score and mix them against the arranged song.
 *  Fast: everything that does not depend on the drums is cached in ctx.A. */
function mixScore(ctx, score, samples, opts) {
  const o = { ...DEFAULTS, ...opts }, A = ctx.A, { N, N_BARS, BEAT, BAR, OB0, barInfo, blocks, drumM, dropM } = A;
  const STEP = BEAT / 4, t = s => OB0 + s * STEP, barOf = s => Math.min(barInfo.length - 1, Math.max(0, Math.floor(s / 16)));
  const clean = (arr) => arr.filter(e => e.s >= 0 && e.s < N_BARS * 16).sort((a, b) => a.s - b.s);
  score = { kick: clean(score.kick), bass: clean(score.bass || []), snare: clean(score.snare), hat: clean(score.hat), bell: clean(score.bell) };
  const hits = score.kick.map(e => { const bar = barOf(e.s), bi = barInfo[bar]; return { t: t(e.s), s: e.s, pc: ((e.pc % 12) + 12) % 12, scale: ctx.scales[bar], lp: ctx.plan[bar].lp, long: bi.kind === "drop", kind: bi.kind, bar }; });
  const bassT = score.bass.map(e => t(e.s));                           // a kick ends where the next kick or bass note starts
  hits.forEach((h, j) => { h.end = Math.min(j + 1 < hits.length ? hits[j + 1].t : h.t + 1, bassT.find(x => x > h.t + 1e-4) ?? Infinity); });
  const layers = { perc: st2(N), fx: st2(N) };
  const snare = samples._snareF || (samples._snareF = filt2(samples.snare, "high", 200)), hat = samples.hat;
  for (const e of score.snare) place(layers.perc, snare, t(e.s), e.g ?? 0.22);
  for (const e of score.hat) place(layers.perc, hat, t(e.s), e.g ?? 0.34);
  const bellSrc = samples._bellF || (samples._bellF = samples.bell.map(c => filt(c, "high", 150, 2)));
  samples._bells = samples._bells || new Map();
  for (const e of score.bell) {
    const k = e.semis ?? 0; if (!samples._bells.has(k)) samples._bells.set(k, repitch2(bellSrc, k));
    place(layers.fx, samples._bells.get(k), t(e.s), e.g ?? 0.5);
  }
  // bed and vocals: duck on the kicks, dip 1-4 kHz under the rapper, then the filter sweeps
  const kickTimes = hits.map(h => h.t), dk25 = duck(N, kickTimes, 0.25, 0.09), dk12 = duck(N, kickTimes, 0.12, 0.07);
  const lift = Math.pow(10, 4.5 / 20) - 1, act = A.act;
  const bed = A.bedPre.map((c, ci) => { const p = A.bedPres[ci], y = zeros(N); for (let i = 0; i < N; i++) y[i] = (c[i] - p[i] * 0.55 * act[i]) * (1 - drumM[i] * (1 - dk25[i])); return y; });
  for (const [a, e, f0, f1, pw, kind] of A.sweeps) for (const c of bed) svf(c, a, e, i => f0 * Math.pow(f1 / f0, Math.pow((i - a) / (e - a), pw)), kind);
  const vox = A.voxComp.map(c => { const y = zeros(N); for (let i = 0; i < N; i++) y[i] = c[i] * (1 - drumM[i] * (1 - dk12[i])) * (1 + dropM[i] * lift) * 1.15; return y; });
  const meta = { tuneCents: ctx.tuneCents, blocks, barInfo, plan: ctx.plan, hits, OB0, BEAT, BAR, N, N_BARS, bpm: 60 / BEAT, key: ctx.key, scale: ctx.scale, arrangement: ctx.arrangement, score, mode: ctx.mode, chop: ctx.chop };
  let bass808 = st2(N);
  if (ctx.mode !== "chop") {                                            // the song's 808, or a dark sub playing its line
    const dk = `${o.bassLo}|${o.bassHi}`;
    if (o.bassVoice === "dark" && A.bassDarkKey !== dk) { A.bassDark = darkSub(A, o.bassLo, o.bassHi, ctx.tuneCents || 0); A.bassDarkKey = dk; }
    const bassSrc = o.bassVoice === "dark" ? A.bassDark : A.bass808;
    const dkB = duck(N, kickTimes, 0.9, 0.08);                           // one low sound at a time: the bass gets out of the kick's way
    bass808 = bassSrc.map(c => Float32Array.from(c, (v, i) => v * (1 - (1 - dropM[i]) * (1 - dkB[i]))));
  }
  if (score.bass.length) { const b = renderBass(score.bass, meta, o); for (let c = 0; c < 2; c++) for (let i = 0; i < N; i++) bass808[c][i] += b[c][i]; }
  const L = { bed, bass808, vocals: vox, perc: layers.perc, fx: layers.fx };
  L.kick = renderKick(samples, meta, o, act);
  score.kick.forEach((e, i) => { e.hz = hits[i].hz; });                // the frequency actually played, for MIDI
  L.sfx = st2(N);
  applyEffects(L, meta, o);
  const dropLift = Math.pow(10, 2 / 20) - 1;                            // drops hit the master 2 dB harder
  for (const lay of Object.values(L)) for (const c of lay) for (let i = 0; i < N; i++) c[i] *= 1 + dropM[i] * dropLift;
  return { layers: L, meta, masks: { drumM, dropM, act }, ctx };
}

/** A dark, deep sub that plays the song's 808 line note for note (pitch track incl. glides,
 *  the 808's own loudness), in place of the 808 itself outside the drops. */
function darkSub(A, fLo = 32.7, fHi = 61.8, cents = 0) {
  const tr = A.out808, N = A.N, y = zeros(N), notes = A.bassNotes || [];
  const frameAt = i => Math.min(tr.t.length - 1, Math.max(0, Math.round((i / SR - tr.t[0]) / (tr.t[1] - tr.t[0] || 1))));
  let rmax = 0; for (let i = 0; i < tr.rms.length; i++) rmax = Math.max(rmax, tr.rms[i]);
  const fold = m => { let f = midiToHz(m + cents / 100); while (f >= fHi) f /= 2; while (f < fLo) f *= 2; return f; };
  let ni = 0, f = 0, a = 0, ph = 0;
  const glide = 1 - Math.exp(-1 / (0.008 * SR)), lvl = 1 - Math.exp(-1 / (0.005 * SR));
  for (let i = 0; i < N; i++) {
    const t = i / SR;
    while (ni < notes.length && notes[ni].e <= t) ni++;
    const on = ni < notes.length && notes[ni].s <= t;
    const fT = on ? fold(notes[ni].m) : (f || 40), aT = on ? tr.rms[frameAt(i)] / (rmax || 1) : 0;
    f += (fT - f) * (f ? glide : 1); a += (aT - a) * lvl;            // 8 ms glide between notes
    ph += 2 * Math.PI * f / SR; y[i] = Math.tanh(1.2 * Math.sin(ph)) * a;
  }
  const lo = filt(filt(y, "low", 160, 4), "high", 25, 2);
  // match the 808 layer's loudness, keep its section mask and sweeps (it is silent in the drops)
  const ref = A.bass808, m = mono(ref);
  let er = 0, ed = 0; for (let i = 0; i < N; i++) { er += m[i] * m[i]; }
  const out = [Float32Array.from(lo), Float32Array.from(lo)];
  for (let i = 0; i < N; i++) { const g = 1 - A.dropM[i]; out[0][i] *= g; out[1][i] *= g; ed += out[0][i] * out[0][i]; }
  for (const [s0, e, f0, f1, pw, kind] of A.sweeps) for (const c of out) svf(c, s0, e, i => f0 * Math.pow(f1 / f0, Math.pow((i - s0) / (e - s0), pw)), kind);
  const g = Math.sqrt(er / (ed || 1)); for (const c of out) for (let i = 0; i < N; i++) c[i] *= g;
  return out;
}

/* ------------------------------------------------------------ effects */
function noise(n, seed) { const r = mulberry(seed), y = zeros(n); for (let i = 0; i < n; i++) y[i] = r() * 2 - 1; return y; }
function mulberry(seed) { let a = seed >>> 0 || 1; return () => { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
/** Remix moves around each drop (riser, impact + tuned boom, vocal echo throw), a tape stop before
 *  the outro, and the song's fade in / fade out. Everything lands on the layers, so the live
 *  preview and the final render hear the same thing. */
function applyEffects(L, meta, o) {
  const { blocks, BEAT, BAR, OB0, N, N_BARS, hits } = meta, sfx = L.sfx;
  const ot = (bar, b = 0) => OB0 + bar * BAR + b * BEAT;
  const drops = blocks.filter((b, i) => b.kind === "drop" && (i === 0 || blocks[i - 1].kind !== "drop"));
  drops.forEach((b, k) => {
    const t = ot(b.out), i0 = Math.round(t * SR);
    if (o.risers && b.out >= 2) {                                       // 2 bars of high-passed noise rising into the drop
      const a = Math.round(ot(b.out - 2) * SR), n = i0 - a;
      for (let c = 0; c < 2; c++) {
        const y = noise(n, 1000 + 7 * k + c);
        svf(y, 0, n, i => 300 * Math.pow(7000 / 300, i / n), "hp");
        for (let i = 0; i < n; i++) { const x = i / n; sfx[c][a + i] += y[i] * 0.16 * x * x * (i > n - 400 ? (n - i) / 400 : 1); }
      }
    }
    if (o.impacts) {                                                    // noise hit + a boom on the drop's first kick note
      const n = Math.round(1.4 * SR), first = hits.find(h => Math.abs(h.t - t) < 0.02), f0 = first?.hz || 46.25;
      for (let c = 0; c < 2; c++) {
        const y = noise(n, 2000 + 7 * k + c);
        svf(y, 0, n, i => 9000 * Math.pow(300 / 9000, i / n), "lp");
        let ph = 0;
        for (let i = 0; i < n && i0 + i < N; i++) {
          const tt = i / SR; ph += 2 * Math.PI * f0 * (1 + 1.5 * Math.exp(-tt / 0.05)) / SR;
          sfx[c][i0 + i] += y[i] * 0.2 * Math.exp(-tt / 0.35) + Math.sin(ph) * 0.3 * Math.exp(-tt / 0.5);
        }
      }
    }
    if (o.echo && b.out >= 1) {                                         // the vocal's last beat before the drop echoes into it
      const a = Math.round((t - BEAT) * SR), n = i0 - a, d = Math.round(0.75 * BEAT * SR);
      const seg = L.vocals.map(c => band(c.slice(a, i0), 400, 5000, 2));
      for (let r = 1; r <= 5; r++) {
        const g = 0.6 * Math.pow(0.45, r), j = a + r * d, pan = r % 2 ? [1, 0.45] : [0.45, 1];
        for (let c = 0; c < 2; c++) for (let i = 0; i < n && j + i < N; i++) sfx[c][j + i] += seg[c][i] * g * pan[c];
      }
    }
  });
  const layers = Object.values(L);
  if (o.tapeStop) {                                                     // brake the whole mix over the last beat before the outro
    const ob = blocks.findIndex(b => b.kind === "outro");
    if (ob > 0 && blocks[ob - 1].kind === "drop") {
      const e = Math.round(ot(blocks[ob].out) * SR), n = Math.round(BEAT * SR), a = e - n;
      for (const lay of layers) for (const c of lay) {
        const out = zeros(n); let read = 0;
        for (let u = 0; u < n; u++) { const i = Math.floor(read), f = read - i; out[u] = (c[a + i] || 0) * (1 - f) + (c[a + i + 1] || 0) * f; read += 1 - u / n; }
        const nf = Math.round(0.012 * SR); for (let j = 0; j < nf; j++) out[n - nf + j] *= (1 - j / nf) ** 2;
        c.set(out, a);
      }
    }
  }
  // fade in over the first bars, fade out over the last bars
  const fi = Math.round(o.fadeInBars * BAR * SR), endS = Math.round(ot(N_BARS) * SR), fo = Math.round(o.fadeOutBars * BAR * SR);
  for (const lay of layers) for (const c of lay) {
    for (let i = 0; i < fi && i < c.length; i++) c[i] *= Math.sin(i / fi * Math.PI / 2) ** 2;
    for (let i = Math.max(0, endS - fo); i < c.length; i++) c[i] *= i >= endS ? 0 : Math.cos((i - (endS - fo)) / fo * Math.PI / 2) ** 2;
  }
}

/** The kick layer alone, so the UI can retune it without rebuilding everything. */
const _kickCache = new Map();
function renderKick(samples, meta, opts, act = null) {
  const o = { ...DEFAULTS, ...opts }, N = meta.N;
  const noteHz = pc => {                                               // folded into the bass window, at the song's tuning
    let hz = 32.703 * Math.pow(2, (((pc % 12) + 12) % 12) / 12 + (meta.tuneCents || 0) / 1200);
    while (hz >= o.bassHi) hz /= 2; while (hz < o.bassLo) hz *= 2; return hz;
  };
  if (o.kickClick === "synth" && !samples._synthKick) {               // a generated attack: fast sine sweep + a 2 ms noise tick
    const n = Math.round(0.4 * SR), y = zeros(n), r = mulberry(99); let ph = 0;
    for (let i = 0; i < n; i++) { const t = i / SR, f = 48 + 170 * Math.exp(-t / 0.018); ph += 2 * Math.PI * f / SR; y[i] = Math.sin(ph) * Math.exp(-t / 0.3) + (t < 0.002 ? (r() * 2 - 1) * 0.5 * (1 - t / 0.002) : 0); }
    samples._synthKick = [y, Float32Array.from(y)];
  }
  const kickSrc = o.kickClick === "synth" ? "synth" : "sample";
  if (samples._kickNormSrc !== kickSrc) { delete samples._kickNorm; samples._kickNormSrc = kickSrc; }
  const base = samples._kickNorm || (samples._kickNorm = (() => {             // peak-normalised, leading silence trimmed
    const y = normPeak(kickSrc === "synth" ? samples._synthKick : samples.kick); let i0 = 0; while (i0 < y[0].length && Math.max(Math.abs(y[0][i0]), Math.abs(y[1][i0])) < 0.02) i0++;
    return y.map(c => c.slice(Math.max(0, i0 - 16)));
  })());
  /** The sample's first ~20 ms (the click; later its pitch is still sliding and sounds sharp),
   *  then a tail generated at exactly `hz`:
   *  a short pitch settle from 1.6x, sustained, tanh-saturated. No resampling of the
   *  tail, so every note is in tune and keeps the same tone. */
  const kickFor = (pc, long) => {
    const hz = noteHz(pc) * Math.pow(2, o.kickTune / 12);
    const dark = o.kickTone === "dark";
    const key = [kickSrc === "synth" ? "synth" : (samples._kickId || "kick"), hz.toFixed(3), long, o.tailLp, o.buzz, o.kickDrive, o.kickTone].join("|");
    if (_kickCache.has(key)) return _kickCache.get(key);
    // Kicks with a fast, wide pitch drop keep their whole drop (it reads as punch) up to where the
    // pitch settles; a slow, narrow slide (the classic kick) would sound sharp, so it keeps 20 ms only.
    const sp = kickSettle(base), nat = sp.tail;
    // the sample's click is only nudged (<= 2 st): big shifts smear it. The generated tail carries the note.
    const shift = Math.max(-2, Math.min(2, 12 * Math.log2(hz / nat) - (sp.fast ? 0 : 2.5)));
    let att = repitch2(base, shift);
    if (dark) att = filt2(att, "low", 2500);                           // a softer, less poppy click
    const lo = filtfilt(mono(att), "low", 150), at = sp.fast ? sp.settle * Math.pow(2, -shift / 12) : 0.02;
    let xs = Math.round(at * SR);                                        // splice on an upward zero crossing
    for (let i = Math.round((at - 0.004) * SR); i < Math.round((at + 0.012) * SR); i++) if (lo[i] < 0 && lo[i + 1] >= 0) { xs = i; break; }
    const len = Math.round((long ? 0.5 : 0.16) * SR + xs), out = st2(len), xf = Math.round(0.004 * SR);
    const ampAt = Math.max(1e-3, (() => { let m = 0; for (let i = xs - 200; i < xs; i++) m = Math.max(m, Math.abs(lo[i])); return m; })());
    const drive = dark ? 1.3 : o.kickDrive, settle = dark ? [0.6, 0.025] : [0.15, 0.01];
    let ph = 0; const th = Math.tanh(drive);
    for (let i = 0; i < len; i++) {
      const t = i / SR, f = hz * (1 + 0.6 * Math.exp(-t / 0.012));
      let tail = 0;
      if (i >= xs - xf) {
        const tt = (i - xs) / SR;
        const env = long ? (tt < 0.05 ? 1 : 0.85 + 0.15 * Math.exp(-(tt - 0.05) / 0.15)) : Math.exp(-Math.max(tt, 0) / 0.04);
        tail = Math.tanh(drive * Math.sin(ph)) / th * env;
        ph += 2 * Math.PI * (i < xs ? hz : hz * (1 + settle[0] * Math.exp(-tt / settle[1]))) / SR;   // the drop into the note
      }
      const w = i < xs - xf ? 0 : i >= xs ? 1 : (i - (xs - xf)) / xf;
      for (let c = 0; c < 2; c++) { const a_ = i < att[c].length ? att[c][i] : 0; out[c][i] = a_ * (1 - w) + tail * w * Math.max(ampAt, 0.5); }
    }
    let y = filt2(out, "low", dark ? Math.min(o.tailLp, 240) : o.tailLp, 2);
    if (dark) y = y.map(c => biquad(c, rbj("lowshelf", 60, 0.7, SR, 3)));
    y = normPeak(y.map(c => { biquad(c, rbj("peak", 2200, 0.6, SR, -4)); return c; }));
    // tune the tail (what is heard as the note), not the click
    for (let k = 0; long && k < 3; k++) {                                // short tails are generated exact; too short to measure
      const tr = yinTrack(mono(y), SR, 28, 160, 1024, 64); let lw = 0, ws = 0;
      for (let i = 0; i < tr.t.length; i++) if (tr.voiced[i] && tr.t[i] > 0.03 && tr.t[i] < (long ? 0.34 : 0.12)) { const w = tr.rms[i]; lw += w * Math.log(tr.f0[i]); ws += w; }
      if (ws <= 0) break;
      const cents = 1200 * Math.log2(Math.exp(lw / ws) / hz);
      if (Math.abs(cents) < 8) break;
      y = repitch2(y, -cents / 100);
    }
    _kickCache.set(key, y); return y;
  };
  const bus = st2(N), hits = meta.hits;
  hits.forEach((h, j) => {
    const gap = (h.end ?? (j + 1 < hits.length ? hits[j + 1].t : h.t + 1)) - h.t;
    h.hz = noteHz(h.pc) * Math.pow(2, o.kickTune / 12);
    const k = kickFor(h.pc, h.long);
    let y = slice2(k, 0, Math.round(Math.min(gap - 0.004, k[0].length / SR) * SR));
    // outside the drops the bass layer ducks under each kick (see mixScore), so the kick keeps its body
    if (h.lp) y = filt2(y, "low", h.lp);
    place(bus, fadeOut(y, 0.006), h.t);
  });
  if (act) for (const c of [0, 1]) { const p = band(bus[c], 900, 4200); for (let i = 0; i < N; i++) bus[c][i] -= p[i] * 0.5 * act[i]; }
  // level: -10.5 dB RMS over the first drop
  const d1 = meta.blocks.find(b => b.kind === "drop");
  if (d1) { const a = Math.round((meta.OB0 + d1.out * meta.BAR) * SR), e = Math.round((meta.OB0 + (d1.out + d1.n) * meta.BAR) * SR); const g = Math.pow(10, (-10.5 - rmsDb(mono(bus), a, e)) / 20); for (const c of bus) for (let i = 0; i < N; i++) c[i] *= g; }
  return bus;
}

/** The score's bass lane: a driven two-saw stab per note an octave over the kick's tail, cut before the next
 *  kick or bass note, so kick and bass never sound together. Level: -12 dB RMS over the first drop. */
function renderBass(events, meta, o) {
  const { N, OB0, BEAT } = meta, bus = zeros(N), kicks = meta.hits.map(h => h.t), times = events.map(e => OB0 + e.s * BEAT / 4);
  events.forEach((e, j) => {
    const t0 = times[j], end = Math.min(kicks.find(x => x > t0 + 1e-4) ?? Infinity, times[j + 1] ?? Infinity, t0 + 2 * BEAT);
    const n = Math.round((end - t0 - 0.004) * SR), i0 = Math.round(t0 * SR);
    let hz = 32.703 * Math.pow(2, pcOf(e.pc) / 12 + (meta.tuneCents || 0) / 1200 + o.kickTune / 12);
    while (hz >= 2 * o.bassHi) hz /= 2; while (hz < 2 * o.bassLo) hz *= 2;
    let a = 0, b = 0.37, ph = 0;
    for (let i = 0; i < n && i0 + i < N; i++) {
      a = (a + hz * 1.004 / SR) % 1; b = (b + hz / 1.004 / SR) % 1; ph += 2 * Math.PI * hz / SR;
      const env = Math.min(1, i / (0.002 * SR)) * (1 - 0.35 * i / n) * Math.min(1, (n - i) / (0.005 * SR));
      bus[i0 + i] += Math.tanh(2.2 * (0.35 * (2 * a - 1) + 0.35 * (2 * b - 1) + 0.6 * Math.sin(ph))) * env;
    }
  });
  const y = filt(filt(bus, "low", 1400, 2), "high", 35, 2), d1 = meta.blocks.find(b => b.kind === "drop");
  const g = d1 ? Math.pow(10, (-12 - rmsDb(y, Math.round((OB0 + d1.out * meta.BAR) * SR), Math.round((OB0 + (d1.out + d1.n) * meta.BAR) * SR))) / 20) : 0.3;
  for (let i = 0; i < N; i++) y[i] *= g;
  return [y, Float32Array.from(y)];
}

/* ------------------------------------------------------------ mix helpers */
function place(bus, y, t, gain = 1) {
  let i = Math.round(t * SR), off = 0; if (i < 0) { off = -i; i = 0; }
  for (let c = 0; c < 2; c++) { const s = y[c], d = bus[c], n = Math.min(s.length - off, d.length - i); for (let k = 0; k < n; k++) d[i + k] += s[off + k] * gain; }
}
function smooth(x, w) { const o = zeros(x.length), h = w >> 1; let s = 0; for (let i = 0; i < x.length + h; i++) { if (i < x.length) s += x[i]; if (i - w >= 0) s -= x[i - w]; const j = i - h; if (j >= 0 && j < x.length) o[j] = s / w; } return o; }
function duck(N, times, depth, tau, att = 0.003) {
  const env = new Float32Array(N).fill(1);
  for (const tk of times) {
    const i0 = Math.max(0, Math.round((tk - att) * SR)), i1 = Math.min(N, Math.round((tk + 5 * tau) * SR));
    for (let i = i0; i < i1; i++) { const t = i / SR - tk, g = t < 0 ? 1 - depth * (t + att) / att : 1 - depth * Math.exp(-t / tau); if (g < env[i]) env[i] = g; }
  }
  return env;
}
/** Zero-delay state-variable filter with a per-sample cutoff, applied in place on [a, e). */
function svf(x, a, e, fAt, kind, q = 0.8) {
  let ic1 = 0, ic2 = 0; const k = 1 / q;
  for (let i = a; i < e && i < x.length; i++) {
    const g = Math.tan(Math.PI * Math.min(Math.max(fAt(i), 20), SR * 0.45) / SR), a1 = 1 / (1 + g * (g + k)), a2 = g * a1, a3 = g * a2;
    const xn = x[i], v3 = xn - ic2, v1 = a1 * ic1 + a2 * v3, v2 = ic2 + a2 * ic1 + a3 * v3;
    ic1 = 2 * v1 - ic1; ic2 = 2 * v2 - ic2; x[i] = kind === "hp" ? xn - k * v1 - v2 : v2;
  }
}
/** Feed-forward compressor, peak detector in dB, like a plugin's default ballistics. */
function compress(x, thr, ratio, attMs, relMs) {
  const at = Math.exp(-1 / (attMs * 1e-3 * SR)), rt = Math.exp(-1 / (relMs * 1e-3 * SR)), y = zeros(x.length);
  let env = 0;
  for (let i = 0; i < x.length; i++) {
    const lv = 20 * Math.log10(Math.abs(x[i]) + 1e-9); const c = lv > env ? at : rt; env = c * env + (1 - c) * lv;
    const over = env - thr, gr = over > 0 ? over * (1 - 1 / ratio) : 0; y[i] = x[i] * Math.pow(10, -gr / 20);
  }
  return y;
}

/* ------------------------------------------------------------ master */
/** BS.1770 integrated loudness (K-weighted, gated). */
function lufs(st) {
  const kw = st.map(c => { const y = Float32Array.from(c); biquad(y, rbj("highshelf", 1681.97, 0.7072, SR, 4.0)); biquad(y, rbj("high", 38.135, 0.5003, SR)); return y; });
  const blk = Math.round(0.4 * SR), hop = Math.round(0.1 * SR), z = [];
  for (let s = 0; s + blk <= kw[0].length; s += hop) { let e = 0; for (const c of kw) for (let i = s; i < s + blk; i++) e += c[i] * c[i]; z.push(e / blk); }
  const L = v => -0.691 + 10 * Math.log10(v + 1e-24);
  let g = z.filter(v => L(v) > -70); if (!g.length) return -70;
  const rel = L(g.reduce((a, b) => a + b, 0) / g.length) - 10; g = g.filter(v => L(v) > rel);
  return L(g.reduce((a, b) => a + b, 0) / g.length);
}
function clipOs(x, drive = 1) { return x.map(c => { const up = resample(c, SR, SR * 2); for (let i = 0; i < up.length; i++) up[i] = Math.tanh(drive * up[i]) / drive; const d = resample(up, SR * 2, SR); const o = zeros(c.length); o.set(d.subarray(0, c.length)); return o; }); }
/** Lookahead limiter on 2x-oversampled peaks, so true peak stays under the ceiling. */
function brickwall(x, ceilDb = -0.3, look = 0.0015) {
  const c = Math.pow(10, ceilDb / 20), n = x[0].length, need = new Float32Array(n);
  const up = x.map(ch => resample(ch, SR, SR * 2));
  for (let i = 0; i < n; i++) { let p = 0; for (const u of up) for (let k = 2 * i; k < 2 * i + 2 && k < u.length; k++) p = Math.max(p, Math.abs(u[k])); for (const ch of x) p = Math.max(p, Math.abs(ch[i])); need[i] = Math.min(1, c / Math.max(p, 1e-9)); }
  const L = Math.round(look * SR), gmin = minFilter(need, 2 * L + 1);
  const rel = smooth(minFilter(gmin, Math.round(0.04 * SR)), Math.round(0.04 * SR)), fast = smooth(gmin, Math.max(1, L));
  return x.map(ch => Float32Array.from(ch, (v, i) => v * Math.min(rel[i], fast[i])));
}
function minFilter(x, w) {   // van Herk / Gil-Werman
  const n = x.length, h = w >> 1, g = new Float32Array(n), hh = new Float32Array(n), o = new Float32Array(n);
  for (let i = 0; i < n; i++) g[i] = i % w === 0 ? x[i] : Math.min(g[i - 1], x[i]);
  for (let i = n - 1; i >= 0; i--) hh[i] = (i === n - 1 || (i + 1) % w === 0) ? x[i] : Math.min(hh[i + 1], x[i]);
  for (let i = 0; i < n; i++) { const a = Math.max(0, i - h), b = Math.min(n - 1, i + h); o[i] = Math.min(hh[a], g[b]); }
  return o;
}
/* Octave balance of S&M HARDTEKK (ANDONIS) + VERITY HARDTEKK (shxnei), dB relative
 * to the loudest band, measured from the full tracks. Only the curve is kept. */
const REF_OCTAVES = [31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000];
const REF_BALANCE = [-5.1, -1.2, -6.9, -9.7, -11.8, -9.8, -7.8, -12.5, -13.2, -21.2];
function octaveBalance(st) {
  const x = mono(st), N = 8192, re = new Float64Array(N), im = new Float64Array(N), w = hann(N), P = new Float64Array(N / 2);
  const step = Math.max(N, Math.floor((x.length - N) / 120));
  for (let s = 0; s + N <= x.length; s += step) { for (let i = 0; i < N; i++) { re[i] = x[s + i] * w[i]; im[i] = 0; } fft(re, im); for (let k = 0; k < N / 2; k++) P[k] += re[k] * re[k] + im[k] * im[k]; }
  const v = REF_OCTAVES.map(c => { let e = 0; for (let k = 1; k < N / 2; k++) { const f = k * SR / N; if (f >= c / Math.SQRT2 && f < c * Math.SQRT2) e += P[k]; } return 10 * Math.log10(e + 1e-20); });
  const m = Math.max(...v); return v.map(a => a - m);
}
/** EQ a stereo mix (in place) toward the reference octave balance; returns the filters applied. */
function matchEq(st, passes = 2, limit = 6) {
  const used = [];
  for (let p = 0; p < passes; p++) {
    const pass = [];
    octaveBalance(st).forEach((v, i) => {
      const g = Math.max(-limit, Math.min(limit, 0.8 * (REF_BALANCE[i] - v)));
      if (Math.abs(g) < 0.3) return;
      const type = i === 0 ? "lowshelf" : i === REF_OCTAVES.length - 1 ? "highshelf" : "peak";
      pass.push({ type, f: type === "lowshelf" ? 45 : type === "highshelf" ? 11000 : REF_OCTAVES[i], q: type === "peak" ? 1.4 : 0.7, g });
    });
    for (const c of st) for (const e of pass) biquad(c, rbj(e.type, e.f, e.q, SR, e.g));
    used.push(...pass);
  }
  return used;
}

/** Sum the layers with per-layer gains (dB) and EQ, high-pass at 25 Hz, match-EQ: what the clipper gets.
 *  Returns the mix and the match-EQ filters applied. */
function mixdown(layers, meta, mixOpts = {}) {
  const N = meta.N, gains = { kick: 0, bass808: 0, bed: 0, vocals: 0, perc: 0, fx: 0, sfx: 0, ...(mixOpts.gains || {}) };
  const eqs = mixOpts.eq || {};
  const mix = st2(N);
  for (const k of Object.keys(gains)) {
    const L = layers[k]; if (!L) continue;
    const g = Math.pow(10, gains[k] / 20), e = eqs[k];
    for (let c = 0; c < 2; c++) {
      let y = L[c];
      if (e && (e.low || e.mid || e.high)) { y = Float32Array.from(y); if (e.low) biquad(y, rbj("lowshelf", 120, 0.7, SR, e.low)); if (e.mid) biquad(y, rbj("peak", 1000, 0.8, SR, e.mid)); if (e.high) biquad(y, rbj("highshelf", 6000, 0.7, SR, e.high)); }
      for (let i = 0; i < N; i++) mix[c][i] += y[i] * g;
    }
  }
  for (const c of mix) biquad(c, rbj("high", 25, Math.SQRT1_2));
  return { mix, eq: mixOpts.matchEq ?? DEFAULTS.matchEq ? matchEq(mix) : [] };
}
/** The render's master in the live preview: the match EQ for the default mix and the loudness it reaches before the clipper. */
function previewTone(layers, meta) { const { mix, eq } = mixdown(layers, meta); return { eq, lufs: lufs(mix) }; }
/** Drive the mix into an oversampled clipper until it reaches the target loudness, then limit true peaks. */
async function master(layers, meta, masks, mixOpts = {}, progress = () => {}) {
  const { mix, eq } = mixdown(layers, meta, mixOpts), N = meta.N;
  log("  match EQ: " + eq.map(e => `${e.g > 0 ? "+" : ""}${e.g.toFixed(1)} dB @ ${e.f} Hz`).join(", "));
  progress(0.2); await tick();
  const target = mixOpts.targetLufs ?? DEFAULTS.targetLufs;
  const run = g => brickwall(brickwall(clipOs(mix.map(c => Float32Array.from(c, v => v * Math.pow(10, g / 20))), 1), -0.8), -0.3);
  let g = target - lufs(mix), out = null, prev = null;
  for (let k = 0; k < 3; k++) {                                       // secant steps: the clipper gives back less than 1 dB per dB
    out = run(g); const l = lufs(out); if (Math.abs(target - l) < 0.15) break;
    const slope = prev ? Math.min(1, Math.max(0.2, (l - prev[1]) / (g - prev[0]))) : 1; prev = [g, l];
    g += (target - l) / slope; progress(0.5 + 0.15 * (k + 1)); await tick();
  }
  const end = Math.min(N, Math.round((meta.OB0 + meta.N_BARS * meta.BAR + 0.5) * SR));
  out = out.map(c => c.slice(0, end));
  fadeOut(out, 0.3); fadeIn(out, 0.01);                                 // the song's fades are in the layers
  return { out, lufs: lufs(out) };
}

/* ------------------------------------------------------------ MIDI */
/** Standard MIDI file of the drum score: kick (tuned notes, ch 1), bass (ch 3, octave 2), snare 38 + hats 42 (ch 10), bell (ch 2, from E4).
 *  Bar 1 starts at tick 0; 480 ticks per beat, 120 per 16th step. */
function encodeMidi(meta) {
  const PPQ = 480, ST = PPQ / 4, sc = meta.score;
  const vlq = n => { const o = [n & 0x7f]; n >>= 7; while (n) { o.unshift((n & 0x7f) | 0x80); n >>= 7; } return o; };
  const str = t => Array.from(t, c => c.charCodeAt(0));
  const track = (name, notes) => {
    const ev = []; for (const [a, b, ch, nt, vel] of notes) { ev.push([a, [0x90 | ch, nt, vel]]); ev.push([b, [0x80 | ch, nt, 0]]); }
    ev.sort((x, y) => x[0] - y[0] || (x[1][0] & 0xf0) - (y[1][0] & 0xf0));
    let d = [0, 0xff, 0x03, ...vlq(name.length), ...str(name)], last = 0;
    for (const [t, m] of ev) { d.push(...vlq(t - last), ...m); last = t; }
    d.push(0, 0xff, 0x2f, 0); return [...str("MTrk"), d.length >>> 24, (d.length >>> 16) & 255, (d.length >>> 8) & 255, d.length & 255, ...d];
  };
  const tempo = Math.round(60e6 / meta.bpm);
  const t0 = [0, 0xff, 0x51, 3, tempo >> 16, (tempo >> 8) & 255, tempo & 255, 0, 0xff, 0x58, 4, 4, 2, 24, 8, 0, 0xff, 0x2f, 0];
  const tracks = [[...str("MTrk"), 0, 0, 0, t0.length, ...t0]];
  const vel = (g, ref) => Math.max(1, Math.min(127, Math.round(g / ref * 110)));
  tracks.push(track("kick", sc.kick.map((e, i) => { const nx = i + 1 < sc.kick.length ? sc.kick[i + 1].s : e.s + 4; return [e.s * ST, Math.min(nx, e.s + 4) * ST, 0, Math.round(hzToMidi(e.hz || 32.703 * Math.pow(2, e.pc / 12))), 110]; })));
  tracks.push(track("bass", (sc.bass || []).map((e, i, a) => [e.s * ST, Math.min(a[i + 1] ? a[i + 1].s : e.s + 2, e.s + 2) * ST, 2, 36 + pcOf(e.pc), 110])));
  tracks.push(track("snare", sc.snare.map(e => [e.s * ST, e.s * ST + ST, 9, 38, vel(e.g ?? 0.22, 0.25)])));
  tracks.push(track("hats", sc.hat.map(e => [e.s * ST, e.s * ST + ST, 9, 42, vel(e.g ?? 0.34, 0.34)])));
  tracks.push(track("bell", sc.bell.map(e => [e.s * ST, e.s * ST + 4 * ST, 1, 64 + (e.semis ?? 0), vel(e.g ?? 0.5, 0.5)])));
  const hdr = [...str("MThd"), 0, 0, 0, 6, 0, 1, 0, tracks.length, PPQ >> 8, PPQ & 255];
  return new Uint8Array([...hdr, ...tracks.flat()]).buffer;
}
/** Read a MIDI file back into a score (as written by encodeMidi, or edited in a DAW at the same tempo). */
function decodeMidi(buf) {
  const b = new Uint8Array(buf), u32 = i => (b[i] << 24 | b[i + 1] << 16 | b[i + 2] << 8 | b[i + 3]) >>> 0;
  const ppq = (b[12] << 8) | b[13], ST = ppq / 4, score = { kick: [], bass: [], snare: [], hat: [], bell: [] };
  let p = 14;
  while (p + 8 <= b.length) {
    const len = u32(p + 4), end = p + 8 + len; let q = p + 8, t = 0, run = 0, name = "";
    while (q < end) {
      let d = 0, c; do { c = b[q++]; d = (d << 7) | (c & 0x7f); } while (c & 0x80); t += d;
      let st = b[q]; if (st & 0x80) q++; else st = run;
      if (st === 0xff) { const ty = b[q++]; let l = 0; do { c = b[q++]; l = (l << 7) | (c & 0x7f); } while (c & 0x80); if (ty === 3) name = String.fromCharCode(...b.slice(q, q + l)).toLowerCase(); q += l; continue; }
      if (st === 0xf0 || st === 0xf7) { let l = 0; do { c = b[q++]; l = (l << 7) | (c & 0x7f); } while (c & 0x80); q += l; continue; }
      run = st; const hi = st & 0xf0, ch = st & 15, d1 = b[q++], d2 = (hi === 0xc0 || hi === 0xd0) ? 0 : b[q++];
      if (hi !== 0x90 || d2 === 0) continue;
      const s = Math.round(t / ST), g = d2 / 110;
      if (ch === 9 || /snare|hat|drum/.test(name)) { if (d1 === 38 || d1 === 40) score.snare.push({ s, g: g * 0.25 }); else if (d1 === 42 || d1 === 44 || d1 === 46) score.hat.push({ s, g: g * 0.34 }); else if (d1 === 36 || d1 === 35) score.kick.push({ s, pc: 2 }); }
      else if (/bass/.test(name) || ch === 2) score.bass.push({ s, pc: ((d1 % 12) + 12) % 12 });
      else if (/bell/.test(name) || ch === 1) score.bell.push({ s, g: g * 0.5, semis: d1 - 64 });
      else score.kick.push({ s, pc: ((d1 % 12) + 12) % 12 });
    }
    p = end;
  }
  return score;
}

/* ------------------------------------------------------------ io */
function encodeWav(st, sr = SR) {
  const n = st[0].length, b = new ArrayBuffer(44 + n * 4), v = new DataView(b);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0, "RIFF"); v.setUint32(4, 36 + n * 4, true); w(8, "WAVE"); w(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 2, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 4, true); v.setUint16(32, 4, true); v.setUint16(34, 16, true); w(36, "data"); v.setUint32(40, n * 4, true);
  for (let i = 0; i < n; i++) for (let c = 0; c < 2; c++) { const s = Math.max(-1, Math.min(1, st[c][i] || 0)); v.setInt16(44 + (i * 2 + c) * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true); }
  return b;
}

const API = { SR, NAMES, DEFAULTS, LOG, separate, separateModel, analyze, autoArrange, parseArrangement, build, mixScore, renderKick, encodeMidi, decodeMidi, shuffleArrangement, master, previewTone, encodeWav, lufs, resample, repitch, stretch, fft, yinTrack, mono };
if (typeof module !== "undefined" && module.exports) module.exports = API; else root.REMIX = API;
})(typeof self !== "undefined" ? self : this);
