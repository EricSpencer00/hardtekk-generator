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
function percentileAbs(a, p) {
  // histogram on |x| in dB steps; exact enough for gain staging
  const B = 2000, h = new Uint32Array(B); let m = peakAbs(a) || 1;
  for (let i = 0; i < a.length; i++) h[Math.min(B - 1, Math.floor(Math.abs(a[i]) / m * (B - 1)))]++;
  let need = p * a.length, c = 0;
  for (let b = 0; b < B; b++) { c += h[b]; if (c >= need) return (b + 0.5) / (B - 1) * m; }
  return m;
}
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
const lr4 = (x, type, f) => filt(filt(x, type, f, 2), type, f, 2);
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
function resample(x, fromRate, toRate) {
  const ratio = fromRate / toRate, cut = Math.min(1, 1 / ratio) * 0.97;
  const n = Math.floor(x.length / ratio), y = zeros(n), half = SINC_ZC / cut;
  for (let i = 0; i < n; i++) {
    const t = i * ratio, c = Math.floor(t);
    let s = 0;
    const a = Math.max(0, Math.ceil(t - half)), b = Math.min(x.length - 1, Math.floor(t + half));
    for (let j = a; j <= b; j++) {
      const d = Math.abs(t - j) * cut * SINC_RES, k = d | 0;
      if (k >= SINC_TAB.length - 1) continue;
      const fr = d - k; s += x[j] * (SINC_TAB[k] + (SINC_TAB[k + 1] - SINC_TAB[k]) * fr);
    }
    y[i] = s * cut;
  }
  return y;
}
/** Varispeed: pitch and length change together, like a sampler. */
const repitch = (x, semis) => Math.abs(semis) < 1e-4 ? Float32Array.from(x) : resample(x, SR * Math.pow(2, semis / 12), SR);
const repitch2 = (s, semis) => [repitch(s[0], semis), repitch(s[1], semis)];

/* ------------------------------------------------------------ separation (no stems given) */
/** HPSS + centre extraction. Returns stems like Demucs would, approximately:
 *  drums = percussive, bass = harmonic below 120 Hz, vocals = centred harmonic
 *  150 Hz - 6 kHz, other = the rest of the harmonic part. */
async function separate(mix, progress = () => {}) {
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
  for (let j = 0; j < L; j++) { const g = 1 / Math.max(norm[j], 1e-3); for (const s of [out.h, out.v]) { s[0][j] *= g; s[1][j] *= g; } }
  // zero-phase split: subtracting a phase-shifted low-pass would double the bass instead of removing it
  const drums = [zeros(L), zeros(L)], bass = [filtfilt(out.h[0], "low", 120, 4), filtfilt(out.h[1], "low", 120, 4)];
  const other = [zeros(L), zeros(L)];
  for (let c = 0; c < 2; c++) for (let j = 0; j < L; j++) {
    drums[c][j] = mix[c][j] - out.h[c][j];
    other[c][j] = out.h[c][j] - bass[c][j] - out.v[c][j];
  }
  return { drums, bass, other, vocals: out.v };
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
  const kick = filtfilt(drums, "low", 120);
  const { o: ok, hop } = onsetFlux(kick, 256, 2048);
  const kp = peaks(ok, Math.round(0.09 * SR / 256), 0.25).map(i => (i * 256 + 1024) / SR);
  let best = { R: -1 };
  for (let bpm = 70; bpm <= 200; bpm += 0.01) { const r = circFit(kp, null, 60 / bpm); if (r.R > best.R) best = { R: r.R, bpm }; }
  let bpm = best.bpm; while (bpm < 100) bpm *= 2; while (bpm >= 200) bpm /= 2;
  // drum onsets refined to the steepest rise of the broadband envelope
  const { o } = onsetFlux(drums, 128, 1024);
  const pk = peaks(o, Math.round(0.05 * SR / 128), 0.2);
  const env = smooth(Float32Array.from(filt(drums, "high", 60, 2), Math.abs), Math.round(0.002 * SR));
  const ft = pk.map(i => {
    const c = i * 128 + 512, a = Math.max(1, c - Math.round(0.04 * SR)), e = Math.min(env.length - 1, c + Math.round(0.01 * SR));
    let bi = a, bd = -1; for (let j = a; j < e; j++) { const d = env[j] - env[j - 1]; if (d > bd) { bd = d; bi = j; } }
    return bi / SR;
  }), fw = pk.map(i => o[i]);
  let per = 60 / bpm, ph = circFit(ft, fw, per).phase;
  // least-squares period and phase on the strong onsets near the grid
  const fwMed = Float32Array.from(fw).sort()[fw.length >> 1] || 0;
  for (let it = 0; it < 3; it++) {
    let n = 0, sk = 0, st_ = 0, skk = 0, skt = 0;
    ft.forEach((t, j) => { if (fw[j] < fwMed) return; const k = Math.round((t - ph) / per), r = t - (ph + k * per); if (Math.abs(r) > 0.03) return; n++; sk += k; st_ += t; skk += k * k; skt += k * t; });
    if (n < 16) break;
    per = (n * skt - sk * st_) / (n * skk - sk * sk); ph = (st_ - per * sk) / n; ph = ((ph % per) + per) % per;
  }
  return { bpm: 60 / per, b0: ph, R: best.R, nKicks: kp.length };
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
      for (let i = 0; i < N; i++) { const j = s + i; re[i] = j < x.length ? x[j] * w[i] : 0; im[i] = 0; }
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

/** Everything about the source we need: grid, map of parts per beat, chords, key, the 808's notes. */
async function analyze(stems, progress = () => {}) {
  const dm = mono(stems.drums), bm = mono(stems.bass), om = mono(stems.other), vm = mono(stems.vocals);
  const grid = beatGrid(dm); progress(0.2); await tick();
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
  let off = 0, bs = -1;
  for (let o = 0; o < 4; o++) { let s = 0; for (let i = o; i < nb; i += 4) s += nov[i] * ((i - o) % 16 === 0 ? 2 : 1); if (s > bs) { bs = s; off = o; } }
  const b0 = grid.b0 + off * beat; nb -= off;
  for (const k in ev) ev[k] = ev[k].slice(off);
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
  return { bpm: grid.bpm, b0, beat, nBars, gridR: grid.R, ev, chords, key, bassPc, tuneCents };
}

/* ------------------------------------------------------------ arrangement */
/** Sections from the song's own 4-bar blocks: [{name, src, n, kind}].
 *  Same shape as the hand-made v10 arrangement: intro with the pre-drop hook
 *  teased, drop, verse, break, build (hook), drop, verse, build, drop, outro. */
function autoArrange(an) {
  const G = Math.floor(an.nBars / 4), sum = (lane, g) => { let s = 0; for (let i = g * 16; i < g * 16 + 16 && i < lane.length; i++) s += lane[i]; return s; };
  const drum = [], voc = [];
  for (let g = 0; g < G; g++) { drum.push(sum(an.ev.kick, g) >= 12); voc.push(sum(an.ev.vocals, g) >= 8); }   // four on the floor
  const runs = []; for (let g = 0; g < G;) { if (!drum[g]) { g++; continue; } let e = g; while (e < G && drum[e]) e++; runs.push([g, e]); g = e; }
  const used = new Set(), take = gs => gs.forEach(g => used.add(g));
  const preDrop = runs.map(([a]) => a - 1).filter(g => g >= 0 && !drum[g]);
  const T = preDrop.filter(g => voc[g]).pop() ?? preDrop.pop() ?? Math.min(G - 1, 2);
  const freeNoDrum = (from, n) => {
    for (let g = from; g + n <= G; g++) { let ok = true; for (let k = 0; k < n; k++) if (drum[g + k] || used.has(g + k) || !voc[g + k] || g + k === T) ok = false; if (ok) return g; }
    for (let g = 0; g + n <= G; g++) { let ok = true; for (let k = 0; k < n; k++) if (drum[g + k] || used.has(g + k) || g + k === T) ok = false; if (ok) return g; }
    return null;
  };
  const S = [];
  const add = (name, g, n, kind) => { if (g === null || g === undefined) return; S.push({ name, src: g * 4, n: n * 4, kind }); take([...Array(n).keys()].map(k => g + k)); };
  add("intro", 0, Math.min(2, G), "intro");
  const r1 = runs[0] || [2, Math.min(G, 6)];
  add("drop1", r1[0], Math.min(4, r1[1] - r1[0]), "drop");
  add("verse1", freeNoDrum(0, 2), 2, "verse");
  add("break1", freeNoDrum(r1[1], 2), 2, "break");
  S.push({ name: "build1", src: T * 4, n: 4, kind: "build" });
  const rl = runs[runs.length - 1] || r1;
  add("drop2", rl[0], Math.min(3, rl[1] - rl[0]), "drop");
  const spare = [...Array(G).keys()].find(g => drum[g] && !used.has(g));
  if (spare !== undefined) add("drop2b", spare, 1, "drop");
  add("verse2", freeNoDrum(0, 2), 2, "verse");
  const b2 = preDrop.find(g => !used.has(g) && g !== T); if (b2 !== undefined) add("build2", b2, 1, "build");
  const rest = runs.map(([a, b]) => [a, b, [...Array(b - a).keys()].filter(k => !used.has(a + k)).length]).sort((x, y) => y[2] - x[2])[0];
  const d3 = rest && rest[2] ? rest[0] + [...Array(rest[1] - rest[0]).keys()].find(k => !used.has(rest[0] + k)) : r1[0];
  S.push({ name: "drop3", src: d3 * 4, n: Math.min(12, (G - d3) * 4), kind: "drop" });
  S.push({ name: "drop3b", src: T * 4, n: 4, kind: "drop" });
  const lastV = [...Array(G).keys()].filter(g => voc[g] && g + 1 < G).pop();
  if (lastV !== undefined) S.push({ name: "outro", src: Math.max(0, lastV - 1) * 4, n: 8, kind: "outro" });
  return { sections: S, teaser: T * 4 };
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
/** "intro:0:8, drop:16:16, ..." -> sections; kind:src:bars, name auto. */
function parseArrangement(txt) {
  const S = []; const count = {};
  for (const part of txt.split(/[,\n]+/).map(s => s.trim()).filter(Boolean)) {
    const [kind, src, n] = part.split(":"); count[kind] = (count[kind] || 0) + 1;
    S.push({ name: kind + count[kind], kind, src: +src, n: +n });
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
  semis: 4,                 // speed the song up: +4 st = 140 -> 176 BPM (v10 B)
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
  // vocals: high-pass + light compression; vocal activity keys the bed's 1-4 kHz dip
  const lift = Math.pow(10, 4.5 / 20) - 1;
  const voxComp = vox.map(c => compress(filt(c, "high", 90, 2), -20, 2.5, 5, 80));
  const act = (() => {
    const m = mono(voxComp); for (let i = 0; i < m.length; i++) m[i] *= 1 + dropM[i] * lift;
    const p = band(m, 900, 4200), e = smooth(Float32Array.from(p, Math.abs), Math.round(0.04 * SR));
    const nz = Float32Array.from(e.filter(v => v > 1e-4)).sort(), ref = nz.length ? nz[Math.floor(nz.length * 0.7)] : 1;
    return smooth(Float32Array.from(e, v => Math.min(1, v / (ref + 1e-9))), Math.round(0.06 * SR));
  })();
  const bedPres = bedPre.map(c => band(c, 900, 4200));
  progress(0.75); await tick();
  const v = { key, blocks, barInfo, N, N_BARS, BEAT, BAR, OB0, ot, out808, drumM, dropM, bedPre, bedPres, bass808, voxComp, act, sweeps };
  cache.arrange.set(key, v);
  while (cache.arrange.size > 2) cache.arrange.delete(cache.arrange.keys().next().value);
  return v;
}

/** Splice + arrangement + all buses. Returns layers and everything the UI needs.
 *  `cache` (kept by the caller) makes speed and groove changes cheap: a groove
 *  change only redoes the drums; a speed seen before skips the speed-up. */
async function build(stems, samples, an, opts = {}, progress = () => {}, cache = {}) {
  const o = { ...DEFAULTS, ...opts };
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
  score = { kick: clean(score.kick), snare: clean(score.snare), hat: clean(score.hat), bell: clean(score.bell) };
  const hits = score.kick.map(e => { const bar = barOf(e.s), bi = barInfo[bar]; return { t: t(e.s), s: e.s, pc: ((e.pc % 12) + 12) % 12, scale: ctx.scales[bar], lp: ctx.plan[bar].lp, long: bi.kind === "drop", kind: bi.kind, bar }; });
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
  const meta = { tuneCents: ctx.tuneCents, blocks, barInfo, plan: ctx.plan, hits, OB0, BEAT, BAR, N, N_BARS, bpm: 60 / BEAT, key: ctx.key, scale: ctx.scale, arrangement: ctx.arrangement, score };
  const dk = `${o.bassLo}|${o.bassHi}`;
  if (o.bassVoice === "dark" && A.bassDarkKey !== dk) { A.bassDark = darkSub(A, o.bassLo, o.bassHi, ctx.tuneCents || 0); A.bassDarkKey = dk; }
  const bassSrc = o.bassVoice === "dark" ? A.bassDark : A.bass808;
  const dkB = duck(N, kickTimes, 0.9, 0.08);                             // one low sound at a time: the bass gets out of the kick's way
  const bass808 = bassSrc.map(c => Float32Array.from(c, (v, i) => v * (1 - (1 - dropM[i]) * (1 - dkB[i]))));
  const L = { bed, bass808, vocals: vox, perc: layers.perc, fx: layers.fx };
  L.kick = renderKick(samples, meta, o, act);
  score.kick.forEach((e, i) => { e.hz = hits[i].hz; });                // the frequency actually played, for MIDI
  L.sfx = st2(N);
  applyEffects(L, meta, o);
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
    const gap = j + 1 < hits.length ? hits[j + 1].t - h.t : 1;
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
/** Maximus-style: 3 bands, each squashed and saturated, summed hot. */
function soundgoodizer(x) {
  return x.map(c => {
    const lo = lr4(c, "low", 120), rest = Float32Array.from(c, (v, i) => v - lo[i]), mid = lr4(rest, "low", 2500), hi = Float32Array.from(rest, (v, i) => v - mid[i]);
    const out = zeros(c.length);
    for (const [b, thr, att, rel, drive, lift] of [[lo, -20, 10, 90, 1.6, 0], [mid, -24, 3, 60, 1.6, 0.5], [hi, -26, 1, 40, 1.3, 0]]) {
      const r0 = rmsDb(b), y = compress(b, thr, 4, att, rel), g = Math.pow(10, (r0 - rmsDb(y) + lift) / 20);
      for (let i = 0; i < y.length; i++) y[i] *= g;
      const pk = percentileAbs(y, 0.999) || 1;
      for (let i = 0; i < y.length; i++) out[i] += Math.tanh(drive * y[i] / pk) * pk / drive;
    }
    return out;
  });
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
/** EQ a stereo mix (in place) toward the reference octave balance; returns the gains used. */
function matchEq(st, passes = 2, limit = 6) {
  const total = new Array(REF_OCTAVES.length).fill(0);
  for (let p = 0; p < passes; p++) {
    const cur = octaveBalance(st);
    const g = cur.map((v, i) => Math.max(-limit, Math.min(limit, 0.8 * (REF_BALANCE[i] - v))));
    for (const c of st) g.forEach((gi, i) => {
      if (Math.abs(gi) < 0.3) return;
      const f = REF_OCTAVES[i], t = i === 0 ? "lowshelf" : i === REF_OCTAVES.length - 1 ? "highshelf" : "peak";
      biquad(c, rbj(t, t === "lowshelf" ? 45 : t === "highshelf" ? 11000 : f, t === "peak" ? 1.4 : 0.7, SR, gi));
    });
    g.forEach((gi, i) => total[i] += gi);
  }
  return total;
}

/** Sum the layers with per-layer gains (dB) and EQ, then the blown-out master. */
async function master(layers, meta, masks, mixOpts = {}, progress = () => {}) {
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
  const lift = Math.pow(10, 2 / 20) - 1;   // drops hit the master harder
  for (let c = 0; c < 2; c++) { for (let i = 0; i < N; i++) mix[c][i] *= 1 + masks.dropM[i] * lift; biquad(mix[c], rbj("high", 25, Math.SQRT1_2)); }
  if (mixOpts.matchEq ?? DEFAULTS.matchEq) {
    const gains = matchEq(mix);
    log("  match EQ (dB @ 31..16k): " + gains.map(g => g.toFixed(1)).join(" "));
  }
  progress(0.2); await tick();
  const sg = soundgoodizer(mix);
  if (mixOpts.matchEq ?? DEFAULTS.matchEq) matchEq(sg, 1, 4);            // saturation adds mids/air back; trim them again progress(0.5); await tick();
  const target = mixOpts.targetLufs ?? DEFAULTS.targetLufs;
  const run = g => brickwall(brickwall(clipOs(sg.map(c => Float32Array.from(c, v => v * Math.pow(10, g / 20))), 1), -0.8), -0.3);
  let g = target - lufs(sg), out = null;
  for (let k = 0; k < 3; k++) { out = run(g); const l = lufs(out); if (Math.abs(target - l) < 0.15) break; g += target - l; progress(0.5 + 0.15 * (k + 1)); await tick(); }
  const end = Math.min(N, Math.round((meta.OB0 + meta.N_BARS * meta.BAR + 0.5) * SR));
  out = out.map(c => c.slice(0, end));
  fadeOut(out, 0.3); fadeIn(out, 0.01);                                 // the song's fades are in the layers
  return { out, lufs: lufs(out) };
}

/* ------------------------------------------------------------ MIDI */
/** Standard MIDI file of the drum score: kick (tuned notes, ch 1), snare 38 + hats 42 (ch 10), bell (ch 2, from E4).
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
  tracks.push(track("snare", sc.snare.map(e => [e.s * ST, e.s * ST + ST, 9, 38, vel(e.g ?? 0.22, 0.25)])));
  tracks.push(track("hats", sc.hat.map(e => [e.s * ST, e.s * ST + ST, 9, 42, vel(e.g ?? 0.34, 0.34)])));
  tracks.push(track("bell", sc.bell.map(e => [e.s * ST, e.s * ST + 4 * ST, 1, 64 + (e.semis ?? 0), vel(e.g ?? 0.5, 0.5)])));
  const hdr = [...str("MThd"), 0, 0, 0, 6, 0, 1, 0, tracks.length, PPQ >> 8, PPQ & 255];
  return new Uint8Array([...hdr, ...tracks.flat()]).buffer;
}
/** Read a MIDI file back into a score (as written by encodeMidi, or edited in a DAW at the same tempo). */
function decodeMidi(buf) {
  const b = new Uint8Array(buf), u32 = i => (b[i] << 24 | b[i + 1] << 16 | b[i + 2] << 8 | b[i + 3]) >>> 0;
  const ppq = (b[12] << 8) | b[13], ST = ppq / 4, score = { kick: [], snare: [], hat: [], bell: [] };
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

const API = { SR, NAMES, DEFAULTS, LOG, separate, analyze, autoArrange, parseArrangement, build, mixScore, renderKick, encodeMidi, decodeMidi, shuffleArrangement, master, encodeWav, lufs, resample, repitch, yinTrack, mono };
if (typeof module !== "undefined" && module.exports) module.exports = API; else root.REMIX = API;
})(typeof self !== "undefined" ? self : this);
