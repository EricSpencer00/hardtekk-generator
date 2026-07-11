/* Hardtekk Generator — fully client-side port of hardtekk.py.
 * Pipeline: detect BPM + key -> speed up to target BPM -> phase-lock a bar grid
 * to the measured downbeat -> analyze structure for drops -> lay a sample-only
 * arrangement (kick / offbeat bass / snare / bell, no synth) -> sidechain + clip.
 * The kick sample plays at its native pitch (C) — never retuned to the song key.
 */
"use strict";

const KEYS = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"];
const MAJOR_PROFILE = [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88];
const MINOR_PROFILE = [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17];

const FFT_N = 2048, HOP = 512;

/* ---------------- small DSP utils ---------------- */

// iterative radix-2 FFT, real input -> magnitude spectrum (first N/2 bins)
function fftMag(re) {
  const n = re.length;
  const im = new Float64Array(n);
  const r = Float64Array.from(re);
  for (let i = 1, j = 0; i < n; i++) {          // bit reversal
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { const t = r[i]; r[i] = r[j]; r[j] = t; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cwr = 1, cwi = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = i + k + len / 2;
        const tr = r[b] * cwr - im[b] * cwi;
        const ti = r[b] * cwi + im[b] * cwr;
        r[b] = r[a] - tr; im[b] = im[a] - ti;
        r[a] += tr; im[a] += ti;
        const nwr = cwr * wr - cwi * wi;
        cwi = cwr * wi + cwi * wr; cwr = nwr;
      }
    }
  }
  const mag = new Float32Array(n / 2);
  for (let i = 0; i < n / 2; i++) mag[i] = Math.hypot(r[i], im[i]);
  return mag;
}

const hann = (() => {
  const w = new Float32Array(FFT_N);
  for (let i = 0; i < FFT_N; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / FFT_N);
  return w;
})();

// spectral-flux onset envelope; also accumulates a chroma vector for key detection
function onsetEnvelope(y, sr, chromaOut) {
  const nFrames = Math.max(1, Math.floor((y.length - FFT_N) / HOP));
  const env = new Float32Array(nFrames);
  let prev = null;
  const frame = new Float64Array(FFT_N);
  // precompute bin -> pitch class
  let pc = null;
  if (chromaOut) {
    pc = new Int8Array(FFT_N / 2).fill(-1);
    for (let b = 1; b < FFT_N / 2; b++) {
      const f = b * sr / FFT_N;
      if (f > 55 && f < 4000) pc[b] = ((Math.round(12 * Math.log2(f / 440) + 69) % 12) + 12) % 12;
    }
  }
  for (let fIdx = 0; fIdx < nFrames; fIdx++) {
    const off = fIdx * HOP;
    for (let i = 0; i < FFT_N; i++) frame[i] = y[off + i] * hann[i];
    const mag = fftMag(frame);
    if (chromaOut && (fIdx & 3) === 0) {         // chroma every 4th frame is plenty
      for (let b = 1; b < FFT_N / 2; b++) if (pc[b] >= 0) chromaOut[pc[b]] += mag[b];
    }
    let flux = 0;
    if (prev) for (let b = 0; b < mag.length; b++) {
      const d = mag[b] - prev[b];
      if (d > 0) flux += d;
    }
    env[fIdx] = flux;
    prev = mag;
  }
  // normalize + light smoothing
  let mx = 0;
  for (const v of env) mx = Math.max(mx, v);
  if (mx > 0) for (let i = 0; i < env.length; i++) env[i] /= mx;
  return env;
}

function interpAt(a, x) {
  const i = Math.floor(x);
  if (i < 0 || i + 1 >= a.length) return 0;
  return a[i] + (a[i + 1] - a[i]) * (x - i);
}

/* BPM via autocorrelation of the onset envelope with a harmonic sum, so the
 * true tempo beats its half-tempo alias. Returns bpm in [70, 200]. */
function detectBPM(env, sr) {
  const fps = sr / HOP;
  const n = env.length;
  const maxLag = Math.min(n - 1, Math.ceil(60 / 60 * fps * 2)); // up to 2s
  const acf = new Float32Array(maxLag);
  let e0 = 1e-9;
  for (const v of env) e0 += v * v;
  for (let lag = 1; lag < maxLag; lag++) {
    let s = 0;
    for (let i = 0; i + lag < n; i++) s += env[i] * env[i + lag];
    acf[lag] = s / e0;
  }
  let bestBpm = 120, bestScore = -1;
  for (let bpm = 70; bpm <= 200; bpm += 0.25) {
    const lag = 60 * fps / bpm;
    const score = interpAt(acf, lag) + 0.5 * interpAt(acf, lag * 2) + 0.25 * interpAt(acf, lag * 4);
    if (score > bestScore) { bestScore = score; bestBpm = bpm; }
  }
  return bestBpm;
}

function detectKey(chroma) {
  const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
  const corr = (a, b) => {
    const ma = mean(a), mb = mean(b);
    let num = 0, da = 0, db = 0;
    for (let i = 0; i < 12; i++) {
      num += (a[i] - ma) * (b[i] - mb);
      da += (a[i] - ma) ** 2; db += (b[i] - mb) ** 2;
    }
    return num / Math.sqrt(da * db + 1e-12);
  };
  let best = { score: -2, root: 0, mode: "major" };
  for (let shift = 0; shift < 12; shift++) {
    const rolled = [];
    for (let i = 0; i < 12; i++) rolled.push(chroma[(i + shift) % 12]);
    for (const [mode, prof] of [["major", MAJOR_PROFILE], ["minor", MINOR_PROFILE]]) {
      const s = corr(rolled, prof);
      if (s > best.score) best = { score: s, root: shift, mode };
    }
  }
  // Tekk kicks punch in the low mids, not the sub: pick the octave of the
  // root whose fundamental lands closest to ~137 Hz (measured off a
  // reference hardtekk master).
  const hz = m => 440 * Math.pow(2, (m - 69) / 12);
  const midi = [best.root + 36, best.root + 48]
    .reduce((a, b) => (Math.abs(hz(a) - 137) < Math.abs(hz(b) - 137) ? a : b));
  return { key: KEYS[best.root], mode: best.mode, rootMidi: midi };
}

/* Downbeat phase on the STRETCHED audio: the grid period is exactly
 * 60/targetBPM (we stretched by an exact ratio), so only the phase is unknown.
 * Score every beat-phase, then every bar-phase, against the onset envelope. */
function detectOffset(env, sr, beatSamples) {
  const fps = sr / HOP;
  const beatFrames = beatSamples / sr * fps;
  let bestO = 0, bestS = -1;
  for (let o = 0; o < beatFrames; o += 1) {
    let s = 0;
    for (let k = o; k < env.length; k += beatFrames) s += interpAt(env, k);
    if (s > bestS) { bestS = s; bestO = o; }
  }
  let bestP = 0; bestS = -1;
  for (let p = 0; p < 4; p++) {
    let s = 0;
    for (let k = bestO + p * beatFrames; k < env.length; k += beatFrames * 4) s += interpAt(env, k);
    if (s > bestS) { bestS = s; bestP = p; }
  }
  return Math.round((bestO + bestP * beatFrames) / fps * sr) % (beatSamples * 4);
}

/* ---------------- synthesis (ports of make_*) ---------------- */

function mulberry(seed) {                    // deterministic rng
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ---------------- structure analysis (ports) ---------------- */

function cleanRuns(high, minFull = 4, minGap = 3) {
  const n = high.length;
  let b = 0;
  while (b < n) {
    if (!high[b]) {
      let j = b;
      while (j < n && !high[j]) j++;
      if (b > 0 && j < n && high[b - 1] && high[j] && (j - b) < minGap)
        for (let k = b; k < j; k++) high[k] = true;
      b = j;
    } else b++;
  }
  b = 0;
  while (b < n) {
    if (high[b]) {
      let j = b;
      while (j < n && high[j]) j++;
      if ((j - b) < minFull) for (let k = b; k < j; k++) high[k] = false;
      b = j;
    } else b++;
  }
  return high;
}

function findSurges(score, w = 4, minSep = 8, maxDrops = 3, minJump = 0.06) {
  const n = score.length, jump = new Float32Array(n);
  const mean = (a, s, e) => {
    if (e <= s) return null;
    let sum = 0;
    for (let i = s; i < e; i++) sum += a[i];
    return sum / (e - s);
  };
  for (let b = 0; b < n; b++) {
    const before = mean(score, Math.max(0, b - w), b);
    const after = mean(score, b, Math.min(n, b + w));
    if (before !== null && after !== null) jump[b] = after - before;
  }
  const order = [...jump.keys()].sort((a, b) => jump[b] - jump[a]);
  const drops = [];
  for (const b of order) {
    if (jump[b] <= minJump) break;
    if (drops.every(d => Math.abs(b - d) >= minSep)) drops.push(b);
    if (drops.length >= maxDrops) break;
  }
  return drops.sort((a, b) => a - b);
}

function percentile(arr, p) {
  const s = Float32Array.from(arr).sort();
  const idx = (p / 100) * (s.length - 1);
  const lo = Math.floor(idx);
  return s[lo] + (s[Math.min(lo + 1, s.length - 1)] - s[lo]) * (idx - lo);
}

/* Per-bar energy of full mix / onsets (drum proxy) / low band (kick+bass) on
 * the phase-locked grid -> full/sparse labels + drop bars. */
function analyzeStructure(y, env, sr, beatSamples, offset) {
  const barLen = beatSamples * 4;
  const nBars = Math.max(Math.floor((y.length - offset) / barLen), 1);
  const fps = sr / HOP;

  // one-pole lowpass at ~200 Hz for the bass band
  const low = new Float32Array(y.length);
  const alpha = 1 - Math.exp(-2 * Math.PI * 200 / sr);
  let acc = 0;
  for (let i = 0; i < y.length; i++) { acc += alpha * (y[i] - acc); low[i] = acc; }

  const eFull = new Float32Array(nBars), eLow = new Float32Array(nBars), ePerc = new Float32Array(nBars);
  for (let b = 0; b < nBars; b++) {
    const a = offset + b * barLen, z = a + barLen;
    if (z > y.length) break;
    let sf = 0, sl = 0;
    for (let i = a; i < z; i++) { sf += y[i] * y[i]; sl += low[i] * low[i]; }
    eFull[b] = Math.sqrt(sf / barLen);
    eLow[b] = Math.sqrt(sl / barLen);
    const fa = Math.floor(a / sr * fps), fz = Math.floor(z / sr * fps);
    let sp = 0;
    for (let f = fa; f < Math.min(fz, env.length); f++) sp += env[f];
    ePerc[b] = sp / Math.max(fz - fa, 1);
  }
  const norm = a => {
    let mx = 1e-9;
    for (const v of a) mx = Math.max(mx, v);
    return a.map(v => v / mx);
  };
  const ef = norm(eFull), ep = norm(ePerc), el = norm(eLow);
  const score = new Float32Array(nBars);
  for (let b = 0; b < nBars; b++) score[b] = 0.34 * ef[b] + 0.33 * ep[b] + 0.33 * el[b];

  let drops = findSurges(score);
  const lo = percentile(score, 30), hi = percentile(score, 85);
  const thr = lo + 0.4 * (hi - lo);
  let high = Array.from(score, v => v > thr);
  high = cleanRuns(high);

  for (const d of drops) {                 // force build -> silence -> DROP shape
    for (const k of [1, 2]) if (d - k >= 0) high[d - k] = false;
    let end = Math.min(nBars, d + 12);
    for (const nd of drops) if (nd > d) { end = Math.min(end, nd); break; }
    for (let b = d; b < end; b++) high[b] = true;
  }
  drops = [];
  for (let b = 2; b < nBars; b++) if (high[b] && !high[b - 1] && !high[b - 2]) drops.push(b);
  return { high, drops };
}

function highFromDrops(nBars, drops) {
  const high = new Array(nBars).fill(false);
  for (const d of drops) {
    let end = Math.min(nBars, d + 12);
    for (const nd of drops) if (nd > d) { end = Math.min(end, nd); break; }
    for (let b = d; b < end; b++) high[b] = true;
    for (const k of [1, 2]) if (d - k >= 0) high[d - k] = false;
  }
  return high;
}

function planFromStructure(high, drops) {
  const modes = high.map(h => (h ? "full" : "sparse"));
  for (const d of drops)
    for (const k of [1, 2])
      if (d - k >= 0 && modes[d - k] === "sparse") modes[d - k] = "build";
  const sections = [];
  let i = 0, variant = 0, first = true;
  while (i < modes.length) {
    const m = modes[i];
    let j = i;
    while (j < modes.length && modes[j] === m) j++;
    const bars = j - i;
    if (m === "full") sections.push({ name: "drop", variant: variant++, bars });
    else if (m === "build") sections.push({ name: "build", variant: 0, bars });
    else sections.push({ name: first ? "intro" : "break", variant: 0, bars });
    first = false;
    i = j;
  }
  return sections;
}

function planFixed(nBars) {
  const sections = [];
  const intro = Math.min(8, nBars);
  sections.push({ name: "intro", variant: 0, bars: intro });
  const build = Math.min(8, nBars - intro);
  if (build > 0) sections.push({ name: "build", variant: 0, bars: build });
  let remaining = nBars - intro - build, variant = 0;
  while (remaining > 0) {
    const d = Math.min(16, remaining);
    sections.push({ name: "drop", variant, bars: d });
    remaining -= d;
    if (remaining <= 0) break;
    const b = Math.min(8, remaining);
    sections.push({ name: "break", variant: 0, bars: b });
    remaining -= b;
    if (remaining <= 0) break;
    const bu = Math.min(4, remaining);
    sections.push({ name: "build", variant: 0, bars: bu });
    remaining -= bu;
    variant++;
  }
  return sections;
}

/* ---------------- arrangement ---------------- */

const patIdx = p => [...p].flatMap((c, i) => (c === "x" ? [i] : []));
const KICK_MAIN = patIdx("x.x.x.x.x.x.xxx.");
const KICK_SPARSE = patIdx("x...x...x...x...");
const KICK_ROLL = patIdx("x.x.x.x.xxx.xxxx");

function sidechainEnv(n, beatSamples, floor = 0.35) {
  const out = new Float32Array(n);
  const duckLen = Math.floor(beatSamples * 0.5);
  for (let i = 0; i < n; i++) {
    const ph = i % beatSamples;
    out[i] = ph < duckLen ? floor + (1 - floor) * (ph / duckLen) ** 2 : 1;
  }
  return out;
}

function smoothGain(g, sr, ms = 30) {
  const w = Math.max(1, Math.floor(sr * ms / 1000));
  const out = new Float32Array(g.length);
  let sum = 0;
  for (let i = 0; i < g.length; i++) {
    sum += g[i];
    if (i >= w) sum -= g[i - w];
    out[i] = sum / Math.min(i + 1, w);
  }
  return out;
}

function buildArrangement(nSamples, sr, beatSamples, sections, offset, samples) {
  const beat = beatSamples / sr;

  // Sample-only: the kick sample stays at its native pitch (C). No retuning —
  // pitch-shifting it up to the song key made it thin and too high. There are
  // no synth oscillators here, so nothing is tuned to the song's key.
  if (!samples.kick) throw new Error("kick-hardtekk_C.wav is required (sample-only mode).");
  let kick = Float32Array.from(samples.kick.slice(0, Math.floor(beat * 0.45 * sr)));  // tight: gaps make punch
  // tekk kicks live in the mids: hard distortion folds the sample's fundamental
  // into harmonics (reference kick has almost no 60-120Hz energy)
  for (let i = 0; i < kick.length; i++) kick[i] = Math.tanh(kick[i] * 3) * 0.95;
  // tekk bassline = the kick sample itself, lowpassed and short, on 16ths
  const stab = Float32Array.from(kick.slice(0, Math.floor(beat * 0.22 * sr)));
  biquadLP(stab, 0, stab.length, 250, sr);
  {
    let smx = 1e-9;
    for (const v of stab) smx = Math.max(smx, Math.abs(v));
    for (let i = 0; i < stab.length; i++) stab[i] = stab[i] / smx * 0.9;
  }
  const snare = samples.snare;
  const kickAlt = samples.kickAlt ? samples.kickAlt.slice(0, Math.floor(beat * 0.45 * sr)) : null;
  const bell = samples.bell;

  const eighth = Math.floor(beatSamples / 2);
  const pad = beatSamples * 12 + (bell ? bell.length : 0);
  const drums = new Float32Array(nSamples + pad);
  const songGain = new Float32Array(nSamples + pad).fill(1);
  const duckBase = sidechainEnv(drums.length, beatSamples);
  const duck = i => duckBase[(((i - offset) % drums.length) + drums.length) % drums.length];
  const filterRegions = [], dropRegions = [], dropStarts = [];
  let dropIndex = 0;

  const put = (buf, i, sig, amp) => {
    if (!sig) return;
    const end = Math.min(i + sig.length, buf.length);
    for (let k = i; k < end; k++) buf[k] += sig[k - i] * amp;
  };

  let bar = 0;
  for (const { name, variant, bars } of sections) {
    const s0 = Math.min(offset + bar * 4 * beatSamples, drums.length);
    const s1 = Math.min(offset + (bar + bars) * 4 * beatSamples, drums.length);
    if (s1 <= s0) break;

    if (name === "intro") {
      // no hats (sample-only): the intro is just the song, untouched
    } else if (name === "build") {
      filterRegions.push([s0, s1]);
      // song pulled back hard, NO kick: the low end must vanish before the
      // drop so the kick's return IS the drop; build stays calmer than drop
      for (let i = s0; i < s1; i++) songGain[i] = 0.6 + 0.15 * (i - s0) / (s1 - s0);
      const totalBeats = bars * 4;
      for (let b = 0; b < totalBeats; b++) {
        const i = s0 + b * beatSamples;
        const frac = b / totalBeats;
        const divs = frac < 0.6 ? 2 : 4;
        if (snare) for (let s = 0; s < divs; s++)
          put(drums, i + Math.floor(s * beatSamples / divs), snare, 0.22 + 0.2 * frac);
      }
      // (no riser: sample-only — the snare build + the breath below carry the tension)
      const gap0 = Math.max(s1 - (beatSamples >> 1), s0);   // half-beat breath before drop
      const gStart = songGain[gap0];
      for (let i = gap0; i < s1; i++) {
        drums[i] = 0;
        songGain[i] = gStart + (0.3 - gStart) * (i - gap0) / Math.max(s1 - gap0, 1);
      }
    } else if (name === "drop") {
      dropStarts.push(s0);
      dropRegions.push([s0, s1]);
      // song stays slightly tucked through the whole drop: at full volume the
      // master clipper intermodulates it against the bass and eats the 16ths
      for (let i = s0; i < s1; i++) songGain[i] = duck(i) * 0.8;
      const rng = mulberry(1000 + dropIndex);
      put(drums, s0, bell, 0.4);
      const totalE = bars * 8;
      for (let e = 0; e < totalE; e++) {
        const i = s0 + e * eighth;
        const step = e % 16, phrase8 = e >> 6, posInPhrase = e % 64;
        // slam at full force from beat one — the reference doesn't ease in
        const amp = 1.0;
        let idx = KICK_MAIN;
        if (posInPhrase >= 48 && (phrase8 % 2 === 1)) idx = KICK_ROLL;
        if (idx.includes(step)) {
          let k = kick;
          if (kickAlt && step >= 12 && rng() < 0.5) k = kickAlt;
          put(drums, i, k, amp);
        }
        const onBeat = e % 2 === 0;
        const beatInBar = (e >> 1) % 4;
        if (snare && onBeat && (beatInBar === 1 || beatInBar === 3)) put(drums, i, snare, 0.5);
        // rolling 16th kickbass between the kicks — the reference's low-band
        // onsets land on every 16th phase, not just the "and"
        if (onBeat) {
          for (const [s16, amp16] of [[1, 0.75], [2, 1.0], [3, 0.85]])
            put(drums, i + Math.floor(s16 * beatSamples / 4), stab, amp16 * amp);
        }
        // (no hats: sample-only — kick + bass + snare carry the drop)
      }
      dropIndex++;
    } else if (name === "break") {
      put(drums, s0, bell, 0.35);   // no hats (sample-only)
    }
    bar += bars;
  }
  return {
    drums: drums.subarray(0, nSamples),
    songGain: smoothGain(songGain.subarray(0, nSamples), sr),
    filterRegions, dropRegions, dropStarts,
  };
}

// RBJ biquads, applied in place over [a, b)
function biquadRun(x, a, b, c) {
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let i = a; i < b; i++) {
    const xn = x[i];
    const yn = (c.b0 * xn + c.b1 * x1 + c.b2 * x2 - c.a1 * y1 - c.a2 * y2) / c.a0;
    x2 = x1; x1 = xn; y2 = y1; y1 = yn;
    x[i] = yn;
  }
}

function biquadHP(x, a, b, fc, sr, q = 0.707) {
  const w0 = 2 * Math.PI * fc / sr;
  const alpha = Math.sin(w0) / (2 * q), cw = Math.cos(w0);
  biquadRun(x, a, b, { b0: (1 + cw) / 2, b1: -(1 + cw), b2: (1 + cw) / 2,
                       a0: 1 + alpha, a1: -2 * cw, a2: 1 - alpha });
}

function biquadLP(x, a, b, fc, sr, q = 0.707) {
  const w0 = 2 * Math.PI * fc / sr;
  const alpha = Math.sin(w0) / (2 * q), cw = Math.cos(w0);
  biquadRun(x, a, b, { b0: (1 - cw) / 2, b1: 1 - cw, b2: (1 - cw) / 2,
                       a0: 1 + alpha, a1: -2 * cw, a2: 1 - alpha });
}

// peaking EQ (used to scoop 60-120Hz mud out of the drum bus)
function biquadPeak(x, fc, gainDb, q, sr) {
  const A = Math.pow(10, gainDb / 40);
  const w0 = 2 * Math.PI * fc / sr;
  const alpha = Math.sin(w0) / (2 * q), cw = Math.cos(w0);
  biquadRun(x, 0, x.length, { b0: 1 + alpha * A, b1: -2 * cw, b2: 1 - alpha * A,
                              a0: 1 + alpha / A, a1: -2 * cw, a2: 1 - alpha / A });
}

function highpassSweep(y, s0, s1, sr, fStart = 150, fEnd = 2400) {
  const n = s1 - s0;
  for (let c = 0; c < 8; c++) {
    const a = s0 + Math.floor(n * c / 8), b = s0 + Math.floor(n * (c + 1) / 8);
    const f = fStart * Math.pow(fEnd / fStart, c / 7);
    biquadHP(y, a, b, f, sr);          // two passes ≈ the 2nd-order butter
    biquadHP(y, a, b, f, sr);
  }
}

/* ---------------- top-level pipeline ---------------- */

async function loadSamples(ctx) {
  const names = {
    kick: "samples/kick-hardtekk_C.wav",
    kickAlt: "samples/hardtekk-kicks-tension-rebellious.wav",
    snare: "samples/snare-hardtekk-distorted-punch_D%23_major.wav",
    bell: "samples/hardtekk-bell-shot.wav",
  };
  const out = {};
  await Promise.all(Object.entries(names).map(async ([k, url]) => {
    try {
      const buf = await (await fetch(url)).arrayBuffer();
      const audio = await ctx.decodeAudioData(buf);
      let d = audio.getChannelData(0).slice();
      let mx = 1e-9;
      for (const v of d) mx = Math.max(mx, Math.abs(v));
      for (let i = 0; i < d.length; i++) d[i] /= mx;
      let hot = 0;                                // trim leading silence so the
      while (hot < d.length && Math.abs(d[hot]) <= 0.05) hot++;   // hit is on-grid
      if (hot > 0 && hot < d.length) d = d.subarray(hot);
      out[k] = d;
    } catch { out[k] = null; }
  }));
  return out;
}

function toMono(buf) {
  const n = buf.length, ch = buf.numberOfChannels;
  const out = new Float32Array(n);
  for (let c = 0; c < ch; c++) {
    const d = buf.getChannelData(c);
    for (let i = 0; i < n; i++) out[i] += d[i] / ch;
  }
  let mx = 1e-9;
  for (const v of out) mx = Math.max(mx, Math.abs(v));
  for (let i = 0; i < n; i++) out[i] /= mx;
  return out;
}

// speed the whole song up by `rate` (pitch rises with it — the hardtekk way)
async function speedUp(mono, sr, rate) {
  const outLen = Math.ceil(mono.length / rate);
  const ctx = new OfflineAudioContext(1, outLen, sr);
  const buf = ctx.createBuffer(1, mono.length, sr);
  buf.copyToChannel(mono, 0);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.playbackRate.value = rate;
  src.connect(ctx.destination);
  src.start();
  const rendered = await ctx.startRendering();
  return rendered.getChannelData(0).slice();
}

function encodeWav(samples, sr) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const ws = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  ws(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); ws(8, "WAVE");
  ws(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true);
  v.setUint16(34, 16, true); ws(36, "data"); v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

const yield_ = () => new Promise(r => setTimeout(r, 0));

/* ---------------- HPSS drum removal ----------------
 * Reused from the stemacle "gold master" (github.com/EricSpencer00/stem-player,
 * app/index.html): complex STFT -> harmonic/percussive soft-mask split via
 * 17-tap median filtering -> ISTFT. We keep only the HARMONIC stem, which drops
 * the song's drums so ours sit alone. The ISTFT normalization floor (1.0) is the
 * fix ported into stemacle-dsp to stop a start-of-playback transient spike. */
const HP_N = 4096, HP_HOP = 1024, HP_BINS = HP_N / 2 + 1;
function hpHann(n) {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / n);
  return w;
}
function hpFFT(re, im) {
  const N = re.length;
  for (let i = 1, j = 0; i < N; i++) { let bit = N >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
  for (let len = 2; len <= N; len <<= 1) {
    const a = -2 * Math.PI / len, wr0 = Math.cos(a), wi0 = Math.sin(a);
    for (let i = 0; i < N; i += len) {
      let wr = 1, wi = 0;
      for (let j = 0; j < len >> 1; j++) {
        const u = i + j, v = u + (len >> 1);
        const tRe = wr * re[v] - wi * im[v], tIm = wr * im[v] + wi * re[v];
        re[v] = re[u] - tRe; im[v] = im[u] - tIm; re[u] += tRe; im[u] += tIm;
        const nr = wr * wr0 - wi * wi0; wi = wr * wi0 + wi * wr0; wr = nr;
      }
    }
  }
}
function hpIFFT(re, im) {
  for (let i = 0; i < im.length; i++) im[i] = -im[i];
  hpFFT(re, im);
  const N = re.length;
  for (let i = 0; i < N; i++) { re[i] /= N; im[i] = -im[i] / N; }
}
async function hpStft(sig, win) {
  const F = Math.floor((sig.length - HP_N) / HP_HOP) + 1;
  const re = [], im = [], fr = new Float32Array(HP_N), fi = new Float32Array(HP_N);
  for (let f = 0; f < F; f++) {
    const s = f * HP_HOP;
    for (let i = 0; i < HP_N; i++) { fr[i] = (s + i < sig.length ? sig[s + i] : 0) * win[i]; fi[i] = 0; }
    hpFFT(fr, fi);
    re.push(new Float32Array(fr.subarray(0, HP_BINS)));
    im.push(new Float32Array(fi.subarray(0, HP_BINS)));
    if ((f & 63) === 0) await yield_();
  }
  return { re, im, F };
}
async function hpIstft(re, im, F, win) {
  const len = (F - 1) * HP_HOP + HP_N, out = new Float32Array(len), nrm = new Float32Array(len);
  const fr = new Float32Array(HP_N), fi = new Float32Array(HP_N);
  for (let f = 0; f < F; f++) {
    for (let b = 0; b < HP_BINS; b++) { fr[b] = re[f][b]; fi[b] = im[f][b]; }
    for (let b = 1; b < HP_BINS - 1; b++) { fr[HP_N - b] = fr[b]; fi[HP_N - b] = -fi[b]; }
    hpIFFT(fr, fi);
    const s = f * HP_HOP;
    for (let i = 0; i < HP_N; i++) { out[s + i] += fr[i] * win[i]; nrm[s + i] += win[i] * win[i]; }
    if ((f & 63) === 0) await yield_();
  }
  for (let i = 0; i < len; i++) out[i] /= Math.max(nrm[i], 1.0);  // floor: no edge spike
  return out;
}
async function hpMedFilter(spec, F, B, L, axis) {
  const out = new Float32Array(F * B), col = new Float32Array(L), h = L >> 1;
  if (axis === 'h') {
    for (let b = 0; b < B; b++) {
      for (let f = 0; f < F; f++) { for (let k = 0; k < L; k++) { const fi = f - h + k; col[k] = (fi >= 0 && fi < F) ? spec[fi * B + b] : 0; } col.sort(); out[f * B + b] = col[h]; }
      if ((b & 31) === 0) await yield_();
    }
  } else {
    for (let f = 0; f < F; f++) {
      for (let b = 0; b < B; b++) { for (let k = 0; k < L; k++) { const bi = b - h + k; col[k] = (bi >= 0 && bi < B) ? spec[f * B + bi] : 0; } col.sort(); out[f * B + b] = col[h]; }
      if ((f & 31) === 0) await yield_();
    }
  }
  return out;
}
// Two-pass refined HPSS, harmonic stem only. Port of hpss_refined() in
// stemacle-dsp: pass 1 uses a wide (31) horizontal kernel for cleaner sustained
// tracking and a narrow (7) vertical kernel for sharper drum onsets; pass 2
// re-HPSSes the harmonic output and pulls any cell still >60% percussive back
// out as drums. Removes far more of the song's kit than the single-pass split.
async function hpssHarmonic(re, im, F, B) {
  const mag = new Float32Array(F * B);
  for (let f = 0; f < F; f++) for (let b = 0; b < B; b++) mag[f * B + b] = re[f][b] ** 2 + im[f][b] ** 2;
  // Pass 1
  const h1 = await hpMedFilter(mag, F, B, 31, 'h');   // sustained → harmonic
  const p1 = await hpMedFilter(mag, F, B, 7, 'v');    // transient → percussive
  const hRe = re.map(() => new Float32Array(B)), hIm = im.map(() => new Float32Array(B));
  const hMag = new Float32Array(F * B);
  for (let f = 0; f < F; f++) for (let b = 0; b < B; b++) {
    const hh = h1[f * B + b], pp = p1[f * B + b], d = hh + pp + 1e-8, hm = hh / d;
    hRe[f][b] = re[f][b] * hm; hIm[f][b] = im[f][b] * hm;
    hMag[f * B + b] = Math.hypot(hRe[f][b], hIm[f][b]);
  }
  // Pass 2: re-HPSS the harmonic; reclassify leaked transients as drums.
  const h2 = await hpMedFilter(hMag, F, B, 31, 'h');
  const p2 = await hpMedFilter(hMag, F, B, 7, 'v');
  for (let f = 0; f < F; f++) for (let b = 0; b < B; b++) {
    const hh = h2[f * B + b], pp = p2[f * B + b], d = hh + pp + 1e-8;
    if (pp / d > 0.60) { const hm = hh / d; hRe[f][b] *= hm; hIm[f][b] *= hm; }
  }
  return { hRe, hIm };
}
// instant-attack / 50ms-release peak limiter (masked stems can spike past full scale)
function hpLimitPeaks(sig, ceiling = 0.98, releaseSamples = Math.round(44100 * 0.05)) {
  let gain = 1;
  for (let i = 0; i < sig.length; i++) {
    const av = Math.abs(sig[i]);
    const needed = av > ceiling ? ceiling / av : 1;
    gain = needed < gain ? needed : Math.min(needed, gain + (1 - gain) / releaseSamples);
    sig[i] *= gain;
  }
  return sig;
}
// Remove the song's drums: harmonic-only copy, same length as the input.
async function removeDrums(mono) {
  const win = hpHann(HP_N);
  const st = await hpStft(mono, win);
  if (st.F < 1) return mono;                       // too short to analyze
  const { hRe, hIm } = await hpssHarmonic(st.re, st.im, st.F, HP_BINS);
  const rec = await hpIstft(hRe, hIm, st.F, win);
  hpLimitPeaks(rec);
  const out = new Float32Array(mono.length);
  out.set(rec.subarray(0, mono.length));
  return out;
}

async function generate(file, targetBPM, dropAt, log) {
  const ctx = new AudioContext();
  log(`loading ${file.name} ...`);
  const [decoded, samples] = await Promise.all([
    ctx.decodeAudioData(await file.arrayBuffer()),
    loadSamples(ctx),
  ]);
  ctx.close();
  const sr = decoded.sampleRate;
  let mono = toMono(decoded);
  await yield_();

  log("analyzing tempo & key ...");
  const chroma = new Float64Array(12);
  const envSrc = onsetEnvelope(mono, sr, chroma);
  const bpm = detectBPM(envSrc, sr);
  const { key, mode } = detectKey(chroma);  // key/mode for display only; nothing is tuned now
  log(`  detected: <span class="hi">${bpm.toFixed(1)} BPM</span>, key <span class="hi">${key} ${mode}</span>`);
  await yield_();

  if (!targetBPM) {
    // pick the tekk tempo like the pros: a clean speed-up ratio of the source
    // (reference remix used exactly 1.25x), landing in 150-195 near ~165
    const cands = [1, 1.25, 4 / 3, 1.5, 2].map(r => bpm * r).filter(t => t >= 150 && t <= 195);
    targetBPM = cands.length
      ? cands.reduce((a, b) => (Math.abs(a - 165) < Math.abs(b - 165) ? a : b))
      : 165;
    log(`  auto tempo: <span class="hi">${targetBPM.toFixed(1)} BPM</span> (${(targetBPM / bpm).toFixed(2)}x speed-up)`);
  }
  const rate = targetBPM / bpm;
  log(`  speeding up ${bpm.toFixed(1)} → ${targetBPM.toFixed(1)} BPM (${rate.toFixed(2)}x)`);
  mono = await speedUp(mono, sr, rate);
  await yield_();

  // grid: period is exact by construction; measure only the downbeat phase
  const beatSamples = Math.round(60 / targetBPM * sr);
  const barLen = beatSamples * 4;
  log("phase-locking grid to the beat ...");
  const env = onsetEnvelope(mono, sr, null);
  const offset = detectOffset(env, sr, beatSamples);
  log(`  grid: <span class="hi">${targetBPM.toFixed(1)} BPM</span>, downbeat offset ${(offset / sr * 1000).toFixed(0)} ms`);
  await yield_();

  const nBars = Math.max(Math.floor((mono.length - offset) / barLen), 1);
  let high, drops, sections;
  if (dropAt && dropAt.length) {
    drops = [...new Set(dropAt.map(t => Math.max(2, Math.round((t * sr - offset) / barLen))))].sort((a, b) => a - b);
    high = highFromDrops(nBars, drops);
    sections = planFromStructure(high, drops);
    log(`  manual drops at ${dropAt.map(t => t.toFixed(0) + "s").join(", ")}`);
  } else {
    log("finding the drops ...");
    ({ high, drops } = analyzeStructure(mono, env, sr, beatSamples, offset));
    // a drop needs a tease: never drop before bar 16
    if (drops.length && nBars > 24) {
      drops = [...new Set(drops.map(d => Math.max(d, 16)))].sort((a, b) => a - b);
      const merged = [drops[0]];
      for (const d of drops.slice(1)) if (d - merged[merged.length - 1] >= 8) merged.push(d);
      drops = merged;
      high = highFromDrops(nBars, drops);
    }
    sections = drops.length ? planFromStructure(high, drops) : planFixed(nBars);
    if (!drops.length) log("  (no clear drop found — using default arrangement)");
  }
  const map = high.map((h, b) => (drops.includes(b) ? '<span class="drop">#</span>' : h ? "#" : ".")).join("");
  log(`  bar map: ${map}`);
  await yield_();

  // Strip the song's OWN drums (HPSS harmonic stem) so ours play over a clean
  // bed. Done after structure analysis, which needs the drums to find the drops.
  log("removing the song's drums (HPSS) — the slow part ...");
  mono = await removeDrums(mono);
  await yield_();

  log(`building arrangement — ${nBars} bars, ${drops.length || "auto"} drop(s) ...`);
  const { drums, songGain, filterRegions, dropRegions, dropStarts } =
    buildArrangement(mono.length, sr, beatSamples, sections, offset, samples);
  for (const [a, b] of filterRegions) highpassSweep(mono, a, b, sr);
  // carve the song's low end out during drops so the tekk kick owns it
  // steep: any low residue smears the low band into a wall and kills the punch
  for (const [a, b] of dropRegions) { biquadHP(mono, a, b, 160, sr); biquadHP(mono, a, b, 160, sr); }
  if (dropStarts.length)
    log(`  drops land at ${dropStarts.map(s => (s / sr).toFixed(0) + "s").join(", ")}`);
  await yield_();

  log("sidechaining + mixing ...");
  // reference master has a hole at 60-120Hz: sub + distorted mids, no mud
  biquadPeak(drums, 90, -9, 1.4, sr);
  const mix = new Float32Array(mono.length);
  for (let i = 0; i < mix.length; i++)
    mix[i] = mono[i] * songGain[i] * 0.75 + drums[i] * 1.0;
  // wall-of-sound master calibrated to a reference hardtekk release:
  // bright 2-8k boost, push into the clipper until LOUD, cap the fizz at 11k
  const bright = mix.slice();
  biquadHP(bright, 0, bright.length, 2000, sr);
  biquadLP(bright, 0, bright.length, 8000, sr);
  let rms = 0;
  for (let i = 0; i < mix.length; i++) { mix[i] += 1.3 * bright[i]; rms += mix[i] * mix[i]; }
  rms = Math.sqrt(rms / mix.length);
  const g = Math.min(Math.max(0.42 / Math.max(rms, 1e-9), 0.8), 6.0);
  const drive = 1.35, td = Math.tanh(drive);
  for (let i = 0; i < mix.length; i++) mix[i] = Math.tanh(mix[i] * g * drive) / td;
  biquadLP(mix, 0, mix.length, 11000, sr);
  let mx = 1e-9;
  for (const v of mix) mx = Math.max(mx, Math.abs(v));
  for (let i = 0; i < mix.length; i++) mix[i] = mix[i] / mx * 0.97;

  return { blob: encodeWav(mix, sr), sr, bpm: targetBPM };
}

/* ---------------- UI ---------------- */

const $ = id => document.getElementById(id);
let currentFile = null;

$("drop").addEventListener("click", () => $("file").click());
$("drop").addEventListener("dragover", e => { e.preventDefault(); $("drop").classList.add("over"); });
$("drop").addEventListener("dragleave", () => $("drop").classList.remove("over"));
$("drop").addEventListener("drop", e => {
  e.preventDefault();
  $("drop").classList.remove("over");
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});
$("file").addEventListener("change", e => { if (e.target.files[0]) setFile(e.target.files[0]); });
$("bpm").addEventListener("input", () => { $("bpmval").textContent = $("bpm").value; });

function setFile(f) {
  currentFile = f;
  $("drop").querySelector(".big").textContent = f.name;
  $("go").disabled = false;
}

$("go").addEventListener("click", async () => {
  if (!currentFile) return;
  $("go").disabled = true;
  $("go").textContent = "cooking ...";
  $("result").style.display = "none";
  const logEl = $("log");
  logEl.style.display = "block";
  logEl.innerHTML = "";
  const log = html => { logEl.innerHTML += html + "\n"; };
  try {
    const dropAt = $("dropat").value.split(",").map(s => parseFloat(s)).filter(v => isFinite(v) && v >= 0);
    const target = $("auto").checked ? null : parseInt($("bpm").value, 10);
    const { blob, bpm } = await generate(currentFile, target, dropAt, log);
    const url = URL.createObjectURL(blob);
    $("player").src = url;
    const base = currentFile.name.replace(/\.[^.]+$/, "");
    $("dl").href = url;
    $("dl").download = `${base}_hardtekk_${Math.round(bpm)}bpm.wav`;
    $("result").style.display = "flex";
    log('<span class="hi">done — press play.</span>');
  } catch (err) {
    log(`<span class="drop">error: ${err.message}</span>`);
    console.error(err);
  } finally {
    $("go").disabled = false;
    $("go").textContent = "generate";
  }
});
