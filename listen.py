#!/usr/bin/env python3
"""My ears: render-and-report objective metrics for a hardtekk render.

I (Claude) can't hear audio, so this is the feedback loop — it turns a WAV into
numbers that map to what a listener notices, and grades each against a target.

Usage:
  # measure a real hardtekk track and lock it in as the target profile
  .venv/bin/python listen.py --reference tekk/some_real_hardtekk.wav

  # grade a render against the saved reference (or genre defaults if none saved)
  .venv/bin/python listen.py output/mytrack_hardtekk.wav [--bpm 170]

Metrics (each maps to a thing we've been tuning):
  lufs          integrated loudness (approx BS.1770 K-weighting)     "louder"
  peak_dbfs     true-ish peak level
  crest_db      peak-minus-RMS — punch vs. over-squashed
  beat_pulse    kick pulse strength every beat, min across thirds    "consistent kick"
  sub_frac      30-80Hz share of energy                               "thump"
  fof_density   fraction of grid beats carrying a low onset (loud)    "four-on-floor"
  roll_density  onsets on off-beat 16ths — high = busy rolls          "sparser rolls"
  drop_contrast loud-section vs quiet-section RMS                     drop dynamics
  air_frac      >8kHz share of energy — high = harsh/fizzy            polish
"""
import sys
import os
import json
import argparse
import numpy as np
import librosa
from scipy.signal import butter, sosfilt, lfilter

SR = 44100
TARGETS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tekk_targets.json")

# Genre-default targets (used until a reference is measured). "dir" = which way
# is better: +1 higher-is-better, -1 lower-is-better, 0 band around target.
GENRE_TARGETS = {
    "lufs":          {"target": -6.0,  "dir": +1, "tol": 1.5},
    "peak_dbfs":     {"target": -0.3,  "dir": 0,  "tol": 0.5},
    "crest_db":      {"target": 8.0,   "dir": 0,  "tol": 3.0},
    "beat_pulse":    {"target": 0.80,  "dir": +1, "tol": 0.15},
    "sub_frac":      {"target": 0.15,  "dir": +1, "tol": 0.06},
    "fof_density":   {"target": 0.85,  "dir": +1, "tol": 0.15},
    "roll_density":  {"target": 0.25,  "dir": -1, "tol": 0.15},
    "drop_contrast": {"target": 1.30,  "dir": +1, "tol": 0.25},
    "air_frac":      {"target": 0.10,  "dir": 0,  "tol": 0.06},
}


def high_shelf(y, sr, f0, gain_db):
    """RBJ high-shelf biquad (used for the K-weighting perceptual tilt)."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / 2 * np.sqrt((A + 1 / A) * (1 / 1.0 - 1) + 2)
    b0 = A * ((A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cw)
    b2 = A * ((A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cw)
    a2 = (A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * alpha
    return lfilter([b0, b1, b2], [a0, a1, a2], y)


def lufs_integrated(y, sr):
    """Approx BS.1770 integrated loudness: K-weight (38Hz HP + ~+4dB shelf),
    400ms gated blocks. Not certified, but consistent for A/B comparison."""
    yk = sosfilt(butter(2, 38, "hp", fs=sr, output="sos"), y)
    yk = high_shelf(yk, sr, 1500.0, 4.0)
    block, hop = int(0.4 * sr), int(0.1 * sr)
    if len(yk) < block:
        return -70.0
    powers = np.array([np.mean(yk[i:i + block] ** 2)
                       for i in range(0, len(yk) - block, hop)])
    lk = -0.691 + 10 * np.log10(powers + 1e-12)
    g = powers[lk > -70]                       # absolute gate
    if not len(g):
        return -70.0
    l_rel = -0.691 + 10 * np.log10(g.mean() + 1e-12) - 10   # relative gate
    g2 = g[(-0.691 + 10 * np.log10(g + 1e-12)) > l_rel]
    g2 = g2 if len(g2) else g
    return float(-0.691 + 10 * np.log10(g2.mean() + 1e-12))


def band_frac(y, sr, lo, hi):
    """Share of total energy in [lo, hi] Hz."""
    if hi >= sr / 2:
        sos = butter(4, lo, "hp", fs=sr, output="sos")
    else:
        sos = butter(4, [lo, hi], "bp", fs=sr, output="sos")
    b = sosfilt(sos, y)
    return float(np.mean(b ** 2) / (np.mean(y ** 2) + 1e-12))


def beat_pulse_min(y, sr, bpm):
    """Onset-envelope autocorrelation at the beat lag, worst of 3 track-thirds —
    a strong steady kick reads near 1.0 everywhere, a sparse one dips in the
    quiet third."""
    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    fps = sr / 512
    lag = int(round(60.0 / bpm * fps))
    vals = []
    for a, b in ((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0)):
        seg = oenv[int(a * len(oenv)):int(b * len(oenv))]
        if len(seg) <= lag:
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, "full")
        mid = len(ac) // 2
        vals.append(ac[mid + lag] / (ac[mid] + 1e-9))
    return float(min(vals)) if vals else 0.0


def analyze(path, bpm=None):
    y, _ = librosa.load(path, sr=SR, mono=True)
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y ** 2)))
    if bpm is None:
        t, _ = librosa.beat.beat_track(y=y, sr=SR)
        bpm = float(np.atleast_1d(t)[0]) or 170.0

    # low band = kick/bass; onsets for four-on-floor + roll density
    low = sosfilt(butter(4, 150, "lp", fs=SR, output="sos"), y)
    oenv = librosa.onset.onset_strength(y=low, sr=SR)
    onsets = librosa.onset.onset_detect(onset_envelope=oenv, sr=SR, units="samples",
                                         backtrack=False)
    bs = 60.0 / bpm * SR
    phases = onsets % bs
    if len(phases):
        ang = phases / bs * 2 * np.pi
        offset = (np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) / (2 * np.pi) * bs) % bs
    else:
        offset = 0.0

    barlen = int(bs * 4)
    n_bars = max(len(y) // barlen, 1)
    bar_rms = np.array([np.sqrt(np.mean(y[b * barlen:(b + 1) * barlen] ** 2))
                        for b in range(n_bars)])
    loud = bar_rms > np.percentile(bar_rms, 60)

    # four-on-floor density and off-beat 16th (roll) density, in loud bars
    fof_hit = fof_tot = roll_hit = roll_tot = 0
    for b in range(n_bars):
        if not loud[b]:
            continue
        for k in range(4):                     # the 4 beats
            pos = b * barlen + offset + k * bs
            fof_tot += 1
            if len(onsets) and np.min(np.abs(onsets - pos)) < 0.03 * SR:
                fof_hit += 1
        for k in range(16):                    # the 16th grid; off-beat 16ths only
            if k % 4 == 0:
                continue
            pos = b * barlen + offset + k * bs / 4
            roll_tot += 1
            if len(onsets) and np.min(np.abs(onsets - pos)) < 0.03 * SR:
                roll_hit += 1
    fof = fof_hit / fof_tot if fof_tot else 0.0
    roll = roll_hit / roll_tot if roll_tot else 0.0

    drop_contrast = (float(bar_rms[loud].mean() / max(bar_rms[~loud].mean(), 1e-9))
                     if (~loud).any() else float("inf"))

    return {
        "bpm": round(bpm, 1),
        "lufs": round(lufs_integrated(y, SR), 2),
        "peak_dbfs": round(20 * np.log10(max(peak, 1e-9)), 2),
        "crest_db": round(20 * np.log10(max(peak, 1e-9) / max(rms, 1e-9)), 2),
        "beat_pulse": round(beat_pulse_min(y, SR, bpm), 2),
        "sub_frac": round(band_frac(y, SR, 30, 80), 3),
        "fof_density": round(fof, 2),
        "roll_density": round(roll, 2),
        "drop_contrast": round(drop_contrast, 2),
        "air_frac": round(band_frac(y, SR, 8000, SR / 2), 3),
    }


def grade(value, spec):
    """Return a one-char verdict for a metric value against its target spec."""
    t, d, tol = spec["target"], spec["dir"], spec["tol"]
    if d == 0:
        return "OK " if abs(value - t) <= tol else "OFF"
    if d > 0:
        return "OK " if value >= t - tol else "LOW"
    return "OK " if value <= t + tol else "HI "


def load_targets():
    if os.path.exists(TARGETS_FILE):
        with open(TARGETS_FILE) as f:
            saved = json.load(f)
        # build specs from the reference's measured values
        specs = {}
        for k, base in GENRE_TARGETS.items():
            if k in saved.get("metrics", {}):
                specs[k] = {**base, "target": saved["metrics"][k]}
            else:
                specs[k] = base
        return specs, saved.get("source", "reference")
    return GENRE_TARGETS, "genre defaults"


def report(metrics, specs, source, title):
    print(f"\n{title}")
    print(f"  bpm {metrics['bpm']}   (targets: {source})")
    order = ["lufs", "peak_dbfs", "crest_db", "beat_pulse", "sub_frac",
             "fof_density", "roll_density", "drop_contrast", "air_frac"]
    for k in order:
        v = metrics[k]
        spec = specs.get(k)
        if spec is None:
            print(f"  {k:14s} {v}")
            continue
        verdict = grade(v, spec)
        arrow = {1: "↑", -1: "↓", 0: "≈"}[spec["dir"]]
        print(f"  [{verdict}] {k:14s} {v:>8}   target {arrow}{spec['target']}")
    print()


KS_MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KS_MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def structure(render, song):
    """Score a remix-app render (render.wav + render.json from tools/remix_node.js, plus its --layers
    WAVs if present) against an offline reference computed from the original song with librosa:
    the song's beats, its backbeat, its section boundaries, its hook and its key."""
    d = json.load(open(render.rsplit(".", 1)[0] + ".json"))
    an = d["analysis"]
    y, _ = librosa.load(song, sr=SR, mono=True)
    hop = 128
    yp = librosa.effects.percussive(y, margin=2)
    env = librosa.onset.onset_strength(y=yp, sr=SR, hop_length=hop)
    # librosa's tempo prior leans to 120 BPM and lands on 2:3 / 3:2 aliases (165 BPM eurodance -> 110);
    # take the candidate whose beat grid holds the onsets tightest, then track beats around it
    t0 = float(librosa.feature.tempo(onset_envelope=env, sr=SR, hop_length=hop)[0])
    tt = librosa.frames_to_time(np.arange(len(env)), sr=SR, hop_length=hop)
    w = np.maximum(env - np.median(env), 0)
    conc = lambda bpm: abs(np.sum(w * np.exp(2j * np.pi * tt * bpm / 60))) / w.sum()
    cands = [c for c in (t0, t0 * 1.5, t0 / 1.5) if 60 <= c <= 200]
    start = max(cands, key=lambda c: max(conc(c + d) for d in np.linspace(-1, 1, 41)))
    _, beats = librosa.beat.beat_track(onset_envelope=env, sr=SR, hop_length=hop, units="time", start_bpm=start)
    per = float(np.median(np.diff(beats)))
    rate = 2 ** (an.get("semis", 4) / 12)
    BEAT, OB0 = d["BEAT"], d["OB0"]
    BAR, SB0 = 4 * BEAT, an["b0"] / rate
    warp = an.get("warp")

    def to_song(t):
        """render time -> original song time (None outside the arranged blocks)"""
        for b in d["blocks"]:
            a = OB0 + b["out"] * BAR
            if a <= t < a + b["n"] * BAR:
                u = (t - a + SB0 + b["src"] * BAR) * rate
                if warp:
                    s, w = np.array(warp["src"]), np.array(warp["dst"])
                    r = (s[-1] - s[0]) / (w[-1] - w[0])
                    u = s[0] + (u - w[0]) * r if u < w[0] else s[-1] + (u - w[-1]) * r if u > w[-1] else float(np.interp(u, w, s))
                return u
        return None

    out = {}
    # 1. kicks on the song's beat: every added kick vs the nearest reference beat or half-beat
    grid = np.sort(np.concatenate([beats, (beats[1:] + beats[:-1]) / 2]))
    offs = []
    for t, *_ in d["hits"]:
        u = to_song(t)
        if u is not None and beats[0] - per < u < beats[-1] + per:
            offs.append(abs(grid[np.argmin(np.abs(grid - u))] - u))
    offs = 1000 * np.array(offs)
    out["kick_grid_ms_median"] = round(float(np.median(offs)), 1)
    out["kick_grid_ms_p90"] = round(float(np.percentile(offs, 90)), 1)
    out["kick_grid_over_15ms"] = round(float(np.mean(offs > 15)), 2)
    # 2. downbeat: the song's kick beats (1, 3) vs snare beats (2, 4) from low vs 1-5 kHz onsets at its beats
    S = np.abs(librosa.stft(yp, n_fft=1024, hop_length=hop))
    fr = librosa.fft_frequencies(sr=SR, n_fft=1024)

    def flux(lo, hi):
        m = np.log1p(100 * S[(fr >= lo) & (fr < hi)])
        return np.concatenate([[0], np.maximum(np.diff(m, axis=1), 0).sum(0)])
    lo, hi = flux(20, 150), flux(1000, 5000)
    bf = librosa.time_to_frames(beats, sr=SR, hop_length=hop)
    pk = lambda x: np.array([x[max(0, f - 3):f + 4].max() for f in bf])
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    dd = z(pk(lo)) - z(pk(hi))
    pairs = np.array([dd[i] - dd[i + 1] for i in range(0, len(dd) - 1, 2)])
    tstat = pairs.mean() / (pairs.std() / np.sqrt(len(pairs)) + 1e-9)
    kick_par = 0 if tstat > 0 else 1
    starts = [to_song(OB0 + b["out"] * BAR + 1e-4) for b in d["blocks"]]
    pars = [int(np.argmin(np.abs(beats - u))) % 2 for u in starts if u is not None]
    out["downbeat_on_kick_beat"] = f"{np.mean([p == kick_par for p in pars]):.0%}" + ("" if abs(tstat) > 3 else f" (weak backbeat evidence, {tstat:+.1f} SE)")
    # 3. sections: reference boundaries = Foote novelty peaks (16-beat checkerboard kernel) on the
    #    self-similarity of beat-synced chroma+MFCC, one per ~8 bars, at least 4 bars apart
    bfr = librosa.time_to_frames(beats, sr=SR, hop_length=512)
    C = librosa.util.sync(librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=512), bfr, pad=False)
    M = librosa.util.sync(librosa.feature.mfcc(y=y, sr=SR, hop_length=512, n_mfcc=13), bfr, pad=False)
    F = np.vstack([librosa.util.normalize(C, axis=0), librosa.util.normalize(M, axis=0)])
    Fn = F / (np.linalg.norm(F, axis=0) + 1e-9); SSM = Fn.T @ Fn
    L = 16; g = np.outer(np.r_[-np.ones(L), np.ones(L)], np.r_[-np.ones(L), np.ones(L)])
    nov = np.array([np.sum(SSM[i - L:i + L, i - L:i + L] * g) if L <= i < len(SSM) - L else 0 for i in range(len(SSM))])
    order = np.argsort(-nov); bnd = []
    for i in order:
        if nov[i] <= 0 or len(bnd) >= max(2, len(beats) // 32): break
        if all(abs(i - j) >= 16 for j in bnd): bnd.append(i)
    bnd = sorted([0] + bnd)
    bt = np.append(beats[np.minimum(bnd, len(beats) - 1)], beats[-1])
    cuts = []
    for b in d["blocks"]:
        for t in (OB0 + b["out"] * BAR + 1e-4, OB0 + (b["out"] + b["n"]) * BAR - 1e-4):
            u = to_song(t)
            if u is not None:
                cuts.append(np.min(np.abs(bt - u)) <= 1.01 * per)
    out["cuts_on_section_lines"] = f"{np.mean(cuts):.0%}"
    # 4. hook: the loudest segment that repeats (chroma similarity to another segment), and its repeats
    rms = librosa.util.sync(librosa.feature.rms(y=y, hop_length=512), librosa.time_to_frames(beats, sr=SR, hop_length=512))[0]
    seg = [(bnd[i], bnd[i + 1] if i + 1 < len(bnd) else len(beats) - 1) for i in range(len(bnd))]
    chs = [librosa.util.normalize(C[:, a:b].mean(1)) if b > a else np.zeros(12) for a, b in seg]
    sim = np.array([[float(chs[i] @ chs[j]) for j in range(len(seg))] for i in range(len(seg))])
    loud = np.array([rms[a:b].mean() if b > a else 0 for a, b in seg]); loud = loud / (loud.max() + 1e-9)
    rep = np.array([max([sim[i, j] for j in range(len(seg)) if abs(i - j) > 1] or [0]) for i in range(len(seg))])
    h = int(np.argmax(loud * rep))
    hook = [i for i in range(len(seg)) if i == h or (sim[h, i] > 0.95 and loud[i] > 0.7)]
    spans = [(beats[seg[i][0]], beats[min(seg[i][1], len(beats) - 1)]) for i in hook]
    inhook = []
    for b in d["blocks"]:
        if b["kind"] != "drop":
            continue
        for q in range(b["n"] * 4):
            u = to_song(OB0 + b["out"] * BAR + (q + 0.5) * BEAT)
            if u is not None:
                inhook.append(any(a <= u < e for a, e in spans))
    out["drop_beats_in_hook"] = f"{np.mean(inhook):.0%}" if inhook else "n/a"
    # 5. kick vs bass: share of the kick's sounding time (<150 Hz) that the bass also sounds
    base = render.rsplit(".", 1)[0]
    if os.path.exists(base + "_kick.wav"):
        import soundfile as sf
        from scipy.signal import butter as bt_, sosfiltfilt
        sos = bt_(4, 150, "lp", fs=SR, output="sos")
        e = []
        for k_ in ("kick", "bass808"):
            x = sosfiltfilt(sos, sf.read(f"{base}_{k_}.wav")[0].mean(1))
            n = len(x) // 441 * 441
            e.append(np.sqrt((x[:n].reshape(-1, 441) ** 2).mean(1)))
        ka, ba = e[0] > 0.1 * np.percentile(e[0], 99), e[1] > 0.1 * np.percentile(e[1], 99)
        out["kick_bass_overlap"] = f"{np.mean(ba[ka]):.0%}"
    # 6. key: Krumhansl on the whole song's chroma vs the engine's key
    ch = librosa.feature.chroma_cqt(y=y, sr=SR).mean(1)
    best = max(((np.corrcoef(np.roll(p, t), ch)[0, 1], t, m) for p, m in ((KS_MAJ, "major"), (KS_MIN, "minor")) for t in range(12)))
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    et, em = an["key"]["tonic"], an["key"]["mode"]
    rel = (best[2] == "major" and em == "minor" and et == (best[1] + 9) % 12) or (best[2] == "minor" and em == "major" and et == (best[1] + 3) % 12)
    out["key"] = f"engine {names[et]} {em}, reference {names[best[1]]} {best[2]}: " + ("match" if (et, em) == (best[1], best[2]) else "relative" if rel else "MISMATCH")
    out["bpm"] = f"engine {an['bpm']:.2f}, reference {60 / per:.2f}"
    return out


def ref_hook_spans(y):
    """Independent hook: the loudest segment that repeats (Foote novelty segments of beat-synced chroma+MFCC,
    chroma similarity > 0.95 to it and loudness > 0.7 of the loudest), as song-time spans."""
    env = librosa.onset.onset_strength(y=librosa.effects.percussive(y, margin=2), sr=SR, hop_length=512)
    _, beats = librosa.beat.beat_track(onset_envelope=env, sr=SR, hop_length=512, units="frames")
    C = librosa.util.sync(librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=512), beats, pad=False)
    M = librosa.util.sync(librosa.feature.mfcc(y=y, sr=SR, hop_length=512, n_mfcc=13), beats, pad=False)
    F = np.vstack([librosa.util.normalize(C, axis=0), librosa.util.normalize(M, axis=0)])
    Fn = F / (np.linalg.norm(F, axis=0) + 1e-9); SSM = Fn.T @ Fn
    L = 16; g = np.outer(np.r_[-np.ones(L), np.ones(L)], np.r_[-np.ones(L), np.ones(L)])
    nov = np.array([np.sum(SSM[i - L:i + L, i - L:i + L] * g) if L <= i < len(SSM) - L else 0 for i in range(len(SSM))])
    bnd = []
    for i in np.argsort(-nov):
        if nov[i] <= 0 or len(bnd) >= max(2, len(beats) // 32): break
        if all(abs(i - j) >= 16 for j in bnd): bnd.append(i)
    bnd = sorted([0] + bnd); bt = librosa.frames_to_time(beats, sr=SR, hop_length=512)
    seg = [(bnd[i], bnd[i + 1] if i + 1 < len(bnd) else len(bt) - 1) for i in range(len(bnd))]
    rms = librosa.util.sync(librosa.feature.rms(y=y, hop_length=512), beats)[0]
    chs = [librosa.util.normalize(C[:, a:b].mean(1)) if b > a else np.zeros(12) for a, b in seg]
    sim = np.array([[float(chs[i] @ chs[j]) for j in range(len(seg))] for i in range(len(seg))])
    loud = np.array([rms[a:b].mean() if b > a else 0 for a, b in seg]); loud = loud / (loud.max() + 1e-9)
    rep = np.array([max([sim[i, j] for j in range(len(seg)) if abs(i - j) > 1] or [0]) for i in range(len(seg))])
    h = int(np.argmax(loud * rep))
    return [(bt[seg[i][0]], bt[min(seg[i][1], len(bt) - 1)]) for i in range(len(seg)) if i == h or (sim[h, i] > 0.95 and loud[i] > 0.7)]


def onsets_refined(x, strong=True):
    """Onsets as the engine defines them: spectral-flux peaks moved to the steepest rise of a 2 ms envelope in the
    40 ms before; `strong` keeps those at or above the median peak strength."""
    env = librosa.onset.onset_strength(y=x, sr=SR, hop_length=128)
    fr = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=128, backtrack=False)
    if strong and len(fr): fr = fr[env[fr] >= np.median(env[fr])]
    from scipy.signal import butter as bt_
    e = np.convolve(np.abs(sosfilt(bt_(2, 60, "hp", fs=SR, output="sos"), x)), np.ones(88) / 88, "same")
    out = []
    for f in fr:
        c = f * 128; a, b = max(1, c - int(0.04 * SR)), min(len(e) - 1, c + int(0.01 * SR))
        out.append((a + int(np.argmax(np.diff(e[a - 1:b])))) / SR if b > a else c / SR)
    return np.array(out)


def card(render, song, stems):
    """Old (continuous) or new (chop) render of tools/remix_node.js against the original song and its true
    (Demucs) stems: timing, stretch, harmony, hook, leakage, kick/bass overlap, loudness, form."""
    import soundfile as sf
    from scipy.signal import sosfiltfilt
    base = render.rsplit(".", 1)[0]; d = json.load(open(base + ".json")); an = d["analysis"]
    chop = d.get("mode") == "chop"; c = d.get("chop") or {}
    BEAT, OB0 = d["BEAT"], d["OB0"]; BAR, STEP = 4 * BEAT, BEAT / 4
    semis = c.get("semis", an.get("semis", 4)); rate = 2 ** (semis / 12)
    out, pct = {}, lambda v: f"{100 * v:.0f}%"
    # render time -> song time, and song time -> render times
    if chop:
        P = c["placed"]
        def to_render(u):
            return [OB0 + p["bar"] * BAR + (u - p["bar0"]) * p["alpha"] / rate for p in P if p["s0"] <= u <= p["t1"]]
        def to_song(t):
            for p in P:
                a = OB0 + p["bar"] * BAR
                if a <= t < a + p["bars"] * BAR: return p["bar0"] + (t - a) * rate / p["alpha"]
    else:
        SB0, warp = an["b0"] / rate, an.get("warp")
        ws, wd = (np.array(warp["src"]), np.array(warp["dst"])) if warp else (None, None)
        def w2s(w):  # warped (rigid) time -> song time
            if not warp: return w
            r = (ws[-1] - ws[0]) / (wd[-1] - wd[0])
            return ws[0] + (w - wd[0]) * r if w < wd[0] else ws[-1] + (w - wd[-1]) * r if w > wd[-1] else float(np.interp(w, wd, ws))
        def s2w(u):
            if not warp: return u
            r = (wd[-1] - wd[0]) / (ws[-1] - ws[0])
            return wd[0] + (u - ws[0]) * r if u < ws[0] else wd[-1] + (u - ws[-1]) * r if u > ws[-1] else float(np.interp(u, ws, wd))
        def to_render(u):
            v = s2w(u) / rate; o = []
            for b in d["blocks"]:
                a = SB0 + b["src"] * BAR
                if a <= v < a + b["n"] * BAR: o.append(OB0 + b["out"] * BAR + v - a)
            return o
        def to_song(t):
            for b in d["blocks"]:
                a = OB0 + b["out"] * BAR
                if a <= t < a + b["n"] * BAR: return w2s((t - a + SB0 + b["src"] * BAR) * rate)
    # 1. grid: every drum hit on the remix's 16th grid, exactly
    ev = [h[0] for h in d["hits"]] + ([OB0 + e["s"] * STEP for k in ("bass", "snare", "hat", "bell") for e in d["score"].get(k, [])] if "score" in d else [])
    res = [abs(t - (OB0 + round((t - OB0) / STEP) * STEP)) * SR for t in ev]
    out["drum_hits_off_grid_samples_max"] = f"{max(res):.3f} ({len(ev)} hits)"
    assert max(res) < 0.5, "a drum hit is off the grid"
    # 2. pocket: each strong vocal onset (true vocal stem) where it plays in the remix, its phase in the remix beat
    #    minus m x its phase in the song's own beat. The song's beat near each onset: a straight grid fitted to the
    #    true drums' kick onsets within +-8 beats (the onsets under 5 kHz when the kick is not significant), period
    #    within 1.5% of the tempo where all drum onsets cluster on beat and 8th. 0 = the vocal sits against our kick
    #    as it sat against the song's kick. Circular, in ms of the remix beat.
    vt, _ = librosa.load(os.path.join(stems, "vocals.wav"), sr=SR, mono=True)
    on = onsets_refined(vt)
    dr, _ = librosa.load(os.path.join(stems, "drums.wav"), sr=SR, mono=True)
    da, dk, dm_ = onsets_refined(dr, strong=False), onsets_refined(sosfiltfilt(butter(4, 150, "lp", fs=SR, output="sos"), dr), strong=False), onsets_refined(sosfiltfilt(butter(4, 5000, "lp", fs=SR, output="sos"), dr), strong=False)
    R_ = lambda t, P: np.abs(np.exp(2j * np.pi * np.outer(1 / np.atleast_1d(P), t)).mean(1))
    bp = np.arange(100, 200, 0.02); P0 = 60 / bp[np.argmax(R_(da, 60 / bp) + R_(da, 30 / bp))]
    # windows of +-8 beats every 4 beats: period and 16th phase from all onsets (beat+8th+16th, ok beyond 2 SE), the
    # beat = the 16th with most kick (+ half snare band) onsets, chosen through the song by Viterbi (0.3 per 16th jumped)
    at = lambda ph, P, t: ph + np.round((t - ph) / P) * P
    win = []
    for cc in np.arange(0, len(dr) / SR, 4 * P0):
        t = da[np.abs(da - cc) <= 8 * P0]
        if len(t) < 8: continue
        Ps = P0 * (1 + np.arange(-0.015, 0.0151, 0.0005)); Pw = Ps[int(np.argmax(R_(t, Ps) + R_(t, Ps / 2) + R_(t, Ps / 4)))]
        z = np.exp(2j * np.pi * 4 * t / Pw).mean()
        if abs(z) <= 2 / np.sqrt(len(t)): continue
        p16 = (np.angle(z) / (2 * np.pi) % 1) * Pw / 4; hit = np.zeros(4)
        for o, w in ((dk, 1.0), (dm_, 0.5)):
            x = (((o[np.abs(o - cc) <= 8 * P0] - p16) / Pw) % 1) * 4; ok = np.abs(x - np.round(x)) < 0.25
            np.add.at(hit, (np.round(x[ok]) % 4).astype(int), w)
        win.append((cc, Pw, [at(p16 + j * Pw / 4, Pw, cc) for j in range(4)], hit / max(hit.sum(), 1e-9)))
    d16 = lambda x, y, P: abs(int(np.round(4 * ((((x - y) / P) + 0.5) % 1 - 0.5))))
    V = [np.array(w[3]) for w in win]; back = []
    for i in range(1, len(win)):
        (c1, P1, B1, _), (c0, P0_, B0, _) = win[i], win[i - 1]
        tr = np.array([[V[i - 1][k] - 0.3 * d16(B1[j], at(B0[k], P0_, c1), P1) for k in range(4)] for j in range(4)])
        back.append(tr.argmax(1)); V[i] = V[i] + tr.max(1)
    beats_w = [None] * len(win)
    if win:
        j = int(np.argmax(V[-1]))
        for i in range(len(win) - 1, -1, -1):
            beats_w[i] = win[i][2][j]
            if i: j = int(back[i - 1][j])
    cw = np.array([w[0] for w in win])
    def phase_at(u):
        if not len(cw): return np.nan
        i = int(np.argmin(np.abs(cw - u)))
        if abs(cw[i] - u) > 4 * P0: return np.nan
        P = win[i][1]; return ((u - at(beats_w[i], P, u)) / P) % 1
    # our beat is where our kick is heard: the steepest rise of the kick layer's 1 ms energy near each hit
    ko = 0.0
    if os.path.exists(base + "_kick.wav"):
        import soundfile as sf_
        kx = sf_.read(base + "_kick.wav")[0].mean(1); ke = np.convolve(kx ** 2, np.ones(44) / 44, "same"); offs = []
        for h in d["hits"][::5]:
            i = int(round(h[0] * SR)); seg = ke[max(0, i - 1320):i + 1320]
            if len(seg) == 2640: offs.append((np.argmax(np.diff(seg)) - 1320) / SR)
        ko = float(np.median(offs)) if offs else 0.0
        out["kick_attack_ms"] = f"{1000 * ko:+.1f} from the grid"
    m = c.get("m", 1) if chop else 1
    wrap = lambda x: (x + 0.5) % 1 - 0.5
    end = OB0 + d["blocks"][-1]["out"] * BAR + d["blocks"][-1]["n"] * BAR
    pk, per_pl, nref = [], {}, 0
    for u in on:
        b = phase_at(u)
        if np.isnan(b): continue
        nref += 1
        if chop:
            for j, p in enumerate(P):
                if p["s0"] <= u <= p["t1"]:
                    t = OB0 + p["bar"] * BAR + (u - p["bar0"]) * p["alpha"] / rate
                    if t < end: e = wrap((t - OB0 - ko) / BEAT - m * b); pk.append(e); per_pl.setdefault(j, []).append((u, e))
        else:
            for t in to_render(u):
                if t < end: pk.append(wrap((t - OB0 - ko) / BEAT - b))
    pa = np.abs(np.array(pk)) * BEAT * 1000
    if not len(pa): pa = np.array([np.nan])
    out["pocket_ms"] = f"median {np.median(pa):.1f}, p90 {np.percentile(pa, 90):.1f}, <=10 ms {pct(np.mean(pa <= 10))}, <=20 ms {pct(np.mean(pa <= 20))} (n {len(pa)}; {nref} of {len(on)} onsets have a song beat; chance median {BEAT * 250:.0f})"
    if per_pl:
        disp = [round(4 * float(np.angle(np.mean(np.exp(2j * np.pi * np.array([e for _, e in v])))) / (2 * np.pi))) for v in per_pl.values()]
        out["phrases_displaced"] = f"{sum(1 for x in disp if x)} of {len(disp)} placements by >=1 16th ({', '.join(f'{x:+d}' for x in disp if x)})"
        dd = [abs(np.mean([e for _, e in sorted(v)[-3:]]) - np.mean([e for _, e in sorted(v)[:3]])) * BEAT * 1000 for v in per_pl.values() if len(v) >= 6]
        if dd: out["within_phrase_drift_ms"] = f"median {np.median(dd):.1f}, p90 {np.percentile(dd, 90):.1f} (first 3 vs last 3 onsets, {len(dd)} placements)"
        # per placement in the drops: its offset (circular mean), drift (first 3 vs last 3 onsets) and spread, in ms;
        # a miss is any of them over 20 ms. Backbeat: is the onset on a kick beat (1/3) in the song and in the remix?
        sb = []                                                         # the song's beats, kick minus snare-band per beat
        kl = sosfiltfilt(butter(4, 120, "lp", fs=SR, output="sos"), dr); sn = sosfiltfilt(butter(4, [1000, 6000], "bp", fs=SR, output="sos"), dr)
        rms = lambda x, t, P: np.sqrt(np.mean(x[max(0, int((t - 0.1 * P) * SR)):max(1, int((t + 0.3 * P) * SR))] ** 2) + 1e-12)
        for i, w in enumerate(win):
            Pw = w[1]; b = at(beats_w[i], Pw, w[0])
            for k in (-2, -1, 0, 1): t = b + k * Pw; sb.append((t, int(round(t / P0)), 20 * np.log10(rms(kl, t, Pw)) - 20 * np.log10(rms(sn, t, Pw))))
        sbt, sbi, sbd = (np.array([x[j] for x in sb]) for j in range(3)) if sb else (np.zeros(0),) * 3
        def parity_at(u):                                             # (song beat index of u, which parity is beat 1/3, evidence in SE)
            if not len(sbt): return None
            near_ = np.abs(sbt - u) <= 16 * P0; ii, dd_ = sbi[near_], sbd[near_]
            if len(ii) < 8: return None
            x = np.where(ii % 2 == 0, 1, -1) * (dd_ - dd_.mean()); se = x.std() / np.sqrt(len(x)) + 1e-9
            return int(sbi[np.argmin(np.abs(sbt - u))]), 0 if x.mean() > 0 else 1, abs(x.mean()) / se
        rows = []
        for j, v in per_pl.items():
            p = P[j]; v = sorted(v); e = np.array([x for _, x in v])
            mu = float(np.angle(np.mean(np.exp(2j * np.pi * e))) / (2 * np.pi)); r = (e - mu + 0.5) % 1 - 0.5
            drift = abs(np.mean(r[-3:]) - np.mean(r[:3])) if len(r) >= 6 else 0.0
            spread = float(np.percentile(np.abs(r), 90))
            u0 = v[0][0]; t0 = OB0 + p["bar"] * BAR + (u0 - p["bar0"]) * p["alpha"] / rate; par = parity_at(u0); bbeat = None
            if par and m == 1 and par[2] > 3:
                song_odd = (par[0] - par[1]) % 2; rem_odd = int(round((t0 - OB0 - ko) / BEAT - e[0])) % 2
                bbeat = song_odd == rem_odd
            ms = lambda x: abs(x) * BEAT * 1000
            cls = []
            a = abs(mu)
            if a > 0.40: cls.append("half-beat")
            elif 0.18 < a <= 0.40: cls.append("16th")
            elif ms(mu) > 20: cls.append("grid phase")
            if ms(drift) > 20: cls.append("tempo drift")
            if ms(spread) > 20 and ms(drift) <= 20: cls.append("scatter")
            if bbeat is False: cls.append("backbeat offset")
            rows.append({"j": j, "i": p["i"], "bar": p["bar"], "kind": p["kind"], "n": len(e), "off_ms": round(ms(mu) * np.sign(mu), 1), "drift_ms": round(ms(drift), 1),
                         "spread_ms": round(ms(spread), 1), "backbeat": bbeat, "bb_se": round(float(par[2]), 1) if par else None, "miss": bool(cls), "why": cls})
        dr_ = [x for x in rows if x["kind"] == "drop"]; mi = [x for x in dr_ if x["miss"]]
        from collections import Counter
        why = Counter(w for x in mi for w in x["why"])
        out["drop_pocket"] = f"{len(mi)} of {len(dr_)} drop placements off by > 20 ms" + (f" ({', '.join(f'{k} {n}' for k, n in why.most_common())})" if mi else "")
        out["placements"] = rows
    if chop:
        se = []
        for p in P:  # the true onset nearest the phrase's onset, against where the engine meant it (bar line + pickup)
            k = np.argmin(np.abs(on - p["on"]))
            if abs(on[k] - p["on"]) < 0.06: se.append(abs(OB0 + p["bar"] * BAR + (on[k] - p["bar0"]) * p["alpha"] / rate - (OB0 + p["bar"] * BAR + p["q"] * BEAT * c["m"])))
        se = 1000 * np.array(se)
        out["phrase_start_vs_bar_ms"] = f"median {np.median(se):.1f}, p90 {np.percentile(se, 90):.1f} (n {len(se)} of {len(P)})" if len(se) else "n/a"
        al = np.array([p["alpha"] for p in P]); u = np.array(sorted({p["i"]: p["alpha"] for p in P}.values()))
        out["stretch"] = f"min {al.min():.2f} median {np.median(al):.2f} max {al.max():.2f}; outside +-12%: {pct(np.mean(np.abs(al - 1) > 0.12))} of placements, {pct(np.mean(np.abs(u - 1) > 0.12))} of phrases"
    # 3. harmony: each bar's bass/kick note (sounding pitch - shift) against the song's own pitch set there
    y, _ = librosa.load(song, sr=SR, mono=True)
    C = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=512); ct = librosa.frames_to_time(np.arange(C.shape[1]), sr=SR, hop_length=512)
    yb, _ = librosa.load(os.path.join(stems, "bass.wav"), sr=11025, mono=True)
    f0 = librosa.yin(yb, fmin=30, fmax=250, sr=11025, frame_length=2048, hop_length=256)
    br = librosa.feature.rms(y=yb, frame_length=2048, hop_length=256)[0][:len(f0)]; bt_ = librosa.frames_to_time(np.arange(len(f0)), sr=11025, hop_length=256)
    def pset(a, b):
        m = C[:, (ct >= a) & (ct < b)]
        if not m.shape[1]: return None
        v = m.mean(1); return set(np.where(v >= 0.5 * v.max())[0])
    def bass_pc(a, b):
        k = (bt_ >= a) & (bt_ < b) & (br > 0.1 * br.max())
        if k.sum() < 4: return None
        h = np.bincount(np.round(12 * np.log2(f0[k] / 440) + 69).astype(int) % 12, weights=br[k], minlength=12); return int(np.argmax(h))
    spans = []
    if chop:
        for r in c["roots"]:
            if r.get("src") and not r["carried"] and r["how"] != "next drop": spans.append(((r["pc"] - semis) % 12, *r["src"]))
    else:
        NM = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        for b in d["blocks"]:
            if b["kind"] != "drop": continue
            for q in range(b["n"]):
                t0 = OB0 + (b["out"] + q) * BAR; hs = [h for h in d["hits"] if t0 <= h[0] < t0 + BAR]
                a, e = to_song(t0 + 1e-4), to_song(t0 + BAR - 1e-4)
                if hs and a is not None and e is not None: spans.append(((NM.index(hs[0][1]) - semis) % 12, a, e))
    ins, same, n_ = [], [], []
    for pc, a, b in spans:
        s = pset(a, b)
        if s is not None: ins.append(pc in s); n_.append(len(s))
        bp = bass_pc(a, b)
        if bp is not None: same.append(pc == bp)
    out["bass_root_in_song_pitch_set"] = f"{pct(np.mean(ins))} of {len(ins)} bars (chance {pct(np.mean(n_) / 12)})" if ins else "n/a"
    out["bass_root_equals_true_bass"] = f"{pct(np.mean(same))} of {len(same)} bars" if same else "n/a"
    # 4. hook: drop bars carrying the engine's hook, and drop bars whose vocal comes from the independent hook
    hs = ref_hook_spans(y); drop_bars = [(b["out"] + q) for b in d["blocks"] if b["kind"] == "drop" for q in range(b["n"])]
    if chop:
        hb = {p["bar"] + k for p in P if p["hook"] and p["kind"] == "drop" for k in range(p["bars"])}
        out["drop_bars_engine_hook"] = pct(len([b for b in drop_bars if b in hb]) / max(1, len(drop_bars)))
    ref = []
    for bar in drop_bars:
        us = [to_song(OB0 + bar * BAR + (k + 0.5) * BEAT) for k in range(4)]; us = [u for u in us if u is not None]
        if us: ref.append(np.mean([any(a <= u < e for a, e in hs) for u in us]) >= 0.5)
    out["drop_bars_in_reference_hook"] = f"{pct(np.mean(ref))} of {len(ref)} drop bars with vocal ({len(drop_bars)} drop bars)" if ref else "n/a"
    if chop:  # repeats: the vocal slices in time order (placed phrases, and each build's stutter hits as one slice)
        seq = sorted([(p["bar"], p["i"]) for p in P] + [(st["bar"], f"stutter{k}") for k, st in enumerate(c.get("stutter") or [{"bar": b["out"] + b["n"] - 2, "hits": 24} for b in d["blocks"] if b["kind"] == "build"]) for _ in range(st["hits"])], key=lambda x: (x[0], str(x[1])))
        run = best = 0; prev = None
        for _, sl in seq: run = run + 1 if sl == prev else 1; best = max(best, run); prev = sl
        drops = [b for b in d["blocks"] if b["kind"] == "drop"]
        dist = [len({p["i"] for p in P if b["out"] <= p["bar"] < b["out"] + b["n"]}) for b in drops]
        mostp = max((sum(1 for p in P if p["i"] == i) for i in {p["i"] for p in P}), default=0)
        same = len(drops) > 1 and all([p["i"] for p in P if drops[0]["out"] <= p["bar"] < drops[0]["out"] + drops[0]["n"]] == [p["i"] for p in P if b["out"] <= p["bar"] < b["out"] + b["n"]] for b in drops[1:])
        out["repeats"] = f"max consecutive plays of one slice {best}; distinct slices per drop {'/'.join(map(str, dist))}; most plays of one phrase {mostp}; drops identical {same}"
    # 5. leakage of the song's drums+bass into the chopped vocal (in-engine separation, measured with --truth)
    if c.get("leak"):
        L = c["leak"]; vl, _ = sf.read(base + "_vocals.wav"); mx, _ = sf.read(render)
        sh = L["db"][1] / max(L["vox"][1], 1e-12); rel = 10 * np.log10(sh * np.mean(vl ** 2) / np.mean(mx ** 2) + 1e-12)
        out["leak_drums_bass_in_vocal"] = f"{pct(sh)} of vocal energy ({pct(L['db'][0] / max(L['vox'][0], 1e-12))} before the 300 Hz high-pass), {rel:.1f} dB vs the mix; high-pass keeps {pct(L['v'][1] / max(L['v'][0], 1e-12))} of what the separator kept of the true vocal"
        if L["bed"]: out["leak_drums_bass_in_backing"] = pct(L["bedDb"] / L["bed"])
    # 6. kick vs bass: share of the kick's sounding time (<150 Hz) that the bass also sounds
    if os.path.exists(base + "_kick.wav"):
        sos = butter(4, 150, "lp", fs=SR, output="sos"); e = []
        for k_ in ("kick", "bass808"):
            x = sosfiltfilt(sos, sf.read(f"{base}_{k_}.wav")[0].mean(1)); n = len(x) // 441 * 441
            e.append(np.sqrt((x[:n].reshape(-1, 441) ** 2).mean(1)))
        ka, ba = e[0] > 0.1 * np.percentile(e[0], 99), e[1] > 0.1 * np.percentile(e[1], 99)
        out["kick_bass_overlap"] = pct(np.mean(ba[ka]))
    mx, _ = sf.read(render)
    out["lufs_peak_nan"] = f"{lufs_integrated(mx.mean(1), SR):.2f} LUFS, peak {20 * np.log10(np.abs(mx).max() + 1e-12):.2f} dBFS, NaN {bool(np.isnan(mx).any())}"
    if d.get("times"): out["render_s"] = " ".join(f"{k} {v:.1f}" for k, v in d["times"].items())
    # 7. form: element entries/exits (kick, bass, backbeat, hats) off the 8-bar lines, outside fills; phrases cut
    nb = d["blocks"][-1]["out"] + d["blocks"][-1]["n"]; fills = d.get("fills") or [False] * nb
    lanes = {"kick": [int((h[0] - OB0) / BAR + 1e-6) for h in d["hits"]]}
    if "score" in d:
        sc = d["score"]
        lanes.update(bass=[e["s"] // 16 for e in sc["bass"]], backbeat=[e["s"] // 16 for e in sc["snare"] if e["s"] % 16 in (4, 12) and abs(e.get("g", 0.22) - 0.22) < 1e-6], hat=[e["s"] // 16 for e in sc["hat"]])
    bad = tot = 0
    for k, bars in lanes.items():
        on_ = [i in set(bars) for i in range(nb)]; prev = None
        for i in range(nb):
            if fills[i]: continue
            if prev is not None and on_[i] != prev:
                tot += 1; bad += i % 8 != 0
            prev = on_[i]
    out["element_changes_off_8bar_lines"] = f"{bad} of {tot} ({'/'.join(lanes)})"
    if chop:
        cut = 0
        for p in P:
            blk = next(b for b in d["blocks"] if b["out"] <= p["bar"] < b["out"] + b["n"])
            cut += p["bar"] + p["bars"] > blk["out"] + blk["n"]
        out["phrases_cut"] = f"{cut} of {len(P)}"
    return out


def main():
    p = argparse.ArgumentParser(description="hardtekk render report card")
    p.add_argument("wav", nargs="?", help="render to grade")
    p.add_argument("--bpm", type=float, default=None, help="target BPM (else detected)")
    p.add_argument("--reference", help="measure a real hardtekk track and save it as the target profile")
    p.add_argument("--song", help="the original song: score the remix-app render's beat, downbeat, cuts, hook, kick/bass and key against it")
    p.add_argument("--stems", help="with --song: the song's true stems (Demucs) for the full card (timing, harmony, hook, form)")
    p.add_argument("--json", action="store_true", help="with --stems: print the card as JSON")
    a = p.parse_args()

    if a.song and a.stems:
        c = card(a.wav, a.song, a.stems)
        if a.json: print(json.dumps(c)); return
        print(f"\nCARD {a.wav}  (reference: {os.path.basename(a.song)})")
        for k, v in c.items():
            if k != "placements": print(f"  {k:32s} {v}")
        for x in c.get("placements", []):
            if x["miss"] and x["kind"] == "drop": print(f"    miss: phrase {x['i']} at bar {x['bar']}: offset {x['off_ms']:+.0f} ms, drift {x['drift_ms']:.0f}, spread {x['spread_ms']:.0f} ({', '.join(x['why'])})")
        return
    if a.song:
        print(f"\nSTRUCTURE {a.wav}  (reference: {os.path.basename(a.song)})")
        for k, v in structure(a.wav, a.song).items():
            print(f"  {k:22s} {v}")
        return

    if a.reference:
        m = analyze(a.reference, a.bpm)
        with open(TARGETS_FILE, "w") as f:
            json.dump({"source": os.path.basename(a.reference), "metrics": m}, f, indent=2)
        report(m, GENRE_TARGETS, "genre defaults", f"REFERENCE {a.reference}")
        print(f"  -> saved target profile to {os.path.basename(TARGETS_FILE)}")
        return

    if not a.wav:
        p.error("give a render to grade, or --reference to set targets")
    specs, source = load_targets()
    m = analyze(a.wav, a.bpm)
    report(m, specs, source, f"RENDER {a.wav}")


if __name__ == "__main__":
    main()
