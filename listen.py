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


def main():
    p = argparse.ArgumentParser(description="hardtekk render report card")
    p.add_argument("wav", nargs="?", help="render to grade")
    p.add_argument("--bpm", type=float, default=None, help="target BPM (else detected)")
    p.add_argument("--reference", help="measure a real hardtekk track and save it as the target profile")
    p.add_argument("--song", help="the original song: score the remix-app render's beat, downbeat, cuts, hook, kick/bass and key against it")
    a = p.parse_args()

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
