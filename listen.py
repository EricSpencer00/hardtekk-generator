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


def main():
    p = argparse.ArgumentParser(description="hardtekk render report card")
    p.add_argument("wav", nargs="?", help="render to grade")
    p.add_argument("--bpm", type=float, default=None, help="target BPM (else detected)")
    p.add_argument("--reference", help="measure a real hardtekk track and save it as the target profile")
    a = p.parse_args()

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
