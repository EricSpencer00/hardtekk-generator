#!/usr/bin/env python3
"""Objective 'does this actually slap like hardtekk' metrics for a render.

Usage: .venv/bin/python score_tekk.py <render.wav> [--bpm 180]

Metrics:
  tempo_err     beat-tracked tempo of the render vs target (half/double folded)
  kick_align    median |low-band onset - nearest grid beat| in ms (want < 15)
  fof_density   fraction of grid beats carrying a low-band onset in loud bars
  pump_depth    RMS envelope modulation depth at beat rate in loud bars (want > 0.25)
  low_own       drums-band (30-120Hz) energy ratio in loud bars vs quiet bars
"""
import sys
import argparse
import numpy as np
import librosa
from scipy.signal import butter, sosfilt

SR = 44100


def fold(t, target):
    while t < target / 1.5:
        t *= 2
    while t > target * 1.5:
        t /= 2
    return t


def score(path, target_bpm=180.0):
    y, _ = librosa.load(path, sr=SR, mono=True)

    # tempo
    tempo, beats = librosa.beat.beat_track(y=y, sr=SR)
    tempo = fold(float(np.atleast_1d(tempo)[0]), target_bpm)
    tempo_err = abs(tempo - target_bpm)

    # low band = kick territory
    sos = butter(4, 150, "lp", fs=SR, output="sos")
    low = sosfilt(sos, y)
    onset_env = librosa.onset.onset_strength(y=low, sr=SR)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=SR, units="samples",
                                        backtrack=False)

    bs = 60.0 / target_bpm * SR
    # grid phase = best offset explaining the low onsets
    phases = (onsets % bs)
    if len(phases):
        # circular median
        ang = phases / bs * 2 * np.pi
        offset = (np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) / (2 * np.pi) * bs) % bs
    else:
        offset = 0.0
    dev = np.abs(((onsets - offset + bs / 2) % bs) - bs / 2)
    kick_align_ms = float(np.median(dev)) / SR * 1000 if len(dev) else float("nan")

    # loud vs quiet bars
    barlen = int(bs * 4)
    n_bars = max(len(y) // barlen, 1)
    rms = np.array([np.sqrt(np.mean(y[b * barlen:(b + 1) * barlen] ** 2)) for b in range(n_bars)])
    loud = rms > np.percentile(rms, 60)

    # four-on-floor density in loud bars
    hits = 0
    total = 0
    for b in range(n_bars):
        if not loud[b]:
            continue
        for k in range(4):
            pos = b * barlen + offset + k * bs
            total += 1
            if len(onsets) and np.min(np.abs(onsets - pos)) < 0.03 * SR:
                hits += 1
    fof = hits / total if total else 0.0

    # pump depth: per-half-beat RMS alternation inside loud bars
    hb = int(bs / 2)
    depths = []
    for b in range(n_bars):
        if not loud[b]:
            continue
        seg = y[b * barlen + int(offset): (b + 1) * barlen + int(offset)]
        if len(seg) < barlen:
            continue
        r = [np.sqrt(np.mean(seg[i * hb:(i + 1) * hb] ** 2)) for i in range(8)]
        on = np.mean(r[0::2])
        off = np.mean(r[1::2])
        depths.append(abs(on - off) / max(on, off, 1e-9))
    pump = float(np.mean(depths)) if depths else 0.0

    # low-end ownership: 30-120Hz energy, loud bars over quiet bars
    sos2 = butter(4, [30, 120], "bp", fs=SR, output="sos")
    sub = sosfilt(sos2, y)
    sub_rms = np.array([np.sqrt(np.mean(sub[b * barlen:(b + 1) * barlen] ** 2)) for b in range(n_bars)])
    lo_own = float(sub_rms[loud].mean() / max(sub_rms[~loud].mean(), 1e-9)) if (~loud).any() else float("inf")

    return {
        "tempo": tempo, "tempo_err": tempo_err,
        "kick_align_ms": kick_align_ms, "fof_density": fof,
        "pump_depth": pump, "low_own": lo_own,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("wav")
    p.add_argument("--bpm", type=float, default=180.0)
    a = p.parse_args()
    m = score(a.wav, a.bpm)
    print(f"{a.wav}")
    print(f"  tempo        {m['tempo']:6.1f} BPM  (err {m['tempo_err']:.1f})")
    print(f"  kick_align   {m['kick_align_ms']:6.1f} ms   (want < 15)")
    print(f"  fof_density  {m['fof_density']:6.2f}      (want > 0.85)")
    print(f"  pump_depth   {m['pump_depth']:6.2f}      (want > 0.25)")
    print(f"  low_own      {m['low_own']:6.2f}      (want > 1.5)")
