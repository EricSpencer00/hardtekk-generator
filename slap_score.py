#!/usr/bin/env python3
"""Does the render SLAP? Composite score against targets measured from a
professional hardtekk release (DJ Beatcoin — I Don't Know hardtekk).

Usage: .venv/bin/python slap_score.py <render.wav> [--bpm 161.5]

Axes (each 0-1, scaled against the reference's numbers):
  drop_wait     first drop lands >= 16 bars in (a drop needs a tease)
  contrast      low-band RMS post-drop / 2 bars pre  (ref: 11.7x, want >= 8)
  density_ratio onset density post/pre drop          (ref: 1.2x,  want >= 1.1)
  hit_density   low-band onsets per beat in the drop (ref: 3.0,   want >= 2)
  attack        median kick attack slope dB/15ms     (ref: 5,     want >= 4)
  loudness      drop RMS                             (ref: 0.53,  want >= 0.4)
  tilt          mid+hi share of drop energy          (ref: 0.74,  want >= 0.55)
"""
import argparse
import numpy as np
import librosa
from scipy.signal import butter, sosfilt

SR = 44100


def analyze(path, bpm):
    y, _ = librosa.load(path, sr=SR, mono=True)
    bs = 60.0 / bpm * SR
    barlen = int(bs * 4)
    n_bars = max(len(y) // barlen, 1)

    low = sosfilt(butter(4, 150, "lp", fs=SR, output="sos"), y)
    lrms = np.array([np.sqrt(np.mean(low[b * barlen:(b + 1) * barlen] ** 2)) for b in range(n_bars)])
    rms = np.array([np.sqrt(np.mean(y[b * barlen:(b + 1) * barlen] ** 2)) for b in range(n_bars)])
    oe = librosa.onset.onset_strength(y=y, sr=SR)
    fpb = barlen / 512
    od = np.array([oe[int(b * fpb):int((b + 1) * fpb)].mean() for b in range(n_bars)])

    # THE drop = biggest sustained low-band step-up
    best, db = -1.0, 2
    for b in range(2, max(n_bars - 4, 3)):
        step = lrms[b:b + 4].mean() - lrms[max(0, b - 2):b].mean()
        if step > best:
            best, db = step, b
    pre = slice(max(0, db - 2), db)
    post = slice(db, min(db + 4, n_bars))

    contrast = lrms[post].mean() / max(lrms[pre].mean(), 1e-9)
    dens_ratio = od[post].mean() / max(od[pre].mean(), 1e-9)

    # low-band hits per beat inside the drop (8 bars)
    seg = low[db * barlen:min((db + 8) * barlen, len(y))]
    oer = librosa.onset.onset_strength(y=seg, sr=SR)
    on = librosa.onset.onset_detect(onset_envelope=oer, sr=SR, units="samples")
    beats = len(seg) / bs
    hit_density = len(on) / max(beats, 1e-9)

    slopes = []
    for o in on:
        a = seg[o:o + int(0.015 * SR)]
        if len(a) > 100 and np.abs(a).max() > 0.01:
            slopes.append(20 * np.log10(np.abs(a).max() / max(np.abs(a[:50]).mean(), 1e-5)))
    attack = float(np.median(slopes)) if slopes else 0.0

    dseg = y[db * barlen:min((db + 8) * barlen, len(y))]
    loud = float(np.sqrt(np.mean(dseg ** 2)))
    S = np.abs(librosa.stft(dseg, n_fft=4096)).mean(axis=1)
    fr = librosa.fft_frequencies(sr=SR, n_fft=4096)
    tilt = S[(fr >= 250) & (fr < 8000)].sum() / max(S.sum(), 1e-9)

    return {
        "drop_bar": db, "drop_s": db * barlen / SR, "n_bars": n_bars,
        "contrast": contrast, "density_ratio": dens_ratio,
        "hit_density": hit_density, "attack": attack,
        "loudness": loud, "tilt": tilt,
    }


def score(m):
    cl = lambda v: float(np.clip(v, 0, 1))
    axes = {
        "drop_wait": cl(m["drop_bar"] / 16),
        "contrast": cl(m["contrast"] / 8),
        "density_ratio": cl((m["density_ratio"] - 0.8) / 0.3),
        "hit_density": cl(m["hit_density"] / 2.0),
        "attack": cl(m["attack"] / 4.0),
        "loudness": cl(m["loudness"] / 0.40),
        "tilt": cl(m["tilt"] / 0.55),
    }
    return axes, 100 * np.mean(list(axes.values()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("wav")
    p.add_argument("--bpm", type=float, default=161.5)
    a = p.parse_args()
    m = analyze(a.wav, a.bpm)
    axes, total = score(m)
    print(f"{a.wav}")
    print(f"  drop at bar {m['drop_bar']} ({m['drop_s']:.0f}s of {m['n_bars']} bars)")
    for k, v in axes.items():
        raw = m.get(k if k in m else "", "")
        print(f"  {k:14s} {v:4.2f}   ({m.get(k, m.get({'drop_wait':'drop_bar'}.get(k,k), 0)):.2f} raw)"
              if k in m or k == "drop_wait" else f"  {k:14s} {v:4.2f}")
    print(f"  SLAP SCORE: {total:.0f}/100")
