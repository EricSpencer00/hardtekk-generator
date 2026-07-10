#!/usr/bin/env python3
"""
Hardtekk Remix Generator
Drop a song in, get a hardtekk remix out.

Recipe (the classic hardtekk formula):
  1. Detect the source BPM and key.
  2. Time-stretch the whole track up to ~180 BPM (keeps pitch).
  3. Slam a distorted 4/4 tekk kick under it, tuned to the song's key.
  4. Offbeat bass stab (also in key), offbeat open hats, 16th closed hats.
  5. Sidechain-duck the original against the kick so it pumps.
  6. Soft-clip the master for that crunchy tekk sound.

Usage:
  python hardtekk.py <song.(mp3|wav|m4a|flac|ogg)> [--bpm 180] [-o out.wav]
"""

import argparse
import os
import sys

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt

SR = 44100

KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def detect_key(y, sr):
    """Return (key_name, mode, root_midi) via chroma correlation."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    best = (-2.0, 0, "major")
    for shift in range(12):
        rolled = np.roll(chroma, -shift)
        for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            score = np.corrcoef(rolled, profile)[0, 1]
            if score > best[0]:
                best = (score, shift, mode)
    _, root, mode = best
    # Put the kick root in a club-friendly octave (F1..E2 range)
    midi = 29 + ((root - 5) % 12)  # F1 = 29
    return KEYS[root], mode, midi


def detect_bpm(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    if tempo <= 0:
        return 120.0
    # normalize into a sane range
    while tempo < 70:
        tempo *= 2
    while tempo > 200:
        tempo /= 2
    return tempo


def soft_clip(x, drive=1.0):
    return np.tanh(x * drive) / np.tanh(drive)


def make_kick(freq, beat_len, sr=SR):
    """Distorted hardtekk kick: fast pitch-sweep sine into hard saturation."""
    n = int(beat_len * sr)
    t = np.arange(n) / sr
    # exponential pitch envelope: start high, drop to root fast
    f_start, f_end = 220.0, freq
    sweep = f_end + (f_start - f_end) * np.exp(-t * 55)
    phase = 2 * np.pi * np.cumsum(sweep) / sr
    body = np.sin(phase)
    # amp envelope: punchy, long enough to feel like a bass-kick
    env = np.exp(-t * 7.0)
    env[: int(0.002 * sr)] *= np.linspace(0, 1, int(0.002 * sr))
    kick = body * env
    # the tekk part: heavy tanh distortion, then a second stage
    kick = np.tanh(kick * 6.0)
    kick = np.tanh(kick * 2.5) * 0.95
    # click transient
    click = np.zeros(n)
    nclick = int(0.004 * sr)
    click[:nclick] = np.random.default_rng(3).uniform(-1, 1, nclick) * np.linspace(1, 0, nclick)
    return kick + click * 0.25


def make_bass_stab(freq, dur, sr=SR):
    """Offbeat saw bass stab, one octave above kick root."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    f = freq * 2
    saw = 2 * ((t * f) % 1.0) - 1
    sub = np.sin(2 * np.pi * f * t)
    env = np.exp(-t * 18)
    stab = (0.6 * saw + 0.6 * sub) * env
    return np.tanh(stab * 3.0) * 0.8


def make_hat(dur, open_hat=False, sr=SR):
    n = int(dur * sr)
    t = np.arange(n) / sr
    noise = np.random.default_rng(7 if open_hat else 11).uniform(-1, 1, n)
    # crude highpass: difference filter applied twice
    for _ in range(2):
        noise = np.diff(noise, prepend=0.0)
    noise /= max(np.max(np.abs(noise)), 1e-9)
    decay = 4.0 if open_hat else 35.0
    return noise * np.exp(-t * decay)


def sidechain_env(n_samples, beat_samples, sr=SR, floor=0.55):
    """Per-beat ducking envelope: dips to `floor` on each kick, recovers by the offbeat."""
    one = np.ones(beat_samples)
    duck_len = int(beat_samples * 0.5)
    duck = floor + (1.0 - floor) * (np.linspace(0, 1, duck_len) ** 2)
    one[:duck_len] = duck
    reps = n_samples // beat_samples + 1
    return np.tile(one, reps)[:n_samples]


def load_samples(root_midi, sr=SR):
    """Load real hardtekk samples from samples/; kick gets pitched from C to the song root."""
    sdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")

    def load(name):
        path = os.path.join(sdir, name)
        if not os.path.exists(path):
            return None
        y, _ = librosa.load(path, sr=sr, mono=True)
        return y / max(np.max(np.abs(y)), 1e-9)

    kick = load("kick-hardtekk_C.wav")
    if kick is not None:
        # sample is in C; shift to song root (nearest, within +/-6 semitones)
        steps = ((root_midi - 0) % 12)
        if steps > 6:
            steps -= 12
        if steps:
            kick = librosa.effects.pitch_shift(kick, sr=sr, n_steps=steps)
    return {
        "kick": kick,
        "kick_alt": load("hardtekk-kicks-tension-rebellious.wav"),
        "snare": load("snare-hardtekk-distorted-punch_D#_major.wav"),
        "bell": load("hardtekk-bell-shot.wav"),
    }


def make_riser(dur, sr=SR):
    """White-noise riser with rising pitch tone for build-ups."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    noise = np.random.default_rng(5).uniform(-1, 1, n)
    noise = np.diff(noise, prepend=0.0)
    noise /= max(np.max(np.abs(noise)), 1e-9)
    amp = (t / dur) ** 2
    f = 200 + 1800 * (t / dur) ** 2
    tone = np.sin(2 * np.pi * np.cumsum(f) / sr)
    return (noise * 0.7 + tone * 0.3) * amp


def plan_sections(n_bars):
    """Arrange the track: intro -> build -> drop, then cycle break/build/drop."""
    sections = []

    def add(kind, bars):
        sections.append((kind, bars))

    intro = min(8, n_bars)
    add("intro", intro)
    build = min(8, n_bars - intro)
    if build > 0:
        add("build", build)
    remaining = n_bars - intro - build
    variant = 0
    while remaining > 0:
        d = min(16, remaining)
        add(("drop", variant), d)
        remaining -= d
        if remaining <= 0:
            break
        b = min(8, remaining)
        add("break", b)
        remaining -= b
        if remaining <= 0:
            break
        bu = min(4, remaining)
        add("build", bu)
        remaining -= bu
        variant += 1
    return sections


def highpass_sweep(seg, sr=SR, f_start=120, f_end=1400):
    """Progressively highpass a segment (for build-ups) in 8 chunks. Gentle end
    so the transition into the drop isn't a hard tonal cliff."""
    out = np.copy(seg)
    n = len(seg)
    for c in range(8):
        a, b = n * c // 8, n * (c + 1) // 8
        f = f_start * (f_end / f_start) ** (c / 7)
        sos = butter(2, f, "hp", fs=sr, output="sos")
        out[a:b] = sosfilt(sos, seg[a:b])
    return out


# ---- Rhythm patterns -------------------------------------------------------
# Grid = 8th notes over 2 bars (16 steps). 'x' = kick, '.' = rest.
# The signature hardtekk kick: steady pulse, then a triplet roll to close the phrase.
KICK_MAIN = "x.x.x.x.x.x.xxx."   # x.x.x.x.x.x.xxx  (the pattern you described)
KICK_SPARSE = "x...x...x...x..."  # downbeats only, for easing in / lighter phrases
KICK_ROLL = "x.x.x.x.xxx.xxxx"    # busier, used sparingly for a fill phrase


def pat_idx(pat):
    return [i for i, c in enumerate(pat) if c == "x"]


def smooth_gain(g, sr=SR, ms=30):
    """Moving-average smooth of an automation curve to kill clicks at section edges."""
    w = max(1, int(sr * ms / 1000))
    k = np.ones(w) / w
    return np.convolve(g, k, mode="same")


def clean_runs(high, min_full=4, min_gap=3):
    """Fill short sparse gaps between full sections, then drop short full runs, so
    we get solid drop blocks instead of stuttery 1-2 bar flickers."""
    high = np.array(high, dtype=bool)
    n = len(high)
    b = 0
    while b < n:  # fill short sparse gaps bounded by full on both sides
        if not high[b]:
            j = b
            while j < n and not high[j]:
                j += 1
            if 0 < b and j < n and high[b - 1] and high[j] and (j - b) < min_gap:
                high[b:j] = True
            b = j
        else:
            b += 1
    b = 0
    while b < n:  # remove short full runs
        if high[b]:
            j = b
            while j < n and high[j]:
                j += 1
            if (j - b) < min_full:
                high[b:j] = False
            b = j
        else:
            b += 1
    return high


def find_surges(score, w=4, min_sep=8, max_drops=3, min_jump=0.06):
    """Find the bars where energy steps up the most (mean of next w bars minus
    mean of previous w bars) — the song's most drop-like moments."""
    n = len(score)
    jump = np.zeros(n)
    for b in range(n):
        before = score[max(0, b - w):b]
        after = score[b:min(n, b + w)]
        if len(before) and len(after):
            jump[b] = after.mean() - before.mean()
    drops = []
    for b in np.argsort(jump)[::-1]:
        if jump[b] <= min_jump:
            break
        if all(abs(int(b) - d) >= min_sep for d in drops):
            drops.append(int(b))
        if len(drops) >= max_drops:
            break
    return sorted(drops)


def detect_grid(y, target_bpm, sr=SR, hop=512):
    """Find the exact beat period AND phase that best align a click grid to the
    song's onsets, searching a tight range around the target. The audio is
    stretched *toward* target_bpm but rarely lands exactly on it (the source BPM
    estimate is imperfect), so a hard-coded target grid drifts ~1% and walks out
    of time over the track. Measuring the true period here keeps us locked.

    Searching only ±11% around the target sidesteps the beat-tracker's octave
    confusion: a half/two-thirds-tempo grid also fits the onsets but isn't ours.
    """
    fallback_bs = int(60.0 / target_bpm * sr)
    _, beats = librosa.beat.beat_track(y=y, sr=sr, units="samples")
    beats = np.asarray(beats, dtype=float)
    if len(beats) < 8:
        return fallback_bs, 0

    # Assign each (jittery) detected beat to an integer grid index, then
    # least-squares fit  beat ≈ A + bs·index. This fits ONE constant tempo to the
    # whole track, averaging out per-beat jitter and (unlike a single global
    # estimate) minimizing total phase error, so the grid doesn't drift.
    bs = 60.0 / target_bpm * sr
    A = beats[0]
    for _ in range(5):
        idx = np.round((beats - A) / bs)
        # drop duplicate indices (tracker doubled a beat) to keep the fit clean
        _, uniq = np.unique(idx, return_index=True)
        bi, ii = beats[uniq], idx[uniq]
        M = np.vstack([np.ones_like(ii), ii]).T
        (A, bs), *_ = np.linalg.lstsq(M, bi, rcond=None)
        resid = bi - (A + bs * ii)
        keep = np.abs(resid) < 0.3 * bs
        if keep.sum() >= 4:
            M = np.vstack([np.ones_like(ii[keep]), ii[keep]]).T
            (A, bs), *_ = np.linalg.lstsq(M, bi[keep], rcond=None)

    grid_bpm = 60.0 * sr / bs
    if not (target_bpm * 0.85 < grid_bpm < target_bpm * 1.15):
        return fallback_bs, 0  # fit went sideways; trust the stretch target
    bs = int(round(bs))
    offset = int(round(A)) % (bs * 4)
    return bs, offset


def analyze_structure(y, bpm, offset=0, sr=SR):
    """Watch the song's stems: split into percussive (drums) and low-band (bass+kick),
    and find where they both drop in after a vocal/sparse section. Returns per-bar
    labels ('full'/'sparse') and the drop bar indices, on a grid anchored at `offset`.
    """
    bs = int(60.0 / bpm * sr)
    barlen = bs * 4

    # --- stem split (lightweight): drums vs tonal, plus the bass band ---
    _, perc = librosa.effects.hpss(y)              # percussive stem ≈ drums
    sos_low = butter(4, 200, "lp", fs=sr, output="sos")
    low = sosfilt(sos_low, y)                       # kick + bassline

    n_bars = max((len(y) - offset) // barlen, 1)
    e_full = np.zeros(n_bars)
    e_perc = np.zeros(n_bars)
    e_low = np.zeros(n_bars)
    for b in range(n_bars):
        a = offset + b * barlen
        z = a + barlen
        if z <= len(y):
            e_full[b] = np.sqrt(np.mean(y[a:z] ** 2))
            e_perc[b] = np.sqrt(np.mean(perc[a:z] ** 2))
            e_low[b] = np.sqrt(np.mean(low[a:z] ** 2))

    ef = e_full / (e_full.max() + 1e-9)
    ep = e_perc / (e_perc.max() + 1e-9)
    el = e_low / (e_low.max() + 1e-9)
    # a "big moment" = loud + drums + bass all together (chorus / drop)
    score = 0.34 * ef + 0.33 * ep + 0.33 * el

    # --- find the song's biggest surges (max sustained step-up in energy) ---
    drops = find_surges(score, min_sep=8, max_drops=3)

    # texture: which bars are "full" vs "sparse"
    lo, hi = np.percentile(score, 30), np.percentile(score, 85)
    high = score > (lo + 0.4 * (hi - lo))
    high = clean_runs(high, min_full=4, min_gap=3)

    # Force the build→silence→DROP shape around each surge, so the contrast is
    # real even when the song itself is flat: pull drums out for 2 bars before,
    # then slam in and hold through the surge.
    for d in drops:
        for k in (1, 2):
            if d - k >= 0:
                high[d - k] = False
        end = min(n_bars, d + 12)
        for nd in drops:
            if nd > d:
                end = min(end, nd)
                break
        high[d:end] = True

    # recompute drop bar list from the final label edges
    drops = [b for b in range(2, n_bars)
             if high[b] and not high[b - 1] and not high[b - 2]]
    return high, drops, score


def plan_from_structure(high, drops):
    """Turn per-bar full/sparse labels into render sections, inserting a 2-bar
    build-up before each detected drop."""
    modes = ["full" if h else "sparse" for h in high]
    for d in drops:
        for k in (1, 2):
            if d - k >= 0 and modes[d - k] == "sparse":
                modes[d - k] = "build"

    sections = []
    i, variant, first = 0, 0, True
    while i < len(modes):
        m = modes[i]
        j = i
        while j < len(modes) and modes[j] == m:
            j += 1
        bars = j - i
        if m == "full":
            sections.append((("drop", variant), bars))
            variant += 1
        elif m == "build":
            sections.append(("build", bars))
        else:
            sections.append(("intro" if first else "break", bars))
        first = False
        i = j
    return sections


def structure_map(high, drops):
    """ASCII overview: # full, . sparse, ^ under detected drop bars."""
    row = "".join("#" if h else "." for h in high)
    mark = "".join("^" if b in drops else " " for b in range(len(high)))
    return row + "\n  " + mark


def build_arrangement(n_samples, bpm, root_midi, sections, offset=0, sr=SR):
    """Return (drums, song_gain, song_filter_regions, drop_starts). Bars are placed
    starting at `offset` so the whole arrangement is phase-locked to the song."""
    beat = 60.0 / bpm
    bs = int(beat * sr)
    freq = librosa.midi_to_hz(root_midi)

    smp = load_samples(root_midi, sr)
    kick = smp["kick"] if smp["kick"] is not None else make_kick(freq, beat * 0.95)
    kick = kick[: int(beat * 0.95 * sr)]
    dbl_kick = kick[: bs // 2]  # short kick for double-kick 8ths
    stab = make_bass_stab(freq, beat * 0.45)
    chat = make_hat(0.05)
    ohat = make_hat(0.18, open_hat=True)
    snare = smp["snare"]
    kick_alt = smp["kick_alt"][: int(beat * 0.95 * sr)] if smp["kick_alt"] is not None else None
    bell = smp["bell"]

    eighth = bs // 2
    main_idx = pat_idx(KICK_MAIN)
    sparse_idx = pat_idx(KICK_SPARSE)
    roll_idx = pat_idx(KICK_ROLL)

    pad = bs * 4 * 3 + (len(bell) if bell is not None else 0)
    drums = np.zeros(n_samples + pad)
    # Song stays present the whole time; sidechain only ducks, never mutes.
    song_gain = np.full(n_samples + pad, 1.0)
    # duck rolled by offset so ducking troughs land on the phase-locked kicks
    duck = np.roll(sidechain_env(len(drums), bs, floor=0.55), offset)
    filter_regions = []
    drop_starts = []
    drop_index = 0

    def put(buf, i, sig, amp):
        if sig is None:
            return
        buf[i : i + len(sig)] += sig * amp

    bar = 0
    for kind, bars in sections:
        s0 = min(offset + bar * 4 * bs, len(drums))
        s1 = min(offset + (bar + bars) * 4 * bs, len(drums))
        if s1 <= s0:
            break
        name = kind if isinstance(kind, str) else kind[0]
        variant = kind[1] if isinstance(kind, tuple) else 0

        if name == "intro":
            song_gain[s0:s1] = 1.0
            half = bar + bars // 2
            for b in range(half * 4, (bar + bars) * 4):
                i = b * bs
                for s in range(4):  # soft 16th hats sneak in
                    put(drums, i + (s * bs) // 4, chat, 0.12)

        elif name == "build":
            filter_regions.append((s0, s1))
            # song swells but stays audible; gentle rise
            song_gain[s0:s1] = np.linspace(0.9, 1.0, s1 - s0)
            total_beats = bars * 4
            for b in range(total_beats):
                i = s0 + b * bs
                if b < total_beats - 2:
                    put(drums, i, kick, 0.75)
                frac = b / total_beats
                divs = 2 if frac < 0.5 else (4 if frac < 0.8 else 8)
                if snare is not None:
                    for s in range(divs):
                        put(drums, i + (s * bs) // divs, snare, 0.3 + 0.35 * frac)
            riser = make_riser(bars * 4 * beat)
            put(drums, s0, riser, 0.45)
            # short half-beat breath before the drop (dip, not a hard cut)
            gap0 = max(s1 - bs // 2, s0)
            drums[gap0:s1] = 0.0
            song_gain[gap0:s1] = np.linspace(song_gain[gap0], 0.3, s1 - gap0)

        elif name == "drop":
            drop_starts.append(s0)
            song_gain[s0:s1] = duck[s0:s1]
            rng = np.random.default_rng(1000 + drop_index)
            put(drums, s0, bell, 0.4)  # bell marks the drop
            total_e = bars * 8  # eighth-note steps in this section

            for e in range(total_e):
                i = s0 + e * eighth
                step = e % 16          # position in the 2-bar kick pattern
                bar_in = e // 8        # bar index within the drop
                phrase8 = e // 64      # 8-bar phrase index
                pos_in_phrase = e % 64

                # ---- ease the FIRST drop in over its first 4 bars ----
                amp = 1.0
                idx = main_idx
                if drop_index == 0 and bar_in < 4:
                    amp = 0.6 + 0.4 * (bar_in / 4)
                    if bar_in < 2:
                        idx = sparse_idx
                # every 8th phrase-end gets a roll fill (last 2 bars of the phrase)
                elif pos_in_phrase >= 48 and (phrase8 % 2 == 1):
                    idx = roll_idx

                if step in idx:
                    # accent the triplet-roll tail with the alt kick sometimes
                    k = kick
                    if kick_alt is not None and step >= 12 and rng.random() < 0.5:
                        k = kick_alt
                    put(drums, i, k, amp)

                # snare on beats 2 & 4 (skip during the ease-in bars)
                on_beat = (e % 2 == 0)
                beat_in_bar = (e // 2) % 4
                if snare is not None and on_beat and beat_in_bar in (1, 3):
                    if not (drop_index == 0 and bar_in < 2):
                        put(drums, i, snare, 0.5)

                # offbeat bass stab on the "and"s; density varies per drop
                if e % 2 == 1:
                    keep = True
                    if variant % 3 == 2:  # sparser bass variant
                        keep = beat_in_bar % 2 == 0
                    if keep:
                        put(drums, i, stab, 0.5 * amp)
                    put(drums, i, ohat, 0.28)  # offbeat open hat

                # closed hats: 16ths on even drops, 8ths on odd (breathing room)
                if variant % 2 == 0 or on_beat:
                    put(drums, i, chat, 0.2 if on_beat else 0.13)

        elif name == "break":
            # drums drop out, song comes forward and breathes
            song_gain[s0:s1] = 1.0
            put(drums, s0, bell, 0.35)
            for b in range(bars * 4):
                if b % 2 == 1:
                    put(drums, s0 + b * bs + bs // 2, ohat, 0.18)

        bar += bars

    song_gain = smooth_gain(song_gain[:n_samples])
    return drums[:n_samples], song_gain, filter_regions, drop_starts


def high_from_drops(n_bars, drops):
    """Build a full/sparse label array from explicit drop bars: 2 sparse run-up
    bars before each drop, full held for 12 bars (or until the next drop)."""
    high = np.zeros(n_bars, dtype=bool)
    for d in drops:
        end = min(n_bars, d + 12)
        for nd in drops:
            if nd > d:
                end = min(end, nd)
                break
        high[d:end] = True
        for k in (1, 2):
            if d - k >= 0:
                high[d - k] = False
    return high


def hardtekk(in_path, out_path, target_bpm=180.0, drop_at=None):
    print(f"Loading {os.path.basename(in_path)} ...")
    y, _ = librosa.load(in_path, sr=SR, mono=True)
    y = y / max(np.max(np.abs(y)), 1e-9)

    bpm = detect_bpm(y, SR)
    key, mode, root_midi = detect_key(y, SR)
    print(f"  Detected: {bpm:.1f} BPM, key {key} {mode}")

    rate = target_bpm / bpm
    print(f"  Stretching {bpm:.1f} -> {target_bpm:.0f} BPM (rate {rate:.2f}x)")
    y = librosa.effects.time_stretch(y, rate=rate)

    print("  Phase-locking grid to the song's measured beats...")
    bs, offset = detect_grid(y, target_bpm)
    grid_bpm = 60.0 * SR / bs
    barlen = bs * 4
    print(f"    measured grid: {grid_bpm:.1f} BPM, downbeat offset {offset/SR*1000:.0f} ms")

    n_bars = max((len(y) - offset) // barlen, 1)
    if drop_at:
        # manual override: place drops at the given remix-timeline seconds
        drops = sorted({max(2, round((t * SR - offset) / barlen)) for t in drop_at})
        high = high_from_drops(n_bars, drops)
        sections = plan_from_structure(high, drops)
        print(f"  Manual drops at {', '.join(f'{t:.0f}s' for t in drop_at)}")
    else:
        print("  Watching stems (drums/bass) to find the real drops...")
        high, drops, _ = analyze_structure(y, grid_bpm, offset)
        if drops:
            sections = plan_from_structure(high, drops)
        else:
            # flat song, no clear contrast: fall back to fixed arrangement
            print("  (no clear drop found — using default arrangement)")
            sections = plan_sections(len(high))
    print("  bar map (# = song's beat in, ^ = drop):")
    print("  " + structure_map(high, drops).replace("\n", "\n  "))

    drums, song_gain, filter_regions, drop_starts = build_arrangement(
        len(y), grid_bpm, root_midi, sections, offset)
    print(f"  {len(high)} bars, {len(drop_starts)} hardtekk drop(s) at "
          + ", ".join(f"{s/SR:.0f}s" for s in drop_starts))

    # highpass-sweep the song during build-ups
    for a, b in filter_regions:
        y[a:b] = highpass_sweep(y[a:b])

    print("  Sidechaining + mixing...")
    # Song sits loud and up front; drums support it rather than bury it.
    mix = y * song_gain * 0.85 + drums * 0.62
    mix = soft_clip(mix, drive=1.15)  # gentler saturation = less harsh/jarring
    mix = mix / max(np.max(np.abs(mix)), 1e-9) * 0.97

    sf.write(out_path, mix.astype(np.float32), SR)
    print(f"  -> {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(description="Hardtekk remix generator")
    p.add_argument("song", help="input audio file (mp3/wav/m4a/flac/ogg)")
    p.add_argument("--bpm", type=float, default=180.0, help="target tekk BPM (default 180)")
    p.add_argument("--drop-at", default=None,
                   help="comma-separated remix-timeline seconds to force drops, "
                        "e.g. --drop-at 30,72 (overrides auto-detection)")
    p.add_argument("-o", "--out", default=None, help="output wav path")
    args = p.parse_args()

    drop_at = None
    if args.drop_at:
        drop_at = [float(x) for x in args.drop_at.split(",") if x.strip()]

    if not os.path.exists(args.song):
        sys.exit(f"File not found: {args.song}")
    out = args.out
    if out is None:
        base = os.path.splitext(os.path.basename(args.song))[0]
        outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(outdir, exist_ok=True)
        out = os.path.join(outdir, f"{base}_hardtekk_{int(args.bpm)}bpm.wav")
    hardtekk(args.song, out, args.bpm, drop_at=drop_at)


if __name__ == "__main__":
    main()
