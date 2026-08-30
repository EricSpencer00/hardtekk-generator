#!/usr/bin/env python3
"""
Hardtekk Remix Generator
Drop a song in, get a hardtekk remix out.

Recipe (the classic hardtekk formula):
  1. Detect the source BPM and key.
  2. Time-stretch the whole track up to ~180 BPM (keeps pitch).
  3. Slam the real distorted 4/4 tekk kick SAMPLE under it, at its native
     pitch — no retuning (pitch-shifting the sample made it thin and too high).
  4. Offbeat bassline = the same kick sample, low-passed and short.
  5. Sidechain-duck the original against the kick so it pumps.
  6. Soft-clip the master for that crunchy tekk sound.

Every pitched voice comes from a sample; there are no synth oscillators.

The arrangement is a score: one row per bar, one lane per voice, 16 steps per
bar. --print-score writes it out and --score renders an edited one back, so any
hit can be moved without a new command-line switch.

Usage:
  python hardtekk.py <song.(mp3|wav|m4a|flac|ogg)> [--bpm 180] [-o out.wav]
  python hardtekk.py song.mp3 --print-score song.tekk
  python hardtekk.py song.mp3 --score song.tekk
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
    # Tekk kicks punch in the low mids, not the sub: pick the octave of the
    # root whose fundamental lands closest to ~137 Hz (measured off a
    # reference hardtekk master — its kick fundamental sits at C#3).
    cands = [root + 36, root + 48]  # C2..B2 / C3..B3 octaves
    midi = min(cands, key=lambda m: abs(librosa.midi_to_hz(m) - 137.0))
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


def remove_drums(y, sr=SR):
    """Strip the song's own drums, keeping only the harmonic (sustained) stem.

    Two-pass refined HPSS, mirroring hpss_refined() in the web app's stemacle-dsp
    (and app.js): pass 1 separates with a wide time kernel (sustained harmonic)
    and a narrow frequency kernel (sharp drum onsets); pass 2 re-runs HPSS on the
    harmonic and pulls any bin still >60% percussive back out as drums. Removes
    far more of the song's kit than a single soft-mask pass."""
    n_fft, hop = 4096, 1024
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    # Pass 1: (harmonic time kernel 31, percussive freq kernel 7)
    mask_h, _ = librosa.decompose.hpss(S, kernel_size=(31, 7), power=2.0, mask=True)
    H = S * mask_h
    # Pass 2: re-HPSS the harmonic; reclassify strongly-percussive bins as drums.
    mask_h2, mask_p2 = librosa.decompose.hpss(H, kernel_size=(31, 7), power=2.0, mask=True)
    H = np.where(mask_p2 > 0.60, H * mask_h2, H)
    return librosa.istft(H, hop_length=hop, length=len(y)).astype(y.dtype)


def soft_clip(x, drive=1.0):
    return np.tanh(x * drive) / np.tanh(drive)


def eq_peak(x, f0, gain_db, q=1.4, sr=SR):
    """RBJ peaking EQ (used to scoop the 60-120Hz mud out of the drum bus,
    matching the notch measured on the reference master)."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2 * q)
    b = np.array([1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A])
    a = np.array([1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A])
    from scipy.signal import lfilter
    return lfilter(b / a[0], a / a[0], x)


def sidechain_env(n_samples, beat_samples, sr=SR, floor=0.35):
    """Per-beat ducking envelope, with the first sample on a kick transient."""
    duck_len = int(beat_samples * 0.5)
    phase = np.arange(n_samples) % beat_samples
    env = np.ones(n_samples)
    in_duck = phase < duck_len
    env[in_duck] = floor + (1.0 - floor) * (phase[in_duck] / duck_len) ** 2
    return env


def load_samples(sr=SR):
    """Load the real hardtekk samples from samples/, each at its NATIVE pitch.

    The kick sample is in C and stays in C — pitch-shifting it up to the song's
    key is what made it sound thin and too high, so we don't retune it at all.
    """
    sdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")

    def load(name):
        path = os.path.join(sdir, name)
        if not os.path.exists(path):
            return None
        y, _ = librosa.load(path, sr=sr, mono=True)
        y = y / max(np.max(np.abs(y)), 1e-9)
        # trim leading silence so the transient lands exactly on the grid
        hot = np.flatnonzero(np.abs(y) > 0.05)
        return y[hot[0]:] if len(hot) else y

    return {
        "kick": load("kick-hardtekk_C.wav"),
        "kick_alt": load("hardtekk-kicks-tension-rebellious.wav"),
        "snare": load("snare-hardtekk-distorted-punch_D#_major.wav"),
        "bell": load("hardtekk-bell-shot.wav"),
    }


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


def highpass_sweep(seg, sr=SR, f_start=150, f_end=2400):
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
    """Return the requested beat period and its best downbeat phase.

    Time-stretching establishes the tempo. Fitting another period to beat
    tracker output lets small detection errors accumulate into drift, so only
    the phase is measured here.
    """
    bs = int(round(60.0 / target_bpm * sr))
    barlen = bs * 4
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    if len(onset) < 4:
        return bs, 0

    def energy(sample):
        frame = sample / hop
        return float(np.interp(frame, np.arange(len(onset)), onset, left=0.0, right=0.0))

    best_offset, best_score = 0, -np.inf
    for offset in range(0, barlen, hop):
        score = 0.0
        for sample in range(offset, len(y), bs):
            score += energy(sample) * (1.25 if ((sample - offset) // bs) % 4 == 0 else 1.0)
        if score > best_score:
            best_offset, best_score = offset, score
    return bs, best_offset % barlen


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


# --- the score ------------------------------------------------------------
# The arrangement is a score: one row per bar, one lane per voice, 16 steps
# (sixteenth notes) per bar. Nothing else decides where a hit lands, so
# --print-score shows the whole plan and --score renders an edited one back.

STEPS = 16
LANES = ("kick", "bass", "snare", "roll", "bell")
HIT, REST = "x", "."


def score_from_sections(sections):
    """Expand render sections into the per-bar score the renderer reads."""
    score = []
    for kind, bars in sections:
        name = kind if isinstance(kind, str) else kind[0]
        for b in range(bars):
            lanes = {k: [REST] * STEPS for k in LANES}
            for s in range(0, STEPS, 4):
                lanes["kick"][s] = HIT                        # four on the floor
            for s in range(2, STEPS, 4):
                lanes["bass"][s] = HIT                        # offbeat bass thump
            if name in ("drop", "build"):
                lanes["snare"][4] = lanes["snare"][12] = HIT  # backbeat on 2 & 4
            if name == "drop" and b % 8 == 7:
                lanes["roll"][9] = lanes["roll"][13] = HIT    # sparse 16th roll
            if b == 0 and name in ("drop", "break"):
                lanes["bell"][0] = HIT                        # bell marks the section
            score.append({"kind": name,
                          "lanes": {k: "".join(v) for k, v in lanes.items()}})
    return score


def format_score(score, bpm, offset, sr=SR):
    """Print the score: bar number, section, then one lane per voice."""
    head = [f"# hardtekk score: {len(score)} bars at {bpm:.1f} bpm, downbeat "
            f"{offset / sr * 1000:.0f} ms, {STEPS} steps per bar",
            "# edit a lane, then render it back with --score FILE",
            "#  bar section " + " ".join(f"{k:<{STEPS}}" for k in LANES)]
    rows = [f"{b:>5} {row['kind']:<7} " + " ".join(row["lanes"][k] for k in LANES)
            for b, row in enumerate(score)]
    return "\n".join(head + rows)


def parse_score(text):
    """Read a score back. Blank lines and # comments are ignored."""
    score = []
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2 + len(LANES):
            raise ValueError(f"score row needs a bar, a section and "
                             f"{len(LANES)} lanes: {line!r}")
        for lane in parts[2:]:
            if len(lane) != STEPS or set(lane) - {HIT, REST}:
                raise ValueError(f"lane must be {STEPS} of '{HIT}' or '{REST}': "
                                 f"{lane!r}")
        score.append({"kind": parts[1], "lanes": dict(zip(LANES, parts[2:]))})
    if not score:
        raise ValueError("score is empty")
    return score


def structure_map(score):
    """ASCII overview of the score: one character per bar."""
    marks = {"intro": "i", "build": "b", "break": ".", "drop": "#"}
    return "".join(marks.get(row["kind"], "?") for row in score)


def build_arrangement(n_samples, bpm, score, offset=0, sr=SR):
    """Return (drums, song_gain, song_filter_regions, drop_regions, drop_starts).

    The score places every hit: bar b, lane L, step s puts lane L's sample at
    bar b's step s. Bars start at `offset`, so the whole arrangement is
    phase-locked to the song. A bar's section only sets the mix — how much the
    song ducks, which bars get swept, and the snare's ramp through a build.

    Sample-only: every voice below is a real sample (or derived from one). There
    are no synth oscillators, so nothing is tuned to the song's key."""
    beat = 60.0 / bpm
    bs = int(beat * sr)

    smp = load_samples(sr)
    if smp["kick"] is None:
        raise SystemExit("samples/kick-hardtekk_C.wav is required (sample-only mode).")
    kick = np.copy(smp["kick"][: int(beat * 0.45 * sr)])  # tight: gaps make punch
    # tekk kicks live in the mids: hard distortion folds the sample's fundamental
    # into harmonics (reference kick has almost no 60-120Hz energy)
    kick = np.tanh(kick * 3.0) * 0.95
    # tekk bassline = the kick sample itself, lowpassed and short, on 16ths
    stab = sosfilt(butter(2, 250, "lp", fs=SR, output="sos"),
                   kick[: int(beat * 0.22 * sr)])
    stab = stab / max(np.max(np.abs(stab)), 1e-9) * 0.9
    snare = smp["snare"]
    kick_alt = smp["kick_alt"][: int(beat * 0.45 * sr)] if smp["kick_alt"] is not None else None
    bell = smp["bell"]

    voices = {"kick": kick, "bass": stab, "snare": snare,
              "roll": kick_alt if kick_alt is not None else kick, "bell": bell}

    pad = bs * 4 * 3 + (len(bell) if bell is not None else 0)
    drums = np.zeros(n_samples + pad)
    # Song stays present the whole time; sidechain only ducks, never mutes.
    song_gain = np.full(n_samples + pad, 1.0)
    # Evaluate phase from the same offset used by every drum event. Rolling a
    # finite envelope wraps its last duck into the start of the track. Shallow
    # floor keeps the song clearly audible while it pumps against the kick.
    duck_base = sidechain_env(len(drums), bs, floor=0.55)
    duck = duck_base[(np.arange(len(drums)) - offset) % bs]
    filter_regions = []
    drop_regions = []
    drop_starts = []

    def put(buf, i, sig, amp):
        if sig is None or i >= len(buf):
            return
        n = min(len(sig), len(buf) - i)
        buf[i : i + n] += sig[:n] * amp

    def level(lane, kind, pos):
        """Hit level. Only the snare ramps, through a build."""
        if lane == "snare" and kind == "build":
            return 0.25 + 0.3 * pos
        return {"kick": 1.0, "bass": 0.9, "snare": 0.5, "roll": 0.85,
                "bell": 0.4 if kind == "drop" else 0.35}[lane]

    # Runs of one section drive the mix; the score drives the hits.
    runs = []
    b = 0
    while b < len(score):
        j = b
        while j < len(score) and score[j]["kind"] == score[b]["kind"]:
            j += 1
        runs.append((score[b]["kind"], b, j - b))
        b = j

    bar_run = []                                # per bar: (kind, bar in run, run length)
    for kind, b0, bars in runs:
        s0 = min(offset + b0 * 4 * bs, len(drums))
        s1 = min(offset + (b0 + bars) * 4 * bs, len(drums))
        bar_run += [(kind, b, bars) for b in range(bars)]
        if s1 <= s0:
            continue

        # The kick + offbeat bass NEVER stop: a steady four-on-the-floor
        # "bass·kick·bass·kick" over the WHOLE song. Sections only change the
        # extras (snares, fills, bell) and how much the song ducks.
        if kind == "drop":
            drop_starts.append(s0)
            drop_regions.append((s0, s1))
            # kick-forward: the song pumps against the kick but stays listenable
            song_gain[s0:s1] = duck[s0:s1] * 0.85
        elif kind == "build":
            filter_regions.append((s0, s1))
            # sweep + tuck the song into the drop; the kick keeps driving under it
            song_gain[s0:s1] = np.linspace(0.85, 0.65, s1 - s0)
        else:  # intro / break — song forward, lighter kit (kick still runs)
            song_gain[s0:s1] = 0.9

    for b, row in enumerate(score):
        base = offset + b * 4 * bs
        if base >= len(drums):
            break
        kind, bar_in_run, run_bars = bar_run[b]
        for lane in LANES:
            for s, c in enumerate(row["lanes"][lane]):
                if c == REST:
                    continue
                pos = (bar_in_run * 8 + s // 2) / (run_bars * 8)
                # a score step is a 16th; round the whole span, not the step, so
                # the last hit of a bar cannot drift off the beat
                put(drums, base + s * bs // 4, voices[lane], level(lane, kind, pos))

    song_gain = smooth_gain(song_gain[:n_samples])
    return drums[:n_samples], song_gain, filter_regions, drop_regions, drop_starts


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


def auto_tempo(bpm):
    """Pick the tekk tempo like the pros do: a clean speed-up ratio of the
    source (reference remix used exactly 1.25x), landing in 150-195 and as
    close to ~165 as possible."""
    cands = [bpm * r for r in (1.0, 1.25, 4 / 3, 1.5, 2.0)]
    cands = [t for t in cands if 150 <= t <= 195]
    return min(cands, key=lambda t: abs(t - 165)) if cands else 165.0


def hardtekk(in_path, out_path, target_bpm=None, drop_at=None, score_path=None,
             print_score=None):
    print(f"Loading {os.path.basename(in_path)} ...")
    y, _ = librosa.load(in_path, sr=SR, mono=True)
    y = y / max(np.max(np.abs(y)), 1e-9)

    bpm = detect_bpm(y, SR)
    key, mode, _ = detect_key(y, SR)  # key/mode for display only; nothing is tuned now
    print(f"  Detected: {bpm:.1f} BPM, key {key} {mode}")

    if target_bpm is None:
        target_bpm = auto_tempo(bpm)
        print(f"  Auto tempo: {target_bpm:.1f} BPM ({target_bpm / bpm:.2f}x speed-up)")

    rate = target_bpm / bpm
    print(f"  Stretching {bpm:.1f} -> {target_bpm:.0f} BPM (rate {rate:.2f}x)")
    y = librosa.effects.time_stretch(y, rate=rate)

    print("  Phase-locking grid to the song's measured beats...")
    bs, offset = detect_grid(y, target_bpm)
    grid_bpm = 60.0 * SR / bs
    barlen = bs * 4
    print(f"    measured grid: {grid_bpm:.1f} BPM, downbeat offset {offset/SR*1000:.0f} ms")

    n_bars = max((len(y) - offset) // barlen, 1)
    if score_path:
        with open(score_path) as f:
            score = parse_score(f.read())
        print(f"  Score: {len(score)} bars read from {os.path.basename(score_path)}")
    else:
        if drop_at:
            # manual override: place drops at the given remix-timeline seconds
            drops = sorted({max(2, round((t * SR - offset) / barlen)) for t in drop_at})
            high = high_from_drops(n_bars, drops)
            sections = plan_from_structure(high, drops)
            print(f"  Manual drops at {', '.join(f'{t:.0f}s' for t in drop_at)}")
        else:
            print("  Watching stems (drums/bass) to find the real drops...")
            high, drops, _ = analyze_structure(y, grid_bpm, offset)
            # a drop needs a tease: never drop before bar 16 (the reference
            # release waits 32 bars of vocal before the first drop)
            if drops and n_bars > 24:
                drops = sorted({max(d, 16) for d in drops})
                merged = [drops[0]]
                for d in drops[1:]:
                    if d - merged[-1] >= 8:
                        merged.append(d)
                drops = merged
                high = high_from_drops(n_bars, drops)
            if drops:
                sections = plan_from_structure(high, drops)
            else:
                # flat song, no clear contrast: fall back to fixed arrangement
                print("  (no clear drop found — using default arrangement)")
                sections = plan_sections(len(high))
        score = score_from_sections(sections)
    print("  bar map (i intro, b build, . break, # drop):")
    print("  " + structure_map(score))

    if print_score:
        text = format_score(score, grid_bpm, offset)
        if print_score == "-":
            print(text)
        else:
            with open(print_score, "w") as f:
                f.write(text + "\n")
            print(f"  Score written to {print_score} — edit it, then --score it back")

    drums, song_gain, filter_regions, drop_regions, drop_starts = build_arrangement(
        len(y), grid_bpm, score, offset)
    print(f"  {len(score)} bars, {len(drop_starts)} hardtekk drop(s) at "
          + ", ".join(f"{s/SR:.0f}s" for s in drop_starts))

    # Strip the song's OWN drums (HPSS harmonic stem) so our tekk kit plays over
    # a clean bed. Done here, after structure analysis, which needs the drums to
    # find the drops.
    print("  Removing the song's drums (HPSS)...")
    y = remove_drums(y)

    # highpass-sweep the song during build-ups
    for a, b in filter_regions:
        y[a:b] = highpass_sweep(y[a:b])

    # carve the song's low end out during drops so the tekk kick owns it —
    # gentler cut (120Hz) leaves more of the song's body so it stays listenable
    sos_hp = butter(4, 120, "hp", fs=SR, output="sos")
    for a, b in drop_regions:
        y[a:b] = sosfilt(sos_hp, y[a:b])

    print("  Sidechaining + mixing...")
    # keep the kick's THUMP: only a light 90Hz scoop (was -9, which thinned it)
    drums = eq_peak(drums, 90, -4.0)
    mix = y * song_gain * 0.9 + drums * 1.0    # song forward but under the kit
    # Wall-of-sound master: bright shelf above ~2k, then push into the clipper
    # until it's LOUD (target RMS bumped for more level).
    sos_hi = butter(1, [2000, 8000], "bp", fs=SR, output="sos")
    mix = mix + 1.3 * sosfilt(sos_hi, mix)
    rms = np.sqrt(np.mean(mix ** 2))
    g = float(np.clip(0.46 / max(rms, 1e-9), 0.8, 6.0))
    mix = soft_clip(mix * g, drive=1.35)
    # tame the clipper's >8k harmonics (reference has only ~12% up there)
    mix = sosfilt(butter(2, 11000, "lp", fs=SR, output="sos"), mix)
    mix = mix / max(np.max(np.abs(mix)), 1e-9) * 0.97
    print(f"  master: gain {g:.2f}x into clipper, RMS {np.sqrt(np.mean(mix**2)):.2f}")

    sf.write(out_path, mix.astype(np.float32), SR)
    print(f"  -> {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(description="Hardtekk remix generator")
    p.add_argument("song", help="input audio file (mp3/wav/m4a/flac/ogg)")
    p.add_argument("--bpm", type=float, default=None,
                   help="target tekk BPM (default: auto — a clean speed-up ratio near 165)")
    p.add_argument("--drop-at", default=None,
                   help="comma-separated remix-timeline seconds to force drops, "
                        "e.g. --drop-at 30,72 (overrides auto-detection)")
    p.add_argument("--print-score", nargs="?", const="-", default=None, metavar="FILE",
                   help="write the bar-by-bar score (default: stdout)")
    p.add_argument("--score", default=None, metavar="FILE",
                   help="render this score file instead of planning one")
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
        tag = f"{int(args.bpm)}bpm" if args.bpm else "auto"
        out = os.path.join(outdir, f"{base}_hardtekk_{tag}.wav")
    hardtekk(args.song, out, args.bpm, drop_at=drop_at, score_path=args.score,
             print_score=args.print_score)


if __name__ == "__main__":
    main()
