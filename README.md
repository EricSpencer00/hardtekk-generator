# hardtekk-generator

Drop a song in, get a hardtekk remix out.

**Try it in your browser (nothing is uploaded — all processing is client-side):**
https://ericspencer.us/hardtekk-generator

## How it works

1. Detects the song's BPM (spectral-flux onset envelope + harmonic-sum
   autocorrelation, which resists half/double-tempo aliasing) and key
   (chroma + Krumhansl profiles).
2. Speeds the whole track up to the target tempo (default 180 BPM) — pitch
   rises with it, the classic hardtekk way.
3. Phase-locks a bar grid to the measured downbeat, so every added element
   lands exactly on the song's beat.
4. Analyzes per-bar energy (full mix / drums / bass band) to find the song's
   real drops, and shapes build → breath → DROP around each one.
5. Writes the arrangement as a **score**: one row per bar, one lane per voice
   (kick, bass, snare, roll, bell), 16 steps per bar. The score places every
   hit, so you can read the whole arrangement and edit any of it.
6. Sidechain-ducks the original against every kick so it pumps, soft-clips
   the master, and hands you a WAV.

## Local CLI (Python)

The same algorithm as a command-line tool:

```sh
cd hardtekk-generator
python3 -m venv .venv && .venv/bin/pip install librosa soundfile numpy scipy
./tekk path/to/song.mp3            # -> output/song_hardtekk_180bpm.wav
./tekk song.mp3 --bpm 190          # faster tekk
./tekk song.mp3 --drop-at 30,72    # force drops at 30s & 72s (remix timeline)
```

It prints a **bar map** so you can see what it planned:

```
iiiiiiiiiiiiiibb############............
i = intro   b = build   . = break   # = drop
```

If a drop feels early/late, pin it with `--drop-at` (or the "force drops"
field on the website). Each forced drop gets a 2-bar build + a half-beat
breath before it.

## The score

The arrangement is a plain-text score. Print it, edit it, render it back:

```sh
./tekk song.mp3 --print-score song.tekk   # write the score (omit the path for stdout)
./tekk song.mp3 --score song.tekk         # render the edited score
```

```
#  bar section kick             bass             snare            roll             bell
   16 drop    x...x...x...x... ..x...x...x...x. ....x.......x... ................ x...............
   23 drop    x...x...x...x... ..x...x...x...x. ....x.......x... .........x...x.. ................
```

`x` is a hit, `.` is a rest, and each lane is 16 sixteenth-note steps. The
score places every hit, so moving a snare or adding a bell needs no new flag.
The section name (`intro`, `build`, `break`, `drop`) sets the mix: how hard the
song ducks, which bars get the highpass sweep, and the snare's build ramp.

The website has the same score in a text box under the controls: generate once
to fill it, edit a lane, then generate again.

## Drop-folder mode

```sh
./watch.sh
```

Then drag any mp3/wav/m4a/flac into `input/` — the remix appears in `output/`
automatically. Ctrl-C to stop.
