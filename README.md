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
5. Lays down the tekk arrangement on the grid: a distorted 4/4 kick **tuned
   to the song's root note** (with the signature `x.x.x.x.x.x.xxx.` roll),
   offbeat saw-bass stabs, offbeat open hats, 16th closed hats, snare builds
   and noise risers.
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

It prints a **bar map** so you can see what it detected:

```
..................################################.......###########..#####
                  ^                                      ^            ^
# = the song's own beat/bass is playing   . = sparse/vocal   ^ = where we drop
```

If a drop feels early/late, read its second off the map and pin it with
`--drop-at` (or the "force drops" field on the website). Each forced drop
gets a 2-bar build + a half-beat breath before it.

## Drop-folder mode

```sh
./watch.sh
```

Then drag any mp3/wav/m4a/flac into `input/` — the remix appears in `output/`
automatically. Ctrl-C to stop.
