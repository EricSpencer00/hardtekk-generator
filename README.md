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

## Remix app (index.html)

The home page is the remix app: drop a song (or its 4 stems: vocals / drums /
bass / other, e.g. from Demucs; the vocal is far cleaner than the in-browser
separation), and it builds a hardtekk remix in the browser.

**Chop mode (default).** The groove exists before the song is looked at: a
fixed tempo (175 BPM default) and hand-written section templates in a fixed
form (intro 8, build 8, drop 16, break 8, build 8, drop 16, outro 8; intro,
build, drop, outro when the song has little vocal). Drops: distorted tonal
kick on every beat, off-beat bass that stops before the next kick, clap on 2
and 4, off-beat hat, 16th hats from bar 9. Builds: backbeat, a snare roll
that doubles, kick in 8ths, a bar of roll, riser, and a stutter of the hook's
first syllable. Breaks: no kick. Elements enter and leave on 8-bar lines. The
song becomes samples: its vocal is cut into phrases, each starts on a bar
line (a pickup lands before it) and is time-stretched with its pitch kept
(WSOLA, onsets locked) from its own tempo onto whole beats; drops play the
hook (the loudest, most-repeated phrase run), the other sections the other
phrases in the song's order. Each bar's kick and bass note is the root of
the source passage its phrase plays there (the song's bass note, else its
chord, else its key), so songs that change key or tempo are followed phrase
by phrase. The song's own drums and bass are never used; its backing plays
high-passed in breaks and builds only. Pitch (+4 st default) is a separate,
deliberate shift.

**Continuous mode** (the mode selector) is the previous engine: the whole
song sped up on its detected beat grid with new drums on top.

Both: double bells, risers, impacts, echo throws, tape stop, fades, a
blown-out master, live mixer (level + 3-band EQ per bus, played through the
render's master), kick picker (CC0 kicks in samples/cc0), drum editor grid
(kick, bass, snare, hats, bell) with MIDI export/import, WAV render. The
previous generator is at classic.html.

Score a render against the original song and its true stems (timing,
stretch, harmony, hook, leakage, kick/bass overlap, loudness, form):
`node tools/remix_node.js out.wav --mix song.wav --truth stems/ --layers`
then `python listen.py out.wav --song song.wav --stems stems/`.
`node tools/stretch_check.js` checks the time stretcher itself.

Engine: remix-engine.js (runs in remix-worker.js; also in Node via
tools/remix_node.js).
