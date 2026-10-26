/* remix-worker.js — runs remix-engine.js off the main thread.
 *
 * Holds the song, stems, analysis and mix buses between calls, so a kick
 * retune or a final render does not redo separation and analysis.
 *
 * in:  {type:"build", mix | stems, samples, opts}  {type:"rebuild", opts}
 *      {type:"kick", opts}  {type:"master", mix}
 * out: {type:"log"|"progress"|"built"|"kick"|"wav"|"error", ...}
 */
importScripts("remix-engine.js");
const R = self.REMIX;
const S = {};
R.LOG.fn = msg => postMessage({ type: "log", msg });
const progress = stage => p => postMessage({ type: "progress", stage, p });

function copyLayers(layers) {
  const out = {}, transfer = [];
  for (const [k, L] of Object.entries(layers)) { out[k] = L.map(c => c.slice()); transfer.push(out[k][0].buffer, out[k][1].buffer); }
  return { out, transfer };
}
function summary(b) {
  const m = b.meta;
  return { bpm: m.bpm, key: m.key, scale: m.scale, bars: m.N_BARS, kicks: m.hits.length, arrangement: m.arrangement,
           blocks: m.blocks.map(x => ({ name: x.name, kind: x.kind, src: x.src, n: x.n, out: x.out })), OB0: m.OB0, BAR: m.BAR,
           N_BARS: m.N_BARS, score: m.score };
}
async function build(opts) {
  S.opts = opts;
  const t0 = Date.now();
  S.b = await R.build(S.stems, S.samples, S.an, opts, progress("building"), S.cache);
  R.LOG.fn(`built in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  const { out, transfer } = copyLayers(S.b.layers);
  postMessage({ type: "built", layers: out, meta: summary(S.b) }, transfer);
}

onmessage = async e => {
  const m = e.data;
  try {
    if (m.type === "build") {
      S.samples = m.samples; S.cache = {};      // new song: drop the speed/arrangement cache
      if (m.stems) S.stems = m.stems;
      else { R.LOG.fn("separating the song (no stems given)"); S.stems = await R.separate(m.mix, progress("separating")); }
      S.an = await R.analyze(S.stems, progress("analysing"));
      R.LOG.fn(`${S.an.bpm.toFixed(2)} BPM, ${R.NAMES[S.an.key.tonic]} ${S.an.key.mode}, ${S.an.nBars} bars`);
      await build(m.opts);
    } else if (m.type === "rebuild") {
      await build(m.opts);
    } else if (m.type === "shuffle") {
      postMessage({ type: "shuffled", arrangement: R.shuffleArrangement(S.an, m.seed) });
    } else if (m.type === "score") {                       // edited drums or FX: re-render only the drums
      const t0 = Date.now();
      if (m.opts) S.opts = { ...S.opts, ...m.opts };
      S.b = R.mixScore(S.b.ctx, m.score, S.samples, S.opts);
      R.LOG.fn(`drums re-rendered in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
      const { out, transfer } = copyLayers(S.b.layers);
      postMessage({ type: "built", layers: out, meta: summary(S.b) }, transfer);
    } else if (m.type === "kick") {
      S.opts = { ...S.opts, ...m.opts };
      if (m.sample) { S.samples.kick = m.sample.data; S.samples._kickId = m.sample.id; delete S.samples._kickNorm; }
      S.b = R.mixScore(S.b.ctx, S.b.meta.score, S.samples, S.opts);       // same path as an edit: fades and FX stay on
      const { out, transfer } = copyLayers(S.b.layers);
      postMessage({ type: "built", layers: out, meta: summary(S.b) }, transfer);
    } else if (m.type === "master") {
      const r = await R.master(S.b.layers, S.b.meta, S.b.masks, m.mix, progress("mastering"));
      const wav = R.encodeWav(r.out);
      postMessage({ type: "wav", wav, lufs: r.lufs }, [wav]);
    }
  } catch (err) {
    postMessage({ type: "error", msg: String(err && (err.stack || err.message) || err) });
  }
};
