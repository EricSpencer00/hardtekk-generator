/* remix-worker.js — runs remix-engine.js off the main thread.
 *
 * Holds the song, stems, analysis and mix buses between calls, so a kick
 * retune or a final render does not redo separation and analysis.
 *
 * in:  {type:"build", mix | stems, samples, opts}   (opts.separator "fast": skip the model)  {type:"rebuild", opts}
 *      {type:"scratch", samples, seed, opts}   (no song: an original track)
 *      {type:"kick", opts}  {type:"master", mix}
 * out: {type:"log"|"progress"|"built"|"kick"|"wav"|"error", ...}
 */
importScripts("remix-engine.js");
const R = self.REMIX;
const S = {};
R.LOG.fn = msg => postMessage({ type: "log", msg });
const progress = stage => p => postMessage({ type: "progress", stage, p });

/* The neural separator: htdemucs (Meta, MIT; ONNX export with the STFT left to the engine by Ghilda, MIT, 174 MB)
 * from its Hugging Face repo, kept in the Cache API, run by onnxruntime-web on WebGPU. Without a WebGPU adapter it
 * is skipped: single-threaded WASM (GitHub Pages cannot enable threads) takes ~12 s per 7.8 s chunk. */
const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/";
const MODEL = "https://huggingface.co/Ghilda/htdemucs-onnx/resolve/850cd89461d0817276337061c29e6abceb86d75f/htdemucs.onnx";
let session = null;
async function modelBytes() {
  const cache = await caches.open("hardtekk-models"), hit = await cache.match(MODEL);
  if (hit) return new Uint8Array(await hit.arrayBuffer());
  const r = await fetch(MODEL); if (!r.ok) throw new Error(`model download: HTTP ${r.status}`);
  const n = +r.headers.get("content-length") || 174266088, parts = [], rd = r.body.getReader(); let got = 0;
  for (;;) { const { done, value } = await rd.read(); if (done) break; parts.push(value); got += value.length; postMessage({ type: "progress", stage: "downloading", p: got / n }); }
  const b = new Blob(parts); await cache.put(MODEL, new Response(b)); return new Uint8Array(await b.arrayBuffer());
}
async function modelStems(mix) {
  if (!session) {
    const gpu = self.navigator.gpu && await navigator.gpu.requestAdapter().catch(() => null);
    if (!gpu) throw new Error("no WebGPU");
    R.LOG.fn("loading the separation model"); const t0 = Date.now();
    importScripts(ORT + "ort.webgpu.min.js"); ort.env.wasm.wasmPaths = ORT;
    session = await ort.InferenceSession.create(await modelBytes(), { executionProviders: ["webgpu"], graphOptimizationLevel: "all" });
    R.LOG.fn(`model ready in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  }
  return R.separateModel(mix, async (x, mag) => {
    const r = await session.run({ mix: new ort.Tensor("float32", x, [1, 2, x.length / 2]), mag: new ort.Tensor("float32", mag, [1, 4, 2048, 336]) });
    return { freq: r.freq.data, time: r.time.data };
  }, progress("separating"));
}

function copyLayers(layers) {
  const out = {}, transfer = [];
  for (const [k, L] of Object.entries(layers)) { out[k] = L.map(c => c.slice()); transfer.push(out[k][0].buffer, out[k][1].buffer); }
  return { out, transfer };
}
function summary(b) {
  const m = b.meta;
  return { bpm: m.bpm, key: m.key, scale: m.scale, bars: m.N_BARS, kicks: m.hits.length, arrangement: m.arrangement,
           blocks: m.blocks.map(x => ({ name: x.name, kind: x.kind, src: x.src, n: x.n, out: x.out })), OB0: m.OB0, BAR: m.BAR,
           N_BARS: m.N_BARS, score: m.score, mode: m.mode };
}
async function build(opts) {
  S.opts = opts;
  const t0 = Date.now();
  S.b = S.seed ? await R.scratch(S.samples, { ...opts, seed: S.seed }, progress("building")) : await R.build(S.stems, S.samples, S.an, opts, progress("building"), S.cache);
  R.LOG.fn(`built in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  sendBuilt();
}
function sendBuilt() {
  const { out, transfer } = copyLayers(S.b.layers);
  postMessage({ type: "built", layers: out, meta: summary(S.b), tone: R.previewTone(S.b.layers, S.b.meta) }, transfer);
}

onmessage = async e => {
  const m = e.data;
  try {
    if (m.type === "scratch") {
      S.samples = m.samples; S.cache = {}; S.stems = S.an = null; S.seed = m.seed;
      await build(m.opts);
    } else if (m.type === "build") {
      S.samples = m.samples; S.cache = {}; S.seed = null;      // new song: drop the speed/arrangement cache
      S.stems = m.stems || null;
      if (!S.stems && m.opts.separator !== "fast") {
        const t0 = Date.now();
        try { S.stems = await modelStems(m.mix); R.LOG.fn(`separated by the model in ${((Date.now() - t0) / 1000).toFixed(1)} s`); }
        catch (err) { session = null; R.LOG.fn(`model separation unavailable (${err.message}): fast separation`); }
      }
      if (!S.stems) { R.LOG.fn("separating the song (fast)"); S.stems = await R.separate(m.mix, progress("separating")); }
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
      sendBuilt();
    } else if (m.type === "kick") {
      const t0 = Date.now();
      S.opts = { ...S.opts, ...m.opts };
      if (m.sample) { S.samples.kick = m.sample.data; S.samples._kickId = m.sample.id; delete S.samples._kickNorm; }
      S.b = R.mixScore(S.b.ctx, S.b.meta.score, S.samples, S.opts);       // same path as an edit: fades and FX stay on
      R.LOG.fn(`kick re-rendered in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
      sendBuilt();
    } else if (m.type === "master") {
      const r = await R.master(S.b.layers, S.b.meta, S.b.masks, m.mix, progress("mastering"));
      const wav = R.encodeWav(r.out);
      postMessage({ type: "wav", wav, lufs: r.lufs }, [wav]);
    }
  } catch (err) {
    postMessage({ type: "error", msg: String(err && (err.stack || err.message) || err) });
  }
};
