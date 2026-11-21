// The heavy work, off the page's main thread so the cat keeps animating: AudioSet's AST (on the graphics chip via
// WebGPU when the browser has it, else the CPU), the pitch contour, and refitting a taught cat's classifier.
import { fitHead, pitchFeatures, pitchLine, unpackColumns } from "./dsp.js";

const TRANSFORMERS = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2";
const SR = 16000;
const say = (text) => postMessage({ type: "status", text });
const model = fetch(new URL("model.json", import.meta.url)).then((r) => r.json());
let ast = null, base = null;

async function gpu() {
  try { return !!(await navigator.gpu?.requestAdapter()); } catch { return false; }
}

// fp16 on WebGPU: ~0.2 s a clip and matches the full model (Cat 0.457 vs 0.477 on a CatMeows clip), but a 174 MB
// download. The 8-bit model is 91 MB and as accurate, but runs ~5 s on one CPU thread (and 18 s on WebGPU).
async function choose(m) {
  const saveData = self.navigator.connection?.saveData;
  return !saveData && (await gpu()) ? { device: "webgpu", dtype: "fp16", mb: 174 } : { device: "wasm", dtype: m.ast.dtype, mb: 91 };
}

function load() {
  if (ast) return ast;
  ast = (async () => {
    const m = await model, t = await import(TRANSFORMERS);
    t.env.allowLocalModels = false;
    const pick = await choose(m);
    const progress = (p) => { if (p.status === "progress" && p.file?.endsWith(".onnx")) say(`Downloading the sound model, once (${pick.mb} MB): ${Math.round(p.progress)}%`); };
    const opts = { revision: m.ast.revision, progress_callback: progress };
    let net;
    const processor = await t.AutoProcessor.from_pretrained(m.ast.id, opts);
    try { net = await t.AutoModelForAudioClassification.from_pretrained(m.ast.id, { ...opts, dtype: pick.dtype, device: pick.device }); }
    catch (e) {  // a WebGPU that fails to start: fall back to the CPU model
      if (pick.device === "wasm") throw e;
      net = await t.AutoModelForAudioClassification.from_pretrained(m.ast.id, { ...opts, dtype: m.ast.dtype, device: "wasm" });
    }
    const ready = { processor, net };
    await run(ready, new Float32Array(SR));  // warm up: the first run compiles the GPU kernels
    return ready;
  })();
  ast.catch(() => { ast = null; });
  return ast;
}

async function run({ processor, net }, wave) {  // AudioSet probabilities, max over consecutive 10 s windows (core.sounds_in)
  let best = null;
  for (let i = 0; i < Math.max(wave.length - SR / 2, 1); i += 10 * SR) {
    const { logits } = await net(await processor(wave.subarray(i, i + 10 * SR)));
    const p = Array.from(logits.data, (z) => 1 / (1 + Math.exp(-z)));
    best = best ? best.map((v, k) => Math.max(v, p[k])) : p;
  }
  return best;
}

async function cached() {  // is the model already on this device? Then warming up costs no download.
  try { return (await caches.keys()).some((k) => k.includes("transformers")); } catch { return false; }
}

onmessage = async ({ data: msg }) => {
  try {
    if (msg.type === "warm") { if (!msg.ifCached || (await cached())) await load(); return postMessage({ id: msg.id, ok: true }); }
    if (msg.type === "analyse") {
      const wave = msg.wave;
      say("Reading the pitch…");
      const pitch = pitchFeatures(wave), line = pitchLine(wave);
      if (!ast) say("Getting the sound model ready…");
      const ready = await load();
      say("Listening for a cat…");
      return postMessage({ id: msg.id, ok: true, result: { audioset: await run(ready, wave), pitch, line } });
    }
    if (msg.type === "fit") {
      const m = await model;
      base ??= unpackColumns(m.teach.X);
      say("Learning from your cat's meows…");
      const X = [...base, ...msg.samples.map((s) => s.feat)], y = [...m.teach.y, ...msg.samples.map((s) => s.label)];
      const w = [...base.map(() => 1), ...msg.samples.map(() => m.teach.weight)];
      return postMessage({ id: msg.id, ok: true, result: fitHead(X, y, w, m.teach.C) });
    }
  } catch (e) {
    postMessage({ id: msg.id, ok: false, error: String(e?.message || e) });
  }
};
