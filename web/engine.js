// Meow Decoder running entirely in the browser (the GitHub Pages version). It answers the same /api/... calls as
// meow_decoder/server.py, so index.html works unchanged. Audio never leaves the device: the AudioSet model runs here
// through transformers.js, cats are kept in this browser's storage, and nothing is shared.
import { SR, headProba } from "./dsp.js";

const STORE = "meowCats";
const MAX_CATS = 20, MAX_SAMPLES = 200;

const model = fetch(new URL("model.json", import.meta.url)).then((r) => r.json());
const clips = new Map();  // recent recordings: token -> {feat, sounds, pitch, line, audioset}
const heads = new Map();  // cat id -> [samples signature, fitted head]
const status = (text) => dispatchEvent(new CustomEvent("meowstatus", { detail: text }));

/* ---------- the heavy work, in web/worker.js ---------- */
let worker = null, calls = 0;
const pending = new Map();
function call(msg, transfer = []) {
  if (!worker) {
    worker = new Worker(new URL("worker.js", import.meta.url), { type: "module" });
    worker.onmessage = ({ data }) => {
      if (data.type === "status") return status(data.text);
      const p = pending.get(data.id); pending.delete(data.id);
      if (p) (data.ok ? p.resolve(data.result) : p.reject(new Error(data.error || "Something went wrong. Try again.")));
    };
  }
  const id = ++calls;
  return new Promise((resolve, reject) => { pending.set(id, { resolve, reject }); worker.postMessage({ ...msg, id }, transfer); });
}
/** Start loading the sound model before it's needed (while recording, or at once if it's already on this device). */
export function warm(ifCached = false) { call({ type: "warm", ifCached }).catch(() => {}); }

/* ---------- audio ---------- */
async function decode(blob) {  // any audio the browser can play -> 16 kHz mono, first 30 s (core.decode)
  // Decode straight at 16 kHz: going through the device rate (44.1/48 kHz) and resampling again leaves artefacts
  // above 4 kHz that moved AudioSet's "Cat" score from 0.48 to 0.12 on an 8 kHz CatMeows clip.
  const Ctx = window.AudioContext || window.webkitAudioContext, bytes = await blob.arrayBuffer();
  let ctx;
  try { ctx = new Ctx({ sampleRate: SR }); } catch { ctx = new Ctx(); }  // very old browsers: device rate, resampled below
  let audio;
  try { audio = await ctx.decodeAudioData(bytes); }
  catch { throw new Error("Couldn't read that file. Try a recording in a common format (m4a, mp3, wav, webm)."); }
  finally { ctx.close(); }
  if (audio.duration < 0.1) throw new Error("That recording is too short.");
  if (audio.sampleRate !== SR) {
    const off = new OfflineAudioContext(1, Math.ceil(Math.min(audio.duration, 30) * SR), SR), src = off.createBufferSource();
    src.buffer = audio; src.connect(off.destination); src.start();
    return (await off.startRendering()).getChannelData(0).slice();  // a copy we can hand to the worker
  }
  const n = Math.min(audio.length, 30 * SR), out = new Float32Array(n);  // mono: the average of the channels, like ffmpeg -ac 1
  for (let c = 0; c < audio.numberOfChannels; c++) { const d = audio.getChannelData(c); for (let i = 0; i < n; i++) out[i] += d[i] / audio.numberOfChannels; }
  return out;
}

/* ---------- ranking (core.Decoder.rank) ---------- */
const logOdds = (p) => { const q = Math.min(Math.max(p, 1e-6), 1 - 1e-6); return Math.log(q / (1 - q)); };
const describe = (m, k) => m.describe[k] || ["🐾", k, k];

function features(m, audiosetP, pitch) {  // what teaching learns from: AudioSet log-odds + pitch contour (export_web.py)
  return [...audiosetP.map(logOdds), ...pitch.map((v, j) => (Number.isNaN(v) ? m.pitch_head.mean[j] : v))];
}

async function headFor(m, cat) {
  if (!cat?.samples.length) return null;
  const sig = cat.samples.map((s) => `${s.id}:${s.label}`).join(",");
  if (heads.get(cat.id)?.[0] === sig) return heads.get(cat.id)[1];
  const h = await call({ type: "fit", samples: cat.samples.map((s) => ({ feat: unpack(s.feat), label: s.label })) });
  heads.set(cat.id, [sig, h]);
  return h;
}

async function rank(entry, cat) {
  const m = await model, A = entry.audioset, idx = m.audioset_index;
  const sounds = Object.fromEntries(Object.entries(idx).map(([k, i]) => [k, Math.round(A[i] * 1000) / 1000]));
  const heard = (k) => {
    const d = m.detectors[k];
    if (!d) return A[idx[k]] >= m.signals[k].threshold ? A[idx[k]] : 0;
    let z = d.intercept;
    for (let j = 0; j < A.length; j++) z += ((logOdds(A[j]) - d.mean[j]) / d.scale[j]) * d.coef[j];
    return z > d.threshold ? 1 / (1 + Math.exp(-(z - d.threshold))) : 0;
  };
  const signals = [];
  for (const [k, s] of Object.entries(m.signals)) { const p = heard(k); if (p) signals.push({ key: k, emoji: s.emoji, title: s.title, meaning: s.meaning, p: Math.round(p * 1000) / 1000 }); }
  const isCat = Math.max(...m.cat_labels.map((k) => A[idx[k]])) >= m.cat_gate;
  const hasMeow = isCat && (A[idx.Meow] >= m.meow_min || !signals.length);
  const head = await headFor(m, cat);
  const h = head || m.pitch_head;
  const p = head ? headProba(head, entry.feat) : headProba(h, entry.pitch.map((v, j) => (Number.isNaN(v) ? h.mean[j] : v)));
  const results = h.classes.map((k, i) => ({ key: k, emoji: describe(m, k)[0], title: describe(m, k)[1], line: describe(m, k)[2], p: Math.round(p[i] * 1e4) / 1e4 }))
    .sort((a, b) => b.p - a.p);
  return { is_cat: isCat, sounds, signals, has_meow: hasMeow, results, guess_from: head ? "taught" : "pitch", pitch_line: entry.line };
}

/* ---------- cats in this browser ---------- */
const load = () => { try { return JSON.parse(localStorage.getItem(STORE)) || {}; } catch { return {}; } };
const save = (cats) => {
  try { localStorage.setItem(STORE, JSON.stringify(cats)); }
  catch { throw new Error("This browser's storage is full. Remove a cat or some taught meows."); }
};
const pack = (v) => btoa(String.fromCharCode(...new Uint8Array(Float32Array.from(v).buffer)));
const unpack = (s) => Array.from(new Float32Array(Uint8Array.from(atob(s), (c) => c.charCodeAt(0)).buffer));
const newId = (n) => (crypto.randomUUID?.() || String(Math.random()).slice(2) + Date.now()).replace(/-/g, "").slice(0, n);
function pub(id, c) {
  const counts = {};
  for (const s of c.samples) counts[s.label] = (counts[s.label] || 0) + 1;
  return { id, name: c.name, photo: c.photo || null, counts, samples: [...c.samples].reverse().slice(0, 50).map(({ id, label, at, hour }) => ({ id, label, at, hour })) };
}
const withId = (cats, id) => (cats[id] ? { ...cats[id], id } : null);
function cleanLabel(m, text) {  // server.clean_label
  const t = String(text).replace(/[\x00-\x1f\x7f]/g, " ").replace(/\s+/g, " ").trim().slice(0, 40);
  for (const [k, [, title]] of Object.entries(m.describe)) if ([k.toLowerCase(), title.toLowerCase()].includes(t.toLowerCase())) return k;
  return t.toLowerCase();
}

/* ---------- the server's API, answered here ---------- */
export async function api(path, opts = {}) {
  const m = await model, method = (opts.method || "GET").toUpperCase(), cats = load();
  const body = typeof opts.body === "string" ? JSON.parse(opts.body) : opts.body;
  let hit;
  if (path === "/api/config") return { contrib: false, local: true, encoder_info: "AudioSet AST (MIT, BSD-3-Clause; Xenova's quantized copy) and a pitch-contour model, running in your browser. Audio never leaves this device. Teaching here learns a little less than the full app (65% instead of 72% after 3 meows per situation)." };
  if (path === "/api/cats" && method === "GET") return Object.entries(cats).map(([id, c]) => pub(id, c));
  if (path === "/api/cats" && method === "POST") {
    const name = String(body.get("name") || "").replace(/[\x00-\x1f\x7f]/g, " ").replace(/\s+/g, " ").trim().slice(0, 40), photo = String(body.get("photo") || "");
    if (!name) throw new Error("Give your cat a name.");
    if (photo && !photo.startsWith("data:image/")) throw new Error("Photo must be an image.");
    if (Object.keys(cats).length >= MAX_CATS) throw new Error(`You can add up to ${MAX_CATS} cats. Remove one first.`);
    const id = newId(8); cats[id] = { name, photo: photo || null, samples: [] }; save(cats);
    return pub(id, cats[id]);
  }
  if ((hit = path.match(/^\/api\/cats\/(\w+)$/)) && method === "DELETE") { delete cats[hit[1]]; heads.delete(hit[1]); save(cats); return { ok: true }; }
  if ((hit = path.match(/^\/api\/cats\/(\w+)\/samples\/(\w+)$/)) && method === "DELETE") {
    const c = cats[hit[1]]; if (!c) throw new Error("That cat isn't here any more.");
    const before = c.samples.length; c.samples = c.samples.filter((s) => s.id !== hit[2]);
    if (c.samples.length === before) throw new Error("That meow was already removed.");
    save(cats); return pub(hit[1], c);
  }
  if (path === "/api/predict") {
    const wave = await decode(body.get("audio"));
    status(`Got ${(wave.length / SR).toFixed(1)} seconds of sound`);
    const { audioset: A, pitch, line } = await call({ type: "analyse", wave }, [wave.buffer]);
    const token = newId(24), entry = { audioset: A, pitch, line, feat: features(m, A, pitch) };
    clips.set(token, entry); if (clips.size > 50) clips.delete(clips.keys().next().value);
    const cat = withId(cats, body.get("cat_id"));
    return { clip: token, taught: cat?.samples.length || 0, ...(await rank(entry, cat)) };
  }
  if (path === "/api/rerank") {
    const entry = clips.get(body.clip); if (!entry) throw new Error("That recording expired. Record the meow again.");
    const cat = withId(cats, body.cat_id);
    return { clip: body.clip, taught: cat?.samples.length || 0, ...(await rank(entry, cat)) };
  }
  if (path === "/api/teach") {
    const label = cleanLabel(m, body.label || ""); if (!label) throw new Error("Say what the cat meant.");
    const c = cats[body.cat_id]; if (!c) throw new Error("Pick or add a cat first.");
    const entry = clips.get(body.clip); if (!entry) throw new Error("That recording expired. Record the meow again.");
    let sample = c.samples.find((s) => s.clip === body.clip);
    if (sample) sample.label = label;  // the same recording taught again: relabel it, never count it twice
    else {
      if (c.samples.length >= MAX_SAMPLES) throw new Error(`${c.name} already has ${MAX_SAMPLES} taught meows. Remove some first.`);
      sample = { id: newId(10), clip: body.clip, label, feat: pack(entry.feat), hour: body.hour ?? null, at: new Date().toISOString().slice(0, 19) };
      c.samples.push(sample);
    }
    save(cats);
    return { ...pub(body.cat_id, c), sample_id: sample.id };
  }
  throw new Error("Something went wrong. Try again.");
}
