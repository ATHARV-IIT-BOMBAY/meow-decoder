// Meow Decoder's signal processing and classifiers, ported line for line from meow_decoder/core.py so the browser
// version gives the same answers. Pure functions on Float32Array audio at 16 kHz; tests/web_parity.mjs checks them
// against the Python code.

export const SR = 16000;

const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
function percentile(values, q) {  // numpy's default (linear interpolation)
  const v = Float64Array.from(values).sort(), pos = (v.length - 1) * q, i = Math.floor(pos);
  return i + 1 < v.length ? v[i] + (v[i + 1] - v[i]) * (pos - i) : v[i];
}
function median(values) { return percentile(values, 0.5); }

/** Sound events [start, end) in samples, loudest first; the whole clip when nothing stands out (core.segments). */
export function segments(wave, sr = SR, maxLen = 4.0) {
  const hop = Math.floor(sr / 100), win = Math.floor(sr / 40), n = wave.length;
  if (n < win) return [[0, n]];
  const frames = Math.floor((n - win) / hop) + 1, db = new Float64Array(frames);
  for (let f = 0; f < frames; f++) {
    let s = 0;
    for (let j = f * hop, e = j + win; j < e; j++) s += wave[j] * wave[j];
    db[f] = 10 * Math.log10(s / win + 1e-10);
  }
  const floor = percentile(db, 0.1);
  let peak = -Infinity; for (const v of db) peak = Math.max(peak, v);
  if (peak - floor < 10) return [[0, n]];
  const th = Math.max(floor + 10, peak - 35), runs = [];
  let start = null;
  for (let i = 0; i <= frames; i++) {
    const a = i < frames && db[i] > th;
    if (a && start === null) start = i;
    else if (!a && start !== null) {
      if (runs.length && start - runs[runs.length - 1][1] < 15) runs[runs.length - 1][1] = i;  // merge gaps under 150 ms
      else runs.push([start, i]);
      start = null;
    }
  }
  const events = [];
  for (let [s, e] of runs) {
    if (e - s < 30) continue;  // under 0.3 s
    const len = Math.round(maxLen * 100);
    if (e - s > len) {  // keep the loudest maxLen seconds
      let best = -Infinity, at = 0, sum = 0;
      for (let k = s; k < s + len; k++) sum += db[k] - floor;
      for (let k = s; ; k++) {
        if (sum > best) { best = sum; at = k - s; }
        if (k + len >= e) break;
        sum += db[k + len] - db[k];
      }
      s += at; e = s + len;
    }
    let top = -Infinity; for (let k = s; k < e; k++) top = Math.max(top, db[k]);
    events.push([Math.max(0, s * hop - Math.floor(sr / 10)), Math.min(n, e * hop + win + Math.floor(sr / 10)), top]);
  }
  if (!events.length) return [[0, n]];
  return events.sort((x, y) => y[2] - x[2]).map(([a, b]) => [a, b]);
}

/** F0 in Hz every `hop` samples (NaN where unvoiced), by YIN (core.pitch_track). */
export function pitchTrack(wave, sr = SR, fmin = 200, fmax = 2000, hop = 160, frame = 1024, threshold = 0.15) {
  if (wave.length < frame) return new Float64Array(0);
  const frames = Math.floor((wave.length - frame) / hop) + 1, tmax = Math.floor(sr / fmin), tmin = Math.floor(sr / fmax), n = frame - tmax;
  const level = new Float64Array(frames);
  for (let f = 0; f < frames; f++) {
    let s = 0, s2 = 0;
    for (let j = f * hop, e = j + frame; j < e; j++) { s += wave[j]; s2 += wave[j] * wave[j]; }
    level[f] = Math.sqrt(Math.max(s2 / frame - (s / frame) ** 2, 0));
  }
  let maxLevel = 0; for (const v of level) maxLevel = Math.max(maxLevel, v);
  const gate = Math.max(1e-4, 0.03 * maxLevel), f0 = new Float64Array(frames).fill(NaN), row = new Float64Array(tmax);
  for (let f = 0; f < frames; f++) {
    if (!(level[f] > gate)) continue;  // silence and faint background have no pitch worth tracking
    const o = f * hop;
    let cum = 0;
    for (let t = 1; t <= tmax; t++) {
      let d = 0;
      for (let j = 0; j < n; j++) { const x = wave[o + j] - wave[o + j + t]; d += x * x; }
      cum += d;
      row[t - 1] = (d * t) / Math.max(cum, 1e-12);
    }
    let t = -1;
    for (let r = tmin - 1; r < tmax; r++) if (row[r] < threshold) { t = r; break; }
    if (t < 0) continue;
    while (t + 1 < tmax && row[t + 1] < row[t]) t++;  // walk down to the local minimum
    const a = row[Math.max(t - 1, 0)], b = row[t], c = row[Math.min(t + 1, tmax - 1)];
    const shift = a - 2 * b + c > 0 ? Math.min(1, Math.max(-1, (0.5 * (a - c)) / (a - 2 * b + c))) : 0;
    f0[f] = sr / (t + 1 + shift);
  }
  return f0;
}

/** One event's pitch in semitones re A4 and times in seconds, cleaned; null when too little is voiced (core.contour). */
export function contour(event) {
  const f0 = pitchTrack(event);
  let st = [], t = [];
  f0.forEach((v, i) => { if (!Number.isNaN(v)) { st.push(12 * Math.log2(v / 440)); t.push(i * 0.01); } });
  if (st.length < 8) return null;
  const m = median(st), keep = st.map((v) => Math.abs(v - m) < 7);  // drop octave errors
  st = st.filter((_, i) => keep[i]); t = t.filter((_, i) => keep[i]);
  if (st.length < 8) return null;
  const padded = [st[0], st[0], ...st, st.at(-1), st.at(-1)];  // smooth tracker jitter over 50 ms
  return [st.map((_, i) => (padded[i] + padded[i + 1] + padded[i + 2] + padded[i + 3] + padded[i + 4]) / 5), t];
}

function slope(x, y) {  // least-squares line, as np.polyfit(x, y, 1)[0]
  const mx = mean(x), my = mean(y);
  let num = 0, den = 0;
  for (let i = 0; i < x.length; i++) { num += (x[i] - mx) * (y[i] - my); den += (x[i] - mx) ** 2; }
  return num / den;
}

/** [rise, slope, peak position] averaged over the loudest three events; NaNs when no meow pitch (core.pitch_features). */
export function pitchFeatures(wave) {
  const rows = [];
  for (const [a, b] of segments(wave).slice(0, 3)) {
    const c = contour(wave.subarray(a, b));
    if (!c) continue;
    const [st, t] = c, third = Math.max(Math.floor(st.length / 3), 1);
    let peak = 0; st.forEach((v, i) => { if (v > st[peak]) peak = i; });
    rows.push([mean(st.slice(-third)) - mean(st.slice(0, third)), slope(t, st), peak / st.length]);
  }
  return rows.length ? [0, 1, 2].map((k) => mean(rows.map((r) => r[k]))) : [NaN, NaN, NaN];
}

/** The loudest meow's pitch for drawing (core.pitch_line). */
export function pitchLine(wave, points = 48) {
  for (const [a, b] of segments(wave).slice(0, 3)) {
    const c = contour(wave.subarray(a, b));
    if (!c) continue;
    const st = c[0].map((v) => v - c[0][0]), k = Math.min(points, st.length), third = Math.max(Math.floor(st.length / 3), 1);
    const idx = Array.from({ length: k }, (_, i) => Math.round(k > 1 ? (i * (st.length - 1)) / (k - 1) : 0));
    return { semitones: idx.map((i) => Math.round(st[i] * 100) / 100), seconds: Math.round(st.length) / 100,
             rise: Math.round((mean(st.slice(-third)) - mean(st.slice(0, third))) * 100) / 100 };
  }
  return null;
}

/* ---------- classifiers ---------- */

export function softmax(z) { const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); }

/** A Head as in core.py: standardize, then multinomial logistic regression. */
export function headProba(h, x) {
  const z = h.intercept.map((b, k) => { let s = b; for (let j = 0; j < x.length; j++) s += ((x[j] - h.mean[j]) / h.scale[j]) * h.coef[k][j]; return s; });
  return softmax(z);
}

/**
 * Head.fit: standardize over all rows, then minimize 0.5 * |W|^2 + C * sum_i w_i * cross-entropy_i (sklearn's objective,
 * intercept not penalized) by accelerated gradient descent with a step from the Lipschitz bound.
 */
export function fitHead(X, y, w, C, iters = 400) {
  const n = X.length, d = X[0].length, classes = [...new Set(y)].sort(), K = classes.length;
  const mean_ = new Float64Array(d), scale = new Float64Array(d);
  for (const r of X) for (let j = 0; j < d; j++) mean_[j] += r[j] / n;
  for (const r of X) for (let j = 0; j < d; j++) scale[j] += (r[j] - mean_[j]) ** 2 / n;
  for (let j = 0; j < d; j++) scale[j] = Math.sqrt(scale[j]) || 1;
  const Z = X.map((r) => Float64Array.from(r, (v, j) => (v - mean_[j]) / scale[j]));
  const Y = y.map((l) => classes.indexOf(l));
  // Lipschitz constant of the gradient: 1 + C * 0.5 * largest eigenvalue of Z' diag(w) Z, by power iteration
  let v = new Float64Array(d).fill(1 / Math.sqrt(d)), lam = 0;
  for (let it = 0; it < 30; it++) {
    const u = new Float64Array(d);
    for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < d; j++) s += Z[i][j] * v[j]; s *= w[i]; for (let j = 0; j < d; j++) u[j] += s * Z[i][j]; }
    lam = Math.hypot(...u); v = u.map((x) => x / lam);
  }
  let sw = 0; for (const x of w) sw += x;
  const step = 1 / (1 + C * 0.5 * (lam + sw));  // + sw: the intercept column of ones
  const size = K * (d + 1), theta = new Float64Array(size), prev = new Float64Array(size), look = new Float64Array(size);
  const grad = (p) => {
    const g = new Float64Array(size);
    for (let k = 0; k < K; k++) for (let j = 0; j < d; j++) g[k * (d + 1) + j] = p[k * (d + 1) + j];  // the L2 term, weights only
    for (let i = 0; i < n; i++) {
      const z = new Array(K);
      for (let k = 0; k < K; k++) { let s = p[k * (d + 1) + d]; for (let j = 0; j < d; j++) s += p[k * (d + 1) + j] * Z[i][j]; z[k] = s; }
      const P = softmax(z);
      for (let k = 0; k < K; k++) {
        const e = C * w[i] * (P[k] - (Y[i] === k ? 1 : 0)), o = k * (d + 1);
        for (let j = 0; j < d; j++) g[o + j] += e * Z[i][j];
        g[o + d] += e;
      }
    }
    return g;
  };
  for (let it = 1; it <= iters; it++) {
    const m = (it - 1) / (it + 2);
    for (let q = 0; q < size; q++) look[q] = theta[q] + m * (theta[q] - prev[q]);
    const g = grad(look);
    prev.set(theta);
    for (let q = 0; q < size; q++) theta[q] = look[q] - step * g[q];
  }
  const coef = classes.map((_, k) => Array.from(theta.subarray(k * (d + 1), k * (d + 1) + d)));
  return { classes, mean: Array.from(mean_), scale: Array.from(scale), coef, intercept: classes.map((_, k) => theta[k * (d + 1) + d]) };
}

/** web/model.json's compact matrices (benchmark/export_web.py int16_columns). */
export function unpackColumns(m) {
  const bin = typeof atob === "function" ? Uint8Array.from(atob(m.data), (c) => c.charCodeAt(0)) : Uint8Array.from(Buffer.from(m.data, "base64"));
  const q = new Int16Array(bin.buffer, bin.byteOffset, m.rows * m.cols);
  return Array.from({ length: m.rows }, (_, i) => Float64Array.from({ length: m.cols }, (_, j) => m.lo[j] + (q[i * m.cols + j] + 32500) * m.step[j]));
}
