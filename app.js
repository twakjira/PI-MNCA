const FEATS_C = ["d", "fc", "l_over_d", "rho_sf", "c_over_d"];
const FRP_LEVELS  = ["GFRP", "CFRP", "BFRP"];
const SURF_LEVELS = ["Ribbed", "Sand-coated", "Smooth"];
const FEATURE_DISPLAY = {
  d:        [{t:"d", sub:false}, {t:" (mm)", sub:false}],
  fc:       [{t:"f", sub:false}, {t:"c", sub:true},  {t:" (MPa)", sub:false}],
  l_over_d: [{t:"l/d", sub:false}],
  rho_sf:   [{t:"ρ", sub:false}, {t:"SF", sub:true}, {t:" (%)", sub:false}],
  c_over_d: [{t:"c/d", sub:false}],
};

const FEATURE_DISPLAY_HTML = {
  d:        "d (mm)",
  fc:       "f<sub>c</sub> (MPa)",
  l_over_d: "l/d",
  rho_sf:   "ρ<sub>SF</sub> (%)",
  c_over_d: "c/d",
};
const DEFAULTS = {
  d: 12, fc: 140, ld: 5, rho: 2.0, cd: 4.0,
  frp: "GFRP", surf: "Ribbed",
};

let STATS = null;
let Z_TR = null;
let META = null;
let Y_TR = null;
let SESSION = null;

const $ = (id) => document.getElementById(id);
const statusEl = () => $("predictor-status");

function setStatus(msg, klass) {
  const el = statusEl();
  el.textContent = msg;
  el.className = "predictor-status" + (klass ? " " + klass : "");
}

const ASSET_VERSION = "v10";

async function loadAssets() {
  const bust = (p) => `${p}?${ASSET_VERSION}`;
  STATS = await fetch(bust("model/stats.json"), { cache: "no-store" }).then(r => r.json());
  Z_TR  = await fetch(bust("model/train_embeddings.json"), { cache: "no-store" }).then(r => r.json());
  META  = await fetch(bust("model/train_meta.json"), { cache: "no-store" }).then(r => r.json());
  Y_TR  = META.map(m => m.y_log_norm);
  ort.env.wasm.wasmPaths =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  const buf = await fetch(bust("model/encoder_v2.onnx"), { cache: "no-store" }).then(r => r.arrayBuffer());
  SESSION = await ort.InferenceSession.create(buf, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
}

function getInputs() {
  return {
    d:        parseFloat($("d").value),
    fc:       parseFloat($("fc").value),
    l_over_d: parseFloat($("ld").value),
    rho_sf:   parseFloat($("rho").value),
    c_over_d: parseFloat($("cd").value),
    frp:      $("frp").value,
    surf:     $("surf").value,
  };
}

function buildFeatureVector(inp, override = {}) {
  const cv = { ...inp, ...override };
  const cont = FEATS_C.map((name, idx) =>
    (cv[name] - STATS.cont_mu[idx]) / STATS.cont_sigma[idx]);
  const frp = FRP_LEVELS.map(l => l === cv.frp ? 1.0 : 0.0);
  const surf = SURF_LEVELS.map(l => l === cv.surf ? 1.0 : 0.0);
  return Float32Array.from(cont.concat(frp, surf));
}

async function encode(vec) {
  const tensor = new ort.Tensor("float32", vec, [1, 11]);
  const out = await SESSION.run({ x: tensor });
  return Array.from(out.z.data);
}

async function encodeBatch(vecs) {
  const N = vecs.length / 11;
  const tensor = new ort.Tensor("float32", vecs, [N, 11]);
  const out = await SESSION.run({ x: tensor });
  const flat = out.z.data; const D = STATS.embedding_dim;
  const rows = [];
  for (let i = 0; i < N; i++) {
    rows.push(Array.from(flat.subarray(i * D, (i + 1) * D)));
  }
  return rows;
}

function attentionWeights(zq) {
  const T = STATS.temperature;
  const D = STATS.embedding_dim;
  const N = Z_TR.length;
  const dists = new Float64Array(N);
  let dmin = Infinity;
  for (let i = 0; i < N; i++) {
    let s = 0.0;
    const row = Z_TR[i];
    for (let k = 0; k < D; k++) {
      const diff = zq[k] - row[k];
      s += diff * diff;
    }
    const d = Math.sqrt(s);
    dists[i] = d;
    if (d < dmin) dmin = d;
  }
  let Z = 0.0;
  const w = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const e = Math.exp(-(dists[i] - dmin) / T);
    w[i] = e;
    Z += e;
  }
  for (let i = 0; i < N; i++) w[i] /= Z;
  return w;
}

function effectiveNeighbors(w) {
  let H = 0;
  for (let i = 0; i < w.length; i++) {
    if (w[i] > 1e-12) H -= w[i] * Math.log(w[i]);
  }
  return Math.exp(H);
}

function predictFromAttention(w, fc) {
  let ybar = 0.0;
  for (let i = 0; i < w.length; i++) ybar += w[i] * Y_TR[i];
  const ylog = ybar * STATS.y_sigma + STATS.y_mu;
  return Math.exp(ylog) * Math.sqrt(fc);
}

function renderResults(tau) {
  const q90 = STATS.conformal_q90_MPa;
  $("r-tau").textContent = tau.toFixed(2);
  $("r-lo").textContent  = Math.max(0, tau - q90).toFixed(2);
  $("r-hi").textContent  = (tau + q90).toFixed(2);
  $("results").classList.remove("hidden");
}

async function runPredict() {
  try {
    const inp = getInputs();
    const vec = buildFeatureVector(inp);
    const zq = await encode(vec);
    const w = attentionWeights(zq);
    const tau = predictFromAttention(w, inp.fc);
    renderResults(tau);
    setStatus("Prediction ready.", "ready");
  } catch (e) {
    console.error(e);
    setStatus("Prediction error: " + (e.message || e), "error");
  }
}

function resetDefaults() {
  $("d").value   = DEFAULTS.d;
  $("fc").value  = DEFAULTS.fc;
  $("ld").value  = DEFAULTS.ld;
  $("rho").value = DEFAULTS.rho;
  $("cd").value  = DEFAULTS.cd;
  $("frp").value  = DEFAULTS.frp;
  $("surf").value = DEFAULTS.surf;
  runPredict();
}

function attachListeners() {
  $("btn-predict").addEventListener("click", runPredict);
  $("btn-reset").addEventListener("click", resetDefaults);
  $("btn-surface").addEventListener("click", runParametric);
}

async function runParametric() {
  const xName = $("surface-x").value;
  const yName = $("surface-y").value;
  if (xName === yName) {
    $("surface-status").textContent = "Pick two different features.";
    $("surface-status").classList.add("error");
    return;
  }
  $("surface-status").classList.remove("error");
  $("surface-status").textContent = "Computing 25 × 25 = 625 predictions …";
  const inp = getInputs();
  const N = 25;
  const xRange = STATS.domain[xName];
  const yRange = STATS.domain[yName];
  const xs = Array.from({ length: N }, (_, i) =>
    xRange[0] + (xRange[1] - xRange[0]) * i / (N - 1));
  const ys = Array.from({ length: N }, (_, j) =>
    yRange[0] + (yRange[1] - yRange[0]) * j / (N - 1));

  const batch = new Float32Array(N * N * 11);
  let row = 0;
  const fcDefault = inp.fc;
  const fcOfRow = new Float32Array(N * N);
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      const override = {}; override[xName] = xs[i]; override[yName] = ys[j];
      const v = buildFeatureVector(inp, override);
      batch.set(v, row * 11);
      fcOfRow[row] = (override.fc !== undefined) ? override.fc : fcDefault;
      row++;
    }
  }
  const t0 = performance.now();
  const Zq = await encodeBatch(batch);
  const tau = new Float32Array(N * N);
  for (let r = 0; r < N * N; r++) {
    const w = attentionWeights(Zq[r]);
    tau[r] = predictFromAttention(w, fcOfRow[r]);
  }
  const dt = performance.now() - t0;

  drawSurface(tau, N, xs, ys, xName, yName);
  $("surface-status").innerHTML =
    `Computed 625 predictions in ${dt.toFixed(0)} ms. ` +
    `τ̂<sub>u</sub> range: ${Math.min(...tau).toFixed(1)} – ${Math.max(...tau).toFixed(1)} MPa.`;
  $("surface-legend").innerHTML =
    `τ̂<sub>u</sub> contour over (${FEATURE_DISPLAY_HTML[xName]}, ${FEATURE_DISPLAY_HTML[yName]}); ` +
    `other features held at predictor values.`;
}

function measureSegs(ctx, segs, mainFont, subFont) {
  let w = 0;
  for (const s of segs) { ctx.font = s.sub ? subFont : mainFont; w += ctx.measureText(s.t).width; }
  return w;
}

function drawSegs(ctx, segs, x, y, mainFont, subFont, align, subDy) {
  const w = measureSegs(ctx, segs, mainFont, subFont);
  let cursor = align === "center" ? x - w / 2 : align === "right" ? x - w : x;
  ctx.textAlign = "left";
  for (const s of segs) {
    ctx.font = s.sub ? subFont : mainFont;
    ctx.fillText(s.t, cursor, y + (s.sub ? subDy : 0));
    cursor += ctx.measureText(s.t).width;
  }
}

function drawSurface(tau, N, xs, ys, xName, yName) {
  const cv = $("surface-canvas"); const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  const left = 118, right = 28, top = 58, bottom = 110;
  const aw = W - left - right, ah = H - top - bottom;
  const cellW = aw / N, cellH = ah / N;

  ctx.clearRect(0, 0, W, H);
  const tmin = Math.min(...tau), tmax = Math.max(...tau);
  const span = tmax - tmin || 1;

  function colour(t) {
    const u = (t - tmin) / span;
    const r = Math.round(255 * u);
    const g = Math.round(80 + 175 * u);
    const b = Math.round(180 * (1 - u));
    return `rgb(${r},${g},${b})`;
  }
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      const t = tau[j * N + i];
      ctx.fillStyle = colour(t);
      const px = left + i * cellW;
      const py = top + (N - 1 - j) * cellH;
      ctx.fillRect(px, py, cellW + 0.5, cellH + 0.5);
    }
  }

  ctx.strokeStyle = "#0f172a"; ctx.lineWidth = 1.8;
  ctx.beginPath();
  ctx.moveTo(left, top); ctx.lineTo(left, top + ah);
  ctx.lineTo(left + aw, top + ah); ctx.stroke();
  ctx.fillStyle = "#0f172a";
  ctx.font = "24px 'Noto Sans', sans-serif";
  ctx.textAlign = "center";
  for (let i = 0; i < N; i += 6) {
    const px = left + (i + 0.5) * cellW;
    ctx.fillText(xs[i].toFixed(1), px, top + ah + 32);
  }
  ctx.textAlign = "right";
  for (let j = 0; j < N; j += 6) {
    const py = top + (N - 1 - j + 0.5) * cellH;
    ctx.fillText(ys[j].toFixed(1), left - 10, py + 8);
  }
  const titleFont = "28px 'Noto Sans', sans-serif";
  const titleSub  = "20px 'Noto Sans', sans-serif";
  drawSegs(ctx, FEATURE_DISPLAY[xName], left + aw / 2, top + ah + 78, titleFont, titleSub, "center", 8);
  ctx.save();
  ctx.translate(32, top + ah / 2);
  ctx.rotate(-Math.PI / 2);
  drawSegs(ctx, FEATURE_DISPLAY[yName], 0, 0, titleFont, titleSub, "center", 8);
  ctx.restore();

  const headFont = "26px 'Noto Sans', sans-serif";
  const headSub  = "18px 'Noto Sans', sans-serif";
  const headSegs = [
    {t:"τ̂", sub:false}, {t:"u", sub:true},
    {t:` (MPa)   ${tmin.toFixed(1)} → ${tmax.toFixed(1)}`, sub:false}
  ];
  drawSegs(ctx, headSegs, left, top - 18, headFont, headSub, "left", 6);
}

(async function main() {
  setStatus("Loading model…");
  try {
    await loadAssets();
    attachListeners();
    $("btn-predict").disabled = false;
    $("btn-surface").disabled = false;
    setStatus("Model loaded. Adjust inputs and press Predict.", "ready");
    await runPredict();
  } catch (e) {
    console.error(e);
    setStatus("Load error: " + (e.message || e), "error");
  }
})();
