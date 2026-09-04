"use strict";

const TARGET_COLORS = {
  court: "#58a6ff",
  net: "#3fb950",
  people: "#a371f7",
  ball: "#39d2c0",
};
const TARGET_LABELS = {
  court: "Court lines",
  net: "Net",
  people: "People",
  ball: "Ball",
};
const DOT_RADIUS = 2.5;

function lossKeyForTarget(target) {
  return "loss_" + target;
}

const chartCanvas = document.getElementById("chartCanvas");
const chartCtx = chartCanvas.getContext("2d");
const statusBar = document.getElementById("statusBar");
const stateLabel = document.getElementById("stateLabel");

const btnStart = document.getElementById("btnStart");
const btnPause = document.getElementById("btnPause");
const btnResume = document.getElementById("btnResume");
const btnReset = document.getElementById("btnReset");
const btnTest = document.getElementById("btnTest");
const btnAnalyzeLabel = document.getElementById("btnAnalyzeLabel");
const btnAnalyzeNN = document.getElementById("btnAnalyzeNN");
const selLabeledImage = document.getElementById("selLabeledImage");
const btnViewRaw = document.getElementById("btnViewRaw");
const btnViewInitial = document.getElementById("btnViewInitial");
const btnViewTrue = document.getElementById("btnViewTrue");
const btnViewTop = document.getElementById("btnViewTop");
const btnViewSide = document.getElementById("btnViewSide");
const lineRefineSlider = document.getElementById("lineRefineSlider");
const cameraFitSlider = document.getElementById("cameraFitSlider");
const ballRefineSlider = document.getElementById("ballRefineSlider");
const smoothRange = document.getElementById("smoothRange");
const smoothValue = document.getElementById("smoothValue");

let lossData = [];
let epochData = [];
let currentState = "idle";
let evtSource = null;
let currentEpoch = 0;
let totalEpochs = 0;
let smoothingWindow = 1;
let labeledImages = [];
const analysisState = {
  data: null,
  includeNN: false,
  view: "raw",
  imgReady: false,
  imageBitmap: null,
  imageData: null,
  initialCorrection: null,
  trueCorrection: null,
  fittedCorrectionPath: [],
  trueCorrectionPath: [],
  modelFitPath: [],
  lineRefinementPath: [],
  fittedCorrection: null,
  ballPath: [],
};

/* ---------- localStorage persistence for target checkboxes ---------- */
const TARGETS_KEY = "train_targets";

function saveTargets() {
  localStorage.setItem(TARGETS_KEY, JSON.stringify(getTargets()));
}

function restoreTargets() {
  try {
    const saved = JSON.parse(localStorage.getItem(TARGETS_KEY));
    if (!Array.isArray(saved)) return;
    document.querySelectorAll('.target-row input[type="checkbox"]').forEach((cb) => {
      cb.checked = saved.includes(cb.value);
    });
  } catch {}
}

document.querySelectorAll('.target-row input[type="checkbox"]').forEach((cb) => {
  cb.addEventListener("change", saveTargets);
});

function getTargets() {
  return [...document.querySelectorAll('.target-row input[type="checkbox"]')]
    .filter((c) => c.checked)
    .map((c) => c.value);
}

function getHyperparams() {
  const lrRaw = document.getElementById("inpLR").value.replace(",", ".");
  const augEl = document.getElementById("inpAugment");
  return {
    epochs: parseInt(document.getElementById("inpEpochs").value) || 50,
    batch_size: parseInt(document.getElementById("inpBatch").value) || 1,
    lr: parseFloat(lrRaw) || 1e-4,
    image_size: parseInt(document.getElementById("inpImgSize").value) || 320,
    augment: augEl ? augEl.checked : true,
  };
}

/* ---------- State & buttons ---------- */
function setState(s) {
  currentState = s;
  btnStart.disabled = s === "running";
  btnPause.disabled = s !== "running";
  btnResume.disabled = s !== "paused";
  btnReset.disabled = s === "running";
  stateLabel.textContent = `State: ${s}`;
}

function setEpochProgress(epoch, total) {
  if (Number.isFinite(epoch)) currentEpoch = epoch;
  if (Number.isFinite(total)) totalEpochs = total;
  renderCounts();
}

document.querySelectorAll('.target-row input[type="checkbox"]').forEach((cb) => {
  cb.addEventListener("change", () => { drawChart(); });
});

/* ---------- API helpers ---------- */
function apiPost(url) {
  return fetch(url, { method: "POST", cache: "no-store" }).then((r) => r.json());
}

/* ---------- Controls ---------- */
btnStart.addEventListener("click", () => {
  try {
    const hp = getHyperparams();
    const tgtList = getTargets();
    const targets = tgtList.join(",");
    if (!targets) { statusBar.textContent = "Select at least one target"; return; }
    activeTrainTargets = new Set(tgtList);
    trainImagesProcessed = 0;
    setState("running");
    statusBar.textContent = "Starting…";
    renderCounts();
    drawChart();
    apiPost(`/api/train/start?epochs=${hp.epochs}&batch_size=${hp.batch_size}&lr=${hp.lr}&image_size=${hp.image_size}&targets=${targets}&resume=false&augment=${hp.augment}`)
      .then((r) => {
        if (r.error) {
          statusBar.textContent = r.error;
          setState("idle");
          return;
        }
        statusBar.textContent =
          "Request accepted — starting a lighter first batch path for faster feedback.";
      })
      .catch((e) => { statusBar.textContent = "Start failed: " + e; setState("idle"); });
  } catch (e) {
    statusBar.textContent = "Error: " + e.message;
    console.error("Start failed", e);
  }
});

btnPause.addEventListener("click", () => {
  setState("paused");
  apiPost("/api/train/pause").then((r) => { if (r.error) { statusBar.textContent = r.error; setState("running"); } });
});

btnResume.addEventListener("click", () => {
  setState("running");
  statusBar.textContent = "Resuming…";
  const hp = getHyperparams();
  const targets = getTargets().join(",");
  apiPost(`/api/train/start?epochs=${hp.epochs}&targets=${targets}&resume=true`)
    .then((r) => {
      if (r.error) {
        statusBar.textContent = r.error;
        setState("paused");
        return;
      }
      statusBar.textContent = "Resume accepted — continuing training.";
    });
});

btnReset.addEventListener("click", () => {
  lossData = [];
  epochData = [];
  resetCounts();
  drawChart();
  setState("idle");
  statusBar.textContent = "Reset.";
  apiPost("/api/train/reset");
});

btnTest.addEventListener("click", async () => {
  const testInfo = document.getElementById("testInfo");
  testInfo.textContent = "Running inference…";
  try {
    const res = await fetch("/api/train/test", { method: "POST", cache: "no-store" });
    const text = await res.text();
    let data;
    try {
      data = text ? JSON.parse(text) : {};
    } catch {
      testInfo.textContent = `Error ${res.status}: ${text.slice(0, 200) || res.statusText}`;
      return;
    }
    let errMsg = data.error;
    if (errMsg == null && data.detail != null) {
      errMsg = Array.isArray(data.detail)
        ? data.detail.map((d) => (d && d.msg) || String(d)).join("; ")
        : String(data.detail);
    }
    if (!res.ok || errMsg) {
      testInfo.textContent = errMsg || `HTTP ${res.status}`;
      return;
    }
    showTestResult(data);
    const nDet = (data.detections || []).length;
    const nCourt = (data.court_points || []).length;
    const nNet = (data.net_points || []).length;
    const hasBall = data.ball ? 1 : 0;
    testInfo.textContent = `${nDet} det, ${nCourt} court pts, ${nNet} net pts, ball ${hasBall ? "found" : "none"} — ${data.image}`;
  } catch (e) {
    testInfo.textContent = String(e.message || e);
  }
});

if (btnAnalyzeLabel) {
  btnAnalyzeLabel.addEventListener("click", () => runLabeledAnalysis(false));
}
if (btnAnalyzeNN) {
  btnAnalyzeNN.addEventListener("click", () => runLabeledAnalysis(true));
}
[
  [btnViewRaw, "raw"],
  [btnViewInitial, "initial"],
  [btnViewTrue, "true"],
  [btnViewTop, "top"],
  [btnViewSide, "side"],
].forEach(([btn, view]) => {
  if (btn) btn.addEventListener("click", () => {
    analysisState.view = view;
    updateAnalysisViewButtons();
    renderAnalysisView();
  });
});
if (lineRefineSlider) lineRefineSlider.addEventListener("input", renderAnalysisView);
if (cameraFitSlider) cameraFitSlider.addEventListener("input", renderAnalysisView);
if (ballRefineSlider) ballRefineSlider.addEventListener("input", renderAnalysisView);

/* ---------- Test overlay ---------- */
function showTestResult(data) {
  document.getElementById("testCard").style.display = "";
  const img = document.getElementById("testImg");
  const cv = document.getElementById("testViewCanvas");
  img.onload = () => {
    cv.width = img.naturalWidth;
    cv.height = img.naturalHeight;
    const ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, cv.width, cv.height);
    drawGeometry(ctx, data);
    drawDetections(ctx, data);
    drawBall(ctx, data);
  };
  img.src = `/api/image?file=${encodeURIComponent(data.image)}`;
  const nCourt = (data.court_points || []).length;
  const nNet = (data.net_points || []).length;
  document.getElementById("testStatus").textContent =
    `${data.image} — ${data.detections.length} det, ${nCourt} court pts, ${nNet} net pts, ${data.ball ? "ball visible" : "ball not visible"}`;
}

async function loadLabeledImages() {
  if (!selLabeledImage) return;
  try {
    const res = await fetch("/api/train/labeled-images", { cache: "no-store" });
    const data = await res.json();
    labeledImages = Array.isArray(data.items) ? data.items : [];
    selLabeledImage.innerHTML = "";
    if (!labeledImages.length) {
      selLabeledImage.innerHTML = `<option value="">No labeled images</option>`;
      return;
    }
    for (const item of labeledImages) {
      const opt = document.createElement("option");
      opt.value = item.image;
      opt.textContent = `${item.image} (${item.court_points} court, ${item.ball_labels} ball)`;
      selLabeledImage.appendChild(opt);
    }
  } catch {
    selLabeledImage.innerHTML = `<option value="">Failed to load</option>`;
  }
}

async function runLabeledAnalysis(includeNN) {
  const testInfo = document.getElementById("testInfo");
  const image = selLabeledImage && selLabeledImage.value;
  if (!image) {
    testInfo.textContent = "Pick a labeled image first.";
    return;
  }
  testInfo.textContent = includeNN ? "Analyzing labels + NN…" : "Analyzing labels…";
  try {
    const res = await fetch(`/api/train/analyze-image?image=${encodeURIComponent(image)}&include_nn=${includeNN ? "true" : "false"}`, { cache: "no-store" });
    const text = await res.text();
    let data = {};
    try { data = text ? JSON.parse(text) : {}; } catch {}
    if (!res.ok) {
      testInfo.textContent = (data && data.detail) ? String(data.detail) : `HTTP ${res.status}`;
      return;
    }
    showAnalysisResult(data, includeNN);
  } catch (e) {
    testInfo.textContent = String(e.message || e);
  }
}

function drawGeometry(ctx, data) {
  function drawPolyline(pts, color, close) {
    if (!pts || pts.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    if (close) ctx.closePath();
    ctx.stroke();
    ctx.fillStyle = color;
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(p[0], p[1], 5, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  if (Array.isArray(data.court_polylines)) {
    for (const poly of data.court_polylines) drawPolyline(poly, "#58a6ff", false);
  } else {
    drawPolyline(data.court_points, "#58a6ff", true);
  }
}

function drawGrid(ctx, lines, color = "#d4a72c") {
  if (!Array.isArray(lines)) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.25;
  for (const seg of lines) {
    if (!Array.isArray(seg) || seg.length < 2) continue;
    ctx.beginPath();
    ctx.moveTo(seg[0][0], seg[0][1]);
    ctx.lineTo(seg[1][0], seg[1][1]);
    ctx.stroke();
  }
}

function sortHelperPointsByProjection(points) {
  if (!Array.isArray(points) || points.length < 2) return Array.isArray(points) ? points.slice() : [];
  const n = points.length;
  let meanX = 0;
  let meanY = 0;
  for (const [x, y] of points) {
    meanX += x;
    meanY += y;
  }
  meanX /= n;
  meanY /= n;
  let sxx = 0, syy = 0, sxy = 0;
  for (const [x, y] of points) {
    const dx = x - meanX;
    const dy = y - meanY;
    sxx += dx * dx;
    syy += dy * dy;
    sxy += dx * dy;
  }
  const theta = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  const dirX = Math.cos(theta);
  const dirY = Math.sin(theta);
  return points
    .map((point) => ({
      point,
      t: (point[0] - meanX) * dirX + (point[1] - meanY) * dirY,
    }))
    .sort((a, b) => a.t - b.t)
    .map((entry) => entry.point);
}

function sampleHelperCurve(points, samplesPerSegment = 12) {
  if (!Array.isArray(points) || points.length < 2) return [];
  const ordered = sortHelperPointsByProjection(points);
  const out = [];
  for (let seg = 0; seg < ordered.length - 1; seg += 1) {
    const p1 = ordered[seg];
    const p2 = ordered[seg + 1];
    const start = seg === 0 ? 0 : 1;
    for (let i = start; i <= samplesPerSegment; i += 1) {
      const t = i / samplesPerSegment;
      out.push([
        p1[0] * (1 - t) + p2[0] * t,
        p1[1] * (1 - t) + p2[1] * t,
      ]);
    }
  }
  return out;
}

function fitLineStats(points) {
  const n = points.length;
  let meanX = 0, meanY = 0;
  for (const [x, y] of points) {
    meanX += x;
    meanY += y;
  }
  meanX /= n;
  meanY /= n;
  let sxx = 0, syy = 0, sxy = 0, error = 0;
  for (const [x, y] of points) {
    const dx = x - meanX;
    const dy = y - meanY;
    sxx += dx * dx;
    syy += dy * dy;
    sxy += dx * dy;
  }
  const theta = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  const dirX = Math.cos(theta);
  const dirY = Math.sin(theta);
  for (const [x, y] of points) {
    const perp = -(x - meanX) * dirY + (y - meanY) * dirX;
    error += perp * perp;
  }
  return { error: error / Math.max(n, 1), theta, meanX, meanY };
}

function weightedLineBendStats(points) {
  if (!Array.isArray(points) || points.length < 3) return { maxAbsDeviation: 0, weightedBendEnergy: 0, bendEnergy: 0 };
  const first = points[0];
  const last = points[points.length - 1];
  const dx = last[0] - first[0];
  const dy = last[1] - first[1];
  const denom = Math.hypot(dx, dy) || 1;
  const absDeviations = [];
  let bendEnergy = 0;
  let weightedBendEnergy = 0;
  for (let i = 1; i < points.length - 1; i += 1) {
    const px = points[i][0] - first[0];
    const py = points[i][1] - first[1];
    const absDeviation = Math.abs((px * dy - py * dx) / denom);
    absDeviations.push(absDeviation);
    bendEnergy += absDeviation * absDeviation;
  }
  const count = absDeviations.length;
  for (let i = 0; i < count; i += 1) {
    const edgeAlpha = count <= 1 ? 1 : Math.abs((i / (count - 1)) * 2 - 1);
    const weight = 1 + edgeAlpha * 1.6;
    weightedBendEnergy += absDeviations[i] * absDeviations[i] * weight;
  }
  return {
    maxAbsDeviation: absDeviations.length ? Math.max(...absDeviations) : 0,
    weightedBendEnergy,
    bendEnergy,
  };
}

function lineCurvatureProfile(points) {
  if (!Array.isArray(points) || points.length < 3) return [];
  const values = [];
  for (let i = 1; i < points.length - 1; i += 1) {
    const prev = points[i - 1];
    const curr = points[i];
    const next = points[i + 1];
    const ax = curr[0] - prev[0];
    const ay = curr[1] - prev[1];
    const bx = next[0] - curr[0];
    const by = next[1] - curr[1];
    const cross = ax * by - ay * bx;
    const aLen = Math.hypot(ax, ay);
    const bLen = Math.hypot(bx, by);
    const cLen = Math.hypot(next[0] - prev[0], next[1] - prev[1]);
    const denom = Math.max(aLen * bLen * cLen, 1e-9);
    values.push(Math.abs((2 * cross) / denom));
  }
  return values;
}

function distortPointForView(x, y, width, height, params) {
  const cx = width / 2 + Number(params?.cx_offset || 0);
  const cy = height / 2 + Number(params?.cy_offset || 0);
  const norm = Math.max(width, height) / 2;
  const lambda = Number(params?.lambda || 0);
  const ux = (x - cx) / norm;
  const uy = (y - cy) / norm;
  let dx = ux;
  let dy = uy;
  for (let i = 0; i < 8; i += 1) {
    const r2 = dx * dx + dy * dy;
    const factor = 1 + lambda * r2;
    dx = ux * factor;
    dy = uy * factor;
  }
  return [cx + dx * norm, cy + dy * norm];
}

function undistortPointForView(x, y, width, height, params) {
  const cx = width / 2 + Number(params?.cx_offset || 0);
  const cy = height / 2 + Number(params?.cy_offset || 0);
  const norm = Math.max(width, height) / 2;
  const dx = (x - cx) / norm;
  const dy = (y - cy) / norm;
  const r2 = dx * dx + dy * dy;
  const factor = 1 + Number(params?.lambda || 0) * r2;
  return [cx + (dx / factor) * norm, cy + (dy / factor) * norm];
}

function rawPointToCorrectedPoint(x, y, width, height, params) {
  const corrected = undistortPointForView(x, y, width, height, params);
  const angle = (Number(params?.rotation_deg || 0) * Math.PI) / 180;
  const dx = corrected[0] - width / 2;
  const dy = corrected[1] - height / 2;
  return [
    width / 2 + dx * Math.cos(angle) - dy * Math.sin(angle),
    height / 2 + dx * Math.sin(angle) + dy * Math.cos(angle),
  ];
}

function estimateTrainerCorrection(polyline, width, height) {
  if (!Array.isArray(polyline) || polyline.length < 4) return null;
  const helperObservations = sortHelperPointsByProjection(polyline);
  if (helperObservations.length < 4) return null;
  const baseline = weightedLineBendStats(helperObservations);
  const baselineCurvature = lineCurvatureProfile(helperObservations);
  const lineImprovesMonotonically = (candidatePoints) => {
    const candidateBend = weightedLineBendStats(candidatePoints);
    const candidateCurvature = lineCurvatureProfile(candidatePoints);
    if (candidateBend.maxAbsDeviation > baseline.maxAbsDeviation + 0.02) return false;
    if (candidateBend.weightedBendEnergy > baseline.weightedBendEnergy + 0.08) return false;
    const edgeCount = Math.max(1, Math.floor(candidatePoints.length * 0.18));
    const candidateDeviations = [];
    const first = candidatePoints[0];
    const last = candidatePoints[candidatePoints.length - 1];
    const dx = last[0] - first[0];
    const dy = last[1] - first[1];
    const denom = Math.hypot(dx, dy) || 1;
    for (let i = 1; i < candidatePoints.length - 1; i += 1) {
      const px = candidatePoints[i][0] - first[0];
      const py = candidatePoints[i][1] - first[1];
      candidateDeviations.push(Math.abs((px * dy - py * dx) / denom));
    }
    const baselineDeviations = [];
    const baseFirst = helperObservations[0];
    const baseLast = helperObservations[helperObservations.length - 1];
    const bdx = baseLast[0] - baseFirst[0];
    const bdy = baseLast[1] - baseFirst[1];
    const bden = Math.hypot(bdx, bdy) || 1;
    for (let i = 1; i < helperObservations.length - 1; i += 1) {
      const px = helperObservations[i][0] - baseFirst[0];
      const py = helperObservations[i][1] - baseFirst[1];
      baselineDeviations.push(Math.abs((px * bdy - py * bdx) / bden));
    }
    for (let i = 0; i < candidateDeviations.length; i += 1) {
      const base = baselineDeviations[i] ?? baseline.maxAbsDeviation;
      const isEdge = i < edgeCount || i >= Math.max(0, candidateDeviations.length - edgeCount);
      const tolerance = isEdge ? 0.03 : 0.08;
      if (candidateDeviations[i] > base + tolerance) return false;
    }
    const curvatureEdgeCount = Math.max(1, Math.floor(candidateCurvature.length * 0.18));
    for (let i = 0; i < candidateCurvature.length; i += 1) {
      const base = baselineCurvature[i] ?? 0;
      const isEdge = i < curvatureEdgeCount || i >= Math.max(0, candidateCurvature.length - curvatureEdgeCount);
      const tolerance = isEdge ? 0.0005 : 0.0015;
      if (candidateCurvature[i] > base + tolerance) return false;
    }
    return true;
  };
  const edgeRadiusSamples = [0.7, 0.82, 0.92, 1.0];
  const regularizedScore = (params, stats) => {
    const lambda = Number(params.lambda || 0);
    const cxOffsetNorm = Number(params.cx_offset || 0) / Math.max(width || 1, 1);
    const cyOffsetNorm = Number(params.cy_offset || 0) / Math.max(height || 1, 1);
    let edgeStretchPenalty = 0;
    for (const radius of edgeRadiusSamples) {
      const factor = 1 + lambda * radius * radius;
      edgeStretchPenalty += Math.pow(Math.max(0, Math.abs(factor - 1) - 0.1), 2);
      edgeStretchPenalty += Math.pow(Math.max(0, 0.62 - factor), 2) * 3.5;
      edgeStretchPenalty += Math.pow(Math.max(0, factor - 1.5), 2) * 3.5;
    }
    const complexityPenalty =
      0.02 * Math.abs(lambda)
      + 0.1 * lambda * lambda
      + 0.05 * Math.abs(cxOffsetNorm)
      + 0.05 * Math.abs(cyOffsetNorm)
      + 0.2 * (cxOffsetNorm * cxOffsetNorm + cyOffsetNorm * cyOffsetNorm);
    return {
      maxAbsDeviation: stats.maxAbsDeviation,
      weightedBendEnergy: stats.weightedBendEnergy,
      edgeStretchPenalty,
      bendEnergy: stats.bendEnergy,
      lineFitError: stats.error,
      complexityPenalty,
    };
  };
  const cmp = (a, b) => {
    const keys = ["maxAbsDeviation", "weightedBendEnergy", "edgeStretchPenalty", "bendEnergy", "lineFitError", "complexityPenalty"];
    for (const key of keys) {
      const d = Number(a[key]) - Number(b[key]);
      if (Math.abs(d) > 1e-9) return d < 0 ? -1 : 1;
    }
    return 0;
  };
  const clamp = (p) => ({
    model: "division",
    lambda: Math.max(-0.28, Math.min(0.28, Number(p.lambda || 0))),
    cx_offset: Math.max(-0.18 * width, Math.min(0.18 * width, Number(p.cx_offset || 0))),
    cy_offset: Math.max(-0.18 * height, Math.min(0.18 * height, Number(p.cy_offset || 0))),
    k1: 0,
    k2: 0,
    k3: 0,
  });
  const objective = (params) => {
    const undistorted = helperObservations.map((p) => undistortPointForView(p[0], p[1], width, height, params));
    if (!lineImprovesMonotonically(undistorted)) {
      return {
        error: Number.POSITIVE_INFINITY,
        theta: 0,
        maxAbsDeviation: Number.POSITIVE_INFINITY,
        weightedBendEnergy: Number.POSITIVE_INFINITY,
        bendEnergy: Number.POSITIVE_INFINITY,
        lineFitError: Number.POSITIVE_INFINITY,
        complexityPenalty: Number.POSITIVE_INFINITY,
        edgeStretchPenalty: Number.POSITIVE_INFINITY,
        objective: {
          maxAbsDeviation: Number.POSITIVE_INFINITY,
          weightedBendEnergy: Number.POSITIVE_INFINITY,
          edgeStretchPenalty: Number.POSITIVE_INFINITY,
          bendEnergy: Number.POSITIVE_INFINITY,
          lineFitError: Number.POSITIVE_INFINITY,
          complexityPenalty: Number.POSITIVE_INFINITY,
        },
      };
    }
    const line = fitLineStats(undistorted);
    const bend = weightedLineBendStats(undistorted);
    return {
      ...line,
      ...bend,
      objective: regularizedScore(params, { ...line, ...bend }),
    };
  };
  let current = clamp({ lambda: 0, cx_offset: 0, cy_offset: 0 });
  let best = objective(current);
  let bestObjective = best.objective;
  const steps = [{ ...current, elapsedMs: 0, diagnostics: { before_max_abs_deviation: baseline.maxAbsDeviation, after_max_abs_deviation: best.maxAbsDeviation, before_bend_energy: baseline.weightedBendEnergy, after_bend_energy: best.weightedBendEnergy } }];
  const t0 = performance.now();
  const coarseLambdaGrid = [-0.26, -0.2, -0.15, -0.1, -0.06, -0.03, 0, 0.03, 0.06, 0.1, 0.15, 0.2, 0.26];
  const coarseCxGrid = [-0.12, -0.06, 0, 0.06, 0.12].map((s) => s * width);
  const coarseCyGrid = [-0.12, -0.06, 0, 0.06, 0.12].map((s) => s * height);
  for (const lambda of coarseLambdaGrid) {
    for (const cxOffset of coarseCxGrid) {
      for (const cyOffset of coarseCyGrid) {
        const cand = clamp({ lambda, cx_offset: cxOffset, cy_offset: cyOffset });
        const score = objective(cand);
        if (cmp(score.objective, bestObjective) < 0) {
          current = cand;
          best = score;
          bestObjective = score.objective;
          steps.push({ ...current, elapsedMs: performance.now() - t0, diagnostics: { before_max_abs_deviation: baseline.maxAbsDeviation, after_max_abs_deviation: best.maxAbsDeviation, before_bend_energy: baseline.weightedBendEnergy, after_bend_energy: best.weightedBendEnergy } });
        }
      }
    }
  }
  const lambdaSteps = [0.08, 0.03, 0.012, 0.005];
  const cxSteps = [0.08, 0.04, 0.02, 0.01].map((s) => s * width);
  const cySteps = [0.08, 0.04, 0.02, 0.01].map((s) => s * height);
  for (let idx = 0; idx < lambdaSteps.length; idx += 1) {
    let improved = true;
    while (improved) {
      improved = false;
      for (const [key, step] of [["lambda", lambdaSteps[idx]], ["cx_offset", cxSteps[idx]], ["cy_offset", cySteps[idx]]]) {
        for (const dir of [-1, 1]) {
          const cand = clamp({ ...current, [key]: current[key] + dir * step });
          const score = objective(cand);
          if (cmp(score.objective, bestObjective) < 0) {
            current = cand;
            best = score;
            bestObjective = score.objective;
            steps.push({ ...current, elapsedMs: performance.now() - t0, diagnostics: { before_max_abs_deviation: baseline.maxAbsDeviation, after_max_abs_deviation: best.maxAbsDeviation, before_bend_energy: baseline.weightedBendEnergy, after_bend_energy: best.weightedBendEnergy } });
            improved = true;
          }
        }
      }
    }
  }
  let rotationDeg = -(best.theta * 180 / Math.PI);
  while (rotationDeg <= -90) rotationDeg += 180;
  while (rotationDeg > 90) rotationDeg -= 180;
  return {
    ...current,
    rotation_deg: rotationDeg,
    path: steps.map((step) => ({ ...step, rotation_deg: rotationDeg })),
    diagnostics: {
      before_max_abs_deviation: baseline.maxAbsDeviation,
      after_max_abs_deviation: best.maxAbsDeviation,
      before_bend_energy: baseline.weightedBendEnergy,
      after_bend_energy: best.weightedBendEnergy,
    },
  };
}

function normalizeViewCorrection(params) {
  if (!params) return null;
  return {
    model: "division",
    lambda: Number(params.lambda || 0),
    cx_offset: Number(params.cx_offset || 0),
    cy_offset: Number(params.cy_offset || 0),
    k1: Number(params.k1 || 0),
    k2: Number(params.k2 || 0),
    k3: Number(params.k3 || 0),
    rotation_deg: Number(params.rotation_deg || 0),
  };
}

function hasMeaningfulSavedCorrection(params) {
  if (!params) return false;
  const lambda = Math.abs(Number(params.lambda || 0));
  const cx = Math.abs(Number(params.cx_offset || 0));
  const cy = Math.abs(Number(params.cy_offset || 0));
  const k1 = Math.abs(Number(params.k1 || 0));
  const k2 = Math.abs(Number(params.k2 || 0));
  const k3 = Math.abs(Number(params.k3 || 0));
  return lambda > 1e-6 || cx > 1e-6 || cy > 1e-6 || k1 > 1e-6 || k2 > 1e-6 || k3 > 1e-6;
}

function buildInitialCorrectionFromRecord(record, width, height) {
  const saved = normalizeViewCorrection(record?.distortion_params || null);
  if (hasMeaningfulSavedCorrection(saved)) return saved;
  const helperPoints = Array.isArray(record?.distortion_helper_points)
    ? record.distortion_helper_points.filter((point) => Array.isArray(point) && point.length >= 2)
    : [];
  if (helperPoints.length >= 4) {
    const fitted = estimateTrainerCorrection(helperPoints, width, height);
    if (fitted) return fitted;
  }
  return saved;
}

function buildTrueCorrectionFromAnalysis(data, width, height) {
  const rawCourtPolylines = (data?.label_record?.geometry_polylines) || [];
  if (rawCourtPolylines.length && rawCourtPolylines[0] && rawCourtPolylines[0].length >= 4) {
    return estimateTrainerCorrection(rawCourtPolylines[0], width, height);
  }
  const cam = data?.label_analysis?.camera_model;
  if (!cam || cam.estimate_status !== "refined") return null;
  return normalizeViewCorrection(cam.distortion || null);
}

function rotationMatrixFromEuler(yawDeg, pitchDeg, rollDeg) {
  const yaw = (Number(yawDeg || 0) * Math.PI) / 180;
  const pitch = (Number(pitchDeg || 0) * Math.PI) / 180;
  const roll = (Number(rollDeg || 0) * Math.PI) / 180;
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const cr = Math.cos(roll), sr = Math.sin(roll);
  const rz = [
    [cr, -sr, 0],
    [sr, cr, 0],
    [0, 0, 1],
  ];
  const rx = [
    [1, 0, 0],
    [0, cp, -sp],
    [0, sp, cp],
  ];
  const ry = [
    [cy, 0, sy],
    [0, 1, 0],
    [-sy, 0, cy],
  ];
  const mul3 = (a, b) => {
    const out = Array.from({ length: 3 }, () => [0, 0, 0]);
    for (let r = 0; r < 3; r += 1) {
      for (let c = 0; c < 3; c += 1) {
        out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
      }
    }
    return out;
  };
  return mul3(mul3(ry, rx), rz);
}

function mat3VecMul(m, v) {
  return [
    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
  ];
}

function mat3TransposeVecMul(m, v) {
  return [
    m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
    m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
    m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
  ];
}

function estimateBallWorldFromCamera(ball, cameraModel) {
  if (!ball || !cameraModel) return null;
  const intr = cameraModel.intrinsics || {};
  const pose = cameraModel.pose || {};
  const fx = Number(intr.fx || 0);
  const fy = Number(intr.fy || 0);
  const cx = Number(intr.cx || 0);
  const cy = Number(intr.cy || 0);
  const radiusPx = Number(ball.radius || ball.radius_prior || 0);
  if (!(fx > 0) || !(fy > 0) || !(radiusPx > 0)) return null;
  const r = rotationMatrixFromEuler(pose.yaw_deg, pose.pitch_deg, pose.roll_deg);
  const ballRadiusM = 0.105;
  const zCam = Math.max(0.05, fx * ballRadiusM / radiusPx);
  const ray = [
    (Number(ball.center_x || 0) - cx) / fx,
    (Number(ball.center_y || 0) - cy) / fy,
    1,
  ];
  const camPoint = ray.map((v) => v * zCam);
  const t = [Number(pose.tx_m || 0), Number(pose.ty_m || 0), Number(pose.tz_m || 0)];
  const shifted = [camPoint[0] - t[0], camPoint[1] - t[1], camPoint[2] - t[2]];
  const world = mat3TransposeVecMul(r, shifted);
  return { x_m: world[0], y_m: world[1], z_m: world[2], depth_m: zCam, radius_px: radiusPx };
}

function projectWorldPointCamera(worldPoint, cameraModel) {
  if (!cameraModel) return null;
  const intr = cameraModel.intrinsics || {};
  const pose = cameraModel.pose || {};
  const fx = Number(intr.fx || 0);
  const fy = Number(intr.fy || 0);
  const cx = Number(intr.cx || 0);
  const cy = Number(intr.cy || 0);
  if (!(fx > 0) || !(fy > 0)) return null;
  const r = rotationMatrixFromEuler(pose.yaw_deg, pose.pitch_deg, pose.roll_deg);
  const t = [Number(pose.tx_m || 0), Number(pose.ty_m || 0), Number(pose.tz_m || 0)];
  const x = Number(worldPoint[0] || 0);
  const y = Number(worldPoint[1] || 0);
  const z = Number(worldPoint[2] || 0);
  const cam = mat3VecMul(r, [x, y, z]);
  const camX = cam[0] + t[0];
  const camY = cam[1] + t[1];
  const camZ = cam[2] + t[2];
  if (!(camZ > 1e-6)) return null;
  return [fx * (camX / camZ) + cx, fy * (camY / camZ) + cy];
}

function orderCourtPointsForGrid(points) {
  if (!Array.isArray(points) || points.length < 4) return null;
  const pts = points.slice(0, 4).map((p) => [Number(p[0]), Number(p[1])]).sort((a, b) => a[1] - b[1]);
  const top = pts.slice(0, 2).sort((a, b) => a[0] - b[0]);
  const bottom = pts.slice(2, 4).sort((a, b) => a[0] - b[0]);
  return [bottom[0], top[0], top[1], bottom[1]];
}

function solveCourtHomography(imagePts) {
  const ordered = orderCourtPointsForGrid(inferCourtQuadFromSupportPolyline(imagePts) || imagePts);
  if (!ordered) return null;
  const world = [[0, 0], [0, 16], [8, 16], [8, 0]];
  const a = [];
  const b = [];
  for (let i = 0; i < 4; i += 1) {
    const [x, y] = world[i];
    const [u, v] = ordered[i];
    a.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u);
    a.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v);
  }
  const solve = (m, rhs) => {
    const n = rhs.length;
    const A = m.map((row, i) => row.concat([rhs[i]]));
    for (let col = 0; col < n; col += 1) {
      let pivot = col;
      for (let r = col + 1; r < n; r += 1) if (Math.abs(A[r][col]) > Math.abs(A[pivot][col])) pivot = r;
      if (Math.abs(A[pivot][col]) < 1e-9) return null;
      [A[col], A[pivot]] = [A[pivot], A[col]];
      const div = A[col][col];
      for (let c = col; c <= n; c += 1) A[col][c] /= div;
      for (let r = 0; r < n; r += 1) {
        if (r === col) continue;
        const factor = A[r][col];
        for (let c = col; c <= n; c += 1) A[r][c] -= factor * A[col][c];
      }
    }
    return A.map((row) => row[n]);
  };
  const h = solve(a, b);
  if (!h) return null;
  return [
    [h[0], h[1], h[2]],
    [h[3], h[4], h[5]],
    [h[6], h[7], 1],
  ];
}

function projectWorldPointH(worldPoint, h) {
  if (!h) return null;
  const x = Number(worldPoint[0] || 0);
  const y = Number(worldPoint[1] || 0);
  const den = h[2][0] * x + h[2][1] * y + h[2][2];
  if (Math.abs(den) < 1e-9) return null;
  return [
    (h[0][0] * x + h[0][1] * y + h[0][2]) / den,
    (h[1][0] * x + h[1][1] * y + h[1][2]) / den,
  ];
}

function buildCameraModelSegments(cameraModel, overrideCourtPoints = null) {
  if (!cameraModel) return [];
  const segs = [];
  const pushSeg = (a, b, role = "court") => {
    const pa = projectWorldPointCamera(a, cameraModel);
    const pb = projectWorldPointCamera(b, cameraModel);
    if (pa && pb) segs.push({ a: pa, b: pb, role });
  };
  pushSeg([0, 0, 0], [8, 0, 0], "court");
  pushSeg([8, 0, 0], [8, 16, 0], "court");
  pushSeg([8, 16, 0], [0, 16, 0], "court");
  pushSeg([0, 16, 0], [0, 0, 0], "court");
  pushSeg([0, 8, 0], [8, 8, 0], "court");
  if (segs.length) return segs;
  const courtPts = inferCourtQuadFromSupportPolyline(overrideCourtPoints || cameraModel?.court_orthopoints || null)
    || overrideCourtPoints
    || cameraModel?.court_orthopoints
    || null;
  const h = solveCourtHomography(courtPts);
  if (!h) return [];
  const fallbackSegs = [];
  const pushHSeg = (a, b, role = "court") => {
    const pa = projectWorldPointH(a, h);
    const pb = projectWorldPointH(b, h);
    if (pa && pb) fallbackSegs.push({ a: pa, b: pb, role });
  };
  pushHSeg([0, 0], [8, 0], "court");
  pushHSeg([8, 0], [8, 16], "court");
  pushHSeg([8, 16], [0, 16], "court");
  pushHSeg([0, 16], [0, 0], "court");
  pushHSeg([0, 8], [8, 8], "court");
  return fallbackSegs;
}

function buildProjectedCourtPolyline(cameraModel) {
  if (!cameraModel) return [];
  const pts = [
    projectWorldPointCamera([0, 0, 0], cameraModel),
    projectWorldPointCamera([0, 16, 0], cameraModel),
    projectWorldPointCamera([8, 16, 0], cameraModel),
    projectWorldPointCamera([8, 0, 0], cameraModel),
  ];
  if (pts.some((p) => !Array.isArray(p))) return [];
  return pts;
}

function lineIntersection2D(a0, a1, b0, b1) {
  const x1 = Number(a0[0]), y1 = Number(a0[1]);
  const x2 = Number(a1[0]), y2 = Number(a1[1]);
  const x3 = Number(b0[0]), y3 = Number(b0[1]);
  const x4 = Number(b1[0]), y4 = Number(b1[1]);
  const den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
  if (Math.abs(den) < 1e-9) return null;
  const px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den;
  const py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den;
  return [px, py];
}

function inferCourtQuadFromSupportPolyline(points) {
  if (!Array.isArray(points) || points.length < 4) return null;
  const pts = points.slice(0, 4).map((p) => [Number(p[0]), Number(p[1])]);
  const c0 = lineIntersection2D(pts[3], pts[0], pts[0], pts[1]);
  const c1 = lineIntersection2D(pts[0], pts[1], pts[1], pts[2]);
  const c2 = lineIntersection2D(pts[1], pts[2], pts[2], pts[3]);
  const c3 = lineIntersection2D(pts[2], pts[3], pts[3], pts[0]);
  if ([c0, c1, c2, c3].some((p) => !Array.isArray(p))) return null;
  return [c0, c1, c2, c3];
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function interpolatePolyline(aPts, bPts, t) {
  if (!Array.isArray(aPts) || !Array.isArray(bPts) || aPts.length < 4 || bPts.length < 4) return bPts || aPts || [];
  const n = Math.min(aPts.length, bPts.length);
  const out = [];
  for (let i = 0; i < n; i += 1) {
    out.push([
      lerp(Number(aPts[i][0]), Number(bPts[i][0]), t),
      lerp(Number(aPts[i][1]), Number(bPts[i][1]), t),
    ]);
  }
  return out;
}

function interpolateCameraModel(seedModel, refinedModel, t) {
  if (!seedModel && !refinedModel) return null;
  if (!seedModel) return refinedModel;
  if (!refinedModel) return seedModel;
  const intrA = seedModel.intrinsics || {};
  const intrB = refinedModel.intrinsics || {};
  const distA = seedModel.distortion || {};
  const distB = refinedModel.distortion || {};
  const poseA = seedModel.pose || {};
  const poseB = refinedModel.pose || {};
  return {
    intrinsics: {
      fx: lerp(Number(intrA.fx || 0), Number(intrB.fx || 0), t),
      fy: lerp(Number(intrA.fy || 0), Number(intrB.fy || 0), t),
      cx: lerp(Number(intrA.cx || 0), Number(intrB.cx || 0), t),
      cy: lerp(Number(intrA.cy || 0), Number(intrB.cy || 0), t),
    },
    distortion: {
      rotation_deg: lerp(Number(distA.rotation_deg || 0), Number(distB.rotation_deg || 0), t),
      k1: lerp(Number(distA.k1 || 0), Number(distB.k1 || 0), t),
      k2: lerp(Number(distA.k2 || 0), Number(distB.k2 || 0), t),
      k3: lerp(Number(distA.k3 || 0), Number(distB.k3 || 0), t),
      p1: lerp(Number(distA.p1 || 0), Number(distB.p1 || 0), t),
      p2: lerp(Number(distA.p2 || 0), Number(distB.p2 || 0), t),
    },
    pose: {
      tx_m: lerp(Number(poseA.tx_m || 0), Number(poseB.tx_m || 0), t),
      ty_m: lerp(Number(poseA.ty_m || 0), Number(poseB.ty_m || 0), t),
      tz_m: lerp(Number(poseA.tz_m || 0), Number(poseB.tz_m || 0), t),
      yaw_deg: lerp(Number(poseA.yaw_deg || 0), Number(poseB.yaw_deg || 0), t),
      pitch_deg: lerp(Number(poseA.pitch_deg || 0), Number(poseB.pitch_deg || 0), t),
      roll_deg: lerp(Number(poseA.roll_deg || 0), Number(poseB.roll_deg || 0), t),
    },
  };
}

function buildCameraModelPath(seedModel, refinedModel, steps = 24) {
  if (refinedModel?.projection_artifact && Array.isArray(refinedModel.projection_artifact.solver_steps) && refinedModel.projection_artifact.solver_steps.length) {
    return refinedModel.projection_artifact.solver_steps;
  }
  if (!refinedModel) return [];
  if (Array.isArray(refinedModel.solver_path) && refinedModel.solver_path.length) {
    return refinedModel.solver_path.map((step) => ({
      intrinsics: { ...(step.intrinsics || refinedModel.intrinsics || {}) },
      distortion: { ...(refinedModel.distortion || {}) },
      pose: { ...(step.pose || refinedModel.pose || {}) },
      court_orthopoints: Array.isArray(step.court_orthopoints) ? step.court_orthopoints.map((p) => [Number(p[0]), Number(p[1])]) : [],
      net_height_m: Number(step.net_height_m || refinedModel.net_height_m || 0),
      objective: step.objective || null,
    }));
  }
  const path = [];
  for (let i = 0; i < steps; i += 1) {
    const t = steps <= 1 ? 1 : i / (steps - 1);
    path.push(interpolateCameraModel(seedModel, refinedModel, t));
  }
  return path;
}

function buildProjectionPathFromAnalysis(labelAnalysis) {
  const artifact = labelAnalysis?.projection_artifact || null;
  if (artifact && Array.isArray(artifact.solver_steps) && artifact.solver_steps.length) {
    return artifact.solver_steps;
  }
  if (artifact?.seed && artifact?.refined) return [artifact.seed, artifact.refined];
  if (artifact?.refined) return [artifact.refined];
  return [];
}

function getProjectionStepCameraModel(step) {
  return step?.camera_model || step || null;
}

function buildLineRefinementPath(imageData, targetPolyline) {
  if (!imageData || !Array.isArray(targetPolyline) || targetPolyline.length < 4) return [];
  const width = imageData.width;
  const height = imageData.height;
  const target = targetPolyline.map((p) => [Number(p[0]), Number(p[1])]);
  const src = imageData.data;
  const colorAt = (x, y) => {
    const ix = Math.max(1, Math.min(width - 2, Math.round(x)));
    const iy = Math.max(1, Math.min(height - 2, Math.round(y)));
    const idx = (iy * width + ix) * 4;
    return { r: src[idx + 0], g: src[idx + 1], b: src[idx + 2] };
  };
  const lineScoreAt = (x, y) => {
    const { r, g, b } = colorAt(x, y);
    return (1.6 * b + 0.15 * g - 0.9 * r);
  };
  const sampleAlongNormal = (pt, tangent, offset) => {
    const tx = tangent[0];
    const ty = tangent[1];
    const len = Math.max(1e-6, Math.hypot(tx, ty));
    const nx = -ty / len;
    const ny = tx / len;
    return [pt[0] + nx * offset, pt[1] + ny * offset];
  };
  const polylineEnergy = (poly) => {
    let support = 0;
    let offsetPenalty = 0;
    let curvaturePenalty = 0;
    let stripePenalty = 0;
    for (let i = 0; i < poly.length; i += 1) {
      const prev = poly[Math.max(0, i - 1)];
      const next = poly[Math.min(poly.length - 1, i + 1)];
      const tangent = [next[0] - prev[0], next[1] - prev[1]];
      const p = poly[i];
      const pEdgeL = sampleAlongNormal(p, tangent, -2.0);
      const pEdgeR = sampleAlongNormal(p, tangent, 2.0);
      const pOuterL = sampleAlongNormal(p, tangent, -4.5);
      const pOuterR = sampleAlongNormal(p, tangent, 4.5);
      const mid = lineScoreAt(p[0], p[1]);
      const edgeL = lineScoreAt(pEdgeL[0], pEdgeL[1]);
      const edgeR = lineScoreAt(pEdgeR[0], pEdgeR[1]);
      const outerL = lineScoreAt(pOuterL[0], pOuterL[1]);
      const outerR = lineScoreAt(pOuterR[0], pOuterR[1]);
      support += mid * 1.0 + edgeL * 2.2 + edgeR * 2.2;
      const stripeContrast = (edgeL + edgeR) - (outerL + outerR) - Math.abs(edgeL - edgeR) * 0.75;
      support += stripeContrast * 1.4;
      stripePenalty += Math.max(0, 6.0 - stripeContrast);
      offsetPenalty += Math.hypot(p[0] - target[i][0], p[1] - target[i][1]) * 2.0;
    }
    for (let i = 1; i < poly.length - 1; i += 1) {
      const ax = poly[i][0] - poly[i - 1][0];
      const ay = poly[i][1] - poly[i - 1][1];
      const bx = poly[i + 1][0] - poly[i][0];
      const by = poly[i + 1][1] - poly[i][1];
      const denom = Math.max(1e-6, Math.hypot(ax, ay) * Math.hypot(bx, by));
      curvaturePenalty += Math.abs(ax * by - ay * bx) / denom * 18.0;
    }
    return {
      support,
      offsetPenalty,
      curvaturePenalty,
      stripePenalty,
      total: support - offsetPenalty - curvaturePenalty - stripePenalty,
    };
  };
  const snapped = target.map((pt, i) => {
    const prev = target[Math.max(0, i - 1)];
    const next = target[Math.min(target.length - 1, i + 1)];
    const tangent = [next[0] - prev[0], next[1] - prev[1]];
    let best = [pt[0], pt[1]];
    let bestDiag = polylineEnergy(target);
    for (let off = -1.5; off <= 1.5001; off += 0.25) {
      const candidate = target.map((p) => [p[0], p[1]]);
      candidate[i] = sampleAlongNormal(pt, tangent, off);
      const candDiag = polylineEnergy(candidate);
      if (candDiag.total > bestDiag.total + 1e-6) {
        best = candidate[i];
        bestDiag = candDiag;
      }
    }
    const displacement = Math.hypot(best[0] - pt[0], best[1] - pt[1]);
    return (bestDiag.total > polylineEnergy(target).total + 0.75 && displacement <= 1.5) ? best : [pt[0], pt[1]];
  });
  const path = [];
  const steps = 24;
  for (let i = 0; i < steps; i += 1) {
    const t = steps <= 1 ? 1 : i / (steps - 1);
    const points = target.map((p, idx) => [
      lerp(p[0], snapped[idx][0], t),
      lerp(p[1], snapped[idx][1], t),
    ]);
    path.push({
      points,
      diagnostics: polylineEnergy(points),
    });
  }
  return path;
}

function formatNum(v, digits = 3) {
  return Number.isFinite(v) ? Number(v).toFixed(digits) : "n/a";
}

function buildModelSummary(labelAnalysis, includeNN, nnAnalysis, fittedCorrection) {
  const cam = labelAnalysis.camera_model || {};
  const seed = labelAnalysis.camera_model_seed || {};
  const intr = cam.intrinsics || {};
  const dist = cam.distortion || {};
  const pose = cam.pose || {};
  const lines = [];
  lines.push(`label init status: ${cam.estimate_status || "unknown"}`);
  if (Array.isArray(cam.notes) && cam.notes.length) {
    lines.push(`notes: ${cam.notes.join(" | ")}`);
  }
  lines.push(`court support polys: ${((labelAnalysis.court_polylines || labelAnalysis.court_points || [])?.length) || 0}`);
  lines.push(`grid segments: ${Array.isArray(labelAnalysis.court_grid) ? labelAnalysis.court_grid.length : 0}`);
  lines.push("");
  lines.push("intrinsics");
  lines.push(`fx ${formatNum(intr.fx, 2)}  fy ${formatNum(intr.fy, 2)}  cx ${formatNum(intr.cx, 2)}  cy ${formatNum(intr.cy, 2)}`);
  lines.push("");
  lines.push("distortion / correction");
  lines.push(`k1 ${formatNum(dist.k1, 6)}  k2 ${formatNum(dist.k2, 6)}  k3 ${formatNum(dist.k3, 6)}  p1 ${formatNum(dist.p1, 6)}  p2 ${formatNum(dist.p2, 6)}`);
  lines.push(`rotation_deg ${formatNum(dist.rotation_deg, 3)}`);
  if (seed.distortion) {
    lines.push(`seed rotation_deg ${formatNum(seed.distortion.rotation_deg, 3)}`);
  }
  if (fittedCorrection) {
    lines.push("");
    lines.push("line-derived correction");
    lines.push(`lambda ${formatNum(fittedCorrection.lambda, 6)}  cx_offset ${formatNum(fittedCorrection.cx_offset, 2)}  cy_offset ${formatNum(fittedCorrection.cy_offset, 2)}  rotation_deg ${formatNum(fittedCorrection.rotation_deg, 3)}`);
    if (fittedCorrection.diagnostics) {
      lines.push(`max bend ${formatNum(fittedCorrection.diagnostics.before_max_abs_deviation, 2)} -> ${formatNum(fittedCorrection.diagnostics.after_max_abs_deviation, 2)}`);
      lines.push(`weighted bend ${formatNum(fittedCorrection.diagnostics.before_bend_energy, 2)} -> ${formatNum(fittedCorrection.diagnostics.after_bend_energy, 2)}`);
    }
    lines.push("corrected-view target: helper back line horizontal");
  }
  lines.push("");
  lines.push("pose");
  lines.push(`tx ${formatNum(pose.tx_m, 3)}m  ty ${formatNum(pose.ty_m, 3)}m  tz ${formatNum(pose.tz_m, 3)}m`);
  lines.push(`yaw ${formatNum(pose.yaw_deg, 3)}  pitch ${formatNum(pose.pitch_deg, 3)}  roll ${formatNum(pose.roll_deg, 3)}`);
  if (labelAnalysis.ball_world) {
    const b = labelAnalysis.ball_world;
    lines.push("");
    lines.push("ball world estimate");
    lines.push(`x ${formatNum(b.x_m, 3)}m  y ${formatNum(b.y_m, 3)}m  z ${formatNum(b.z_m, 3)}m  depth ${formatNum(b.depth_m, 3)}m`);
  }
  if (includeNN && nnAnalysis && nnAnalysis.camera_model) {
    const nnCam = nnAnalysis.camera_model || {};
    lines.push("");
    lines.push(`nn status: ${nnCam.estimate_status || "unknown"}`);
    lines.push(`nn reproj: ${formatNum(nnCam.reprojection_error_px, 2)}px`);
  }
  return lines.join("\n");
}

function getSourceImageDims(img) {
  if (!img) return { width: 0, height: 0 };
  return {
    width: Number(img.naturalWidth || img.width || 0),
    height: Number(img.naturalHeight || img.height || 0),
  };
}

function buildCorrectedRaster(img, params) {
  const dims = getSourceImageDims(img);
  if (!img || !dims.width || !dims.height) return null;
  const w = dims.width;
  const h = dims.height;
  const off = document.createElement("canvas");
  off.width = w;
  off.height = h;
  const offCtx = off.getContext("2d");
  offCtx.drawImage(img, 0, 0);
  const src = offCtx.getImageData(0, 0, w, h);
  const correctedCanvas = document.createElement("canvas");
  correctedCanvas.width = w;
  correctedCanvas.height = h;
  const correctedCtx = correctedCanvas.getContext("2d");
  const dst = correctedCtx.createImageData(w, h);
  const srcData = src.data;
  const dstData = dst.data;
  const cx = w * 0.5;
  const cy = h * 0.5;
  const rot = (Number(params?.rotation_deg || 0) * Math.PI) / 180;
  const cos = Math.cos(-rot);
  const sin = Math.sin(-rot);

  function sampleBilinear(x, y, ch) {
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const x1 = x0 + 1;
    const y1 = y0 + 1;
    if (x0 < 0 || y0 < 0 || x1 >= w || y1 >= h) return 0;
    const tx = x - x0;
    const ty = y - y0;
    const i00 = (y0 * w + x0) * 4 + ch;
    const i10 = (y0 * w + x1) * 4 + ch;
    const i01 = (y1 * w + x0) * 4 + ch;
    const i11 = (y1 * w + x1) * 4 + ch;
    const a = srcData[i00] * (1 - tx) + srcData[i10] * tx;
    const b = srcData[i01] * (1 - tx) + srcData[i11] * tx;
    return a * (1 - ty) + b * ty;
  }

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const rx = dx * cos - dy * sin;
      const ry = dx * sin + dy * cos;
      const correctedX = cx + rx;
      const correctedY = cy + ry;
      const [sx, sy] = distortPointForView(correctedX, correctedY, w, h, params);
      const idx = (y * w + x) * 4;
      if (sx < 0 || sy < 0 || sx >= w - 1 || sy >= h - 1) {
        dstData[idx + 0] = 12;
        dstData[idx + 1] = 17;
        dstData[idx + 2] = 23;
        dstData[idx + 3] = 255;
        continue;
      }
      dstData[idx + 0] = sampleBilinear(sx, sy, 0);
      dstData[idx + 1] = sampleBilinear(sx, sy, 1);
      dstData[idx + 2] = sampleBilinear(sx, sy, 2);
      dstData[idx + 3] = 255;
    }
  }
  correctedCtx.putImageData(dst, 0, 0);
  return {
    width: w,
    height: h,
    canvas: correctedCanvas,
    imageData: correctedCtx.getImageData(0, 0, w, h),
  };
}

function renderCorrectedView(canvas, img, params, rawOverlay = {}) {
  const dims = getSourceImageDims(img);
  if (!canvas || !img || !dims.width || !dims.height) return;
  const w = dims.width;
  const h = dims.height;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const raster = buildCorrectedRaster(img, params);
  if (!raster) return;
  const fitToContent = rawOverlay.fitToContent !== false;
  const fitPoints = [
    [0, 0],
    [w, 0],
    [w, h],
    [0, h],
  ];
  for (const seg of rawOverlay.grid || []) {
    if (Array.isArray(seg?.[0])) fitPoints.push(seg[0]);
    if (Array.isArray(seg?.[1])) fitPoints.push(seg[1]);
  }
  for (const seg of rawOverlay.modelSegments || []) {
    if (seg?.role !== "court") continue;
    if (Array.isArray(seg?.a)) fitPoints.push(seg.a);
    if (Array.isArray(seg?.b)) fitPoints.push(seg.b);
  }
  for (const poly of rawOverlay.courtPolylines || []) {
    for (const pt of poly || []) fitPoints.push(pt);
  }
  for (const pt of rawOverlay.netGeometry?.topCurve || []) fitPoints.push(pt);
  for (const pt of rawOverlay.netGeometry?.leftAntenna || []) fitPoints.push(pt);
  for (const pt of rawOverlay.netGeometry?.rightAntenna || []) fitPoints.push(pt);
  for (const pt of rawOverlay.refinedPolyline || []) fitPoints.push(pt);
  if (rawOverlay.ball && Number.isFinite(rawOverlay.ball.center_x) && Number.isFinite(rawOverlay.ball.center_y)) {
    fitPoints.push([rawOverlay.ball.center_x, rawOverlay.ball.center_y]);
  }
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  fitPoints.forEach((p) => {
    if (!Array.isArray(p) || p.length < 2) return;
    minX = Math.min(minX, p[0]);
    minY = Math.min(minY, p[1]);
    maxX = Math.max(maxX, p[0]);
    maxY = Math.max(maxY, p[1]);
  });
  if (!Number.isFinite(minX) || !Number.isFinite(minY) || !Number.isFinite(maxX) || !Number.isFinite(maxY)) {
    minX = 0; minY = 0; maxX = w; maxY = h;
  }
  let viewScale = 1;
  let offsetX = 0;
  let offsetY = 0;
  if (fitToContent) {
    const margin = 18;
    const boundW = Math.max(1, maxX - minX);
    const boundH = Math.max(1, maxY - minY);
    viewScale = Math.min((w - margin * 2) / boundW, (h - margin * 2) / boundH, 1.1);
    offsetX = (w - boundW * viewScale) * 0.5 - minX * viewScale;
    offsetY = (h - boundH * viewScale) * 0.5 - minY * viewScale;
  }
  const presentPoint = (p) => [p[0] * viewScale + offsetX, p[1] * viewScale + offsetY];

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#05080d";
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(raster.canvas, offsetX, offsetY, w * viewScale, h * viewScale);
  ctx.save();
  if (Array.isArray(rawOverlay.grid)) {
    ctx.strokeStyle = "#d4a72c";
    ctx.lineWidth = 1.25;
    for (const seg of rawOverlay.grid) {
      if (!Array.isArray(seg) || seg.length < 2) continue;
      const a = presentPoint(seg[0]);
      const b = presentPoint(seg[1]);
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.stroke();
    }
  }
  if (Array.isArray(rawOverlay.modelSegments)) {
    for (const seg of rawOverlay.modelSegments) {
      if (seg?.role !== "court") continue;
      if (!seg || !Array.isArray(seg.a) || !Array.isArray(seg.b)) continue;
      const a = presentPoint(seg.a);
      const b = presentPoint(seg.b);
      ctx.strokeStyle = "#f85149";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.stroke();
      ctx.fillStyle = ctx.strokeStyle;
      ctx.beginPath();
      ctx.arc(a[0], a[1], 3.5, 0, Math.PI * 2);
      ctx.arc(b[0], b[1], 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  if (Array.isArray(rawOverlay.courtPolylines)) {
    ctx.strokeStyle = "#58a6ff";
    ctx.fillStyle = "#58a6ff";
    ctx.lineWidth = 3;
    for (const poly of rawOverlay.courtPolylines) {
      if (!Array.isArray(poly) || poly.length < 2) continue;
      ctx.beginPath();
      poly.forEach((pt, i) => {
        const p = presentPoint(pt);
        if (i === 0) ctx.moveTo(p[0], p[1]);
        else ctx.lineTo(p[0], p[1]);
      });
      ctx.stroke();
      for (const pt of poly) {
        const p = presentPoint(pt);
        ctx.beginPath();
        ctx.arc(p[0], p[1], 6, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
  if (rawOverlay.netGeometry) {
    ctx.strokeStyle = "#3fb950";
    ctx.lineWidth = 3;
    const drawPair = (pair) => {
      if (!Array.isArray(pair) || pair.length < 2) return;
      const a = presentPoint(pair[0]);
      const b = presentPoint(pair[1]);
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.stroke();
    };
    drawPair(rawOverlay.netGeometry.leftAntenna);
    drawPair(rawOverlay.netGeometry.rightAntenna);
    if (Array.isArray(rawOverlay.netGeometry.topCurve) && rawOverlay.netGeometry.topCurve.length >= 2) {
      ctx.beginPath();
      rawOverlay.netGeometry.topCurve.forEach((pt, i) => {
        const p = presentPoint(pt);
        if (i === 0) ctx.moveTo(p[0], p[1]);
        else ctx.lineTo(p[0], p[1]);
      });
      ctx.stroke();
    }
  }
  if (Array.isArray(rawOverlay.helperCurve) && rawOverlay.helperCurve.length >= 2) {
    ctx.strokeStyle = "#f2cc60";
    ctx.lineWidth = 2.25;
    ctx.setLineDash([10, 8]);
    ctx.beginPath();
    rawOverlay.helperCurve.forEach((pt, i) => {
      const p = presentPoint(pt);
      if (i === 0) ctx.moveTo(p[0], p[1]);
      else ctx.lineTo(p[0], p[1]);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }
  if (rawOverlay.ball && Number.isFinite(rawOverlay.ball.center_x) && Number.isFinite(rawOverlay.ball.center_y)) {
    const p = presentPoint([rawOverlay.ball.center_x, rawOverlay.ball.center_y]);
    const radius = Number(rawOverlay.ball.radius || 0);
    ctx.strokeStyle = "#39d2c0";
    ctx.fillStyle = "#39d2c0";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(p[0], p[1], Math.max(8, radius * viewScale), 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(p[0], p[1], 3.5, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function updateAnalysisViewButtons() {
  const mapping = { raw: btnViewRaw, initial: btnViewInitial, true: btnViewTrue, top: btnViewTop, side: btnViewSide };
  Object.entries(mapping).forEach(([key, btn]) => {
    if (btn) btn.classList.toggle("active", analysisState.view === key);
  });
}

function buildBallRefinementPath(imageData, ball) {
  if (!imageData || !ball || !Number.isFinite(ball.center_x) || !Number.isFinite(ball.center_y)) return [];
  const cropSize = 64;
  const cropX = Math.max(0, Math.min(imageData.width - cropSize, Math.round(ball.center_x - cropSize / 2)));
  const cropY = Math.max(0, Math.min(imageData.height - cropSize, Math.round(ball.center_y - cropSize / 2)));
  const hint = { cx: ball.center_x - cropX, cy: ball.center_y - cropY };
  const yellow = new Float32Array(cropSize * cropSize);
  for (let y = 0; y < cropSize; y += 1) {
    for (let x = 0; x < cropSize; x += 1) {
      const si = ((cropY + y) * imageData.width + (cropX + x)) * 4;
      const r = imageData.data[si + 0];
      const g = imageData.data[si + 1];
      const b = imageData.data[si + 2];
      yellow[y * cropSize + x] = r + g - 1.2 * b;
    }
  }
  const threshold = 310;
  const mask = new Uint8Array(cropSize * cropSize);
  for (let i = 0; i < mask.length; i += 1) if (yellow[i] > threshold) mask[i] = 1;
  const visited = new Uint8Array(cropSize * cropSize);
  const seeds = [];
  for (let dy = -2; dy <= 2; dy += 1) {
    for (let dx = -2; dx <= 2; dx += 1) {
      const x = Math.round(hint.cx + dx);
      const y = Math.round(hint.cy + dy);
      if (x < 0 || y < 0 || x >= cropSize || y >= cropSize) continue;
      const idx = y * cropSize + x;
      if (mask[idx]) seeds.push({ x, y, d2: dx * dx + dy * dy });
    }
  }
  seeds.sort((a, b) => a.d2 - b.d2);
  const blob = [];
  if (seeds.length) {
    const queue = [seeds[0]];
    visited[seeds[0].y * cropSize + seeds[0].x] = 1;
    while (queue.length) {
      const node = queue.pop();
      blob.push(node);
      for (let oy = -1; oy <= 1; oy += 1) {
        for (let ox = -1; ox <= 1; ox += 1) {
          if (!ox && !oy) continue;
          const nx = node.x + ox;
          const ny = node.y + oy;
          if (nx < 0 || ny < 0 || nx >= cropSize || ny >= cropSize) continue;
          const nidx = ny * cropSize + nx;
          if (!mask[nidx] || visited[nidx]) continue;
          visited[nidx] = 1;
          queue.push({ x: nx, y: ny });
        }
      }
    }
  }
  let sumW = 0;
  let sumX = 0;
  let sumY = 0;
  for (const { x, y } of blob) {
    const idx = y * cropSize + x;
    const yv = Math.max(0, yellow[idx] - threshold);
    const dx = x - hint.cx;
    const dy = y - hint.cy;
    const spatial = Math.exp(-(dx * dx + dy * dy) / (2 * 1.8 * 1.8));
    const belowBoost = y >= hint.cy ? 1.25 : 0.85;
    const weight = Math.max(1, yv) * spatial * belowBoost;
    sumW += weight;
    sumX += weight * x;
    sumY += weight * y;
  }
  const areaRadius = blob.length ? Math.sqrt(blob.length / Math.PI) * 1.15 : 2.7;
  const seed = sumW > 1e-6
    ? { cx: sumX / sumW, cy: sumY / sumW, r: Math.max(2.2, Math.min(4.0, areaRadius)) }
    : { cx: hint.cx, cy: hint.cy + 1.0, r: 2.7 };
  const initial = {
    cx: hint.cx,
    cy: hint.cy,
    r: Math.max(2.2, Math.min(4.0, blob.length ? areaRadius : 2.7)),
  };
  let current = { cx: seed.cx, cy: seed.cy, r: seed.r, score: 0, elapsedMs: 0 };
  const score = (cx, cy, r) => {
    let blobHit = 0;
    let blobMiss = 0;
    let insideSupport = 0;
    let asymPenalty = 0;
    let count = 0;
    const rr = r * r;
    const bandIn = Math.max(0, (r - 1.1) * (r - 1.1));
    const bandOut = (r + 1.1) * (r + 1.1);
    for (let y = Math.max(1, Math.floor(cy - r - 2)); y <= Math.min(cropSize - 2, Math.ceil(cy + r + 2)); y += 1) {
      for (let x = Math.max(1, Math.floor(cx - r - 2)); x <= Math.min(cropSize - 2, Math.ceil(cx + r + 2)); x += 1) {
        const dx = x - cx;
        const dy = y - cy;
        const d2 = dx * dx + dy * dy;
        if (d2 > bandOut) continue;
        const idx = y * cropSize + x;
        const yellowVal = yellow[idx];
        const blobPx = visited[idx] ? 1 : 0;
        if (d2 <= rr) {
          insideSupport += yellowVal;
          blobHit += blobPx;
        } else if (d2 >= bandIn) {
          blobMiss += blobPx;
        }
        const sx = Math.round(cx - dx);
        const sy = Math.round(cy - dy);
        if (sx >= 1 && sy >= 1 && sx < cropSize - 1 && sy < cropSize - 1 && d2 <= rr) {
          asymPenalty += Math.abs(yellowVal - yellow[sy * cropSize + sx]);
          count += 1;
        }
      }
    }
    if (!count) return { total: -1e9, insideSupport: 0, blobHit: 0, blobMiss: 0, asymPenalty: 0, radiusPenalty: 0, hintPenalty: 0, downPenalty: 0 };
    const hintPenalty = Math.hypot(cx - hint.cx, cy - hint.cy);
    const downPenalty = Math.max(0, hint.cy - cy) * 4.0;
    const radiusPenalty = Math.abs(r - areaRadius) * 4.0;
    return {
      insideSupport: insideSupport * 0.0025,
      blobHit: blobHit * 9.0,
      blobMiss: blobMiss * 5.0,
      asymPenalty: (asymPenalty / count) * 0.18,
      radiusPenalty,
      hintPenalty: hintPenalty * 11.0,
      downPenalty,
      total: insideSupport * 0.0025 + blobHit * 9.0 - blobMiss * 5.0 - (asymPenalty / count) * 0.18 - radiusPenalty - hintPenalty * 11.0 - downPenalty,
    };
  };
  current.diagnostics = score(current.cx, current.cy, current.r);
  current.score = current.diagnostics.total;
  const path = [{
    ...initial,
    diagnostics: score(initial.cx, initial.cy, initial.r),
    score: score(initial.cx, initial.cy, initial.r).total,
    elapsedMs: 0,
    center_x: cropX + initial.cx,
    center_y: cropY + initial.cy,
    radius: initial.r,
  }];
  path.push({ ...current, center_x: cropX + current.cx, center_y: cropY + current.cy, radius: current.r });
  const best = { ...current, elapsedMs: 3.5 };
  const finalPath = [];
  const finalSteps = 20;
  for (let i = 0; i < finalSteps; i += 1) {
    const t = finalSteps <= 1 ? 1 : i / (finalSteps - 1);
    finalPath.push({
      cx: lerp(initial.cx, best.cx, t),
      cy: lerp(initial.cy, best.cy, t),
      r: lerp(initial.r, best.r, t),
      score: lerp(path[0].score, best.score, t),
      diagnostics: {
        insideSupport: lerp(path[0].diagnostics.insideSupport, best.diagnostics.insideSupport, t),
        blobHit: lerp(path[0].diagnostics.blobHit, best.diagnostics.blobHit, t),
        blobMiss: lerp(path[0].diagnostics.blobMiss, best.diagnostics.blobMiss, t),
        asymPenalty: lerp(path[0].diagnostics.asymPenalty, best.diagnostics.asymPenalty, t),
        radiusPenalty: lerp(path[0].diagnostics.radiusPenalty, best.diagnostics.radiusPenalty, t),
        hintPenalty: lerp(path[0].diagnostics.hintPenalty, best.diagnostics.hintPenalty, t),
        downPenalty: lerp(path[0].diagnostics.downPenalty, best.diagnostics.downPenalty, t),
        total: lerp(path[0].diagnostics.total, best.diagnostics.total, t),
      },
      elapsedMs: lerp(0, Number(best.elapsedMs || 0), t),
      center_x: cropX + lerp(initial.cx, best.cx, t),
      center_y: cropY + lerp(initial.cy, best.cy, t),
      radius: lerp(initial.r, best.r, t),
    });
  }
  return finalPath;
}

function buildNetRefinementGeometry(netPoints) {
  if (!Array.isArray(netPoints) || netPoints.length < 5) return null;
  const leftTop = [Number(netPoints[0][0]), Number(netPoints[0][1])];
  const leftBase = [Number(netPoints[1][0]), Number(netPoints[1][1])];
  const mid = [Number(netPoints[2][0]), Number(netPoints[2][1])];
  const rightBase = [Number(netPoints[3][0]), Number(netPoints[3][1])];
  const rightTop = [Number(netPoints[4][0]), Number(netPoints[4][1])];
  const curveAnchors = [leftBase, mid, rightBase];
  const ax = curveAnchors[0][0], ay = curveAnchors[0][1];
  const bx = curveAnchors[1][0], by = curveAnchors[1][1];
  const cx = curveAnchors[2][0], cy = curveAnchors[2][1];
  const denom = (ax - bx) * (ax - cx) * (bx - cx);
  let curve = [];
  if (Math.abs(denom) > 1e-6) {
    const A = (cx * (by - ay) + bx * (ay - cy) + ax * (cy - by)) / denom;
    const B = (cx * cx * (ay - by) + bx * bx * (cy - ay) + ax * ax * (by - cy)) / denom;
    const C = (bx * cx * (bx - cx) * ay + cx * ax * (cx - ax) * by + ax * bx * (ax - bx) * cy) / denom;
    const x0 = Math.min(ax, cx);
    const x1 = Math.max(ax, cx);
    for (let i = 0; i <= 48; i += 1) {
      const t = i / 48;
      const x = x0 + (x1 - x0) * t;
      const y = A * x * x + B * x + C;
      curve.push([x, y]);
    }
  } else {
    curve = [leftBase, mid, rightBase];
  }
  return {
    leftAntenna: [leftTop, leftBase],
    rightAntenna: [rightTop, rightBase],
    topCurve: curve,
  };
}

function buildSummaryChips(data, fittedCorrection) {
  const host = document.getElementById("analysisSummary");
  if (!host) return;
  host.innerHTML = "";
  const chips = [];
  const ballWorld = data?.label_analysis?.ball_world;
  if (fittedCorrection?.diagnostics) chips.push(`helper bend ${fittedCorrection.diagnostics.before_max_abs_deviation.toFixed(1)} -> ${fittedCorrection.diagnostics.after_max_abs_deviation.toFixed(1)}`);
  if (fittedCorrection && Number.isFinite(fittedCorrection.rotation_deg)) chips.push(`helper rot ${fittedCorrection.rotation_deg.toFixed(2)}deg`);
  if (ballWorld && Number.isFinite(ballWorld.z_m)) chips.push(`ball ${ballWorld.z_m.toFixed(2)}m high`);
  for (const text of chips) {
    const chip = document.createElement("span");
    chip.className = "analysis-chip";
    chip.textContent = text;
    host.appendChild(chip);
  }
}

function drawTopProjection(canvas, projectionStep = null, ballWorld = null) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0d1117";
  ctx.fillRect(0, 0, w, h);
  const pad = 50;
  const scale = Math.min((w - pad * 2) / 8, (h - pad * 2) / 16);
  const courtW = 8 * scale;
  const courtH = 16 * scale;
  const originX = (w - courtW) / 2;
  const originY = (h - courtH) / 2;
  const topView = projectionStep?.world_views?.top || null;
  ctx.strokeStyle = "#d4a72c";
  ctx.lineWidth = 3;
  ctx.strokeRect(originX, originY, courtW, courtH);
  const midline = topView?.midline || [[0, 8], [8, 8]];
  ctx.beginPath();
  ctx.moveTo(originX + midline[0][0] * scale, originY + midline[0][1] * scale);
  ctx.lineTo(originX + midline[1][0] * scale, originY + midline[1][1] * scale);
  ctx.stroke();
  const b = (ballWorld && Number.isFinite(ballWorld.x_m) && Number.isFinite(ballWorld.y_m))
    ? [ballWorld.x_m, ballWorld.y_m]
    : topView?.ball;
  if (b && Number.isFinite(b[0]) && Number.isFinite(b[1])) {
    const x = originX + Math.max(0, Math.min(8, b[0])) * scale;
    const y = originY + Math.max(0, Math.min(16, b[1])) * scale;
    ctx.fillStyle = "#39d2c0";
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, Math.PI * 2);
    ctx.fill();
  }
  const cam = topView?.camera || null;
  if (cam && Number.isFinite(cam[0]) && Number.isFinite(cam[1])) {
    const camX = originX + Math.max(-4, Math.min(12, Number(cam[0]))) * scale;
    const camY = originY + Math.max(-4, Math.min(20, Number(cam[1]))) * scale;
    ctx.fillStyle = "#f85149";
    ctx.beginPath();
    ctx.arc(camX, camY, 7, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawSideProjection(canvas, projectionStep = null, ballWorld = null) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0d1117";
  ctx.fillRect(0, 0, w, h);
  const padX = 50, padY = 30;
  const scale = Math.min((w - padX * 2) / 16, (h - padY * 2) / 12);
  const courtL = 16 * scale;
  const originX = (w - courtL) / 2;
  const groundY = h - padY;
  const sideView = projectionStep?.world_views?.side || null;
  ctx.strokeStyle = "#d4a72c";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(originX, groundY);
  ctx.lineTo(originX + courtL, groundY);
  ctx.stroke();
  const net = sideView?.net || [[8, 0], [8, 2.43]];
  const netX = originX + Number(net[0][0] || 8) * scale;
  ctx.beginPath();
  ctx.moveTo(netX, groundY);
  ctx.lineTo(netX, groundY - Number(net[1][1] || 2.43) * scale);
  ctx.stroke();
  const b = (ballWorld && Number.isFinite(ballWorld.y_m) && Number.isFinite(ballWorld.z_m))
    ? [ballWorld.y_m, ballWorld.z_m]
    : sideView?.ball;
  if (b && Number.isFinite(b[0]) && Number.isFinite(b[1])) {
    const x = originX + Math.max(0, Math.min(16, b[0])) * scale;
    const y = groundY - Math.max(0, Math.min(12, b[1])) * scale;
    ctx.fillStyle = "#39d2c0";
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, Math.PI * 2);
    ctx.fill();
  }
  const cam = sideView?.camera || null;
  if (cam && Number.isFinite(cam[0]) && Number.isFinite(cam[1])) {
    const camX = originX + Math.max(-4, Math.min(20, Number(cam[0]))) * scale;
    const camY = groundY - Math.max(0, Math.min(12, Number(cam[1]))) * scale;
    ctx.fillStyle = "#f85149";
    ctx.beginPath();
    ctx.arc(camX, camY, 7, 0, Math.PI * 2);
    ctx.fill();
  }
}

function updateDiagnosticsPanel(lineStep, ballStep) {
  const host = document.getElementById("diagnosticsPanel");
  if (!host) return;
  const lines = [];
  if (lineStep?.diagnostics) {
    const d = lineStep.diagnostics;
    lines.push("Line objective");
    lines.push(`total: ${Number(d.total).toFixed(2)}`);
    lines.push(`support: +${Number(d.support).toFixed(2)}`);
    lines.push(`offset penalty: -${Number(d.offsetPenalty).toFixed(2)}`);
    lines.push(`curvature penalty: -${Number(d.curvaturePenalty).toFixed(2)}`);
    lines.push(`stripe penalty: -${Number(d.stripePenalty).toFixed(2)}`);
  }
  if (ballStep?.diagnostics) {
    const d = ballStep.diagnostics;
    if (lines.length) lines.push("");
    lines.push("Ball objective");
    lines.push(`total: ${Number(d.total).toFixed(2)}`);
    lines.push(`inside support: +${Number(d.insideSupport).toFixed(2)}`);
    lines.push(`blob hit: +${Number(d.blobHit).toFixed(2)}`);
    lines.push(`blob miss: -${Number(d.blobMiss).toFixed(2)}`);
    lines.push(`asym penalty: -${Number(d.asymPenalty).toFixed(2)}`);
    lines.push(`radius penalty: -${Number(d.radiusPenalty).toFixed(2)}`);
    lines.push(`hint penalty: -${Number(d.hintPenalty).toFixed(2)}`);
    lines.push(`upward penalty: -${Number(d.downPenalty).toFixed(2)}`);
  }
  host.textContent = lines.join("\n");
}

function renderAnalysisView() {
  if (!analysisState.data || !analysisState.imgReady) return;
  const canvas = document.getElementById("testViewCanvas");
  const ctx = canvas.getContext("2d");
  const img = analysisState.imageBitmap;
  if (!img) return;
  const dims = getSourceImageDims(img);
  if (!dims.width || !dims.height) return;
  canvas.width = dims.width;
  canvas.height = dims.height;
  if (analysisState.view === "raw") {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, dims.width, dims.height);
    document.getElementById("metricView").textContent = "Raw";
    return;
  }
  try {
    const rawCourtPolylines = (analysisState.data.label_record && analysisState.data.label_record.geometry_polylines) || [];
    const initialParams = analysisState.initialCorrection
      || normalizeViewCorrection(analysisState.data?.label_record?.distortion_params)
      || { lambda: 0, cx_offset: 0, cy_offset: 0, rotation_deg: 0 };
    const linePath = analysisState.lineRefinementPath || [];
    const lineEntry = linePath[Math.min(Number(lineRefineSlider?.value || 0), Math.max(0, linePath.length - 1))] || null;
    const lineStep = lineEntry?.points || rawCourtPolylines[0] || [];
    const modelFitPath = analysisState.modelFitPath || [];
    const modelIndex = Math.min(Number(cameraFitSlider?.value || 0), Math.max(0, modelFitPath.length - 1));
    const projectionArtifact = analysisState.data?.label_analysis?.projection_artifact || null;
    const modelStep = modelFitPath[modelIndex]
      || projectionArtifact?.refined
      || null;
    const ballPath = analysisState.ballPath || [];
    const ballStep = ballPath[Math.min(Number(ballRefineSlider?.value || 0), Math.max(0, ballPath.length - 1))]
      || (analysisState.data.label_analysis && analysisState.data.label_analysis.ball);
    const correctionParams = initialParams;
    const activeCameraModel = getProjectionStepCameraModel(modelStep)
      || projectionArtifact?.seed?.camera_model
      || analysisState.data?.label_analysis?.camera_model_seed
      || analysisState.data?.label_analysis?.camera_model
      || null;
    const currentBallWorld = estimateBallWorldFromCamera(ballStep, activeCameraModel) || analysisState.data?.label_analysis?.ball_world || null;
    const netGeometry = buildNetRefinementGeometry((analysisState.data.label_record && analysisState.data.label_record.net_points) || []);
    const modelSegments = modelStep?.model_segments || projectionArtifact?.seed?.model_segments || [];
    if (analysisState.view === "initial" || analysisState.view === "true") {
      renderCorrectedView(canvas, img, correctionParams, {
        fitToContent: analysisState.view === "true",
        grid: analysisState.view === "true"
          ? (modelStep?.court_grid || projectionArtifact?.refined?.court_grid || analysisState.data.label_analysis?.court_grid || [])
          : [],
        courtPolylines: (lineStep.length >= 2 ? [lineStep] : rawCourtPolylines),
        modelSegments,
        netGeometry,
        ball: ballStep || null,
      });
    } else if (analysisState.view === "top") {
      drawTopProjection(canvas, modelStep || projectionArtifact?.refined || null, currentBallWorld);
    } else if (analysisState.view === "side") {
      drawSideProjection(canvas, modelStep || projectionArtifact?.refined || null, currentBallWorld);
    }
    document.getElementById("metricBallHeight").textContent = Number.isFinite(currentBallWorld?.z_m) ? `${currentBallWorld.z_m.toFixed(2)} m` : "-";
    document.getElementById("metricBallDepth").textContent = Number.isFinite(currentBallWorld?.depth_m) ? `${currentBallWorld.depth_m.toFixed(2)} m` : "-";
    document.getElementById("metricLineBend").textContent = analysisState.trueCorrection?.diagnostics ? `${analysisState.trueCorrection.diagnostics.after_max_abs_deviation.toFixed(1)} px` : "model";
    document.getElementById("lineRefineTime").textContent = `${(Math.min(Number(lineRefineSlider?.value || 0), Math.max(0, linePath.length - 1)) / Math.max(1, linePath.length - 1) * 8).toFixed(2)} ms`;
    document.getElementById("cameraFitTime").textContent = `${(Math.min(Number(cameraFitSlider?.value || 0), Math.max(0, modelFitPath.length - 1)) / Math.max(1, modelFitPath.length - 1) * 16).toFixed(2)} ms`;
    document.getElementById("ballRefineTime").textContent = `${Number(ballStep?.elapsedMs || 0).toFixed(2)} ms`;
    updateDiagnosticsPanel(lineEntry, ballStep);
  } catch (err) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, dims.width, dims.height);
    const testInfo = document.getElementById("testInfo");
    if (testInfo) testInfo.textContent = `Analysis draw fallback: ${String(err && err.message ? err.message : err)}`;
  }
  document.getElementById("metricView").textContent = ({ raw: "Raw", initial: "Initial", true: "Pinhole", top: "Top", side: "Side" })[analysisState.view] || "Raw";
}

function showAnalysisResult(data, includeNN) {
  document.getElementById("testCard").style.display = "";
  analysisState.data = data;
  analysisState.includeNN = includeNN;
  analysisState.view = "raw";
  analysisState.imgReady = false;
  analysisState.imageBitmap = null;
  analysisState.initialCorrection = null;
  analysisState.trueCorrection = null;
  analysisState.trueCorrectionPath = [];
  analysisState.modelFitPath = [];
  analysisState.lineRefinementPath = [];
  const finishWithHtmlImage = (url) => {
    const imgEl = new Image();
    imgEl.onload = () => finishWithImage(imgEl);
    imgEl.onerror = () => {
      analysisState.imgReady = false;
      document.getElementById("testInfo").textContent = "Image load failed in browser fallback path.";
      const canvas = document.getElementById("testViewCanvas");
      const ctx = canvas.getContext("2d");
      canvas.width = 960;
      canvas.height = 540;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#05080d";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    };
    imgEl.src = `${url}${url.includes("?") ? "&" : "?"}t=${Date.now()}`;
  };
  const finishWithImage = (img) => {
    const dims = getSourceImageDims(img);
    if (!dims.width || !dims.height) {
      analysisState.imgReady = false;
      document.getElementById("testInfo").textContent = "Loaded image has no pixel dimensions.";
      return;
    }
    analysisState.imageBitmap = img;
    analysisState.imgReady = true;
    analysisState.initialCorrection = buildInitialCorrectionFromRecord(data.label_record, dims.width, dims.height);
    analysisState.trueCorrection = buildTrueCorrectionFromAnalysis(data, dims.width, dims.height);
    analysisState.trueCorrectionPath = analysisState.trueCorrection?.path || [];
    analysisState.modelFitPath = buildProjectionPathFromAnalysis(data?.label_analysis || null);
    analysisState.fittedCorrection = analysisState.initialCorrection;
    analysisState.fittedCorrectionPath = analysisState.fittedCorrection?.path || [];
    const correctedRaster = buildCorrectedRaster(img, analysisState.initialCorrection || { lambda: 0, cx_offset: 0, cy_offset: 0, rotation_deg: 0 });
    analysisState.imageData = correctedRaster?.imageData || null;
    analysisState.lineRefinementPath = buildLineRefinementPath(analysisState.imageData, ((data.label_record && data.label_record.geometry_polylines) || [])[0] || null);
    analysisState.ballPath = buildBallRefinementPath(analysisState.imageData, data?.label_analysis?.ball || null);
    if (lineRefineSlider) {
      const sliderPath = analysisState.lineRefinementPath;
      lineRefineSlider.max = String(Math.max(0, sliderPath.length - 1));
      lineRefineSlider.value = "0";
      lineRefineSlider.disabled = sliderPath.length <= 1;
    }
    if (cameraFitSlider) {
      const sliderPath = analysisState.modelFitPath;
      cameraFitSlider.max = String(Math.max(0, sliderPath.length - 1));
      cameraFitSlider.value = "0";
      cameraFitSlider.disabled = sliderPath.length <= 1;
    }
    if (ballRefineSlider) {
      ballRefineSlider.max = String(Math.max(0, analysisState.ballPath.length - 1));
      ballRefineSlider.value = "0";
      ballRefineSlider.disabled = analysisState.ballPath.length <= 1;
    }
    buildSummaryChips(data, analysisState.fittedCorrection);
    updateAnalysisViewButtons();
    renderAnalysisView();
  };
  fetch(data.image_url, { cache: "no-store" })
    .then((res) => {
      if (!res.ok) throw new Error(`Image load failed: HTTP ${res.status}`);
      return res.blob();
    })
    .then((blob) => {
      if (typeof createImageBitmap === "function") return createImageBitmap(blob);
      throw new Error("createImageBitmap unavailable");
    })
    .then((bitmap) => finishWithImage(bitmap))
    .catch((err) => {
      document.getElementById("testInfo").textContent = `Bitmap path failed, using browser image fallback: ${String(err.message || err)}`;
      finishWithHtmlImage(data.image_url);
    });
  const labelBallWorld = data?.label_analysis?.ball_world;
  const labelCam = data?.label_analysis?.camera_model || {};
  const labelErr = Number(labelCam.reprojection_error_px || 0).toFixed(2);
  const parts = [`${data.image}`, `label reproj ${labelErr}px`];
  if (labelBallWorld && Number.isFinite(labelBallWorld.z_m)) {
    parts.push(`ball z ${labelBallWorld.z_m.toFixed(2)}m`);
    parts.push(`depth ${labelBallWorld.depth_m.toFixed(2)}m`);
  }
  document.getElementById("testStatus").textContent = parts.join(" — ");
  document.getElementById("testInfo").textContent = data?.label_record?.distortion_helper_points?.length
    ? "Initial pinhole-like replays the saved corrected labeling view. Camera model fit uses the refined blue lines as its target."
    : "Initial pinhole-like replays the saved corrected labeling view. Camera model fit uses the refined blue lines as its target.";
}

const DET_COLORS = { ball_in_play: "#39d2c0", ball_other: "#8b949e", player: "#a371f7", referee: "#d4a72c", other_person: "#6e7681" };
function drawDetections(ctx, data) {
  for (const det of data.detections) {
    const [x1, y1, x2, y2] = det.box;
    const col = DET_COLORS[det.label] || "#58a6ff";
    ctx.strokeStyle = col; ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.font = "bold 13px sans-serif"; ctx.fillStyle = col;
    ctx.fillText(`${det.label} ${(det.score * 100).toFixed(0)}%`, x1, y1 - 4);
  }
}

function drawBall(ctx, data) {
  if (!data.ball || !data.ball.visible) return;
  const cx = data.ball.center_x;
  const cy = data.ball.center_y;
  let radiusPx = 0;
  if (Number.isFinite(data.ball.radius) && data.ball.radius > 0) {
    radiusPx = data.ball.radius;
  } else if (Number.isFinite(data.ball.radius_prior) && data.ball.radius_prior > 0) {
    const prior = data.ball.radius_prior;
    radiusPx = prior <= 0.25 ? prior * Math.max(ctx.canvas.width, ctx.canvas.height) : prior;
  }
  if (!(radiusPx > 0)) return;
  ctx.strokeStyle = "#39d2c0";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.arc(cx, cy, radiusPx, 0, Math.PI * 2);
  ctx.stroke();
}

/* ---------- Dataset counts + progress ---------- */
let datasetCounts = null;
let trainImagesProcessed = 0;
let trainDatasetSize = 0;
let activeTrainTargets = new Set();

function loadDatasetCounts() {
  fetch("/api/train/dataset-counts", { cache: "no-store" })
    .then((r) => r.json())
    .then((c) => { datasetCounts = c; renderCounts(); })
    .catch(() => {});
}

function renderCounts() {
  if (!datasetCounts) return;
  document.querySelectorAll(".target-count").forEach((el) => {
    const key = el.dataset.key;
    const total = datasetCounts[key] || 0;
    const epochSuffix =
      totalEpochs > 0 ? ` | E${currentEpoch}/${totalEpochs}` : "";
    if (activeTrainTargets.has(key) && currentState === "running") {
      const done = Math.min(trainImagesProcessed, trainDatasetSize || total);
      el.innerHTML = `${done} <span class="label-text">/ ${trainDatasetSize || total}${epochSuffix}</span>`;
    } else if (activeTrainTargets.has(key) && totalEpochs > 0) {
      el.innerHTML = `<span class="label-text">${total} labeled${epochSuffix}</span>`;
    } else {
      el.innerHTML = `<span class="label-text">${total} labeled${epochSuffix}</span>`;
    }
  });
}

function resetCounts() {
  trainImagesProcessed = 0;
  trainDatasetSize = 0;
  activeTrainTargets.clear();
  currentEpoch = 0;
  totalEpochs = 0;
  renderCounts();
}

function smoothSeries(values, windowSize) {
  if (windowSize <= 1) return values.slice();
  const out = new Array(values.length);
  const half = Math.floor(windowSize / 2);
  for (let i = 0; i < values.length; i++) {
    const a = Math.max(0, i - half);
    const b = Math.min(values.length - 1, i + half);
    let sum = 0;
    let n = 0;
    for (let j = a; j <= b; j++) {
      sum += values[j];
      n += 1;
    }
    out[i] = n ? sum / n : values[i];
  }
  return out;
}

/* ---------- Chart ---------- */
function drawChart() {
  const dpr = window.devicePixelRatio || 1;
  const rect = chartCanvas.getBoundingClientRect();
  chartCanvas.width = rect.width * dpr;
  chartCanvas.height = rect.height * dpr;
  chartCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = rect.width, H = rect.height;
  const PAD = { top: 20, right: 20, bottom: 30, left: 55 };
  const cw = W - PAD.left - PAD.right, ch = H - PAD.top - PAD.bottom;

  chartCtx.clearRect(0, 0, W, H);
  chartCtx.fillStyle = "#0d1117";
  chartCtx.fillRect(0, 0, W, H);

  if (lossData.length === 0) {
    chartCtx.fillStyle = "#484f58";
    chartCtx.font = "14px sans-serif";
    chartCtx.textAlign = "center";
    const line1 =
      currentState === "running"
        ? "Preparing model and first batch…"
        : "Start training or load history to plot loss.";
    const line2 =
      currentState === "running"
        ? "This should reach the first real loss point much sooner than the old baseline-first flow."
        : "";
    chartCtx.fillText(line1, W / 2, H / 2 - (line2 ? 10 : 0));
    if (line2) {
      chartCtx.font = "12px sans-serif";
      chartCtx.fillStyle = "#6e7681";
      chartCtx.fillText(line2, W / 2, H / 2 + 12);
    }
    return;
  }

  const activeTargets = getTargets();
  const lossKeys = activeTargets.map(lossKeyForTarget);

  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (const d of lossData) {
    for (const k of lossKeys) {
      const v = d[k];
      if (!Number.isFinite(v)) continue;
      if (v < minY) minY = v;
      if (v > maxY) maxY = v;
    }
  }
  if (!Number.isFinite(minY) || !Number.isFinite(maxY)) {
    minY = 0;
    maxY = 0.01;
  }
  const spread = Math.max(maxY - minY, Math.max(maxY, 0.01) * 0.08, 0.0025);
  const padY = spread * 0.18;
  const yMin = Math.max(0, minY - padY);
  const yMax = maxY + padY;

  const xScale = lossData.length > 1 ? cw / (lossData.length - 1) : 0;
  const yScale = ch / Math.max(yMax - yMin, 1e-6);

  // Grid
  chartCtx.strokeStyle = "#21262d"; chartCtx.lineWidth = 1;
  const nGrid = 5;
  for (let i = 0; i <= nGrid; i++) {
    const y = PAD.top + (ch / nGrid) * i;
    chartCtx.beginPath(); chartCtx.moveTo(PAD.left, y); chartCtx.lineTo(PAD.left + cw, y); chartCtx.stroke();
    chartCtx.fillStyle = "#484f58"; chartCtx.font = "11px sans-serif"; chartCtx.textAlign = "right";
    const value = yMax - ((yMax - yMin) / nGrid) * i;
    chartCtx.fillText(value.toFixed(3), PAD.left - 6, y + 4);
  }
  const xGrid = Math.min(10, Math.max(2, lossData.length - 1));
  for (let i = 0; i <= xGrid; i++) {
    const x = PAD.left + (cw / xGrid) * i;
    chartCtx.beginPath();
    chartCtx.moveTo(x, PAD.top);
    chartCtx.lineTo(x, PAD.top + ch);
    chartCtx.stroke();
  }

  // Epoch markers
  for (const ep of epochData) {
    const stepIdx = lossData.findIndex((d) => d.epoch === ep.epoch);
    if (stepIdx < 0) continue;
    const x = PAD.left + stepIdx * xScale;
    chartCtx.strokeStyle = "#4b5563"; chartCtx.beginPath();
    chartCtx.moveTo(x, PAD.top); chartCtx.lineTo(x, PAD.top + ch); chartCtx.stroke();
    chartCtx.fillStyle = "#484f58"; chartCtx.font = "10px sans-serif"; chartCtx.textAlign = "center";
    chartCtx.fillText(`E${ep.epoch}`, x, PAD.top + ch + 14);
  }

  // One line per active target
  for (const target of activeTargets) {
    const k = lossKeyForTarget(target);
    const vals = lossData.map((d) => d[k] || 0);
    let allZero = true;
    for (const v of vals) if (v > 0.0001) { allZero = false; break; }
    if (allZero) continue;

    const col = TARGET_COLORS[target];
    const plotted = smoothSeries(vals, smoothingWindow);

    // Lines
    if (plotted.length > 1) {
      chartCtx.strokeStyle = col; chartCtx.lineWidth = 2; chartCtx.beginPath();
      for (let i = 0; i < plotted.length; i++) {
        const x = PAD.left + i * xScale, y = PAD.top + ch - (plotted[i] - yMin) * yScale;
        if (i === 0) chartCtx.moveTo(x, y); else chartCtx.lineTo(x, y);
      }
      chartCtx.stroke();
    }

    // Dots
    const dotR = Math.max(2, Math.min(DOT_RADIUS + 1, 200 / Math.max(vals.length, 1)));
    chartCtx.fillStyle = col;
    for (let i = 0; i < plotted.length; i++) {
      const x = PAD.left + i * xScale, y = PAD.top + ch - (plotted[i] - yMin) * yScale;
      chartCtx.beginPath(); chartCtx.arc(x, y, dotR, 0, Math.PI * 2); chartCtx.fill();
    }
  }
}

/* ---------- SSE ---------- */
function connectSSE() {
  if (evtSource) evtSource.close();
  evtSource = new EventSource("/api/train/stream");
  evtSource.onmessage = (e) => { try { handleEvent(JSON.parse(e.data)); } catch {} };
  evtSource.onerror = () => { setTimeout(connectSSE, 3000); };
}

function handleEvent(d) {
  if (d.type === "state" || (d.state && typeof d.state === "string")) setState(d.state || d.type);
  if (d.type === "info") {
    statusBar.textContent = d.message;
    if (d.targets) d.targets.forEach((t) => activeTrainTargets.add(t));
    if (d.dataset_size) trainDatasetSize = d.dataset_size;
    if (Number.isFinite(d.total_epochs)) setEpochProgress(currentEpoch, d.total_epochs);
    drawChart();
  }
  if (d.type === "loss" || d.type === "baseline") {
    lossData.push(d);
    trainImagesProcessed = d.images_processed || 0;
    if (d.dataset_size) trainDatasetSize = d.dataset_size;
    if (Number.isFinite(d.epoch)) setEpochProgress(d.epoch, totalEpochs);
    renderCounts();
    const prefix = d.type === "baseline" ? "Baseline" : `Step ${d.step} | Epoch ${d.epoch} | ${d.images_processed}/${d.dataset_size} images`;
    const parts = [];
    if (d.loss_court > 0) parts.push(`court=${d.loss_court.toFixed(4)}`);
    if (d.loss_net > 0) parts.push(`net=${d.loss_net.toFixed(4)}`);
    if (d.loss_people > 0) parts.push(`people=${d.loss_people.toFixed(4)}`);
    if (d.loss_ball > 0) parts.push(`ball=${d.loss_ball.toFixed(4)}`);
    statusBar.textContent = `${prefix} | ${parts.join(" ")}`;
    drawChart();
  }
  if (d.type === "epoch") {
    epochData.push(d);
    setEpochProgress(d.epoch, d.total_epochs);
    drawChart();
    statusBar.textContent = `Epoch ${d.epoch}/${d.total_epochs} complete`;
  }
  if (d.type === "error") statusBar.textContent = `Error: ${d.message}`;
  if (d.type === "done") { statusBar.textContent = `Training complete (${d.epochs} epochs)`; setState("idle"); renderCounts(); }
  if (d.type === "reset") { lossData = []; epochData = []; resetCounts(); drawChart(); }
}

/* ---------- Init ---------- */
async function loadExisting() {
  statusBar.textContent = "Loading history…";
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), 120000);
  try {
    const res = await fetch("/api/train/losses", { cache: "no-store", signal: ctrl.signal });
    clearTimeout(t);
    const data = await res.json();
    if (!res.ok) {
      const detail = data.detail != null
        ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail))
        : (data.error || `HTTP ${res.status}`);
      statusBar.textContent = `Could not load history: ${detail}`;
      drawChart();
      return;
    }
    if (data.losses) {
      lossData = data.losses;
      if (lossData.length > 0) {
        const last = lossData[lossData.length - 1];
        trainImagesProcessed = last.images_processed || 0;
        trainDatasetSize = last.dataset_size || 0;
      }
    }
    if (data.epochs) epochData = data.epochs;
    if (data.targets) data.targets.forEach((t) => activeTrainTargets.add(t));
    drawChart();
    renderCounts();
    statusBar.textContent = "";
  } catch (e) {
    clearTimeout(t);
    statusBar.textContent = e.name === "AbortError"
      ? "Timed out loading training history — server may still be loading PyTorch; retry refresh."
      : `Could not load history: ${e.message || e}`;
    drawChart();
  }
  try {
    const res = await fetch("/api/train/status", { cache: "no-store" });
    const data = await res.json();
    if (data.state) setState(data.state);
    setEpochProgress(data.epoch || 0, data.total_epochs || 0);
  } catch {}
}

function initTrainPage() {
  restoreTargets();
  if (smoothRange) {
    smoothRange.addEventListener("input", () => {
      smoothingWindow = parseInt(smoothRange.value, 10) || 1;
      if (smoothValue) smoothValue.textContent = smoothingWindow === 1 ? "1 (off)" : `${smoothingWindow}x`;
      drawChart();
    });
    smoothingWindow = parseInt(smoothRange.value, 10) || 1;
    if (smoothValue) smoothValue.textContent = smoothingWindow === 1 ? "1 (off)" : `${smoothingWindow}x`;
  }
  drawChart();
  setState("idle");
  loadDatasetCounts();
  loadLabeledImages();
  loadExisting().finally(() => {
    connectSSE();
  });
  window.addEventListener("resize", drawChart);
}

initTrainPage();
