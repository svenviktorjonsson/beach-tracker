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
const smoothRange = document.getElementById("smoothRange");
const smoothValue = document.getElementById("smoothValue");

let lossData = [];
let epochData = [];
let currentState = "idle";
let evtSource = null;
let currentEpoch = 0;
let totalEpochs = 0;
let smoothingWindow = 1;

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
    image_size: parseInt(document.getElementById("inpImgSize").value) || 384,
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
    testInfo.textContent = `${nDet} det, ${nCourt} court pts, ${nNet} net pts — ${data.image}`;
  } catch (e) {
    testInfo.textContent = String(e.message || e);
  }
});

/* ---------- Test overlay ---------- */
function showTestResult(data) {
  document.getElementById("testCard").style.display = "";
  const img = document.getElementById("testImg");
  const cv = document.getElementById("testCanvas");
  img.onload = () => {
    cv.width = img.naturalWidth;
    cv.height = img.naturalHeight;
    const ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, cv.width, cv.height);
    drawGeometry(ctx, data);
    drawDetections(ctx, data);
  };
  img.src = `/api/image?file=${encodeURIComponent(data.image)}`;
  const nCourt = (data.court_points || []).length;
  const nNet = (data.net_points || []).length;
  document.getElementById("testStatus").textContent =
    `${data.image} — ${data.detections.length} det, ${nCourt} court pts, ${nNet} net pts`;
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
  drawPolyline(data.court_points, "#58a6ff", true);
  drawPolyline(data.net_points, "#3fb950", false);
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
  chartCtx.scale(dpr, dpr);
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

  let maxY = 0.01;
  for (const d of lossData) for (const k of lossKeys) { const v = d[k] || 0; if (v > maxY) maxY = v; }
  maxY *= 1.1;

  const xScale = lossData.length > 1 ? cw / (lossData.length - 1) : 0;
  const yScale = ch / maxY;

  // Grid
  chartCtx.strokeStyle = "#21262d"; chartCtx.lineWidth = 1;
  const nGrid = 5;
  for (let i = 0; i <= nGrid; i++) {
    const y = PAD.top + (ch / nGrid) * i;
    chartCtx.beginPath(); chartCtx.moveTo(PAD.left, y); chartCtx.lineTo(PAD.left + cw, y); chartCtx.stroke();
    chartCtx.fillStyle = "#484f58"; chartCtx.font = "11px sans-serif"; chartCtx.textAlign = "right";
    chartCtx.fillText((maxY - (maxY / nGrid) * i).toFixed(3), PAD.left - 6, y + 4);
  }

  // Epoch markers
  for (const ep of epochData) {
    const stepIdx = lossData.findIndex((d) => d.epoch === ep.epoch);
    if (stepIdx < 0) continue;
    const x = PAD.left + stepIdx * xScale;
    chartCtx.strokeStyle = "#30363d"; chartCtx.beginPath();
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
        const x = PAD.left + i * xScale, y = PAD.top + ch - plotted[i] * yScale;
        if (i === 0) chartCtx.moveTo(x, y); else chartCtx.lineTo(x, y);
      }
      chartCtx.stroke();
    }

    // Dots
    const dotR = Math.max(2, Math.min(DOT_RADIUS + 1, 200 / Math.max(vals.length, 1)));
    chartCtx.fillStyle = col;
    for (let i = 0; i < plotted.length; i++) {
      const x = PAD.left + i * xScale, y = PAD.top + ch - plotted[i] * yScale;
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
  loadExisting().finally(() => {
    connectSSE();
  });
  window.addEventListener("resize", drawChart);
}

initTrainPage();
