const cv = document.getElementById("cv");
const ctx = cv.getContext("2d");
const canvasWrap = document.getElementById("canvasWrap");
const focusOverlay = document.getElementById("focusOverlay");
const focusMarkers = Array.from(document.querySelectorAll("#focusOverlay .focus-marker"));
const statusEl = document.getElementById("status");
const instructionBanner = document.getElementById("instructionBanner");
const labelerSummary = document.getElementById("labelerSummary");
const btnContinueStep = document.getElementById("btnContinueStep");
const btnCompleteNext = document.getElementById("btnCompleteNext");
const btnSkipStep = document.getElementById("btnSkipStep");
const sessionQuestionButtons = document.getElementById("sessionQuestionButtons");
const distortionHint = document.getElementById("distortionHint");
const personRoleButtons = document.getElementById("personRoleButtons");
const wizardStepTitle = document.getElementById("wizardStepTitle");
const stepGeometryPanel = document.getElementById("stepGeometryPanel");
const stepPeoplePanel = document.getElementById("stepPeoplePanel");
const stepBallPanel = document.getElementById("stepBallPanel");
const stepIgnorePanel = document.getElementById("stepIgnorePanel");

const STEP = {
  SESSION: "session",
  GEOMETRY: "geometry",
  FOCUS: "focus",
  PEOPLE: "people",
  BALL: "ball",
  IGNORE: "ignore",
};
const STEP_ORDER = [STEP.SESSION, STEP.GEOMETRY, STEP.PEOPLE, STEP.BALL, STEP.IGNORE];
const SESSION_SUBSTEP = {
  GENDER: "gender",
  BALL_IN_PLAY: "ball_in_play",
  BALL_VISIBLE: "ball_visible",
};
const GEOMETRY_SUBSTEP = {
  DISTORTION: "distortion",
  COURT: "court",
  NET: "net",
};
const PEOPLE_SUBSTEP = {
  PLAYERS: "players",
  REFEREES: "referees",
  PARTICIPANTS: "participants",
};
  const PASS_N = 1;
  const DISTORTION_GUIDE_COLOR = "#79c0ff";
  const MIN_DISTORTION_HELPER_POINTS = 4;
  const DISTORTION_PREVIEW_POINTS = 5;
const BEACH_COURT_WIDTH_M = 8.0;
const BEACH_COURT_LENGTH_M = 16.0;
const ANTENNA_ABOVE_NET_M = 0.8;
const NET_PANEL_HEIGHT_M = 1.0;
const LABELER_RESUME_VERSION = "v7";
const DISABLE_LABELER_MEMORY = true;
const LAST_IMAGE_KEY = `beach-tracker:${LABELER_RESUME_VERSION}:last-image:pass:${PASS_N}`;
const LAST_WIZARD_KEY = `beach-tracker:${LABELER_RESUME_VERSION}:last-wizard:pass:${PASS_N}`;
const FOCUS_CORNER_DRAG_MARGIN_PX = 24;
const CROPPED_RENDER_PADDING_PX = 16;
const BALL_HINT_RADIUS_PX = 1;
const BALL_HINT_DRAW_RADIUS_PX = 9;
const FALLBACK_ORIGINS = Array.isArray(window.__LABELER_FALLBACK_ORIGINS__)
  ? window.__LABELER_FALLBACK_ORIGINS__.filter((origin) => typeof origin === "string" && origin)
  : [];
let offlineRedirectStarted = false;

const state = {
  currentName: null,
  image: null,
  imageW: 0,
  imageH: 0,
  viewScale: 1,
  renderCanvas: document.createElement("canvas"),
  rawCanvas: document.createElement("canvas"),
  rawCtx: null,
  renderCtx: null,
  renderDirty: false,
  drawRect: { x: 0, y: 0, w: 1, h: 1 },
  renderMeta: null,
  wizardStep: STEP.SESSION,
  sessionSubstep: SESSION_SUBSTEP.GENDER,
  geometrySubstep: GEOMETRY_SUBSTEP.DISTORTION,
  peopleSubstep: PEOPLE_SUBSTEP.PLAYERS,
  sessionType: null,
  genderCategory: null,
  ballInPlay: null,
  ballVisible: null,
  distortionParams: { model: "division", lambda: 0, cx_offset: 0, cy_offset: 0, k1: 0, k2: 0, k3: 0, rotation_deg: 0 },
  cameraModel: null,
  courtModelQuad: null,
  courtCenterAxisPoints: null,
  courtCenterAxisLocked: false,
  centerLineT: 0.5,
  netHeightM: 2.35,
  leftNetDxPx: 0,
  rightNetDxPx: 0,
  leftAntennaDxPx: 0,
  rightAntennaDxPx: 0,
  geometryPolylines: [],
  netPolylines: [],
  currentPolylineIndex: null,
  people: [],
  ball: null,
  ballZoomRect: null,
  focusRegion: [],
  focusRegionCanvasDefault: false,
  focusRegionCanvasPoints: null,
  ignorePoints: [],
  inferred: null,
  history: [],
  drag: null,
  distortionPoints: [],
  distortionHelperPoints: [],
  hoverPoint: null,
  suggestionTimer: null,
  autoSaveTimer: null,
  suspendAutoSave: false,
  isDirty: false,
};

let carryForwardGeometryPreset = null;

state.rawCtx = state.rawCanvas.getContext("2d");
state.renderCtx = state.renderCanvas.getContext("2d");

function setStatus(msg) {
  if (statusEl) statusEl.textContent = msg || "";
}

function setInstruction(msg) {
  if (instructionBanner) instructionBanner.textContent = msg || "";
}

function rememberCurrentImage(name) {
  if (DISABLE_LABELER_MEMORY) return;
  try {
    if (name) localStorage.setItem(LAST_IMAGE_KEY, name);
    else localStorage.removeItem(LAST_IMAGE_KEY);
  } catch (_) {
    // ignore storage failures
  }
}

function getRememberedImage() {
  if (DISABLE_LABELER_MEMORY) return null;
  try {
    return localStorage.getItem(LAST_IMAGE_KEY);
  } catch (_) {
    return null;
  }
}

function rememberWizardProgress() {
  if (DISABLE_LABELER_MEMORY) return;
  try {
    if (!state.currentName) return;
    localStorage.setItem(LAST_WIZARD_KEY, JSON.stringify({
      image: state.currentName,
      wizardStep: state.wizardStep,
      sessionSubstep: state.sessionSubstep,
      geometrySubstep: state.geometrySubstep,
      peopleSubstep: state.peopleSubstep,
    }));
  } catch (_) {
    // ignore storage failures
  }
}

function getRememberedWizardProgress(name) {
  if (DISABLE_LABELER_MEMORY) return null;
  try {
    const raw = localStorage.getItem(LAST_WIZARD_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || parsed.image !== name) return null;
    return parsed;
  } catch (_) {
    return null;
  }
}

function clearRememberedWizardProgress() {
  try {
    localStorage.removeItem(LAST_WIZARD_KEY);
    localStorage.removeItem(LAST_IMAGE_KEY);
  } catch (_) {
    // ignore storage failures
  }
}

function offlineRedirectUrl() {
  for (const origin of FALLBACK_ORIGINS) {
    if (!origin || origin === window.location.origin) continue;
    return `${origin}${window.location.pathname}${window.location.search}${window.location.hash}`;
  }
  return null;
}

function maybeRedirectToOfflineLabeler(error = null) {
  if (offlineRedirectStarted) return false;
  if (!String(window.location.hostname || "").endsWith("trycloudflare.com")) return false;
  const isOffline = navigator.onLine === false;
  const isNetworkError =
    !error || error instanceof TypeError || /Failed to fetch|NetworkError/i.test(String(error?.message || error));
  if (!isOffline && !isNetworkError) return false;
  const target = offlineRedirectUrl();
  if (!target) return false;
  offlineRedirectStarted = true;
  window.location.replace(target);
  return true;
}

window.addEventListener("offline", () => {
  maybeRedirectToOfflineLabeler();
});

async function apiFetch(url, options) {
  try {
    const res = await fetch(url, { credentials: "same-origin", ...options });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res;
  } catch (error) {
    maybeRedirectToOfflineLabeler(error);
    throw error;
  }
}

function deepClone(obj) {
  return JSON.parse(JSON.stringify(obj));
}

function markDirty() {
  state.isDirty = true;
}

function markClean() {
  state.isDirty = false;
}

function clonePointArray(points) {
  return Array.isArray(points)
    ? points.map((point) => [Number(point[0] || 0), Number(point[1] || 0)])
    : null;
}

function hasMeaningfulDistortionParams(params) {
  if (!params) return false;
  return Math.abs(Number(params.lambda || 0)) > 1e-6
    || Math.abs(Number(params.cx_offset || 0)) > 1e-6
    || Math.abs(Number(params.cy_offset || 0)) > 1e-6
    || Math.abs(Number(params.k1 || 0)) > 1e-6
    || Math.abs(Number(params.k2 || 0)) > 1e-6
    || Math.abs(Number(params.k3 || 0)) > 1e-6
    || Math.abs(Number(params.rotation_deg || 0)) > 1e-6;
}

function hasMeaningfulCourtModel(record) {
  return Array.isArray(record?.camera_model?.court_model_quad)
    && record.camera_model.court_model_quad.length === 4;
}

function scaleQuadToImage(quad, srcW, srcH, dstW, dstH) {
  if (!Array.isArray(quad) || quad.length !== 4 || !(srcW > 0) || !(srcH > 0) || !(dstW > 0) || !(dstH > 0)) return null;
  const scaleX = dstW / srcW;
  const scaleY = dstH / srcH;
  return quad.map((point) => [Number(point[0] || 0) * scaleX, Number(point[1] || 0) * scaleY]);
}

function scalePointsToImage(points, srcW, srcH, dstW, dstH) {
  if (!Array.isArray(points) || !(srcW > 0) || !(srcH > 0) || !(dstW > 0) || !(dstH > 0)) return null;
  const scaleX = dstW / srcW;
  const scaleY = dstH / srcH;
  return points.map((point) => [Number(point[0] || 0) * scaleX, Number(point[1] || 0) * scaleY]);
}

function captureCarryForwardGeometryPreset() {
  if (!(state.imageW > 0) || !(state.imageH > 0)) return;
  if (!activeCourtQuad()) return;
  carryForwardGeometryPreset = {
    sourceImageW: state.imageW,
    sourceImageH: state.imageH,
    courtModelQuad: clonePointArray(activeCourtQuad() || state.courtModelQuad),
    courtCenterAxisPoints: clonePointArray(state.courtCenterAxisPoints),
    courtCenterAxisLocked: Boolean(state.courtCenterAxisLocked),
    centerLineT: Number(state.centerLineT || 0.5),
    netHeightM: Number(state.netHeightM || currentNetHeightMeters()),
    leftNetDxPx: Number(state.leftNetDxPx || 0),
    rightNetDxPx: Number(state.rightNetDxPx || 0),
    leftAntennaDxPx: Number(state.leftAntennaDxPx || 0),
    rightAntennaDxPx: Number(state.rightAntennaDxPx || 0),
    cameraModel: state.cameraModel ? deepClone(state.cameraModel) : null,
  };
}

function applyCarryForwardGeometryPreset() {
  if (!carryForwardGeometryPreset || !(state.imageW > 0) || !(state.imageH > 0)) return false;
  const preset = carryForwardGeometryPreset;
  state.courtModelQuad = constrainCourtProjectionQuad(
    scaleQuadToImage(
      preset.courtModelQuad,
      Number(preset.sourceImageW || state.imageW),
      Number(preset.sourceImageH || state.imageH),
      state.imageW,
      state.imageH,
    ),
  );
  state.centerLineT = Number(preset.centerLineT ?? 0.5);
  state.courtCenterAxisPoints = scalePointsToImage(
    preset.courtCenterAxisPoints,
    Number(preset.sourceImageW || state.imageW),
    Number(preset.sourceImageH || state.imageH),
    state.imageW,
    state.imageH,
  ) || null;
  state.courtCenterAxisLocked = Boolean(preset.courtCenterAxisLocked && state.courtCenterAxisPoints);
  state.netHeightM = Number(preset.netHeightM ?? currentNetHeightMeters());
  state.leftNetDxPx = Number(preset.leftNetDxPx || 0) * (state.imageW / Math.max(preset.sourceImageW || state.imageW, 1));
  state.rightNetDxPx = Number(preset.rightNetDxPx || 0) * (state.imageW / Math.max(preset.sourceImageW || state.imageW, 1));
  state.leftAntennaDxPx = Number(preset.leftAntennaDxPx || 0) * (state.imageW / Math.max(preset.sourceImageW || state.imageW, 1));
  state.rightAntennaDxPx = Number(preset.rightAntennaDxPx || 0) * (state.imageW / Math.max(preset.sourceImageW || state.imageW, 1));
  state.cameraModel = preset.cameraModel ? deepClone(preset.cameraModel) : null;
  return true;
}

function pushHistory() {
  markDirty();
  state.history.push(
    deepClone({
      wizardStep: state.wizardStep,
      sessionSubstep: state.sessionSubstep,
      geometrySubstep: state.geometrySubstep,
      peopleSubstep: state.peopleSubstep,
      sessionType: state.sessionType,
      genderCategory: state.genderCategory,
      ballInPlay: state.ballInPlay,
      ballVisible: state.ballVisible,
      distortionPoints: state.distortionPoints,
      distortionParams: state.distortionParams,
      cameraModel: state.cameraModel,
      courtModelQuad: state.courtModelQuad,
      courtCenterAxisPoints: state.courtCenterAxisPoints,
      courtCenterAxisLocked: state.courtCenterAxisLocked,
      centerLineT: state.centerLineT,
      netHeightM: state.netHeightM,
      leftNetDxPx: state.leftNetDxPx,
      rightNetDxPx: state.rightNetDxPx,
      leftAntennaDxPx: state.leftAntennaDxPx,
      rightAntennaDxPx: state.rightAntennaDxPx,
      geometryPolylines: state.geometryPolylines,
      netPolylines: state.netPolylines,
      currentPolylineIndex: state.currentPolylineIndex,
      people: state.people,
      ball: state.ball,
      ballZoomRect: state.ballZoomRect,
      focusRegion: state.focusRegion,
      focusRegionCanvasDefault: state.focusRegionCanvasDefault,
      focusRegionCanvasPoints: state.focusRegionCanvasPoints,
      ignorePoints: state.ignorePoints,
    }),
  );
  if (state.history.length > 120) state.history.shift();
}

function undo() {
  const snap = state.history.pop();
  if (!snap) return;
  Object.assign(state, snap);
  syncUi();
  scheduleBaseRender();
  render();
}

function origin() {
  return [state.imageW / 2, state.imageH / 2];
}

function distortionStrength() {
  const { lambda, cx_offset, cy_offset, k1, k2, k3 } = state.distortionParams;
  return Math.max(
    Math.abs(lambda || 0),
    Math.abs((cx_offset || 0) / Math.max(state.imageW || 1, 1)),
    Math.abs((cy_offset || 0) / Math.max(state.imageH || 1, 1)),
    Math.abs(k1 || 0),
    Math.abs(k2 || 0),
    Math.abs(k3 || 0),
  );
}

function rotationRadians() {
  return ((state.distortionParams.rotation_deg || 0) * Math.PI) / 180;
}

function currentNetHeightMeters() {
  if (state.genderCategory === "men") return 2.43;
  if (state.genderCategory === "mixed") return 2.35;
  return 2.24;
}

function currentAntennaHeightMeters() {
  return currentNetHeightMeters() + ANTENNA_ABOVE_NET_M;
}

function lineAngleDegrees(a, b) {
  return (Math.atan2(b[1] - a[1], b[0] - a[0]) * 180) / Math.PI;
}

function wrapDegrees(value) {
  let out = value;
  while (out <= -180) out += 360;
  while (out > 180) out -= 360;
  return out;
}

function averagePoint(points) {
  if (!Array.isArray(points) || points.length === 0) return [0, 0];
  const sum = points.reduce((acc, point) => [acc[0] + point[0], acc[1] + point[1]], [0, 0]);
  return [sum[0] / points.length, sum[1] / points.length];
}

function netAnchorWorldPoints() {
  return [
    [-BEACH_COURT_WIDTH_M / 2, 0, currentAntennaHeightMeters()],
    [-BEACH_COURT_WIDTH_M / 2, 0, 0],
    [0, 0, currentNetHeightMeters()],
    [BEACH_COURT_WIDTH_M / 2, 0, 0],
    [BEACH_COURT_WIDTH_M / 2, 0, currentAntennaHeightMeters()],
  ];
}

function labeledNetAnchorPoints() {
  const points = state.netPolylines.flatMap((line) => line || []).slice(0, 5);
  if (points.length < 5) return [];
  const labels = [
    "left_antenna_top",
    "left_antenna_base",
    "net_top_center",
    "right_antenna_base",
    "right_antenna_top",
  ];
  const world = netAnchorWorldPoints();
  return labels.map((label, index) => ({
    label,
    image: [points[index][0], points[index][1]],
    world: world[index],
  }));
}

function primaryCourtSupportPolyline() {
  const candidates = state.geometryPolylines
    .filter((line) => Array.isArray(line) && line.length >= 4)
    .sort((a, b) => b.length - a.length);
  return candidates[0] || null;
}

function defaultCourtModelQuad() {
  if (!state.imageW || !state.imageH) return null;
  const w = state.imageW;
  const h = state.imageH;
  return [
    [0.10 * w, 0.86 * h],
    [0.28 * w, 0.50 * h],
    [0.72 * w, 0.50 * h],
    [0.90 * w, 0.86 * h],
  ];
}

function courtWorldQuad() {
  return [
    [0, 0],
    [0, BEACH_COURT_LENGTH_M],
    [BEACH_COURT_WIDTH_M, BEACH_COURT_LENGTH_M],
    [BEACH_COURT_WIDTH_M, 0],
  ];
}

function activeCourtQuad() {
  if (Array.isArray(state.courtModelQuad) && state.courtModelQuad.length === 4) {
    const quad = state.courtModelQuad.map((point) => [Number(point[0] || 0), Number(point[1] || 0)]);
    return isUsableCourtProjectionQuad(quad) ? quad : constrainCourtProjectionQuad(quad);
  }
  return null;
}

function isUsableCourtProjectionQuad(quad) {
  if (!Array.isArray(quad) || quad.length !== 4) return false;
  const points = quad.map((point) => [Number(point?.[0]), Number(point?.[1])]);
  if (points.some((point) => !Number.isFinite(point[0]) || !Number.isFinite(point[1]))) return false;
  const crosses = [];
  for (let i = 0; i < 4; i += 1) {
    const a = points[i];
    const b = points[(i + 1) % 4];
    const c = points[(i + 2) % 4];
    const abX = b[0] - a[0];
    const abY = b[1] - a[1];
    const bcX = c[0] - b[0];
    const bcY = c[1] - b[1];
    if (Math.hypot(abX, abY) < 8) return false;
    crosses.push(abX * bcY - abY * bcX);
  }
  const hasPositive = crosses.some((value) => value > 1e-3);
  const hasNegative = crosses.some((value) => value < -1e-3);
  if (hasPositive && hasNegative) return false;
  const twiceArea = points.reduce((sum, point, index) => {
    const next = points[(index + 1) % 4];
    return sum + point[0] * next[1] - point[1] * next[0];
  }, 0);
  return Math.abs(twiceArea) >= 100;
}

function setCourtModelQuadIfUsable(quad) {
  if (!isUsableCourtProjectionQuad(quad)) return false;
  state.courtModelQuad = quad.map((point) => [Number(point[0]), Number(point[1])]);
  return true;
}

function constrainCourtProjectionQuad(quad) {
  if (!Array.isArray(quad) || quad.length !== 4) return null;
  const points = quad.map((point) => [Number(point[0] || 0), Number(point[1] || 0)]);
  const [nearLeft, farLeft, farRight, nearRight] = points;
  const farCenter = [(farLeft[0] + farRight[0]) * 0.5, (farLeft[1] + farRight[1]) * 0.5];
  const nearCenter = [(nearLeft[0] + nearRight[0]) * 0.5, (nearLeft[1] + nearRight[1]) * 0.5];
  const farWidth = Math.max(1, Math.hypot(farRight[0] - farLeft[0], farRight[1] - farLeft[1]));
  let nearWidth = Math.max(1, Math.hypot(nearRight[0] - nearLeft[0], nearRight[1] - nearLeft[1]));
  const minNearWidth = farWidth * 1.02;
  if (nearWidth < minNearWidth) {
    const nearDirNorm = Math.max(1e-6, nearWidth);
    const nearDir = [(nearRight[0] - nearLeft[0]) / nearDirNorm, (nearRight[1] - nearLeft[1]) / nearDirNorm];
    const grow = (minNearWidth - nearWidth) * 0.5;
    nearLeft[0] -= nearDir[0] * grow;
    nearLeft[1] -= nearDir[1] * grow;
    nearRight[0] += nearDir[0] * grow;
    nearRight[1] += nearDir[1] * grow;
    nearWidth = minNearWidth;
  }
  if (nearCenter[1] <= farCenter[1] + 12) {
    const lift = farCenter[1] + 12 - nearCenter[1];
    nearLeft[1] += lift;
    nearRight[1] += lift;
  }
  const leftLen = Math.hypot(nearLeft[0] - farLeft[0], nearLeft[1] - farLeft[1]);
  const rightLen = Math.hypot(nearRight[0] - farRight[0], nearRight[1] - farRight[1]);
  const sideMin = 12;
  if (leftLen < sideMin) {
    nearLeft[1] += sideMin - leftLen;
  }
  if (rightLen < sideMin) {
    nearRight[1] += sideMin - rightLen;
  }
  if (nearLeft[0] >= nearRight[0] - 10) {
    const cx = (nearLeft[0] + nearRight[0]) * 0.5;
    nearLeft[0] = cx - Math.max(nearWidth * 0.5, 20);
    nearRight[0] = cx + Math.max(nearWidth * 0.5, 20);
  }
  return [nearLeft, farLeft, farRight, nearRight];
}

function ensureCourtModelQuad() {
  if (activeCourtQuad()) return;
  const supportPolyline = primaryCourtSupportPolyline();
  const inferred = inferCourtQuadFromSupportPolyline(supportPolyline);
  state.courtModelQuad = constrainCourtProjectionQuad(inferred || defaultCourtModelQuad());
}

function sampledCourtSupportAnchors(points) {
  if (!Array.isArray(points) || points.length < 4) return null;
  const lastIndex = points.length - 1;
  const idx1 = Math.max(1, Math.min(lastIndex - 2, Math.round(lastIndex / 3)));
  const idx2 = Math.max(idx1 + 1, Math.min(lastIndex - 1, Math.round((2 * lastIndex) / 3)));
  const indices = [0, idx1, idx2, lastIndex];
  return indices.map((index) => [Number(points[index][0] || 0), Number(points[index][1] || 0)]);
}

function lineIntersection2d(a0, a1, b0, b1) {
  const x1 = Number(a0?.[0] || 0);
  const y1 = Number(a0?.[1] || 0);
  const x2 = Number(a1?.[0] || 0);
  const y2 = Number(a1?.[1] || 0);
  const x3 = Number(b0?.[0] || 0);
  const y3 = Number(b0?.[1] || 0);
  const x4 = Number(b1?.[0] || 0);
  const y4 = Number(b1?.[1] || 0);
  const den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
  if (Math.abs(den) < 1e-9) return null;
  const px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den;
  const py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den;
  return [px, py];
}

function inferCourtQuadFromSupportPolyline(points) {
  const explicitQuad = activeCourtQuad();
  if (explicitQuad) return explicitQuad;
  const pts = sampledCourtSupportAnchors(points);
  if (!pts) return null;
  const [p0, p1, p2, p3] = pts;
  const quadCenter = (quad) => lineIntersection2d(quad[0], quad[2], quad[1], quad[3]) || averagePoint(quad);
  const targetSupportFraction = 0.80;
  const targetScale = 1 / Math.max(targetSupportFraction, 1e-6);
  let bestQuad = [p0, p1, p2, p3];
  let bestScore = Number.POSITIVE_INFINITY;
  for (let leftScale = 1.2; leftScale <= 3.8; leftScale += 0.08) {
    for (let rightScale = 1.2; rightScale <= 3.8; rightScale += 0.08) {
      const nearLeft = [
        p1[0] + (p0[0] - p1[0]) * leftScale,
        p1[1] + (p0[1] - p1[1]) * leftScale,
      ];
      const nearRight = [
        p2[0] + (p3[0] - p2[0]) * rightScale,
        p2[1] + (p3[1] - p2[1]) * rightScale,
      ];
      const quad = [nearLeft, p1, p2, nearRight];
      const center = quadCenter(quad);
      let score = 0;
      score += 22 * Math.pow((1 / leftScale) - targetSupportFraction, 2);
      score += 22 * Math.pow((1 / rightScale) - targetSupportFraction, 2);
      score += 8 * Math.pow(leftScale - rightScale, 2);
      score += 2.5 * Math.pow(leftScale - targetScale, 2);
      score += 2.5 * Math.pow(rightScale - targetScale, 2);
      if (score < bestScore) {
        bestScore = score;
        bestQuad = quad;
      }
    }
  }
  return constrainCourtProjectionQuad(bestQuad);
}

function solveCourtHomography(worldPts, imagePts) {
  if (!Array.isArray(worldPts) || !Array.isArray(imagePts) || worldPts.length < 4 || imagePts.length < 4) return null;
  const rows = [];
  const rhs = [];
  for (let i = 0; i < 4; i += 1) {
    const [x, y] = worldPts[i];
    const [u, v] = imagePts[i];
    rows.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    rhs.push(u);
    rows.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    rhs.push(v);
  }
  const solveLinear = (matrix, vector) => {
    const n = vector.length;
    const a = matrix.map((row, rowIndex) => row.slice().concat(vector[rowIndex]));
    for (let col = 0; col < n; col += 1) {
      let pivot = col;
      for (let row = col + 1; row < n; row += 1) {
        if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
      }
      if (Math.abs(a[pivot][col]) < 1e-9) return null;
      if (pivot !== col) {
        const tmp = a[col];
        a[col] = a[pivot];
        a[pivot] = tmp;
      }
      const pivotVal = a[col][col];
      for (let k = col; k <= n; k += 1) a[col][k] /= pivotVal;
      for (let row = 0; row < n; row += 1) {
        if (row === col) continue;
        const factor = a[row][col];
        if (Math.abs(factor) < 1e-12) continue;
        for (let k = col; k <= n; k += 1) a[row][k] -= factor * a[col][k];
      }
    }
    return a.map((row) => row[n]);
  };
  const solution = solveLinear(rows, rhs);
  if (!solution) return null;
  return [
    [solution[0], solution[1], solution[2]],
    [solution[3], solution[4], solution[5]],
    [solution[6], solution[7], 1],
  ];
}

function solveLinearSystem(matrix, vector) {
  const n = vector.length;
  const a = matrix.map((row, rowIndex) => row.slice().concat(vector[rowIndex]));
  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < n; row += 1) {
      if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    }
    if (Math.abs(a[pivot][col]) < 1e-9) return null;
    [a[col], a[pivot]] = [a[pivot], a[col]];
    const pivotValue = a[col][col];
    for (let k = col; k <= n; k += 1) a[col][k] /= pivotValue;
    for (let row = 0; row < n; row += 1) {
      if (row === col) continue;
      const factor = a[row][col];
      for (let k = col; k <= n; k += 1) a[row][k] -= factor * a[col][k];
    }
  }
  return a.map((row) => row[n]);
}

function solveWeightedCourtHomography(observations) {
  const ata = Array.from({ length: 8 }, () => Array(8).fill(0));
  const atb = Array(8).fill(0);
  observations.forEach(({ world, image, weight = 1 }) => {
    const [x, y] = world;
    const [u, v] = image;
    const rows = [
      [[x, y, 1, 0, 0, 0, -u * x, -u * y], u],
      [[0, 0, 0, x, y, 1, -v * x, -v * y], v],
    ];
    rows.forEach(([row, rhs]) => {
      for (let i = 0; i < 8; i += 1) {
        atb[i] += weight * row[i] * rhs;
        for (let j = 0; j < 8; j += 1) ata[i][j] += weight * row[i] * row[j];
      }
    });
  });
  const solution = solveLinearSystem(ata, atb);
  return solution ? [
    [solution[0], solution[1], solution[2]],
    [solution[3], solution[4], solution[5]],
    [solution[6], solution[7], 1],
  ] : null;
}

function fitCourtProjection(startQuad, targets, centerAxisTargets = null) {
  const worlds = courtWorldQuad();
  const observations = worlds.map((world, index) => ({
    world,
    image: targets[index] || startQuad[index],
    weight: targets[index] ? 100000 : 12,
  }));
  if (Array.isArray(centerAxisTargets) && centerAxisTargets.length === 2) {
    observations.push(
      { world: [0, BEACH_COURT_LENGTH_M * 0.5], image: centerAxisTargets[0], weight: 12000 },
      { world: [BEACH_COURT_WIDTH_M, BEACH_COURT_LENGTH_M * 0.5], image: centerAxisTargets[1], weight: 12000 },
    );
  }
  const homography = solveWeightedCourtHomography(observations);
  const quad = worlds.map((world) => projectWorldPointH(world, homography));
  return quad.every(Boolean) && isUsableCourtProjectionQuad(quad) ? quad : null;
}

function centerAxisFromQuad(quad) {
  const homography = solveCourtHomography(courtWorldQuad(), quad);
  if (!homography) return null;
  const points = [
    projectWorldPointH([0, BEACH_COURT_LENGTH_M * 0.5], homography),
    projectWorldPointH([BEACH_COURT_WIDTH_M, BEACH_COURT_LENGTH_M * 0.5], homography),
  ];
  return points.every(Boolean) ? points : null;
}

function projectWorldPointH(worldPoint, homography) {
  if (!homography) return null;
  const x = Number(worldPoint?.[0] || 0);
  const y = Number(worldPoint?.[1] || 0);
  const den = homography[2][0] * x + homography[2][1] * y + homography[2][2];
  if (Math.abs(den) < 1e-9) return null;
  const u = (homography[0][0] * x + homography[0][1] * y + homography[0][2]) / den;
  const v = (homography[1][0] * x + homography[1][1] * y + homography[1][2]) / den;
  return [u, v];
}

function buildCourtModelSegmentsFromHomography(homography) {
  if (!homography) return [];
  const worldSegments = [
    [[0, 0], [BEACH_COURT_WIDTH_M, 0]],
    [[BEACH_COURT_WIDTH_M, 0], [BEACH_COURT_WIDTH_M, BEACH_COURT_LENGTH_M]],
    [[BEACH_COURT_WIDTH_M, BEACH_COURT_LENGTH_M], [0, BEACH_COURT_LENGTH_M]],
    [[0, BEACH_COURT_LENGTH_M], [0, 0]],
    [[0, BEACH_COURT_LENGTH_M * 0.5], [BEACH_COURT_WIDTH_M, BEACH_COURT_LENGTH_M * 0.5]],
  ];
  return worldSegments
    .map(([a, b]) => {
      const pa = projectWorldPointH(a, homography);
      const pb = projectWorldPointH(b, homography);
      return pa && pb ? { a: pa, b: pb } : null;
    })
    .filter(Boolean);
}

function estimateInitialCameraModel() {
  if (!state.imageW || !state.imageH) return null;
  const imageCenterX = state.imageW / 2;
  const imageCenterY = state.imageH / 2;
  const maxDim = Math.max(state.imageW, state.imageH, 1);
  const fovDegrees = 60;
  const fGuess = (0.5 * maxDim) / Math.tan((fovDegrees * Math.PI) / 360);
  const sources = ["distortion_fit"];
  const notes = [
    "Initial pinhole+radtan seed derived from labeler distortion fit and visible geometry.",
    "Pose values are coarse heuristics and intended as optimizer starting values, not final calibration.",
  ];
  let rollDeg = Number(state.distortionParams.rotation_deg || 0);
  let yawDeg = 0;
  let pitchDeg = 0;
  let tzMeters = 12;
  const netAnchors = labeledNetAnchorPoints();
  if (netAnchors.length >= 5) {
    sources.push("net_points");
    const leftVertical = lineAngleDegrees(netAnchors[1].image, netAnchors[0].image);
    const rightVertical = lineAngleDegrees(netAnchors[3].image, netAnchors[4].image);
    const avgVertical = wrapDegrees((leftVertical + rightVertical) / 2);
    rollDeg = wrapDegrees(avgVertical - 90);
    const netCenter = averagePoint(netAnchors.map((anchor) => anchor.image));
    yawDeg = ((netCenter[0] - imageCenterX) / Math.max(state.imageW, 1)) * 40;
    pitchDeg = ((imageCenterY - netCenter[1]) / Math.max(state.imageH, 1)) * 30;
    const leftBase = netAnchors[1].image;
    const rightBase = netAnchors[3].image;
    const spanPx = Math.max(16, Math.hypot(rightBase[0] - leftBase[0], rightBase[1] - leftBase[1]));
    tzMeters = Math.max(4, (fGuess * BEACH_COURT_WIDTH_M) / spanPx);
    notes.push("Net anchors supplied 3D reference points for the initial pose seed.");
  } else {
    notes.push("Net anchors are incomplete; pose seed falls back to image-center heuristics.");
  }
  if (Array.isArray(state.focusRegion) && state.focusRegion.length >= 4) {
    sources.push("focus_region");
  }
  if (state.geometryPolylines.some((line) => Array.isArray(line) && line.length >= 2)) {
    sources.push("court_lines");
  }
  if (state.geometrySubstep === GEOMETRY_SUBSTEP.COURT || state.wizardStep === STEP.GEOMETRY) {
    ensureCourtModelQuad();
  }
  const supportPolyline = primaryCourtSupportPolyline();
  const supportQuad = activeCourtQuad() || inferCourtQuadFromSupportPolyline(supportPolyline);
  const worldQuad = [
    [0, 0],
    [0, BEACH_COURT_LENGTH_M],
    [BEACH_COURT_WIDTH_M, BEACH_COURT_LENGTH_M],
    [BEACH_COURT_WIDTH_M, 0],
  ];
  const courtHomography = supportQuad ? solveCourtHomography(worldQuad, supportQuad) : null;
  const modelGeometry = courtModelGeometry();
  const courtModelSegments = modelGeometry
    ? modelGeometry.segments.map((segment) => ({
      a: [segment.a[0], segment.a[1]],
      b: [segment.b[0], segment.b[1]],
      role: segment.role,
    }))
    : buildCourtModelSegmentsFromHomography(courtHomography);
  if (supportPolyline && supportPolyline.length >= 4) {
    sources.push("court_support_polyline");
    notes.push("Court model is fit live from the main support polyline so camera fit and line fit share the same observations.");
  }
  return {
    version: "camera-editor-v2",
    estimate_status: "initial_guess",
    estimate_sources: sources,
    confidence: netAnchors.length >= 5 ? 0.42 : 0.18,
    intrinsics: {
      fx: fGuess,
      fy: fGuess,
      cx: imageCenterX,
      cy: imageCenterY,
      skew: 0,
      image_width: state.imageW,
      image_height: state.imageH,
    },
      distortion: {
        model: state.distortionParams.model || "division",
        lambda: Number(state.distortionParams.lambda || 0),
        cx_offset: Number(state.distortionParams.cx_offset || 0),
        cy_offset: Number(state.distortionParams.cy_offset || 0),
        k1: Number(state.distortionParams.k1 || 0),
        k2: Number(state.distortionParams.k2 || 0),
        k3: Number(state.distortionParams.k3 || 0),
      p1: 0,
      p2: 0,
      rotation_deg: Number(state.distortionParams.rotation_deg || 0),
    },
    pose: {
      tx_m: 0,
      ty_m: -BEACH_COURT_LENGTH_M / 2,
      tz_m: tzMeters,
      yaw_deg: yawDeg,
      pitch_deg: pitchDeg,
      roll_deg: rollDeg,
    },
    net_anchors: netAnchors,
    focus_region: Array.isArray(state.focusRegion)
      ? state.focusRegion.filter((point) => Array.isArray(point) && point.length >= 2).map((point) => [point[0], point[1]])
      : [],
    court_polylines: state.geometryPolylines
      .filter((line) => Array.isArray(line) && line.length >= 2)
      .map((line) => line.map((point) => [point[0], point[1]])),
    court_center_hint: projectedCourtMidlinePoint(supportQuad),
    court_model_quad: supportQuad
      ? supportQuad.map((point) => [point[0], point[1]])
      : [],
    center_axis_points: Array.isArray(state.courtCenterAxisPoints)
      ? state.courtCenterAxisPoints.map((point) => [point[0], point[1]])
      : [],
    center_axis_locked: Boolean(state.courtCenterAxisLocked),
    coordinate_system: {
      origin: "court_center_ground",
      x_axis: "across_court_along_center_line",
      y_axis: "along_left_sideline_away_from_camera",
      z_axis: "up",
      handedness: "right",
    },
    center_line_t: Number(state.centerLineT || 0.5),
    net_height_m: Number(state.netHeightM || currentNetHeightMeters()),
    left_net_dx_px: Number(state.leftNetDxPx || 0),
    right_net_dx_px: Number(state.rightNetDxPx || 0),
    left_antenna_dx_px: Number(state.leftAntennaDxPx || 0),
    right_antenna_dx_px: Number(state.rightAntennaDxPx || 0),
    court_support_polyline: supportPolyline
      ? supportPolyline.map((point) => [point[0], point[1]])
      : [],
    court_quad_image: supportQuad
      ? supportQuad.map((point) => [point[0], point[1]])
      : [],
    court_homography: courtHomography,
    court_model_segments: courtModelSegments,
    notes,
  };
}

function refreshCameraModelEstimate() {
  state.cameraModel = estimateInitialCameraModel();
}

function geometryLineCount() {
  const lines = state.geometrySubstep === GEOMETRY_SUBSTEP.NET ? state.netPolylines : state.geometryPolylines;
  return lines.filter((line) => Array.isArray(line) && line.length >= 2).length;
}

function geometryPointCount() {
  const lines = state.geometrySubstep === GEOMETRY_SUBSTEP.NET ? state.netPolylines : state.geometryPolylines;
  return lines.reduce((sum, line) => (
    sum + (Array.isArray(line) ? line.length : 0)
  ), 0);
}

function activePolyline() {
  if (state.currentPolylineIndex == null) return null;
  const lines = state.geometrySubstep === GEOMETRY_SUBSTEP.NET ? state.netPolylines : state.geometryPolylines;
  return lines[state.currentPolylineIndex] || null;
}

function defaultFocusRegionImage() {
  if (state.renderMeta && cv.width > 0 && cv.height > 0) {
    return [
      canvasPixelsToImagePoint(0, 0),
      canvasPixelsToImagePoint(cv.width, 0),
      canvasPixelsToImagePoint(cv.width, cv.height),
      canvasPixelsToImagePoint(0, cv.height),
    ];
  }
  return [
    [0, 0],
    [state.imageW, 0],
    [state.imageW, state.imageH],
    [0, state.imageH],
  ];
}

function defaultFocusRegionCanvasPoints() {
  return [
    [0, 0],
    [cv.width, 0],
    [cv.width, cv.height],
    [0, cv.height],
  ];
}

function defaultFocusRegion() {
  state.focusRegionCanvasDefault = true;
  state.focusRegionCanvasPoints = defaultFocusRegionCanvasPoints();
  return defaultFocusRegionImage();
}

function resetFocusRegionToCurrentViewDefault() {
  state.focusRegionCanvasDefault = true;
  state.focusRegionCanvasPoints = defaultFocusRegionCanvasPoints();
  state.focusRegion = defaultFocusRegionImage();
}

function ensureDefaultFocusRegion() {
  if (!Array.isArray(state.focusRegion) || state.focusRegion.length < 4) {
    state.focusRegion = defaultFocusRegion();
    return;
  }
  const xs = state.focusRegion.map((point) => point?.[0]).filter((value) => Number.isFinite(value));
  const ys = state.focusRegion.map((point) => point?.[1]).filter((value) => Number.isFinite(value));
  if (xs.length < 4 || ys.length < 4) {
    state.focusRegion = defaultFocusRegion();
    return;
  }
  const spanX = Math.max(...xs) - Math.min(...xs);
  const spanY = Math.max(...ys) - Math.min(...ys);
  if (spanX < 24 || spanY < 24) {
    state.focusRegion = defaultFocusRegion();
  }
}

function ensureEditableFocusRegionPoints() {
  if (Array.isArray(state.focusRegionCanvasPoints) && state.focusRegionCanvasPoints.length >= 4) {
    return;
  }
  if (state.focusRegionCanvasDefault || !Array.isArray(state.focusRegion) || state.focusRegion.length < 4) {
    state.focusRegionCanvasPoints = defaultFocusRegionCanvasPoints();
    return;
  }
  state.focusRegionCanvasPoints = state.focusRegion.slice(0, 4).map((point) => imageToCanvasPoint(point[0], point[1]));
}

function commitFocusRegionCanvasPoints() {
  ensureEditableFocusRegionPoints();
  state.focusRegion = state.focusRegionCanvasPoints.slice(0, 4).map(([x, y]) => {
    const [ix, iy] = canvasPixelsToImagePoint(x, y);
    return [
      Math.max(0, Math.min(state.imageW, ix)),
      Math.max(0, Math.min(state.imageH, iy)),
    ];
  });
  state.focusRegionCanvasDefault = false;
}

function focusRegionDisplayPoints(points) {
  if (state.wizardStep === STEP.FOCUS && state.focusRegionCanvasDefault) {
    return defaultFocusRegionCanvasPoints();
  }
  if (Array.isArray(state.focusRegionCanvasPoints) && state.focusRegionCanvasPoints.length >= 4) {
    return state.focusRegionCanvasPoints;
  }
  if (state.focusRegionCanvasDefault && Array.isArray(points) && points.length >= 4) {
    return defaultFocusRegionCanvasPoints();
  }
  return (points || []).map((point) => imageToCanvasPoint(point[0], point[1]));
}

function hitCanvasDefaultFocusCorner(clientX, clientY) {
  const [localX, localY] = clientToCanvasPixels(clientX, clientY);
  const corners = focusRegionDisplayPoints(state.focusRegion);
  const radius = 18 / Math.max(state.viewScale, 0.01);
  let best = null;
  corners.forEach(([x, y], pointIndex) => {
    const dist = Math.hypot(localX - x, localY - y);
    if (dist <= radius && (!best || dist < best.dist)) {
      best = { pointIndex, dist };
    }
  });
  return best;
}

function materializeCanvasDefaultFocusRegion() {
  commitFocusRegionCanvasPoints();
}

function hitFocusCorner(ix, iy) {
  const radius = 16 / Math.max(state.viewScale, 0.01);
  let best = null;
  state.focusRegion.forEach((point, pointIndex) => {
    if (!Array.isArray(point) || point.length < 2) return;
    const dist = Math.hypot(point[0] - ix, point[1] - iy);
    if (dist <= radius && (!best || dist < best.dist)) {
      best = { pointIndex, dist };
    }
  });
  return best;
}

function allGeometryPoints() {
  const lines = state.geometrySubstep === GEOMETRY_SUBSTEP.NET ? state.netPolylines : state.geometryPolylines;
  return lines.flatMap((line) => line || []);
}

function currentGeometryCollection() {
  return state.geometrySubstep === GEOMETRY_SUBSTEP.NET ? state.netPolylines : state.geometryPolylines;
}

function mergeGeometryPoint(ix, iy) {
  const radius = 14 / Math.max(state.viewScale, 0.01);
  let best = null;
  for (const point of allGeometryPoints()) {
    if (!Array.isArray(point) || point.length < 2) continue;
    const dist = Math.hypot(point[0] - ix, point[1] - iy);
    if (dist <= radius && (!best || dist < best.dist)) {
      best = { point, dist };
    }
  }
  return best ? [best.point[0], best.point[1]] : [ix, iy];
}

function hitGeometryPointRef(ix, iy) {
  const radius = 16 / Math.max(state.viewScale, 0.01);
  let best = null;
  currentGeometryCollection().forEach((line, lineIndex) => {
    if (!Array.isArray(line)) return;
    line.forEach((point, pointIndex) => {
      if (!Array.isArray(point) || point.length < 2) return;
      const dist = Math.hypot(point[0] - ix, point[1] - iy);
      if (dist <= radius && (!best || dist < best.dist)) {
        best = { lineIndex, pointIndex, point, dist };
      }
    });
  });
  return best;
}

function pointToSegmentDistance(px, py, ax, ay, bx, by) {
  const abx = bx - ax;
  const aby = by - ay;
  const ab2 = abx * abx + aby * aby;
  if (ab2 <= 1e-9) return { dist: Math.hypot(px - ax, py - ay), t: 0 };
  const t = Math.max(0, Math.min(1, ((px - ax) * abx + (py - ay) * aby) / ab2));
  const qx = ax + abx * t;
  const qy = ay + aby * t;
  return { dist: Math.hypot(px - qx, py - qy), t };
}

function hitGeometrySegment(ix, iy) {
  const radius = 12 / Math.max(state.viewScale, 0.01);
  let best = null;
  currentGeometryCollection().forEach((line, lineIndex) => {
    if (!Array.isArray(line) || line.length < 2) return;
    for (let i = 0; i < line.length - 1; i += 1) {
      const a = line[i];
      const b = line[i + 1];
      const hit = pointToSegmentDistance(ix, iy, a[0], a[1], b[0], b[1]);
      if (hit.dist <= radius && (!best || hit.dist < best.dist)) {
        best = { lineIndex, insertIndex: i + 1, dist: hit.dist };
      }
    }
  });
  return best;
}

function setSessionType(value) {
  state.sessionType = value;
  state.peopleSubstep = value === "training" ? PEOPLE_SUBSTEP.PARTICIPANTS : PEOPLE_SUBSTEP.PLAYERS;
  if (value === "training") {
    state.people = state.people.map((p) => ({ ...p, role: "participant" }));
  }
  syncUi();
}

function setGender(value) {
  markDirty();
  state.genderCategory = value;
  state.sessionSubstep = SESSION_SUBSTEP.BALL_IN_PLAY;
  syncUi();
  scheduleAutoSave();
  render();
}

function setBallInPlay(value) {
  markDirty();
  state.ballInPlay = value;
  state.ballVisible = null;
  state.sessionSubstep = SESSION_SUBSTEP.BALL_VISIBLE;
  syncUi();
  scheduleAutoSave();
  render();
}

function setBallVisible(value) {
  markDirty();
  state.ballVisible = value;
  state.wizardStep = STEP.GEOMETRY;
  state.geometrySubstep = GEOMETRY_SUBSTEP.DISTORTION;
  syncUi();
  scheduleAutoSave();
  render();
}

function buildChoiceButtons(container, values, getLabel, onClick) {
  container.innerHTML = "";
  values.forEach((value) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "btn btn-choice";
    button.dataset.value = value;
    button.textContent = getLabel(value);
    button.addEventListener("click", () => onClick(value));
    container.appendChild(button);
  });
}

function updateChoiceButtons(container, activeValue) {
  container.querySelectorAll("button").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.value === activeValue);
  });
}

function currentPersonRole() {
  if (state.peopleSubstep === PEOPLE_SUBSTEP.REFEREES) return "referee";
  if (state.peopleSubstep === PEOPLE_SUBSTEP.PARTICIPANTS) return "participant";
  return "player";
}

function syncSessionQuestionUi() {
  let values = ["women", "men", "mixed"];
  let getLabel = (v) => v.charAt(0).toUpperCase() + v.slice(1);
  let onClick = setGender;

  if (state.sessionSubstep === SESSION_SUBSTEP.BALL_IN_PLAY) {
    values = ["true", "false"];
    getLabel = (v) => (v === "true" ? "Yes" : "No");
    onClick = (value) => setBallInPlay(value === "true");
  } else if (state.sessionSubstep === SESSION_SUBSTEP.BALL_VISIBLE) {
    values = ["true", "false"];
    getLabel = (v) => (v === "true" ? "Visible" : "Not visible");
    onClick = (value) => setBallVisible(value === "true");
  }

  buildChoiceButtons(sessionQuestionButtons, values, getLabel, onClick);
}

function activePanel(step) {
  sessionQuestionButtons.classList.toggle("hidden", step !== STEP.SESSION);
  stepGeometryPanel.classList.toggle("hidden", step !== STEP.GEOMETRY);
  stepPeoplePanel.classList.toggle("hidden", step !== STEP.PEOPLE);
  stepBallPanel.classList.toggle("hidden", step !== STEP.BALL);
  stepIgnorePanel.classList.toggle("hidden", step !== STEP.IGNORE);
}

  function canContinueFromStep(step) {
  if (step === STEP.SESSION) {
    if (!state.genderCategory) return false;
    if (state.ballInPlay === null) return false;
    if (state.ballVisible === null) return false;
    return true;
  }
    if (step === STEP.GEOMETRY) {
      if (state.geometrySubstep === GEOMETRY_SUBSTEP.DISTORTION) {
        return state.distortionPoints.length >= MIN_DISTORTION_HELPER_POINTS;
      }
    return state.geometrySubstep === GEOMETRY_SUBSTEP.COURT
      ? !!activeCourtQuad()
      : geometryPointCount() >= 3;
  }
  if (step === STEP.FOCUS) {
    return state.focusRegion.length >= 4;
  }
  if (step === STEP.BALL) {
    return !!state.ball;
  }
  return true;
}

function syncUi() {
  rememberWizardProgress();
  if (labelerSummary) {
    const summaryItems = [];
    const sessionType = state.sessionType || state.inferred?.session_type || null;
    if (sessionType) {
      summaryItems.push({
        text: sessionType === "training" ? "Training" : "Game",
        tone: "default",
      });
    }
    if (state.genderCategory) {
      summaryItems.push({
        text: state.genderCategory.charAt(0).toUpperCase() + state.genderCategory.slice(1),
        tone: "default",
      });
    }
    if (state.ballInPlay !== null) {
      summaryItems.push({
        text: state.ballInPlay ? "In play" : "Not in play",
        tone: "default",
      });
    }
    if (state.ballVisible !== null) {
      summaryItems.push({
        text: state.ballVisible ? "Ball visible" : "Ball hidden",
        tone: "default",
      });
    }
    const lineCount = geometryLineCount();
    if (lineCount > 0) {
      summaryItems.push({
        text: `Lines ${lineCount}`,
        tone: "default",
      });
    }

    labelerSummary.innerHTML = "";
    summaryItems.forEach((item, index) => {
      if (index > 0) {
        const dot = document.createElement("span");
        dot.className = "labeler-summary-dot";
        dot.textContent = "·";
        labelerSummary.appendChild(dot);
      }
      const bit = document.createElement("span");
      bit.className = `labeler-summary-item${item.tone === "warn" ? " is-warn" : ""}`;
      bit.textContent = item.text;
      labelerSummary.appendChild(bit);
    });
  }

  personRoleButtons.classList.add("hidden");

  if (distortionHint) {
    distortionHint.textContent = state.geometrySubstep === GEOMETRY_SUBSTEP.DISTORTION
      ? `Click more than ${DISTORTION_PREVIEW_POINTS} support points on one bent rigid horizontal line. The previous frame's correction is reused as a starting guess, then Continue refits the pinhole bootstrap (${state.distortionPoints.length}/${DISTORTION_PREVIEW_POINTS + 1})`
      : "";
  }
  const contextBits = [];
  if (state.inferred?.session_type) contextBits.push(state.inferred.session_type);

  activePanel(state.wizardStep);

  if (state.wizardStep === STEP.SESSION) {
    syncSessionQuestionUi();
    wizardStepTitle.textContent = state.sessionSubstep === SESSION_SUBSTEP.GENDER
      ? "Who is playing?"
      : state.sessionSubstep === SESSION_SUBSTEP.BALL_IN_PLAY
        ? "Is the ball in play?"
        : "Is the ball visible?";
    setInstruction("");
    } else if (state.wizardStep === STEP.GEOMETRY) {
    if (state.geometrySubstep === GEOMETRY_SUBSTEP.DISTORTION) {
        wizardStepTitle.textContent = "Adjust the carried-over fisheye view by clicking more than 5 support points on one rigid horizontal line";
        setInstruction("");
    } else {
      wizardStepTitle.textContent = state.geometrySubstep === GEOMETRY_SUBSTEP.NET
        ? "Drag the green net vertically and drag the cyan antennas horizontally to match the image"
        : "Place the red court/net model from the carried-over start, then refine the court from image evidence";
      setInstruction("Court: reuse the previous frame's camera guess and drag the projected court, net, and antennas into place. These human-set model parameters are saved directly as the training target.");
    }
  } else if (state.wizardStep === STEP.PEOPLE) {
    if (state.sessionType === "training") {
      wizardStepTitle.textContent = "Draw participant boxes";
      setInstruction("");
    } else if (state.peopleSubstep === PEOPLE_SUBSTEP.REFEREES) {
      wizardStepTitle.textContent = "Draw referee boxes";
      setInstruction("");
    } else {
      wizardStepTitle.textContent = "Draw player boxes";
      setInstruction("");
    }
  } else if (state.wizardStep === STEP.BALL) {
    wizardStepTitle.textContent = "Click roughly where the active ball is";
    setInstruction("");
  } else if (state.wizardStep === STEP.FOCUS) {
    ensureDefaultFocusRegion();
    wizardStepTitle.textContent = "Drag the markers to set the focus region";
    setInstruction("");
  } else {
    wizardStepTitle.textContent = "Mark all the balls in the scene that we want to ignore";
    setInstruction("");
  }

    const stepIndex = STEP_ORDER.indexOf(state.wizardStep);
    btnContinueStep.disabled = state.wizardStep === STEP.IGNORE
      ? !state.currentName
      : !canContinueFromStep(state.wizardStep);
    const showContinue =
      (state.wizardStep === STEP.GEOMETRY) ||
      state.wizardStep === STEP.PEOPLE ||
      state.wizardStep === STEP.BALL ||
      state.wizardStep === STEP.FOCUS ||
      state.wizardStep === STEP.IGNORE;
  btnContinueStep.classList.toggle("hidden", !showContinue);
  btnContinueStep.textContent = state.wizardStep === STEP.IGNORE
    ? "Submit & next image"
    : "Continue";
  btnSkipStep.disabled = false;
  btnSkipStep.classList.toggle("hidden", false);
  btnSkipStep.textContent = "Skip step";
  btnCompleteNext.disabled = !state.currentName;
  btnCompleteNext.textContent = state.wizardStep === STEP.IGNORE
    ? (state.isDirty ? "Submit & next image" : "Next image")
    : (state.isDirty ? "Save & next image" : "Next image");
  btnCompleteNext.classList.toggle(
    "hidden",
    (state.wizardStep === STEP.GEOMETRY && state.geometrySubstep === GEOMETRY_SUBSTEP.DISTORTION) ||
    state.wizardStep === STEP.IGNORE,
  );
}

function getDisplaySize() {
  const outerW = Math.max(320, canvasWrap.clientWidth);
  const outerH = Math.max(320, canvasWrap.clientHeight);
  const distortionPad = 7;
  const maxW = Math.max(220, outerW - distortionPad * 2);
  const maxH = Math.max(220, outerH - distortionPad * 2);
  const rawScale = Math.min(maxW / state.imageW, maxH / state.imageH);
  const drawW = Math.max(1, Math.round(state.imageW * rawScale));
  const drawH = Math.max(1, Math.round(state.imageH * rawScale));
  const overscan = 1.45 + distortionStrength() * 0.9;
  return {
    width: outerW,
    height: outerH,
    maxW,
    maxH,
    drawW,
    drawH,
    workW: Math.max(drawW, Math.round(drawW * overscan)),
    workH: Math.max(drawH, Math.round(drawH * overscan)),
    bboxW: Math.max(drawW, Math.round(drawW * overscan)),
    bboxH: Math.max(drawH, Math.round(drawH * overscan)),
    rawScale,
  };
}

function scheduleBaseRender() {
  state.renderDirty = true;
  requestAnimationFrame(() => {
    if (!state.renderDirty || !state.image) return;
    state.renderDirty = false;
    rebuildBaseImage();
    if (state.wizardStep === STEP.FOCUS && state.focusRegionCanvasDefault) {
      resetFocusRegionToCurrentViewDefault();
    }
    scheduleSuggestionDetection();
    render();
  });
}

function scheduleSuggestionDetection() {
  if (state.suggestionTimer) {
    clearTimeout(state.suggestionTimer);
  }
}

function rebuildBaseImage() {
  const size = getDisplaySize();
  state.viewScale = size.rawScale;
  state.rawCanvas.width = size.drawW;
  state.rawCanvas.height = size.drawH;
  state.rawCtx.clearRect(0, 0, size.drawW, size.drawH);
  state.rawCtx.drawImage(state.image, 0, 0, size.drawW, size.drawH);
  const srcImage = state.rawCtx.getImageData(0, 0, size.drawW, size.drawH);
  const srcData = srcImage.data;
  const params = { ...state.distortionParams };
  const angle = Math.abs(rotationRadians());
  const rotW = Math.ceil(Math.abs(size.workW * Math.cos(angle)) + Math.abs(size.workH * Math.sin(angle)));
  const rotH = Math.ceil(Math.abs(size.workW * Math.sin(angle)) + Math.abs(size.workH * Math.cos(angle)));
  const rotatedCanvas = document.createElement("canvas");
  rotatedCanvas.width = Math.max(1, rotW);
  rotatedCanvas.height = Math.max(1, rotH);
  const rotatedCtx = rotatedCanvas.getContext("2d");
  const rotatedImage = rotatedCtx.createImageData(rotatedCanvas.width, rotatedCanvas.height);
  const out = rotatedImage.data;
  for (let y = 0; y < rotatedCanvas.height; y += 1) {
    for (let x = 0; x < rotatedCanvas.width; x += 1) {
      const warped = combinedRotatedToSourcePoint(
        x,
        y,
        size.drawW,
        size.drawH,
        size.workW,
        size.workH,
        rotatedCanvas.width,
        rotatedCanvas.height,
        params,
      );
      const sx = warped[0];
      const sy = warped[1];
      const di = (y * rotatedCanvas.width + x) * 4;
      if (sx < 0 || sy < 0 || sx >= size.drawW - 1 || sy >= size.drawH - 1) {
        out[di] = 0;
        out[di + 1] = 0;
        out[di + 2] = 0;
        out[di + 3] = 0;
        continue;
      }
      sampleBilinear(srcData, size.drawW, size.drawH, sx, sy, out, di);
    }
  }
  rotatedCtx.putImageData(rotatedImage, 0, 0);

  const correctedData = rotatedCtx.getImageData(0, 0, rotatedCanvas.width, rotatedCanvas.height);
  let minX = rotatedCanvas.width;
  let minY = rotatedCanvas.height;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < rotatedCanvas.height; y += 1) {
    for (let x = 0; x < rotatedCanvas.width; x += 1) {
      const alpha = correctedData.data[(y * rotatedCanvas.width + x) * 4 + 3];
      if (alpha > 0) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }
  const cropX = maxX >= minX ? minX : 0;
  const cropY = maxY >= minY ? minY : 0;
  const cropW = maxX >= minX ? (maxX - minX + 1) : rotatedCanvas.width;
  const cropH = maxY >= minY ? (maxY - minY + 1) : rotatedCanvas.height;
  const paddedCropX = Math.max(0, cropX - CROPPED_RENDER_PADDING_PX);
  const paddedCropY = Math.max(0, cropY - CROPPED_RENDER_PADDING_PX);
  const paddedCropW = maxX >= minX
    ? Math.min(rotatedCanvas.width, cropW + CROPPED_RENDER_PADDING_PX * 2)
    : rotatedCanvas.width;
  const paddedCropH = maxY >= minY
    ? Math.min(rotatedCanvas.height, cropH + CROPPED_RENDER_PADDING_PX * 2)
    : rotatedCanvas.height;
  const paddedWidth = Math.max(1, Math.min(rotatedCanvas.width - paddedCropX, paddedCropW));
  const paddedHeight = Math.max(1, Math.min(rotatedCanvas.height - paddedCropY, paddedCropH));
  const drawCropW = paddedWidth;
  const drawCropH = paddedHeight;
  const displayScale = Math.min(size.maxW / drawCropW, size.maxH / drawCropH);
  const displayW = Math.max(1, Math.round(drawCropW * displayScale));
  const displayH = Math.max(1, Math.round(drawCropH * displayScale));
  cv.width = displayW;
  cv.height = displayH;
  state.renderCanvas.width = displayW;
  state.renderCanvas.height = displayH;

  state.renderCtx.clearRect(0, 0, displayW, displayH);
  state.renderCtx.drawImage(
    rotatedCanvas,
    paddedCropX,
    paddedCropY,
    paddedWidth,
    paddedHeight,
    0,
    0,
    displayW,
    displayH,
  );
  state.drawRect = { x: 0, y: 0, w: displayW, h: displayH };
  state.renderMeta = {
    drawW: size.drawW,
    drawH: size.drawH,
    rawScaleX: size.drawW / state.imageW,
    rawScaleY: size.drawH / state.imageH,
    workW: size.workW,
    workH: size.workH,
    bboxW: rotatedCanvas.width,
    bboxH: rotatedCanvas.height,
    cropX: paddedCropX,
    cropY: paddedCropY,
    cropW: paddedWidth,
    cropH: paddedHeight,
    paddedCropX,
    paddedCropY,
    paddedWidth,
    paddedHeight,
    displayScale,
    offsetX: 0,
    offsetY: 0,
  };
}

function imageToCanvasPointBase(x, y) {
  const meta = state.renderMeta;
  if (!meta) return [0, 0];
  const rawX = x * meta.rawScaleX;
  const rawY = y * meta.rawScaleY;
  const rotated = combinedSourceToRotatedPoint(
    rawX,
    rawY,
    meta.drawW,
    meta.drawH,
    meta.workW,
    meta.workH,
    meta.bboxW,
    meta.bboxH,
    state.distortionParams,
  );
  return [
    meta.offsetX + (rotated[0] - meta.cropX) * meta.displayScale,
    meta.offsetY + (rotated[1] - meta.cropY) * meta.displayScale,
  ];
}

function imageToCanvasPoint(x, y) {
  return imageToCanvasPointBase(x, y);
}

function canvasPixelsToImagePointApprox(x, y) {
  const meta = state.renderMeta;
  if (!meta) return [0, 0];
  const rotX = meta.cropX + (x - meta.offsetX) / meta.displayScale;
  const rotY = meta.cropY + (y - meta.offsetY) / meta.displayScale;
  const pseudoDistorted = combinedRotatedToSourcePoint(
    rotX,
    rotY,
    meta.drawW,
    meta.drawH,
    meta.workW,
    meta.workH,
    meta.bboxW,
    meta.bboxH,
    state.distortionParams,
  );
  const rawX = pseudoDistorted[0];
  const rawY = pseudoDistorted[1];
  return [
    rawX / meta.rawScaleX,
    rawY / meta.rawScaleY,
  ];
}

function canvasPixelsToImagePoint(x, y) {
  if (!state.renderMeta) return [0, 0];
  let [guessX, guessY] = canvasPixelsToImagePointApprox(x, y);
  const imageStep = Math.max(0.5, Math.max(state.imageW, state.imageH) / 1500);
  for (let i = 0; i < 6; i += 1) {
    const [projectedX, projectedY] = imageToCanvasPoint(guessX, guessY);
    const errX = x - projectedX;
    const errY = y - projectedY;
    if (Math.hypot(errX, errY) <= 0.2) break;

    const [stepXCanvasX, stepXCanvasY] = imageToCanvasPoint(guessX + imageStep, guessY);
    const [stepYCanvasX, stepYCanvasY] = imageToCanvasPoint(guessX, guessY + imageStep);
    const j00 = (stepXCanvasX - projectedX) / imageStep;
    const j10 = (stepXCanvasY - projectedY) / imageStep;
    const j01 = (stepYCanvasX - projectedX) / imageStep;
    const j11 = (stepYCanvasY - projectedY) / imageStep;
    const det = j00 * j11 - j01 * j10;
    if (!Number.isFinite(det) || Math.abs(det) < 1e-6) break;

    const deltaX = (errX * j11 - errY * j01) / det;
    const deltaY = (j00 * errY - j10 * errX) / det;
    const limitedDeltaX = Math.max(-128, Math.min(128, deltaX));
    const limitedDeltaY = Math.max(-128, Math.min(128, deltaY));
    guessX += limitedDeltaX;
    guessY += limitedDeltaY;
  }
  return [guessX, guessY];
}

function canvasDeltaToImageDelta(anchorImagePoint, deltaCanvasX, deltaCanvasY) {
  if (!Array.isArray(anchorImagePoint) || anchorImagePoint.length < 2) {
    return [deltaCanvasX, deltaCanvasY];
  }
  const imageStep = Math.max(0.5, Math.max(state.imageW || 1, state.imageH || 1) / 1500);
  const [ax, ay] = anchorImagePoint;
  const [projectedX, projectedY] = imageToCanvasPoint(ax, ay);
  const [stepXCanvasX, stepXCanvasY] = imageToCanvasPoint(ax + imageStep, ay);
  const [stepYCanvasX, stepYCanvasY] = imageToCanvasPoint(ax, ay + imageStep);
  const j00 = (stepXCanvasX - projectedX) / imageStep;
  const j10 = (stepXCanvasY - projectedY) / imageStep;
  const j01 = (stepYCanvasX - projectedX) / imageStep;
  const j11 = (stepYCanvasY - projectedY) / imageStep;
  const det = j00 * j11 - j01 * j10;
  if (!Number.isFinite(det) || Math.abs(det) < 1e-6) {
    return [deltaCanvasX, deltaCanvasY];
  }
  return [
    (deltaCanvasX * j11 - deltaCanvasY * j01) / det,
    (j00 * deltaCanvasY - j10 * deltaCanvasX) / det,
  ];
}

function canvasToImagePoint(clientX, clientY) {
  const rect = cv.getBoundingClientRect();
  const x = ((clientX - rect.left) / rect.width) * cv.width;
  const y = ((clientY - rect.top) / rect.height) * cv.height;
  return canvasPixelsToImagePoint(x, y);
}

function clientToCanvasPixels(clientX, clientY) {
  const rect = cv.getBoundingClientRect();
  return [
    ((clientX - rect.left) / rect.width) * cv.width,
    ((clientY - rect.top) / rect.height) * cv.height,
  ];
}

function isCanvasPixelInside(x, y) {
  return x >= 0 && y >= 0 && x <= cv.width && y <= cv.height;
}

function drawGeometryPolyline(points, color, isActive, options = {}) {
  if (!Array.isArray(points) || points.length === 0) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = Number.isFinite(options.width) ? options.width : (isActive ? 3 : 2);
  if (Array.isArray(options.dash) && options.dash.length > 0) {
    ctx.setLineDash(options.dash);
  }
  if (points.length > 1) {
    ctx.beginPath();
    points.forEach((point, index) => {
      const [x, y] = imageToCanvasPoint(point[0], point[1]);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  const pointRadius = options.pointRadius == null ? (isActive ? 5 : 4) : Number(options.pointRadius || 0);
  if (pointRadius > 0) {
    points.forEach((point) => {
      const [x, y] = imageToCanvasPoint(point[0], point[1]);
      ctx.beginPath();
      ctx.arc(x, y, pointRadius, 0, Math.PI * 2);
      ctx.fill();
    });
  }
  ctx.restore();
}

function drawGeometryPreview(points, hoverPoint, color) {
  if (!Array.isArray(points) || points.length === 0 || !hoverPoint) return;
  const last = points[points.length - 1];
  const [x1, y1] = imageToCanvasPoint(last[0], last[1]);
  const [x2, y2] = imageToCanvasPoint(hoverPoint[0], hoverPoint[1]);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.setLineDash([10, 7]);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.restore();
}

function drawSuggestedLine(line) {
  ctx.save();
  ctx.strokeStyle = DISTORTION_GUIDE_COLOR;
  ctx.lineWidth = 3;
  ctx.beginPath();
  line.points.forEach((point, index) => {
    if (index === 0) ctx.moveTo(point[0], point[1]);
    else ctx.lineTo(point[0], point[1]);
  });
  ctx.stroke();
  if (line.id != null && line.points[0]) {
    ctx.fillStyle = line.color || DISTORTION_GUIDE_COLOR;
    ctx.font = "700 14px Segoe UI";
    ctx.textBaseline = "bottom";
    ctx.fillText(String(line.id), line.points[0][0] + 6, line.points[0][1] - 6);
  }
  ctx.restore();
}

function drawDistortionPreviewCurve(preview) {
  if (!preview || !Array.isArray(preview.samples) || preview.samples.length < 2) return;
  ctx.save();
  ctx.strokeStyle = "#f2cc60";
  ctx.lineWidth = 2.5;
  ctx.setLineDash([10, 8]);
  ctx.beginPath();
  preview.samples.forEach((point, index) => {
    const [x, y] = imageToCanvasPoint(point[0], point[1]);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.setLineDash([]);
  const center = preview.samples[Math.floor(preview.samples.length / 2)];
  if (center) {
    const [rawX, rawY] = imageToCanvasPoint(center[0], center[1]);
    ctx.fillStyle = "#f2cc60";
    ctx.font = "700 12px Segoe UI";
    ctx.textBaseline = "bottom";
    const metrics = ctx.measureText(preview.kind);
    const textWidth = metrics.width || 0;
    const x = Math.max(8, Math.min(cv.width - textWidth - 8, rawX - textWidth / 2));
    const y = Math.max(18, Math.min(cv.height - 8, rawY - 8));
    ctx.fillText(preview.kind, x, y);
  }
  ctx.restore();
}

function drawDistortionSupportPoints(points) {
  if (!Array.isArray(points)) return;
  points.forEach((point) => {
    const [x, y] = imageToCanvasPoint(point[0], point[1]);
    ctx.save();
    ctx.fillStyle = DISTORTION_GUIDE_COLOR;
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  });
}

function hitDistortionPointRef(ix, iy) {
  const radius = 16 / Math.max(state.viewScale, 0.01);
  let best = null;
  state.distortionPoints.forEach((point, pointIndex) => {
    if (!Array.isArray(point) || point.length < 2) return;
    const dist = Math.hypot(point[0] - ix, point[1] - iy);
    if (dist <= radius && (!best || dist < best.dist)) {
      best = { pointIndex, dist };
    }
  });
  return best;
}

function hitCourtCenterHintRef(ix, iy) {
  const point = projectedCourtMidlinePoint(activeCourtQuad());
  if (!Array.isArray(point) || point.length < 2) return null;
  const radius = 18 / Math.max(state.viewScale, 0.01);
  const dist = Math.hypot(point[0] - ix, point[1] - iy);
  return dist <= radius ? { dist } : null;
}

function drawCourtCenterHint(point) {
  if (!Array.isArray(point) || point.length < 2) return;
  const [x, y] = imageToCanvasPoint(point[0], point[1]);
  ctx.save();
  ctx.strokeStyle = "#f2cc60";
  ctx.fillStyle = "#f2cc60";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(x - 10, y);
  ctx.lineTo(x + 10, y);
  ctx.moveTo(x, y - 10);
  ctx.lineTo(x, y + 10);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(x, y, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.font = "700 12px Segoe UI";
  ctx.textBaseline = "bottom";
  ctx.fillText("court midline", x + 8, y - 8);
  ctx.restore();
}

function projectedCourtMidlinePoint(quad) {
  if (!Array.isArray(quad) || quad.length !== 4) return null;
  const homography = solveCourtHomography(courtWorldQuad(), quad);
  return projectWorldPointH([BEACH_COURT_WIDTH_M * 0.5, BEACH_COURT_LENGTH_M * 0.5], homography);
}

function hitCourtModelHandleRef(ix, iy) {
  const quad = activeCourtQuad();
  if (!quad) return null;
  const cornerRadius = 18 / Math.max(state.viewScale, 0.01);
  let best = null;
  quad.forEach((point, index) => {
    const dist = Math.hypot(point[0] - ix, point[1] - iy);
    if (dist <= cornerRadius && (!best || dist < best.dist)) {
      best = { type: "corner", index, dist };
    }
  });
  const midline = projectedCourtMidlinePoint(quad);
  if (midline) {
    const dist = Math.hypot(midline[0] - ix, midline[1] - iy);
    if (dist <= cornerRadius && (!best || dist < best.dist)) {
      best = { type: "midline", dist };
    }
  }
  return best;
}

function drawCourtModelHandles(quad) {
  if (!Array.isArray(quad) || quad.length !== 4) return;
  ctx.save();
  ctx.strokeStyle = "#ff4d4f";
  ctx.fillStyle = "#ff4d4f";
  ctx.lineWidth = 2;
  quad.forEach((point) => {
    const [x, y] = imageToCanvasPoint(point[0], point[1]);
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(x, y, 9, 0, Math.PI * 2);
    ctx.stroke();
  });
  const midline = projectedCourtMidlinePoint(quad);
  if (midline) {
    const [x, y] = imageToCanvasPoint(midline[0], midline[1]);
    ctx.strokeStyle = "#f2cc60";
    ctx.fillStyle = "#f2cc60";
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = "700 12px Segoe UI";
    ctx.textBaseline = "bottom";
    ctx.fillText("drag midline", x + 8, y - 8);
  }
  ctx.restore();
}

function lerpPoint(a, b, t) {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
  ];
}

function pointToSegmentDistance(point, a, b) {
  const px = Number(point?.[0] || 0);
  const py = Number(point?.[1] || 0);
  const ax = Number(a?.[0] || 0);
  const ay = Number(a?.[1] || 0);
  const bx = Number(b?.[0] || 0);
  const by = Number(b?.[1] || 0);
  const abx = bx - ax;
  const aby = by - ay;
  const denom = (abx * abx) + (aby * aby);
  let t = 0;
  if (denom > 1e-9) {
    t = ((px - ax) * abx + (py - ay) * aby) / denom;
    t = Math.max(0, Math.min(1, t));
  }
  const qx = ax + abx * t;
  const qy = ay + aby * t;
  return {
    dist: Math.hypot(px - qx, py - qy),
    point: [qx, qy],
    t,
  };
}

function clampCanvasPoint(x, y, pad = 14) {
  return [
    Math.max(pad, Math.min(cv.width - pad, x)),
    Math.max(pad, Math.min(cv.height - pad, y)),
  ];
}

function projectHandleToCanvasEdge(anchorCanvasPoint, targetCanvasPoint, pad = 14) {
  const minX = pad;
  const minY = pad;
  const maxX = cv.width - pad;
  const maxY = cv.height - pad;
  const ax = anchorCanvasPoint[0];
  const ay = anchorCanvasPoint[1];
  const tx = targetCanvasPoint[0];
  const ty = targetCanvasPoint[1];
  const vx = tx - ax;
  const vy = ty - ay;
  const candidates = [];
  if (Math.abs(vx) > 1e-6) {
    const tMin = (minX - ax) / vx;
    const yMin = ay + tMin * vy;
    if (tMin >= 0 && tMin <= 1 && yMin >= minY && yMin <= maxY) candidates.push([tMin, minX, yMin]);
    const tMax = (maxX - ax) / vx;
    const yMax = ay + tMax * vy;
    if (tMax >= 0 && tMax <= 1 && yMax >= minY && yMax <= maxY) candidates.push([tMax, maxX, yMax]);
  }
  if (Math.abs(vy) > 1e-6) {
    const tMin = (minY - ay) / vy;
    const xMin = ax + tMin * vx;
    if (tMin >= 0 && tMin <= 1 && xMin >= minX && xMin <= maxX) candidates.push([tMin, xMin, minY]);
    const tMax = (maxY - ay) / vy;
    const xMax = ax + tMax * vx;
    if (tMax >= 0 && tMax <= 1 && xMax >= minX && xMax <= maxX) candidates.push([tMax, xMax, maxY]);
  }
  if (!candidates.length) {
    return clampCanvasPoint(tx, ty, pad);
  }
  candidates.sort((a, b) => a[0] - b[0]);
  return [candidates[0][1], candidates[0][2]];
}

function modelHandleDisplayPoint(imagePoint, anchorImagePoint = null) {
  const [cx, cy] = imageToCanvasPoint(imagePoint[0], imagePoint[1]);
  const offscreen = cx < 0 || cy < 0 || cx > cv.width || cy > cv.height;
  let dx = cx;
  let dy = cy;
  if (offscreen) {
    if (Array.isArray(anchorImagePoint) && anchorImagePoint.length >= 2) {
      const anchorCanvas = imageToCanvasPoint(anchorImagePoint[0], anchorImagePoint[1]);
      [dx, dy] = projectHandleToCanvasEdge(anchorCanvas, [cx, cy], 14);
    } else {
      [dx, dy] = clampCanvasPoint(cx, cy);
    }
  }
  return { canvas: [dx, dy], rawCanvas: [cx, cy], offscreen };
}

function drawArrowHandle(canvasPoint, direction, color) {
  const [x, y] = canvasPoint;
  const dirLen = Math.max(1e-6, Math.hypot(direction[0], direction[1]));
  const ux = direction[0] / dirLen;
  const uy = direction[1] / dirLen;
  const px = -uy;
  const py = ux;
  const tipX = x;
  const tipY = y;
  const backX = x - ux * 14;
  const backY = y - uy * 14;
  ctx.save();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(tipX, tipY);
  ctx.lineTo(backX + px * 7, backY + py * 7);
  ctx.lineTo(backX - px * 7, backY - py * 7);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawArrowGuide(fromPoint, toPoint, color, dashed = true) {
  const dx = toPoint[0] - fromPoint[0];
  const dy = toPoint[1] - fromPoint[1];
  const len = Math.max(1e-6, Math.hypot(dx, dy));
  const ux = dx / len;
  const uy = dy / len;
  const endX = toPoint[0] - ux * 16;
  const endY = toPoint[1] - uy * 16;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.setLineDash(dashed ? [10, 6] : []);
  ctx.beginPath();
  ctx.moveTo(fromPoint[0], fromPoint[1]);
  ctx.lineTo(endX, endY);
  ctx.stroke();
  ctx.restore();
  drawArrowHandle(toPoint, [dx, dy], color);
}

function adjustCourtProjectionDepth(delta) {
  const quad = activeCourtQuad();
  if (!quad) return false;
  const [nearLeft, farLeft, farRight, nearRight] = quad.map((point) => [point[0], point[1]]);
  const leftVec = [nearLeft[0] - farLeft[0], nearLeft[1] - farLeft[1]];
  const rightVec = [nearRight[0] - farRight[0], nearRight[1] - farRight[1]];
  const leftLen = Math.hypot(leftVec[0], leftVec[1]);
  const rightLen = Math.hypot(rightVec[0], rightVec[1]);
  if (leftLen < 1e-6 || rightLen < 1e-6) return false;
  const scale = Math.max(0.72, Math.min(1.35, 1 + delta));
  const nextQuad = [
    [farLeft[0] + leftVec[0] * scale, farLeft[1] + leftVec[1] * scale],
    farLeft,
    farRight,
    [farRight[0] + rightVec[0] * scale, farRight[1] + rightVec[1] * scale],
  ];
  state.courtModelQuad = constrainCourtProjectionQuad(nextQuad);
  refreshCameraModelEstimate();
  markDirty();
  scheduleAutoSave();
  render();
  return true;
}

function courtModelGeometry() {
  const quad = activeCourtQuad();
  if (!quad) return null;
  const homography = solveCourtHomography(courtWorldQuad(), quad);
  if (!homography) return null;
  const nearLeft = projectWorldPointH([0, 0], homography);
  const farLeft = projectWorldPointH([0, BEACH_COURT_LENGTH_M], homography);
  const farRight = projectWorldPointH([BEACH_COURT_WIDTH_M, BEACH_COURT_LENGTH_M], homography);
  const nearRight = projectWorldPointH([BEACH_COURT_WIDTH_M, 0], homography);
  const centerLeft = projectWorldPointH([0, BEACH_COURT_LENGTH_M * 0.5], homography);
  const centerRight = projectWorldPointH([BEACH_COURT_WIDTH_M, BEACH_COURT_LENGTH_M * 0.5], homography);
  if (!nearLeft || !farLeft || !farRight || !nearRight || !centerLeft || !centerRight) return null;
  const centerLineWidthPx = Math.hypot(centerRight[0] - centerLeft[0], centerRight[1] - centerLeft[1]);
  const pxPerMeter = Math.max(6, centerLineWidthPx / BEACH_COURT_WIDTH_M);
  const netHeightM = Math.max(2.2, Math.min(2.5, Number(state.netHeightM || currentNetHeightMeters())));
  const netDirNorm = Math.max(1e-6, centerLineWidthPx);
  const netDir = [(centerRight[0] - centerLeft[0]) / netDirNorm, (centerRight[1] - centerLeft[1]) / netDirNorm];
  const centerAxisExtensionPx = pxPerMeter * 3;
  const centerAxisLeft = [centerLeft[0] - netDir[0] * centerAxisExtensionPx, centerLeft[1] - netDir[1] * centerAxisExtensionPx];
  const centerAxisRight = [centerRight[0] + netDir[0] * centerAxisExtensionPx, centerRight[1] + netDir[1] * centerAxisExtensionPx];
  const netHalfExtraPx = pxPerMeter * 0.2;
  const netTopYOffsetPx = pxPerMeter * netHeightM;
  const netBottomYOffsetPx = pxPerMeter * Math.max(netHeightM - NET_PANEL_HEIGHT_M, 0);
  const antennaAboveNetPx = pxPerMeter * ANTENNA_ABOVE_NET_M;
  const antennaInsetPx = pxPerMeter * 0.2;
  const netTopLeft = [centerLeft[0] - netDir[0] * netHalfExtraPx, centerLeft[1] - netDir[1] * netHalfExtraPx - netTopYOffsetPx];
  const netTopRight = [centerRight[0] + netDir[0] * netHalfExtraPx, centerRight[1] + netDir[1] * netHalfExtraPx - netTopYOffsetPx];
  const netBottomLeft = [centerLeft[0] - netDir[0] * netHalfExtraPx, centerLeft[1] - netDir[1] * netHalfExtraPx - netBottomYOffsetPx];
  const netBottomRight = [centerRight[0] + netDir[0] * netHalfExtraPx, centerRight[1] + netDir[1] * netHalfExtraPx - netBottomYOffsetPx];
  const leftNetDx = Number(state.leftNetDxPx || 0);
  const rightNetDx = Number(state.rightNetDxPx || 0);
  const movedNetTopLeft = [netTopLeft[0] + netDir[0] * leftNetDx, netTopLeft[1] + netDir[1] * leftNetDx];
  const movedNetBottomLeft = [netBottomLeft[0] + netDir[0] * leftNetDx, netBottomLeft[1] + netDir[1] * leftNetDx];
  const movedNetTopRight = [netTopRight[0] + netDir[0] * rightNetDx, netTopRight[1] + netDir[1] * rightNetDx];
  const movedNetBottomRight = [netBottomRight[0] + netDir[0] * rightNetDx, netBottomRight[1] + netDir[1] * rightNetDx];
  const leftDx = Number(state.leftAntennaDxPx || 0);
  const rightDx = Number(state.rightAntennaDxPx || 0);
  const leftAntennaBase = [netBottomLeft[0] + netDir[0] * antennaInsetPx, netBottomLeft[1] + netDir[1] * antennaInsetPx];
  const rightAntennaBase = [netBottomRight[0] - netDir[0] * antennaInsetPx, netBottomRight[1] - netDir[1] * antennaInsetPx];
  const leftAntennaBottom = [leftAntennaBase[0] + netDir[0] * leftDx, leftAntennaBase[1] + netDir[1] * leftDx];
  const rightAntennaBottom = [rightAntennaBase[0] + netDir[0] * rightDx, rightAntennaBase[1] + netDir[1] * rightDx];
  const leftAntennaTop = [leftAntennaBottom[0], netTopLeft[1] - antennaAboveNetPx];
  const rightAntennaTop = [rightAntennaBottom[0], netTopRight[1] - antennaAboveNetPx];
  const nodes = [
    { id: "nearLeft", point: nearLeft, kind: "corner" },
    { id: "farLeft", point: farLeft, kind: "corner" },
    { id: "farRight", point: farRight, kind: "corner" },
    { id: "nearRight", point: nearRight, kind: "corner" },
    { id: "centerLeft", point: centerLeft, kind: "center-joint" },
    { id: "centerRight", point: centerRight, kind: "center-joint" },
  ];
  const segments = [
    { role: "court", a: nearLeft, b: centerLeft },
    { role: "court", a: centerLeft, b: farLeft },
    { role: "court", a: farLeft, b: farRight },
    { role: "court", a: farRight, b: centerRight },
    { role: "court", a: centerRight, b: nearRight },
    { role: "court", a: nearRight, b: nearLeft },
    { role: "center", a: centerAxisLeft, b: centerAxisRight },
    { role: "net", a: movedNetTopLeft, b: movedNetTopRight },
    { role: "net", a: movedNetBottomLeft, b: movedNetBottomRight },
    { role: "net", a: movedNetTopLeft, b: movedNetBottomLeft },
    { role: "net", a: movedNetTopRight, b: movedNetBottomRight },
    { role: "antenna", a: leftAntennaBottom, b: leftAntennaTop },
    { role: "antenna", a: rightAntennaBottom, b: rightAntennaTop },
  ];
  return {
    quad: [nearLeft, farLeft, farRight, nearRight],
    homography,
    nodes,
    segments,
    centerLeft,
    centerRight,
    centerAxisLeft,
    centerAxisRight,
    netTopLeft: movedNetTopLeft,
    netTopRight: movedNetTopRight,
    netBottomLeft: movedNetBottomLeft,
    netBottomRight: movedNetBottomRight,
    leftAntennaBottom,
    rightAntennaBottom,
    leftAntennaTop,
    rightAntennaTop,
    pxPerMeter,
    netHeightM,
  };
}

function drawInteractiveModelGeometry() {
  const model = courtModelGeometry();
  if (!model) return;
  const colors = {
    court: "#ff4d4f",
    center: "#ff4d4f",
    net: "#2fbf71",
    antenna: "#49b6ff",
  };
  model.segments.forEach((segment) => {
    drawGeometryPolyline([segment.a, segment.b], colors[segment.role] || "#ffffff", false, {
      width: segment.role === "court" ? 3 : 2.5,
      dash: segment.role === "court" ? [10, 6] : [],
      pointRadius: 0,
    });
  });
  const offscreenGuideTargets = {
    nearLeft: model.centerLeft,
    nearRight: model.centerRight,
    farLeft: model.centerLeft,
    farRight: model.centerRight,
  };
  ctx.save();
  model.nodes.forEach((node) => {
    const anchor = offscreenGuideTargets[node.id] || null;
    const handle = modelHandleDisplayPoint(node.point, anchor);
    if (handle.offscreen) {
      const anchorCanvas = anchor ? imageToCanvasPoint(anchor[0], anchor[1]) : null;
      if (anchorCanvas) {
        drawArrowGuide(anchorCanvas, handle.canvas, colors.court);
      } else {
        const dir = [
          handle.canvas[0] - handle.rawCanvas[0],
          handle.canvas[1] - handle.rawCanvas[1],
        ];
        drawArrowHandle(handle.canvas, dir, colors.court);
      }
      return;
    }
    ctx.fillStyle = colors.court;
    ctx.beginPath();
    ctx.arc(handle.canvas[0], handle.canvas[1], node.kind === "corner" ? 5 : 4, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.restore();
  const drawnOffscreen = new Set();
  model.segments.forEach((segment) => {
    const dashed = segment.role === "court";
    [
      [segment.a, segment.b],
      [segment.b, segment.a],
    ].forEach(([endpoint, anchor]) => {
      const key = `${segment.role}:${Math.round(endpoint[0])},${Math.round(endpoint[1])}:${Math.round(anchor[0])},${Math.round(anchor[1])}`;
      if (drawnOffscreen.has(key)) return;
      const handle = modelHandleDisplayPoint(endpoint, anchor);
      if (!handle.offscreen) return;
      drawArrowGuide(imageToCanvasPoint(anchor[0], anchor[1]), handle.canvas, colors[segment.role] || "#ffffff", dashed);
      drawnOffscreen.add(key);
    });
  });
}

function refineCourtModelFit() {
  const model = courtModelGeometry();
  if (!model || !state.renderCanvas || !state.renderCtx) return false;
  const width = state.renderCanvas.width;
  const height = state.renderCanvas.height;
  if (!(width > 4 && height > 4)) return false;
  const img = state.renderCtx.getImageData(0, 0, width, height);
  const src = img.data;
  const initialQuad = (state.courtModelQuad || []).map((p) => [p[0], p[1]]);
  const initialNetHeight = Number(state.netHeightM || currentNetHeightMeters());
  const initialLeftNetDx = Number(state.leftNetDxPx || 0);
  const initialRightNetDx = Number(state.rightNetDxPx || 0);
  const initialLeftDx = Number(state.leftAntennaDxPx || 0);
  const initialRightDx = Number(state.rightAntennaDxPx || 0);
  const grayAt = (x, y) => {
    const ix = Math.max(1, Math.min(width - 2, Math.round(x)));
    const iy = Math.max(1, Math.min(height - 2, Math.round(y)));
    const idx = (iy * width + ix) * 4;
    return src[idx] * 0.299 + src[idx + 1] * 0.587 + src[idx + 2] * 0.114;
  };
  const lineScoreAt = (x, y, nx, ny) => {
    const c = grayAt(x, y);
    const n1L = grayAt(x + nx * 1.25, y + ny * 1.25);
    const n1R = grayAt(x - nx * 1.25, y - ny * 1.25);
    const n2L = grayAt(x + nx * 2.5, y + ny * 2.5);
    const n2R = grayAt(x - nx * 2.5, y - ny * 2.5);
    const fL = grayAt(x + nx * 5.0, y + ny * 5.0);
    const fR = grayAt(x - nx * 5.0, y - ny * 5.0);
    const narrowEdge = Math.abs(n1L - n1R) * 1.8;
    const centerStripe = Math.abs((n1L + n1R) - 2 * c) * 1.2;
    const mediumBand = Math.abs(n2L - n2R) * 0.9;
    const broadBand = Math.abs(fL - fR) * 0.8 + Math.abs((fL + fR) - (n2L + n2R)) * 0.35;
    return narrowEdge + centerStripe - mediumBand - broadBand;
  };
  const sampleSegmentScore = (aImg, bImg) => {
    const a = imageToCanvasPoint(aImg[0], aImg[1]);
    const b = imageToCanvasPoint(bImg[0], bImg[1]);
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const len = Math.max(1e-6, Math.hypot(dx, dy));
    const nx = -dy / len;
    const ny = dx / len;
    const steps = Math.max(6, Math.round(len / 18));
    const sampleScores = [];
    for (let i = 0; i <= steps; i += 1) {
      const t = i / steps;
      const x = a[0] + dx * t;
      const y = a[1] + dy * t;
      sampleScores.push(lineScoreAt(x, y, nx, ny));
    }
    const mean = sampleScores.reduce((sum, value) => sum + value, 0) / Math.max(sampleScores.length, 1);
    const variance = sampleScores.reduce((sum, value) => sum + ((value - mean) ** 2), 0) / Math.max(sampleScores.length, 1);
    const weakPenalty = sampleScores.reduce((sum, value) => sum + Math.max(0, 10 - value), 0) / Math.max(sampleScores.length, 1);
    return mean - Math.sqrt(variance) * 0.65 - weakPenalty * 0.8;
  };
  const buildScore = (quad, netHeightM, leftNetDx, rightNetDx, leftDx, rightDx) => {
    const prevQuad = state.courtModelQuad;
    const prevNetHeight = state.netHeightM;
    const prevLeftNetDx = state.leftNetDxPx;
    const prevRightNetDx = state.rightNetDxPx;
    const prevLeftDx = state.leftAntennaDxPx;
    const prevRightDx = state.rightAntennaDxPx;
    state.courtModelQuad = constrainCourtProjectionQuad(quad);
    state.netHeightM = netHeightM;
    state.leftNetDxPx = leftNetDx;
    state.rightNetDxPx = rightNetDx;
    state.leftAntennaDxPx = leftDx;
    state.rightAntennaDxPx = rightDx;
    const m = courtModelGeometry();
    state.courtModelQuad = prevQuad;
    state.netHeightM = prevNetHeight;
    state.leftNetDxPx = prevLeftNetDx;
    state.rightNetDxPx = prevRightNetDx;
    state.leftAntennaDxPx = prevLeftDx;
    state.rightAntennaDxPx = prevRightDx;
    if (!m) return -Infinity;
    let score = 0;
    m.segments.forEach((segment) => {
      if (segment.role !== "court") return;
      score += sampleSegmentScore(segment.a, segment.b) * 1.2;
    });
    const baseQuad = initialQuad.length === 4 ? initialQuad : quad;
    for (let i = 0; i < 4; i += 1) {
      score -= Math.hypot(quad[i][0] - baseQuad[i][0], quad[i][1] - baseQuad[i][1]) * 0.35;
    }
    return score;
  };
  let quad = (state.courtModelQuad || []).map((p) => [p[0], p[1]]);
  if (quad.length !== 4) return false;
  const leftNetDx = Number(state.leftNetDxPx || 0);
  const rightNetDx = Number(state.rightNetDxPx || 0);
  const leftDx = Number(state.leftAntennaDxPx || 0);
  const rightDx = Number(state.rightAntennaDxPx || 0);
  const netHeight = Number(state.netHeightM || currentNetHeightMeters());
  let bestScore = buildScore(quad, netHeight, leftNetDx, rightNetDx, leftDx, rightDx);
  const pointMoves = [2.0, 1.0, 0.5, 0.25];
  for (const stepPx of pointMoves) {
    let improved = true;
    while (improved) {
      improved = false;
      let bestLocal = { score: bestScore, quad };
      for (let i = 0; i < 4; i += 1) {
        for (const [mx, my] of [[stepPx, 0], [-stepPx, 0], [0, stepPx], [0, -stepPx]]) {
          const candQuad = quad.map((p) => [p[0], p[1]]);
          candQuad[i] = [candQuad[i][0] + mx, candQuad[i][1] + my];
          const constrained = constrainCourtProjectionQuad(candQuad);
          const s = buildScore(constrained, netHeight, leftNetDx, rightNetDx, leftDx, rightDx);
          if (s > bestLocal.score + 1e-6) bestLocal = { score: s, quad: constrained };
        }
      }
      if (bestLocal.score > bestScore + 1e-6) {
        ({ quad } = bestLocal);
        bestScore = bestLocal.score;
        improved = true;
      }
    }
  }
  state.courtModelQuad = constrainCourtProjectionQuad(quad);
  state.leftNetDxPx = leftNetDx;
  state.rightNetDxPx = rightNetDx;
  state.leftAntennaDxPx = leftDx;
  state.rightAntennaDxPx = rightDx;
  state.courtRefinedReady = true;
  refreshCameraModelEstimate();
  markDirty();
  scheduleAutoSave();
  setStatus(`Refined court fit score ${bestScore.toFixed(1)}. Check the red court overlay and Continue again if it looks right.`);
  render();
  return true;
}

function hitInteractiveModelHandle(ix, iy) {
  const model = courtModelGeometry();
  if (!model) return null;
  const radius = 16 / Math.max(state.viewScale, 0.01);
  let best = null;
  const [cx, cy] = imageToCanvasPoint(ix, iy);
  const tryHitPoint = (type, payload, point, anchorPoint = null) => {
    const handle = modelHandleDisplayPoint(point, anchorPoint);
    const dist = handle.offscreen
      ? Math.hypot(handle.canvas[0] - cx, handle.canvas[1] - cy)
      : Math.hypot(point[0] - ix, point[1] - iy);
    if (dist <= radius && (!best || best.priority > 0 || dist < best.dist)) {
      best = { type, dist, priority: 0, ...payload };
    }
  };
  const tryHitSegment = (type, payload, a, b) => {
    const hit = pointToSegmentDistance([ix, iy], a, b);
    if (hit.dist <= radius && (!best || (best.priority === 1 && hit.dist < best.dist))) {
      best = { type, dist: hit.dist, priority: 1, ...payload };
    }
  };
  model.nodes.forEach((node, index) => {
    if (node.kind === "corner") {
      const anchor = node.id === "nearLeft" || node.id === "farLeft"
        ? model.centerLeft
        : node.id === "nearRight" || node.id === "farRight"
          ? model.centerRight
          : null;
      tryHitPoint("court-corner", { cornerIndex: index }, node.point, anchor);
    }
    if (node.id === "centerLeft") tryHitPoint("center-endpoint", { endpoint: "left" }, node.point, model.centerRight);
    if (node.id === "centerRight") tryHitPoint("center-endpoint", { endpoint: "right" }, node.point, model.centerLeft);
  });
  tryHitSegment("center-line", {}, model.centerAxisLeft, model.centerAxisRight);
  tryHitSegment("court-side", { side: "left" }, model.quad[0], model.quad[1]);
  tryHitSegment("court-side", { side: "far" }, model.quad[1], model.quad[2]);
  tryHitSegment("court-side", { side: "right" }, model.quad[3], model.quad[2]);
  tryHitSegment("court-side", { side: "near" }, model.quad[0], model.quad[3]);
  tryHitSegment("net-side", { side: "left" }, model.netTopLeft, model.netBottomLeft);
  tryHitSegment("net-side", { side: "right" }, model.netTopRight, model.netBottomRight);
  tryHitSegment("net-line", {}, model.netTopLeft, model.netTopRight);
  tryHitSegment("left-antenna", {}, model.leftAntennaBottom, model.leftAntennaTop);
  tryHitSegment("right-antenna", {}, model.rightAntennaBottom, model.rightAntennaTop);
  if (!best) {
    tryHitPoint("left-antenna", {}, model.leftAntennaTop, model.leftAntennaBottom);
    tryHitPoint("right-antenna", {}, model.rightAntennaTop, model.rightAntennaBottom);
  }
  return best;
}

function modelDragAnchorPoint(hitModel, startQuad) {
  if (!hitModel || !Array.isArray(startQuad) || startQuad.length !== 4) return null;
  if (hitModel.type === "court-corner") {
    const idx = Math.max(0, Math.min(3, Number(hitModel.cornerIndex || 0)));
    return startQuad[idx];
  }
  if (hitModel.type === "court-side") {
    if (hitModel.side === "left") return [(startQuad[0][0] + startQuad[1][0]) * 0.5, (startQuad[0][1] + startQuad[1][1]) * 0.5];
    if (hitModel.side === "right") return [(startQuad[3][0] + startQuad[2][0]) * 0.5, (startQuad[3][1] + startQuad[2][1]) * 0.5];
    if (hitModel.side === "near") return [(startQuad[0][0] + startQuad[3][0]) * 0.5, (startQuad[0][1] + startQuad[3][1]) * 0.5];
    if (hitModel.side === "far") return [(startQuad[1][0] + startQuad[2][0]) * 0.5, (startQuad[1][1] + startQuad[2][1]) * 0.5];
  }
  if (hitModel.type === "center-endpoint") {
    return hitModel.endpoint === "left"
      ? [(startQuad[0][0] + startQuad[1][0]) * 0.5, (startQuad[0][1] + startQuad[1][1]) * 0.5]
      : [(startQuad[3][0] + startQuad[2][0]) * 0.5, (startQuad[3][1] + startQuad[2][1]) * 0.5];
  }
  if (hitModel.type === "center-line") {
    const left = [(startQuad[0][0] + startQuad[1][0]) * 0.5, (startQuad[0][1] + startQuad[1][1]) * 0.5];
    const right = [(startQuad[3][0] + startQuad[2][0]) * 0.5, (startQuad[3][1] + startQuad[2][1]) * 0.5];
    return [(left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5];
  }
  const model = courtModelGeometry();
  if (!model) return null;
  if (hitModel.type === "net-line") {
    return [(model.netTopLeft[0] + model.netTopRight[0]) * 0.5, (model.netTopLeft[1] + model.netTopRight[1]) * 0.5];
  }
  if (hitModel.type === "net-side") {
    return hitModel.side === "left"
      ? [(model.netTopLeft[0] + model.netBottomLeft[0]) * 0.5, (model.netTopLeft[1] + model.netBottomLeft[1]) * 0.5]
      : [(model.netTopRight[0] + model.netBottomRight[0]) * 0.5, (model.netTopRight[1] + model.netBottomRight[1]) * 0.5];
  }
  if (hitModel.type === "left-antenna") {
    return [(model.leftAntennaTop[0] + model.leftAntennaBottom[0]) * 0.5, (model.leftAntennaTop[1] + model.leftAntennaBottom[1]) * 0.5];
  }
  if (hitModel.type === "right-antenna") {
    return [(model.rightAntennaTop[0] + model.rightAntennaBottom[0]) * 0.5, (model.rightAntennaTop[1] + model.rightAntennaBottom[1]) * 0.5];
  }
  return null;
}

function sampleBilinear(srcData, width, height, x, y, out, index) {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(width - 1, x0 + 1);
  const y1 = Math.min(height - 1, y0 + 1);
  const tx = x - x0;
  const ty = y - y0;
  const i00 = (y0 * width + x0) * 4;
  const i10 = (y0 * width + x1) * 4;
  const i01 = (y1 * width + x0) * 4;
  const i11 = (y1 * width + x1) * 4;
  for (let c = 0; c < 4; c += 1) {
    const a = srcData[i00 + c] * (1 - tx) + srcData[i10 + c] * tx;
    const b = srcData[i01 + c] * (1 - tx) + srcData[i11 + c] * tx;
    out[index + c] = Math.round(a * (1 - ty) + b * ty);
  }
}

function drawRect(rect, color, dashed = false) {
  const [x1, y1] = imageToCanvasPoint(rect.x, rect.y);
  const [x2, y2] = imageToCanvasPoint(rect.x + rect.w, rect.y + rect.h);
  const x = Math.min(x1, x2);
  const y = Math.min(y1, y2);
  const w = Math.max(1, Math.abs(x2 - x1));
  const h = Math.max(1, Math.abs(y2 - y1));
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  if (dashed) ctx.setLineDash([8, 5]);
  ctx.strokeRect(x, y, w, h);
  ctx.restore();
}

function currentPeopleColor() {
  if (state.sessionType === "training") return "#a371f7";
  return state.peopleSubstep === PEOPLE_SUBSTEP.REFEREES ? "#f2cc60" : "#a371f7";
}

function currentRectPreviewColor() {
  if (state.wizardStep === STEP.PEOPLE) {
    return currentPeopleColor();
  }
  return "#ffffff";
}

function makeBallHint(ix, iy) {
  return {
    center_x: ix,
    center_y: iy,
    radius: BALL_HINT_RADIUS_PX,
    x: ix - BALL_HINT_RADIUS_PX,
    y: iy - BALL_HINT_RADIUS_PX,
    w: BALL_HINT_RADIUS_PX * 2,
    h: BALL_HINT_RADIUS_PX * 2,
  };
}

function drawBallHint(ball, color) {
  if (!ball) return;
  const [x, y] = imageToCanvasPoint(ball.center_x, ball.center_y);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(x, y, BALL_HINT_DRAW_RADIUS_PX, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x - 7, y);
  ctx.lineTo(x + 7, y);
  ctx.moveTo(x, y - 7);
  ctx.lineTo(x, y + 7);
  ctx.stroke();
  ctx.restore();
}

function drawCross(point, color) {
  const [x, y] = imageToCanvasPoint(point[0], point[1]);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x - 6, y - 6);
  ctx.lineTo(x + 6, y + 6);
  ctx.moveTo(x + 6, y - 6);
  ctx.lineTo(x - 6, y + 6);
  ctx.stroke();
  ctx.restore();
}

function shouldShowGeometry() {
  return state.wizardStep === STEP.GEOMETRY;
}

function shouldShowPeople() {
  return state.wizardStep === STEP.PEOPLE;
}

function shouldShowBall() {
  return state.wizardStep === STEP.BALL;
}

function shouldShowIgnore() {
  return state.wizardStep === STEP.IGNORE;
}

function shouldShowFocusMask() {
  return hasVisibleFocusRegion();
}

function shouldShowFocusHelpers() {
  return state.wizardStep === STEP.FOCUS;
}

function hasVisibleFocusRegion() {
  if (!Array.isArray(state.focusRegion) || state.focusRegion.length < 4) return false;
  if (state.focusRegionCanvasDefault) return false;
  return true;
}

function visiblePeopleForCurrentStep() {
  if (state.wizardStep !== STEP.PEOPLE) return [];
  if (state.sessionType === "training") {
    return state.people.filter((person) => person.role === "participant");
  }
  if (state.peopleSubstep === PEOPLE_SUBSTEP.REFEREES) {
    return state.people.filter((person) => person.role === "referee");
  }
  return state.people.filter((person) => person.role === "player");
}

function drawFocusRegion(points) {
  if (!Array.isArray(points) || points.length === 0) return;
  const displayPoints = focusRegionDisplayPoints(points);
  ctx.save();
  ctx.fillStyle = shouldShowFocusHelpers()
    ? "rgba(13, 17, 23, 0.42)"
    : "rgba(13, 17, 23, 0.28)";
  ctx.beginPath();
  ctx.rect(0, 0, cv.width, cv.height);
  displayPoints.forEach(([x, y], index) => {
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  if (points.length >= 3) ctx.closePath();
  ctx.fill("evenodd");
  ctx.strokeStyle = "#ff4fd8";
  ctx.lineWidth = shouldShowFocusHelpers() ? 2 : 1.75;
  ctx.setLineDash([12, 8]);
  ctx.beginPath();
  displayPoints.forEach(([x, y], index) => {
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  if (points.length >= 3) ctx.closePath();
  ctx.stroke();
  ctx.restore();
}

function updateFocusOverlay(points) {
  if (!focusOverlay) return;
  const show = shouldShowFocusHelpers() && Array.isArray(points) && points.length >= 4;
  focusOverlay.classList.toggle("hidden", !show);
  if (!show) return;
  const canvasRect = cv.getBoundingClientRect();
  const wrapRect = canvasWrap.getBoundingClientRect();
  const scaleX = canvasRect.width / Math.max(cv.width, 1);
  const scaleY = canvasRect.height / Math.max(cv.height, 1);
  focusRegionDisplayPoints(points).slice(0, 4).forEach(([x, y], index) => {
    const marker = focusMarkers[index];
    if (!marker) return;
    marker.style.left = `${canvasRect.left - wrapRect.left + x * scaleX}px`;
    marker.style.top = `${canvasRect.top - wrapRect.top + y * scaleY}px`;
  });
}

function render() {
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!state.image) {
    updateFocusOverlay([]);
    return;
  }
  ctx.drawImage(state.renderCanvas, 0, 0);
  if (shouldShowGeometry() && state.geometrySubstep === GEOMETRY_SUBSTEP.DISTORTION) {
    const preview = bestDistortionHelperPreview(state.distortionPoints);
    if (preview) {
      drawDistortionPreviewCurve(preview);
    }
    drawDistortionSupportPoints(state.distortionPoints);
  }
  if (shouldShowGeometry()) {
    if (state.geometrySubstep === GEOMETRY_SUBSTEP.COURT) {
      drawInteractiveModelGeometry();
      state.geometryPolylines.forEach((line, index) => {
        drawGeometryPolyline(line, "#58a6ff", index === state.currentPolylineIndex, {
          width: 1.5,
          pointRadius: 3,
        });
      });
    } else if (state.geometrySubstep === GEOMETRY_SUBSTEP.NET) {
      drawInteractiveModelGeometry();
    }
  }
  if (shouldShowGeometry() && state.geometrySubstep !== GEOMETRY_SUBSTEP.DISTORTION) {
    const currentLine = activePolyline();
    if (currentLine && currentLine.length > 0 && state.hoverPoint) {
      drawGeometryPreview(
        currentLine,
        state.hoverPoint,
        state.geometrySubstep === GEOMETRY_SUBSTEP.NET ? "#3fb950" : "#58a6ff",
      );
    }
  }
  if (shouldShowFocusMask()) {
    ensureDefaultFocusRegion();
    if (state.focusRegionCanvasDefault) {
      state.focusRegionCanvasPoints = defaultFocusRegionCanvasPoints();
    }
    drawFocusRegion(state.focusRegion);
  }
  updateFocusOverlay(state.focusRegion);

  if (shouldShowPeople()) {
    visiblePeopleForCurrentStep().forEach((person) => {
      const color = person.role === "referee" ? "#f2cc60" : "#a371f7";
      drawRect(person, color);
    });
  }
  if (shouldShowBall() && state.ball) drawBallHint(state.ball, "#3fb950");
  if (shouldShowIgnore()) {
    state.ignorePoints.forEach((point) => drawCross(point, "#f85149"));
  }

  if (state.drag?.type === "rect") drawRect(state.drag.rect, currentRectPreviewColor(), true);
}

function distortPointWithParams(x, y, width, height, params) {
  const offsetScaleX = width / Math.max(state.imageW || width || 1, 1);
  const offsetScaleY = height / Math.max(state.imageH || height || 1, 1);
  const cx = width / 2 + Number(params.cx_offset || 0) * offsetScaleX;
  const cy = height / 2 + Number(params.cy_offset || 0) * offsetScaleY;
  const norm = Math.max(width, height) / 2;
  if ((params.model || "") === "division" || Number.isFinite(params.lambda)) {
    const ux = (x - cx) / norm;
    const uy = (y - cy) / norm;
    const lambda = Number(params.lambda || 0);
    let dx = ux;
    let dy = uy;
    for (let i = 0; i < 10; i += 1) {
      const r2 = dx * dx + dy * dy;
      const factor = 1 + lambda * r2;
      dx = ux * factor;
      dy = uy * factor;
    }
    return [cx + dx * norm, cy + dy * norm];
  }
  const dx = (x - cx) / norm;
  const dy = (y - cy) / norm;
  const r2 = dx * dx + dy * dy;
  const factor = 1 + params.k1 * r2 + params.k2 * r2 * r2 + params.k3 * r2 * r2 * r2;
  return [cx + dx * factor * norm, cy + dy * factor * norm];
}

function undistortPointWithParams(x, y, width, height, params) {
  const offsetScaleX = width / Math.max(state.imageW || width || 1, 1);
  const offsetScaleY = height / Math.max(state.imageH || height || 1, 1);
  const cx = width / 2 + Number(params.cx_offset || 0) * offsetScaleX;
  const cy = height / 2 + Number(params.cy_offset || 0) * offsetScaleY;
  const norm = Math.max(width, height) / 2;
  if ((params.model || "") === "division" || Number.isFinite(params.lambda)) {
    const dx = (x - cx) / norm;
    const dy = (y - cy) / norm;
    const r2 = dx * dx + dy * dy;
    const factor = 1 + Number(params.lambda || 0) * r2;
    return [cx + (dx / factor) * norm, cy + (dy / factor) * norm];
  }
  let ux = (x - cx) / norm;
  let uy = (y - cy) / norm;
  const dx = ux;
  const dy = uy;
  for (let i = 0; i < 8; i += 1) {
    const r2 = ux * ux + uy * uy;
    const factor = 1 + params.k1 * r2 + params.k2 * r2 * r2 + params.k3 * r2 * r2 * r2;
    ux = dx / factor;
    uy = dy / factor;
  }
  return [cx + ux * norm, cy + uy * norm];
}

function combinedSourceToRotatedPoint(x, y, drawW, drawH, workW, workH, bboxW, bboxH, params) {
  const corrected = undistortPointWithParams(x, y, drawW, drawH, params);
  const localX = corrected[0] - drawW / 2;
  const localY = corrected[1] - drawH / 2;
  const angle = rotationRadians();
  return [
    bboxW / 2 + localX * Math.cos(angle) - localY * Math.sin(angle),
    bboxH / 2 + localX * Math.sin(angle) + localY * Math.cos(angle),
  ];
}

function combinedRotatedToSourcePoint(x, y, drawW, drawH, workW, workH, bboxW, bboxH, params) {
  const dx = x - bboxW / 2;
  const dy = y - bboxH / 2;
  const angle = -rotationRadians();
  const correctedDrawX = drawW / 2 + dx * Math.cos(angle) - dy * Math.sin(angle);
  const correctedDrawY = drawH / 2 + dx * Math.sin(angle) + dy * Math.cos(angle);
  return distortPointWithParams(correctedDrawX, correctedDrawY, drawW, drawH, params);
}

function refineSuggestedLine(theta, rho, points, width, height) {
  const cos = Math.cos(theta);
  const sin = Math.sin(theta);
  const dirX = -sin;
  const dirY = cos;
  const support = [];
  for (const point of points) {
    const dist = Math.abs(point.x * cos + point.y * sin - rho);
    if (dist <= 3.5) {
      const t = point.x * dirX + point.y * dirY;
      support.push({ t, mag: point.mag });
    }
  }
  if (support.length < 60) return null;
  support.sort((a, b) => a.t - b.t);
  const tMin = support[0].t;
  const tMax = support[support.length - 1].t;
  if (tMax - tMin < Math.min(width, height) * 0.28) return null;

  const span = intersectLineWithRect(cos, sin, rho, width, height);
  if (!span) return null;
  const baseX = cos * rho;
  const baseY = sin * rho;
  const stepCount = Math.max(72, Math.round((tMax - tMin) / 3));
  const searchRadius = Math.max(4, Math.round(Math.min(width, height) * 0.035));
  const samples = [];
  const offsets = [];
  const pointLookup = new Map(points.map((point) => [`${point.x},${point.y}`, point.mag]));
  for (let i = 0; i <= stepCount; i += 1) {
    const alpha = i / stepCount;
    const t = span.t1 + (span.t2 - span.t1) * alpha;
    let best = null;
    for (let offset = -searchRadius; offset <= searchRadius; offset += 1) {
      const x = baseX + dirX * t + cos * offset;
      const y = baseY + dirY * t + sin * offset;
      const xi = Math.round(x);
      const yi = Math.round(y);
      if (xi < 1 || yi < 1 || xi >= width - 1 || yi >= height - 1) continue;
      const mag = pointLookup.get(`${xi},${yi}`) || 0;
      if (!best || mag > best.mag) {
        best = { x, y, mag, offset };
      }
    }
    if (best && best.mag > 0) {
      offsets.push(best.offset);
    } else {
      offsets.push(null);
    }
  }
  const validOffsets = offsets.filter((value) => value !== null);
  if (validOffsets.length < 18) return null;
  const bend = Math.max(...validOffsets) - Math.min(...validOffsets);
  if (bend < 3.5) return null;
  const filledOffsets = interpolateOffsets(offsets);
  const smoothOffsets = smoothOffsetsPass(filledOffsets, 7);
  for (let i = 0; i <= stepCount; i += 1) {
    const alpha = i / stepCount;
    const t = span.t1 + (span.t2 - span.t1) * alpha;
    const offset = smoothOffsets[i];
    const x = baseX + dirX * t + cos * offset;
    const y = baseY + dirY * t + sin * offset;
    samples.push([x, y]);
  }
  return { points: samples, theta, rho };
}

function interpolateOffsets(offsets) {
  const values = offsets.slice();
  let lastKnown = null;
  for (let i = 0; i < values.length; i += 1) {
    if (values[i] !== null) {
      lastKnown = i;
      continue;
    }
    let nextKnown = null;
    for (let j = i + 1; j < values.length; j += 1) {
      if (values[j] !== null) {
        nextKnown = j;
        break;
      }
    }
    if (lastKnown === null && nextKnown === null) {
      values[i] = 0;
    } else if (lastKnown === null) {
      values[i] = values[nextKnown];
    } else if (nextKnown === null) {
      values[i] = values[lastKnown];
    } else {
      const alpha = (i - lastKnown) / (nextKnown - lastKnown);
      values[i] = values[lastKnown] + (values[nextKnown] - values[lastKnown]) * alpha;
    }
  }
  return values;
}

function smoothOffsetsPass(values, radius) {
  return values.map((_, index) => {
    let sum = 0;
    let weightSum = 0;
    for (let offset = -radius; offset <= radius; offset += 1) {
      const pos = index + offset;
      if (pos < 0 || pos >= values.length) continue;
      const weight = radius + 1 - Math.abs(offset);
      sum += values[pos] * weight;
      weightSum += weight;
    }
    return weightSum ? sum / weightSum : values[index];
  });
}

function intersectLineWithRect(cos, sin, rho, width, height) {
  const pts = [];
  if (Math.abs(sin) > 1e-6) {
    const yLeft = (rho - 0 * cos) / sin;
    const yRight = (rho - (width - 1) * cos) / sin;
    if (yLeft >= 0 && yLeft <= height - 1) pts.push({ x: 0, y: yLeft });
    if (yRight >= 0 && yRight <= height - 1) pts.push({ x: width - 1, y: yRight });
  }
  if (Math.abs(cos) > 1e-6) {
    const xTop = (rho - 0 * sin) / cos;
    const xBottom = (rho - (height - 1) * sin) / cos;
    if (xTop >= 0 && xTop <= width - 1) pts.push({ x: xTop, y: 0 });
    if (xBottom >= 0 && xBottom <= width - 1) pts.push({ x: xBottom, y: height - 1 });
  }
  const deduped = [];
  for (const pt of pts) {
    if (!deduped.some((existing) => Math.hypot(existing.x - pt.x, existing.y - pt.y) < 1)) {
      deduped.push(pt);
    }
  }
  if (deduped.length < 2) return null;
  let bestA = deduped[0];
  let bestB = deduped[1];
  let bestDist = 0;
  for (let i = 0; i < deduped.length; i += 1) {
    for (let j = i + 1; j < deduped.length; j += 1) {
      const dist = Math.hypot(deduped[i].x - deduped[j].x, deduped[i].y - deduped[j].y);
      if (dist > bestDist) {
        bestDist = dist;
        bestA = deduped[i];
        bestB = deduped[j];
      }
    }
  }
  const dirX = -sin;
  const dirY = cos;
  return {
    t1: bestA.x * dirX + bestA.y * dirY,
    t2: bestB.x * dirX + bestB.y * dirY,
  };
}

function fitLineStats(points) {
  const n = points.length;
  const meanX = points.reduce((sum, p) => sum + p[0], 0) / n;
  const meanY = points.reduce((sum, p) => sum + p[1], 0) / n;
  let sxx = 0;
  let syy = 0;
  let sxy = 0;
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
  let error = 0;
  for (const [x, y] of points) {
    const perp = -(x - meanX) * dirY + (y - meanY) * dirX;
    error += perp * perp;
  }
  return { error: error / n, theta, meanX, meanY, dirX, dirY };
}

function sampledLineBendStats(points) {
  if (!Array.isArray(points) || points.length < 3) {
    return { maxAbsDeviation: 0, bendEnergy: 0, absDeviations: [] };
  }
  const first = points[0];
  const last = points[points.length - 1];
  const dx = last[0] - first[0];
  const dy = last[1] - first[1];
  const denom = Math.hypot(dx, dy) || 1;
  const absDeviations = [];
  let bendEnergy = 0;
  for (let i = 1; i < points.length - 1; i += 1) {
    const px = points[i][0] - first[0];
    const py = points[i][1] - first[1];
    const signedDeviation = (px * dy - py * dx) / denom;
    const absDeviation = Math.abs(signedDeviation);
    absDeviations.push(absDeviation);
    bendEnergy += absDeviation * absDeviation;
  }
  return {
    maxAbsDeviation: absDeviations.length ? Math.max(...absDeviations) : 0,
    bendEnergy,
    absDeviations,
  };
}

function weightedLineBendStats(points) {
  if (!Array.isArray(points) || points.length < 3) {
    return {
      maxAbsDeviation: 0,
      bendEnergy: 0,
      weightedBendEnergy: 0,
      absDeviations: [],
    };
  }
  const base = sampledLineBendStats(points);
  const count = base.absDeviations.length;
  let weightedBendEnergy = 0;
  for (let i = 0; i < count; i += 1) {
    const edgeAlpha = count <= 1 ? 1 : Math.abs((i / (count - 1)) * 2 - 1);
    const weight = 1 + edgeAlpha * 1.6;
    weightedBendEnergy += base.absDeviations[i] * base.absDeviations[i] * weight;
  }
  return {
    ...base,
    weightedBendEnergy,
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

function lineResidualStats(points, line) {
  if (!Array.isArray(points) || !points.length || !line) {
    return { maxResidual: Number.POSITIVE_INFINITY, endpointResidual: Number.POSITIVE_INFINITY, residuals: [] };
  }
  const dirX = Math.cos(line.theta);
  const dirY = Math.sin(line.theta);
  const residuals = points.map(([x, y]) => (
    Math.abs(-(x - line.meanX) * dirY + (y - line.meanY) * dirX)
  ));
  const endpointResidual = residuals.length >= 2
    ? Math.max(residuals[0], residuals[residuals.length - 1])
    : (residuals[0] ?? 0);
  return {
    maxResidual: residuals.length ? Math.max(...residuals) : 0,
    endpointResidual,
    residuals,
  };
}

function sampleHelperCurve(points, samplesPerSegment = 10) {
  if (!Array.isArray(points) || points.length < 2) return Array.isArray(points) ? points.slice() : [];
  const out = [];
  for (let seg = 0; seg < points.length - 1; seg += 1) {
    const p1 = points[seg];
    const p2 = points[seg + 1];
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

function buildDistortionHelperSamples(points) {
  const orderedPoints = sortHelperPointsByProjection(points);
  return sampleHelperCurve(orderedPoints, 12);
}

function buildDistortionHelperObservations(points) {
  return sortHelperPointsByProjection(points);
}

function helperCurveLocalFrame(points) {
  if (!Array.isArray(points) || points.length < 2) return null;
  const stats = fitLineStats(points);
  if (!Number.isFinite(stats.theta)) return null;
  const dir = [Math.cos(stats.theta), Math.sin(stats.theta)];
  const normal = [-dir[1], dir[0]];
  const mid = averagePoint(points);
  const locals = points.map((point) => {
    const px = point[0] - mid[0];
    const py = point[1] - mid[1];
    return {
      x: px * dir[0] + py * dir[1],
      y: px * normal[0] + py * normal[1],
    };
  });
  return { mid, dir, normal, locals };
}

function sortHelperPointsByProjection(points) {
  const frame = helperCurveLocalFrame(points);
  if (!frame) return Array.isArray(points) ? points.slice() : [];
  return points
    .map((point) => {
      const px = point[0] - frame.mid[0];
      const py = point[1] - frame.mid[1];
      return {
        point,
        t: px * frame.dir[0] + py * frame.dir[1],
      };
    })
    .sort((a, b) => a.t - b.t)
    .map((entry) => entry.point);
}

function helperCurveSpanForImage(frame, width, height) {
  if (!frame || !Number.isFinite(width) || !Number.isFinite(height)) return null;
  const corners = [
    [0, 0],
    [width, 0],
    [width, height],
    [0, height],
  ];
  const projections = corners.map((point) => {
    const px = point[0] - frame.mid[0];
    const py = point[1] - frame.mid[1];
    return px * frame.dir[0] + py * frame.dir[1];
  });
  const minX = Math.min(...projections);
  const maxX = Math.max(...projections);
  const pad = (maxX - minX) * 0.04;
  return { minX: minX - pad, maxX: maxX + pad };
}

function solveLinear3x3(rows, rhs) {
  const a = rows.map((row, i) => [...row, rhs[i]]);
  for (let col = 0; col < 3; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < 3; row += 1) {
      if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    }
    if (Math.abs(a[pivot][col]) < 1e-9) return null;
    if (pivot !== col) {
      const tmp = a[col];
      a[col] = a[pivot];
      a[pivot] = tmp;
    }
    const scale = a[col][col];
    for (let j = col; j < 4; j += 1) a[col][j] /= scale;
    for (let row = 0; row < 3; row += 1) {
      if (row === col) continue;
      const factor = a[row][col];
      if (!factor) continue;
      for (let j = col; j < 4; j += 1) a[row][j] -= factor * a[col][j];
    }
  }
  return [a[0][3], a[1][3], a[2][3]];
}

function solveLinear2x2(a00, a01, a11, b0, b1) {
  const det = a00 * a11 - a01 * a01;
  if (Math.abs(det) < 1e-9) return null;
  return [
    (b0 * a11 - b1 * a01) / det,
    (a00 * b1 - a01 * b0) / det,
  ];
}

function distortionPreviewFromParams(points, params) {
  const helperObservations = buildDistortionHelperObservations(points);
  if (!Array.isArray(helperObservations) || helperObservations.length < 3) return null;
  const undistorted = helperObservations.map((point) => (
    undistortPointWithParams(point[0], point[1], state.imageW, state.imageH, params)
  ));
  const first = undistorted[0];
  const last = undistorted[undistorted.length - 1];
  const spanDx = last[0] - first[0];
  const spanDy = last[1] - first[1];
  const spanLen = Math.hypot(spanDx, spanDy);
  if (!(spanLen > 1e-6)) return null;
  const dirX = spanDx / spanLen;
  const dirY = spanDy / spanLen;
  const steps = Math.max(72, helperObservations.length * 16);
  const samples = [];
  for (let i = 0; i <= steps; i += 1) {
    const t = spanLen * (i / steps);
    const ux = first[0] + dirX * t;
    const uy = first[1] + dirY * t;
    samples.push(
      distortPointWithParams(ux, uy, state.imageW, state.imageH, params),
    );
  }
  return {
    kind: "camera model fit + rotation",
    samples,
  };
}

function bestDistortionHelperPreview(points) {
  if (!Array.isArray(points) || points.length < DISTORTION_PREVIEW_POINTS) return null;
  const fitted = estimateDistortionFromPoints(points);
  if (!fitted) return null;
  return distortionPreviewFromParams(points, fitted);
}

function estimateDistortionFromPoints(points) {
  if (points.length < MIN_DISTORTION_HELPER_POINTS) return null;
  const helperObservations = buildDistortionHelperObservations(points);
  if (!Array.isArray(helperObservations) || helperObservations.length < 3) return null;
  const helperSamples = buildDistortionHelperSamples(points);
  if (!Array.isArray(helperSamples) || helperSamples.length < 6) return null;
  const baselineBend = weightedLineBendStats(helperSamples);
  const baselineCurvature = lineCurvatureProfile(helperSamples);
  const lineImprovesMonotonically = (candidatePoints) => {
    const candidateBend = weightedLineBendStats(candidatePoints);
    const candidateCurvature = lineCurvatureProfile(candidatePoints);
    if (candidateBend.maxAbsDeviation > baselineBend.maxAbsDeviation + 0.02) return false;
    if (candidateBend.weightedBendEnergy > baselineBend.weightedBendEnergy + 0.08) return false;
    const edgeCount = Math.max(2, Math.floor(candidateBend.absDeviations.length * 0.18));
    for (let i = 0; i < candidateBend.absDeviations.length; i += 1) {
      const base = baselineBend.absDeviations[i] ?? baselineBend.maxAbsDeviation;
      const isEdge =
        i < edgeCount || i >= Math.max(0, candidateBend.absDeviations.length - edgeCount);
      const tolerance = isEdge ? 0.03 : 0.08;
      if (candidateBend.absDeviations[i] > base + tolerance) return false;
    }
    const curvatureEdgeCount = Math.max(2, Math.floor(candidateCurvature.length * 0.18));
    for (let i = 0; i < candidateCurvature.length; i += 1) {
      const base = baselineCurvature[i] ?? 0;
      const isEdge =
        i < curvatureEdgeCount || i >= Math.max(0, candidateCurvature.length - curvatureEdgeCount);
      const tolerance = isEdge ? 0.0005 : 0.0015;
      if (candidateCurvature[i] > base + tolerance) return false;
    }
    return true;
  };
  const edgeRadiusSamples = [0.7, 0.82, 0.92, 1.0];
  const regularizedScore = (params, stats) => {
    const lambda = Number(params.lambda || 0);
    const cxOffsetNorm = Number(params.cx_offset || 0) / Math.max(state.imageW || 1, 1);
    const cyOffsetNorm = Number(params.cy_offset || 0) / Math.max(state.imageH || 1, 1);
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
      + 0.14 * Math.abs(cxOffsetNorm)
      + 0.14 * Math.abs(cyOffsetNorm)
      + 1.2 * (cxOffsetNorm * cxOffsetNorm + cyOffsetNorm * cyOffsetNorm);
    return {
      endpointResidual: stats.endpointResidual,
      maxResidual: stats.maxResidual,
      maxAbsDeviation: stats.maxAbsDeviation,
      weightedBendEnergy: stats.weightedBendEnergy,
      bendEnergy: stats.bendEnergy,
      lineFitError: stats.error,
      edgeStretchPenalty,
      complexityPenalty,
    };
  };
  const compareObjectives = (a, b) => {
    const keys = [
      "endpointResidual",
      "maxResidual",
      "maxAbsDeviation",
      "weightedBendEnergy",
      "edgeStretchPenalty",
      "bendEnergy",
      "lineFitError",
      "complexityPenalty",
    ];
    for (const key of keys) {
      const delta = Number(a?.[key] ?? Number.POSITIVE_INFINITY) - Number(b?.[key] ?? Number.POSITIVE_INFINITY);
      if (Math.abs(delta) > 1e-9) return delta < 0 ? -1 : 1;
    }
    return 0;
  };
  const score = (params) => {
    const undistortedSamples = helperSamples.map((point) => (
      undistortPointWithParams(point[0], point[1], state.imageW, state.imageH, params)
    ));
    if (!lineImprovesMonotonically(undistortedSamples)) {
      return {
        error: Number.POSITIVE_INFINITY,
        theta: 0,
        undistorted: undistortedSamples,
        objective: {
          endpointResidual: Number.POSITIVE_INFINITY,
          maxResidual: Number.POSITIVE_INFINITY,
          maxAbsDeviation: Number.POSITIVE_INFINITY,
          weightedBendEnergy: Number.POSITIVE_INFINITY,
          bendEnergy: Number.POSITIVE_INFINITY,
          lineFitError: Number.POSITIVE_INFINITY,
          edgeStretchPenalty: Number.POSITIVE_INFINITY,
          complexityPenalty: Number.POSITIVE_INFINITY,
        },
      };
    }
    const undistortedObservations = helperObservations.map((point) => (
      undistortPointWithParams(point[0], point[1], state.imageW, state.imageH, params)
    ));
    const stats = fitLineStats(undistortedObservations);
    const residualStats = lineResidualStats(undistortedObservations, stats);
    if (residualStats.endpointResidual > 1.5 || residualStats.maxResidual > 2.0) {
      return {
        error: Number.POSITIVE_INFINITY,
        theta: 0,
        undistorted: undistortedSamples,
        objective: {
          endpointResidual: Number.POSITIVE_INFINITY,
          maxResidual: Number.POSITIVE_INFINITY,
          maxAbsDeviation: Number.POSITIVE_INFINITY,
          weightedBendEnergy: Number.POSITIVE_INFINITY,
          bendEnergy: Number.POSITIVE_INFINITY,
          lineFitError: Number.POSITIVE_INFINITY,
          edgeStretchPenalty: Number.POSITIVE_INFINITY,
          complexityPenalty: Number.POSITIVE_INFINITY,
        },
      };
    }
    const bendStats = weightedLineBendStats(undistortedSamples);
    const mergedStats = { ...stats, ...residualStats, ...bendStats, undistorted: undistortedSamples };
    return { ...mergedStats, objective: regularizedScore(params, mergedStats) };
  };
  const clampParams = (params) => ({
    model: "division",
    lambda: Math.max(-0.28, Math.min(0.28, Number(params.lambda || 0))),
    // Allow only a small center adjustment; one helper line cannot support large shifts.
    cx_offset: Math.max(-0.06 * state.imageW, Math.min(0.06 * state.imageW, Number(params.cx_offset || 0))),
    cy_offset: Math.max(-0.06 * state.imageH, Math.min(0.06 * state.imageH, Number(params.cy_offset || 0))),
    k1: 0,
    k2: 0,
    k3: 0,
  });
  let current = clampParams({ lambda: 0, cx_offset: 0, cy_offset: 0 });
  let currentScore = score(current).objective;
  const coarseLambdaGrid = [-0.26, -0.2, -0.15, -0.1, -0.06, -0.03, 0, 0.03, 0.06, 0.1, 0.15, 0.2, 0.26];
  const coarseCxGrid = [-0.04, -0.02, 0, 0.02, 0.04].map((step) => step * state.imageW);
  const coarseCyGrid = [-0.04, -0.02, 0, 0.02, 0.04].map((step) => step * state.imageH);
  for (const lambda of coarseLambdaGrid) {
    for (const cxOffset of coarseCxGrid) {
      for (const cyOffset of coarseCyGrid) {
        const candidate = clampParams({ lambda, cx_offset: cxOffset, cy_offset: cyOffset });
        const candidateScore = score(candidate).objective;
        if (compareObjectives(candidateScore, currentScore) < 0) {
          current = candidate;
          currentScore = candidateScore;
        }
      }
    }
  }
  const lambdaSteps = [0.08, 0.03, 0.012, 0.005, 0.002];
  const cxSteps = [0.02, 0.01, 0.005, 0.0025].map((step) => step * state.imageW);
  const cySteps = [0.02, 0.01, 0.005, 0.0025].map((step) => step * state.imageH);
  const stepSchedule = Math.max(lambdaSteps.length, cxSteps.length, cySteps.length);
  for (let idx = 0; idx < stepSchedule; idx += 1) {
    const lambdaStep = lambdaSteps[Math.min(idx, lambdaSteps.length - 1)];
    const cxStep = cxSteps[Math.min(idx, cxSteps.length - 1)];
    const cyStep = cySteps[Math.min(idx, cySteps.length - 1)];
    let improved = true;
    while (improved) {
      improved = false;
      for (const [key, step] of [["lambda", lambdaStep], ["cx_offset", cxStep], ["cy_offset", cyStep]]) {
        for (const dir of [-1, 1]) {
          const candidate = clampParams({ ...current, [key]: current[key] + dir * step });
          const candidateScore = score(candidate).objective;
          if (compareObjectives(candidateScore, currentScore) < 0) {
            current = candidate;
            currentScore = candidateScore;
            improved = true;
          }
        }
      }
    }
  }
  const finalStats = score(current);
  let rotationDeg = -(finalStats.theta * 180 / Math.PI);
  while (rotationDeg <= -90) rotationDeg += 180;
  while (rotationDeg > 90) rotationDeg -= 180;
  return {
    ...current,
    rotation_deg: rotationDeg,
    diagnostics: {
      before_max_abs_deviation: baselineBend.maxAbsDeviation,
      before_bend_energy: baselineBend.bendEnergy,
      before_weighted_bend_energy: baselineBend.weightedBendEnergy,
      after_max_abs_deviation: finalStats.maxAbsDeviation,
      after_bend_energy: finalStats.bendEnergy,
      after_weighted_bend_energy: finalStats.weightedBendEnergy,
      objective: finalStats.objective,
    },
  };
}

function clampRect(rect, options = {}) {
  const { allowOutside = false, square = false } = options;
  let { x, y, w, h } = rect;
  if (square) {
    const size = Math.max(Math.abs(w), Math.abs(h), 1);
    w = w < 0 ? -size : size;
    h = h < 0 ? -size : size;
  }
  if (w < 0) {
    x += w;
    w *= -1;
  }
  if (h < 0) {
    y += h;
    h *= -1;
  }
  if (allowOutside) {
    return { x, y, w: Math.max(1, w), h: Math.max(1, h) };
  }
  x = Math.max(0, Math.min(x, state.imageW));
  y = Math.max(0, Math.min(y, state.imageH));
  w = Math.max(1, Math.min(w, state.imageW - x));
  h = Math.max(1, Math.min(h, state.imageH - y));
  return { x, y, w, h };
}

function aspectLockedRect(startX, startY, endX, endY, aspect) {
  const dx = endX - startX;
  const dy = endY - startY;
  const signX = dx < 0 ? -1 : 1;
  const signY = dy < 0 ? -1 : 1;
  const absDx = Math.abs(dx);
  const absDy = Math.abs(dy);
  if (absDx === 0 && absDy === 0) {
    return { x: startX, y: startY, w: 1, h: 1 };
  }
  let width = absDx;
  let height = absDy;
  if (height === 0 || width / Math.max(height, 1e-6) > aspect) {
    height = width / aspect;
  } else {
    width = height * aspect;
  }
  return {
    x: startX,
    y: startY,
    w: signX * width,
    h: signY * height,
  };
}

function onPointerDown(ev) {
  ev.preventDefault();
  if (!state.image) return;
  if (ev.button !== 0) return;
  const [canvasX, canvasY] = clientToCanvasPixels(ev.clientX, ev.clientY);
  const allowOutsideStart =
    state.wizardStep === STEP.FOCUS ||
    state.wizardStep === STEP.BALL;
  if (!allowOutsideStart && !isCanvasPixelInside(canvasX, canvasY)) return;
  const [ix, iy] = canvasToImagePoint(ev.clientX, ev.clientY);

  if (state.wizardStep === STEP.GEOMETRY) {
      if (state.geometrySubstep === GEOMETRY_SUBSTEP.DISTORTION) {
        const hitPoint = hitDistortionPointRef(ix, iy);
        pushHistory();
        if (hitPoint) {
          state.drag = {
            type: "distortion-point",
            pointIndex: hitPoint.pointIndex,
          };
          render();
          return;
        }
        state.distortionPoints.push([ix, iy]);
        syncUi();
        scheduleAutoSave();
        render();
      return;
    }
    if (state.geometrySubstep === GEOMETRY_SUBSTEP.DISTORTION) {
      return;
    }
    if (state.geometrySubstep === GEOMETRY_SUBSTEP.COURT || state.geometrySubstep === GEOMETRY_SUBSTEP.NET) {
      ensureCourtModelQuad();
      pushHistory();
      let hitModel = hitInteractiveModelHandle(ix, iy);
      if (hitModel?.type === "court-side") {
        const sideCorners = { left: [0, 1], far: [1, 2], right: [3, 2], near: [0, 3] }[hitModel.side];
        const quad = activeCourtQuad();
        const offscreen = sideCorners?.filter((index) => {
          const point = imageToCanvasPoint(quad[index][0], quad[index][1]);
          return !isCanvasPixelInside(point[0], point[1]);
        }) || [];
        if (offscreen.length === 1) hitModel = { type: "court-corner", cornerIndex: offscreen[0] };
      }
      if (hitModel) {
        const startQuad = activeCourtQuad();
        state.drag = {
          type: hitModel.type,
          side: hitModel.side || null,
          cornerIndex: Number.isInteger(hitModel.cornerIndex) ? hitModel.cornerIndex : null,
          endpoint: hitModel.endpoint || null,
          startX: ix,
          startY: iy,
          startCanvasX: canvasX,
          startCanvasY: canvasY,
          startQuad: startQuad,
          startCenterAxisPoints: clonePointArray(state.courtCenterAxisPoints),
          startCenterLineT: state.centerLineT,
          startNetHeightM: state.netHeightM,
          startLeftNetDxPx: state.leftNetDxPx,
          startRightNetDxPx: state.rightNetDxPx,
          startLeftAntennaDxPx: state.leftAntennaDxPx,
          startRightAntennaDxPx: state.rightAntennaDxPx,
          anchorPoint: modelDragAnchorPoint(hitModel, startQuad),
        };
        syncUi();
        render();
        return;
      }
      render();
      return;
    }
  }

  if (state.wizardStep === STEP.FOCUS) {
    ensureDefaultFocusRegion();
    pushHistory();
    ensureEditableFocusRegionPoints();
    const hitCorner = hitCanvasDefaultFocusCorner(ev.clientX, ev.clientY);
    if (hitCorner) {
      state.drag = {
        type: "focus-corner",
        pointIndex: hitCorner.pointIndex,
      };
      render();
      return;
    }
    syncUi();
    scheduleAutoSave();
    render();
    return;
  }

  if (state.wizardStep === STEP.BALL) {
    pushHistory();
    state.ball = makeBallHint(ix, iy);
    state.ballZoomRect = null;
    syncUi();
    scheduleAutoSave();
    render();
    return;
  }

  if (state.wizardStep === STEP.PEOPLE) {
    state.drag = {
      type: "rect",
      startX: ix,
      startY: iy,
      rect: { x: ix, y: iy, w: 1, h: 1 },
    };
    render();
    return;
  }

  if (state.wizardStep === STEP.IGNORE) {
    pushHistory();
    state.ignorePoints.push([ix, iy]);
    syncUi();
    scheduleAutoSave();
    render();
  }
}

function onPointerMove(ev) {
  if (state.drag?.type === "focus-corner") {
    ev.preventDefault();
  }
  const [ix, iy] = canvasToImagePoint(ev.clientX, ev.clientY);
  if (
    state.image &&
    state.wizardStep === STEP.GEOMETRY &&
    state.geometrySubstep !== GEOMETRY_SUBSTEP.DISTORTION &&
    !state.drag
  ) {
    state.hoverPoint = [ix, iy];
    render();
  }
  if (!state.image || !state.drag) return;
  if (state.drag.type === "distortion-point") {
    if (state.distortionPoints[state.drag.pointIndex]) {
      state.distortionPoints[state.drag.pointIndex] = [ix, iy];
    }
    render();
    return;
  }
  if (state.drag.type === "geometry-point") {
    const line = currentGeometryCollection()[state.drag.lineIndex];
    if (line && line[state.drag.pointIndex]) {
      line[state.drag.pointIndex] = [ix, iy];
    }
    if (state.geometrySubstep === GEOMETRY_SUBSTEP.COURT) refreshCameraModelEstimate();
    render();
    return;
  }
  if (["court-side", "court-corner", "center-endpoint", "center-line", "net-line", "net-side", "left-antenna", "right-antenna"].includes(state.drag.type)) {
    const startQuad = Array.isArray(state.drag.startQuad) ? state.drag.startQuad.map((point) => [point[0], point[1]]) : activeCourtQuad();
    if (!startQuad) return;
    const [canvasX, canvasY] = clientToCanvasPixels(ev.clientX, ev.clientY);
    const deltaCanvasX = canvasX - Number(state.drag.startCanvasX || 0);
    const deltaCanvasY = canvasY - Number(state.drag.startCanvasY || 0);
    const [dx, dy] = canvasDeltaToImageDelta(state.drag.anchorPoint, deltaCanvasX, deltaCanvasY);
    if (state.drag.type === "court-corner") {
      const idx = Math.max(0, Math.min(3, Number(state.drag.cornerIndex || 0)));
      const targets = {};
      targets[idx] = [startQuad[idx][0] + dx, startQuad[idx][1] + dy];
      const lockedAxis = state.courtCenterAxisLocked ? state.drag.startCenterAxisPoints : null;
      const quad = fitCourtProjection(startQuad, targets, lockedAxis);
      if (quad && setCourtModelQuadIfUsable(quad) && state.courtCenterAxisLocked) {
        state.courtCenterAxisPoints = centerAxisFromQuad(quad);
      }
      state.courtRefinedReady = false;
    } else if (state.drag.type === "court-side") {
      const quad = startQuad.map((point) => [point[0], point[1]]);
      if (state.drag.side === "left") {
        quad[0] = [quad[0][0] + dx, quad[0][1] + dy];
        quad[1] = [quad[1][0] + dx, quad[1][1] + dy];
      } else if (state.drag.side === "right") {
        quad[3] = [quad[3][0] + dx, quad[3][1] + dy];
        quad[2] = [quad[2][0] + dx, quad[2][1] + dy];
      } else if (state.drag.side === "near") {
        quad[0] = [quad[0][0] + dx, quad[0][1] + dy];
        quad[3] = [quad[3][0] + dx, quad[3][1] + dy];
      } else if (state.drag.side === "far") {
        quad[1] = [quad[1][0] + dx, quad[1][1] + dy];
        quad[2] = [quad[2][0] + dx, quad[2][1] + dy];
      }
      setCourtModelQuadIfUsable(quad);
      state.courtRefinedReady = false;
    } else if (state.drag.type === "center-endpoint") {
      const startH = solveCourtHomography(courtWorldQuad(), startQuad);
      const axisTargets = state.drag.startCenterAxisPoints || [
        projectWorldPointH([0, BEACH_COURT_LENGTH_M * 0.5], startH),
        projectWorldPointH([BEACH_COURT_WIDTH_M, BEACH_COURT_LENGTH_M * 0.5], startH),
      ];
      const endpointIndex = state.drag.endpoint === "left" ? 0 : 1;
      axisTargets[endpointIndex] = [axisTargets[endpointIndex][0] + dx, axisTargets[endpointIndex][1] + dy];
      const quad = fitCourtProjection(startQuad, {}, axisTargets);
      if (quad) {
        setCourtModelQuadIfUsable(quad);
        state.courtCenterAxisPoints = centerAxisFromQuad(quad);
        state.courtCenterAxisLocked = true;
      }
      state.courtRefinedReady = false;
    } else if (state.drag.type === "center-line") {
      const startH = solveCourtHomography(courtWorldQuad(), startQuad);
      const startAxis = state.drag.startCenterAxisPoints || [
        projectWorldPointH([0, BEACH_COURT_LENGTH_M * 0.5], startH),
        projectWorldPointH([BEACH_COURT_WIDTH_M, BEACH_COURT_LENGTH_M * 0.5], startH),
      ];
      const axisTargets = startAxis.map((point) => [point[0] + dx, point[1] + dy]);
      const quad = fitCourtProjection(startQuad, {}, axisTargets);
      if (quad) {
        setCourtModelQuadIfUsable(quad);
        state.courtCenterAxisPoints = centerAxisFromQuad(quad);
        state.courtCenterAxisLocked = true;
      }
      state.courtRefinedReady = false;
    } else if (state.drag.type === "net-line") {
      const model = courtModelGeometry();
      const pxPerMeter = Math.max(1, Number(model?.pxPerMeter || 1));
      state.netHeightM = Math.max(2.2, Math.min(2.5, Number(state.drag.startNetHeightM || currentNetHeightMeters()) - (dy / pxPerMeter)));
      state.courtRefinedReady = false;
    } else if (state.drag.type === "net-side") {
      if (state.drag.side === "left") {
        state.leftNetDxPx = Number(state.drag.startLeftNetDxPx || 0) + dx;
      } else {
        state.rightNetDxPx = Number(state.drag.startRightNetDxPx || 0) + dx;
      }
      state.courtRefinedReady = false;
    } else if (state.drag.type === "left-antenna") {
      state.leftAntennaDxPx = Number(state.drag.startLeftAntennaDxPx || 0) + dx;
      state.courtRefinedReady = false;
    } else if (state.drag.type === "right-antenna") {
      state.rightAntennaDxPx = Number(state.drag.startRightAntennaDxPx || 0) + dx;
      state.courtRefinedReady = false;
    }
    refreshCameraModelEstimate();
    render();
    return;
  }
  if (state.drag.type === "focus-corner") {
    const [canvasX, canvasY] = clientToCanvasPixels(ev.clientX, ev.clientY);
    const dragX = Math.max(
      -FOCUS_CORNER_DRAG_MARGIN_PX,
      Math.min(canvasX, cv.width + FOCUS_CORNER_DRAG_MARGIN_PX),
    );
    const dragY = Math.max(
      -FOCUS_CORNER_DRAG_MARGIN_PX,
      Math.min(canvasY, cv.height + FOCUS_CORNER_DRAG_MARGIN_PX),
    );
    ensureEditableFocusRegionPoints();
    if (state.focusRegionCanvasPoints?.[state.drag.pointIndex]) {
      state.focusRegionCanvasPoints[state.drag.pointIndex] = [dragX, dragY];
    }
    commitFocusRegionCanvasPoints();
    render();
    return;
  }
  state.drag.rect = clampRect(
    {
      x: state.drag.startX,
      y: state.drag.startY,
      w: ix - state.drag.startX,
      h: iy - state.drag.startY,
    },
    {
      allowOutside: false,
      square: false,
    },
  );
  render();
}

function onPointerUp() {
  if (!state.drag) return;
  if (state.drag.type === "distortion-point") {
    state.drag = null;
    syncUi();
    scheduleAutoSave();
    render();
    return;
  }
  if (state.drag.type === "geometry-point") {
    state.drag = null;
    if (state.geometrySubstep === GEOMETRY_SUBSTEP.COURT) refreshCameraModelEstimate();
    syncUi();
    scheduleAutoSave();
    render();
    return;
  }
  if (["court-side", "court-corner", "center-endpoint", "center-line", "net-line", "net-side", "left-antenna", "right-antenna"].includes(state.drag.type)) {
    state.drag = null;
    refreshCameraModelEstimate();
    syncUi();
    scheduleAutoSave();
    render();
    return;
  }
  if (state.drag.type === "focus-corner") {
    state.drag = null;
    syncUi();
    scheduleAutoSave();
    render();
    return;
  }
  const rect = clampRect(
    state.drag.rect,
    {
      allowOutside: false,
      square: false,
    },
  );
  state.drag = null;
  if (rect.w < 3 || rect.h < 3) {
    render();
    return;
  }
  pushHistory();
  if (state.wizardStep === STEP.PEOPLE) {
    state.people.push({ ...rect, role: currentPersonRole() });
  }
  syncUi();
  scheduleAutoSave();
  render();
}

function onPointerLeave() {
  if (state.hoverPoint) {
    state.hoverPoint = null;
    render();
  }
}

function skipCurrentLine() {
  if (state.wizardStep !== STEP.GEOMETRY || state.geometrySubstep === GEOMETRY_SUBSTEP.DISTORTION) return;
  pushHistory();
  if (state.currentPolylineIndex != null) {
    const lines = currentGeometryCollection();
    const line = lines[state.currentPolylineIndex];
    if (Array.isArray(line) && line.length < 2) {
      lines.splice(state.currentPolylineIndex, 1);
    }
  }
  state.currentPolylineIndex = null;
  syncUi();
  scheduleAutoSave();
  render();
}

function skipCurrentStep() {
  if (state.wizardStep === STEP.GEOMETRY) {
    if (state.geometrySubstep === GEOMETRY_SUBSTEP.DISTORTION) {
      state.distortionPoints = [];
      ensureDefaultFocusRegion();
      state.wizardStep = STEP.FOCUS;
      syncUi();
      render();
      return;
    }
    skipCurrentLine();
    return;
  }
  if (state.wizardStep === STEP.PEOPLE) {
    goContinueStep();
    return;
  }
  if (state.wizardStep === STEP.BALL) {
    state.ball = null;
    state.ballZoomRect = null;
    scheduleAutoSave();
    goContinueStep();
    return;
  }
  if (state.wizardStep === STEP.IGNORE) {
    state.ignorePoints = [];
    syncUi();
    scheduleAutoSave();
    render();
    return;
  }
  if (state.wizardStep === STEP.FOCUS) {
    state.focusRegion = defaultFocusRegion();
    syncUi();
    scheduleAutoSave();
    render();
    return;
  }
  if (state.wizardStep === STEP.SESSION) {
    if (state.sessionSubstep === SESSION_SUBSTEP.GENDER) {
      state.sessionSubstep = SESSION_SUBSTEP.BALL_IN_PLAY;
      syncUi();
      render();
      return;
    }
    if (state.sessionSubstep === SESSION_SUBSTEP.BALL_IN_PLAY) {
      state.wizardStep = STEP.GEOMETRY;
      state.geometrySubstep = GEOMETRY_SUBSTEP.DISTORTION;
      syncUi();
      render();
      return;
    }
    if (state.sessionSubstep === SESSION_SUBSTEP.BALL_VISIBLE) {
      state.wizardStep = STEP.GEOMETRY;
      state.geometrySubstep = GEOMETRY_SUBSTEP.DISTORTION;
      syncUi();
      render();
    }
  }
}

function onCanvasContextMenu(ev) {
  if (state.drag) {
    ev.preventDefault();
    state.drag = null;
    render();
    return;
  }
  if (state.wizardStep !== STEP.GEOMETRY || state.geometrySubstep === GEOMETRY_SUBSTEP.DISTORTION) return;
  ev.preventDefault();
  skipCurrentLine();
}

function normalizeBallForSave() {
  if (!state.ball) return null;
  return {
    center_x: state.ball.center_x,
    center_y: state.ball.center_y,
    radius: state.ball.radius,
    x: state.ball.x,
    y: state.ball.y,
    w: state.ball.w,
    h: state.ball.h,
  };
}

async function saveLabel(complete) {
  if (!state.currentName) return null;
  if (
    (state.wizardStep === STEP.FOCUS || Array.isArray(state.focusRegionCanvasPoints))
    && !state.focusRegionCanvasDefault
  ) {
    commitFocusRegionCanvasPoints();
  }
  refreshCameraModelEstimate();
  const serializedFocusRegion = state.focusRegionCanvasDefault
    ? []
    : state.focusRegion.filter((point) => Array.isArray(point) && point.length >= 2);
  const payload = {
    image: state.currentName,
    session_type: state.sessionType,
    gender_category: state.genderCategory,
    ball_in_play: state.ballInPlay,
    ball_visible: state.ballVisible,
    wizard_step: state.wizardStep,
    session_substep: state.sessionSubstep,
    geometry_substep: state.geometrySubstep,
    people_substep: state.peopleSubstep,
      distortion_k: Number(state.distortionParams.k1 || 0),
      distortion_params: {
        model: state.distortionParams.model || "division",
        lambda: Number(state.distortionParams.lambda || 0),
        cx_offset: Number(state.distortionParams.cx_offset || 0),
        cy_offset: Number(state.distortionParams.cy_offset || 0),
        k1: Number(state.distortionParams.k1 || 0),
        k2: Number(state.distortionParams.k2 || 0),
        k3: Number(state.distortionParams.k3 || 0),
      rotation_deg: Number(state.distortionParams.rotation_deg || 0),
    },
    distortion_helper_points: state.distortionHelperPoints,
    complete: !!complete,
    geometry_polylines: state.geometryPolylines.filter((line) => Array.isArray(line) && line.length > 0),
    net_points: state.netPolylines
      .flatMap((line) => (Array.isArray(line) ? line : []))
      .filter((point) => Array.isArray(point) && point.length >= 2),
    net_polylines: state.netPolylines.filter((line) => Array.isArray(line) && line.length > 0),
    focus_region: serializedFocusRegion,
    camera_model: state.cameraModel,
    people: state.people,
    ball: normalizeBallForSave(),
    ignore_points: state.ignorePoints,
  };
  const res = await apiFetch(`/api/v2/label?pass=${PASS_N}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  markClean();
  syncUi();
  setStatus(complete ? "Saved and completed." : "Draft saved.");
  return data;
}

function scheduleAutoSave() {
  if (!state.currentName || state.suspendAutoSave) return;
  if (state.autoSaveTimer) clearTimeout(state.autoSaveTimer);
  state.autoSaveTimer = setTimeout(async () => {
    state.autoSaveTimer = null;
    if (!state.currentName || state.suspendAutoSave) return;
    try {
      await saveLabel(false);
    } catch (_err) {
      setStatus("Draft save failed.");
    }
  }, 700);
}

async function skipImage() {
  if (!state.currentName) return;
  const res = await apiFetch(
    `/api/v2/skip-image?image=${encodeURIComponent(state.currentName)}&pass=${PASS_N}`,
    { method: "POST" },
  );
  const data = await res.json();
  if (data.next_image) {
    rememberCurrentImage(data.next_image);
    await loadImage(data.next_image);
  } else {
    rememberCurrentImage(null);
    setStatus("No more images in this pass.");
  }
}

function applyRecord(record) {
  state.sessionType = record.session_type || record.inferred?.session_type || "training";
  state.peopleSubstep = state.sessionType === "training" ? PEOPLE_SUBSTEP.PARTICIPANTS : PEOPLE_SUBSTEP.PLAYERS;
  state.genderCategory = record.gender_category || null;
  state.ballInPlay = typeof record.ball_in_play === "boolean" ? record.ball_in_play : null;
  state.ballVisible = typeof record.ball_visible === "boolean" ? record.ball_visible : null;
  state.sessionSubstep = !state.genderCategory
    ? SESSION_SUBSTEP.GENDER
    : state.ballInPlay === null
      ? SESSION_SUBSTEP.BALL_IN_PLAY
      : state.ballVisible === null
        ? SESSION_SUBSTEP.BALL_VISIBLE
        : SESSION_SUBSTEP.GENDER;
  state.distortionParams = {
    model: record.distortion_params?.model || (Number(record.distortion_params?.lambda ?? NaN) === Number(record.distortion_params?.lambda ?? NaN) ? "division" : "polynomial"),
    lambda: Number(record.distortion_params?.lambda ?? 0),
    cx_offset: Number(record.distortion_params?.cx_offset ?? 0),
    cy_offset: Number(record.distortion_params?.cy_offset ?? 0),
    k1: Number(record.distortion_params?.k1 ?? record.distortion_k ?? 0),
    k2: Number(record.distortion_params?.k2 ?? 0),
    k3: Number(record.distortion_params?.k3 ?? 0),
    rotation_deg: Number(record.distortion_params?.rotation_deg ?? 0),
  };
  state.geometryPolylines = Array.isArray(record.geometry_polylines)
    ? record.geometry_polylines.map((line) => (
      Array.isArray(line)
        ? line
          .filter((point) => Array.isArray(point) && point.length >= 2)
          .map((point) => [Number(point[0]), Number(point[1])])
        : []
    )).filter((line) => line.length > 0)
    : [];
  const loadedNetPoints = Array.isArray(record.net_points)
    ? record.net_points
      .filter((point) => Array.isArray(point) && point.length >= 2)
      .map((point) => [Number(point[0]), Number(point[1])])
    : [];
  state.netPolylines = loadedNetPoints.length > 0
    ? [loadedNetPoints]
    : Array.isArray(record.net_polylines)
      ? record.net_polylines.map((line) => (
        Array.isArray(line)
          ? line
            .filter((point) => Array.isArray(point) && point.length >= 2)
            .map((point) => [Number(point[0]), Number(point[1])])
          : []
      )).filter((line) => line.length > 0)
      : [];
  state.currentPolylineIndex = null;
  state.ballZoomRect = null;
  state.focusRegion = Array.isArray(record.focus_region)
    ? record.focus_region
      .filter((point) => Array.isArray(point) && point.length >= 2)
      .map((point) => [Number(point[0]), Number(point[1])])
    : [];
  state.cameraModel = record.camera_model || null;
  state.courtModelQuad = Array.isArray(record.camera_model?.court_model_quad) && record.camera_model.court_model_quad.length === 4
    ? record.camera_model.court_model_quad.map((point) => [Number(point[0]), Number(point[1])])
    : null;
  state.courtCenterAxisPoints = Array.isArray(record.camera_model?.center_axis_points) && record.camera_model.center_axis_points.length === 2
    ? record.camera_model.center_axis_points.map((point) => [Number(point[0]), Number(point[1])])
    : null;
  state.courtCenterAxisLocked = Boolean(record.camera_model?.center_axis_locked && state.courtCenterAxisPoints);
  state.centerLineT = Number(record.camera_model?.center_line_t ?? 0.5);
  state.netHeightM = Number(record.camera_model?.net_height_m ?? currentNetHeightMeters());
  state.leftNetDxPx = Number(record.camera_model?.left_net_dx_px ?? 0);
  state.rightNetDxPx = Number(record.camera_model?.right_net_dx_px ?? 0);
  const savedModelVersion = String(record.camera_model?.version || "");
  const useSavedAntennaOffsets = savedModelVersion === "camera-editor-v2";
  state.leftAntennaDxPx = useSavedAntennaOffsets ? Number(record.camera_model?.left_antenna_dx_px ?? 0) : 0;
  state.rightAntennaDxPx = useSavedAntennaOffsets ? Number(record.camera_model?.right_antenna_dx_px ?? 0) : 0;
  state.focusRegionCanvasDefault = false;
  state.focusRegionCanvasPoints = null;
  state.people = Array.isArray(record.people) ? record.people : [];
  state.ball = record.ball ? { ...record.ball } : null;
  state.ignorePoints = Array.isArray(record.ignore_points) ? record.ignore_points : [];
  state.inferred = record.inferred || null;
  state.wizardStep = STEP.SESSION;
  state.geometrySubstep = GEOMETRY_SUBSTEP.DISTORTION;
  applyWizardProgress({
    wizardStep: record.wizard_step,
    sessionSubstep: record.session_substep,
    geometrySubstep: record.geometry_substep,
    peopleSubstep: record.people_substep,
  });
  if (state.wizardStep === STEP.SESSION && !record.wizard_step) {
    applyWizardProgress(inferWizardProgressFromState());
  }
  state.history = [];
  state.distortionPoints = [];
  state.distortionHelperPoints = Array.isArray(record.distortion_helper_points)
    ? record.distortion_helper_points
        .filter((point) => Array.isArray(point) && point.length >= 2)
        .map((point) => [Number(point[0]), Number(point[1])])
    : [];
  const shouldUseCarryForward = !hasMeaningfulCourtModel(record)
    && !hasMeaningfulDistortionParams(record.distortion_params)
    && state.geometryPolylines.length === 0
    && state.netPolylines.length === 0;
  if (shouldUseCarryForward) {
    applyCarryForwardGeometryPreset();
  }
  state.hoverPoint = null;
  refreshCameraModelEstimate();
  markClean();
}

function restoreWizardProgressForCurrentImage(allowResume = false) {
  clearRememberedWizardProgress();
}

function loadImageElement(name) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Failed to load ${name}`));
    image.src = `/api/image?file=${encodeURIComponent(name)}`;
  });
}

async function loadImage(name) {
  setStatus("Loading image…");
  state.suspendAutoSave = true;
  const [record, image] = await Promise.all([
    apiFetch(`/api/v2/label/${encodeURIComponent(name)}?pass=${PASS_N}`).then((r) => r.json()),
    loadImageElement(name),
  ]);
  state.currentName = name;
  rememberCurrentImage(name);
  state.image = image;
  state.imageW = image.naturalWidth;
  state.imageH = image.naturalHeight;
  applyRecord(record);
  restoreWizardProgressForCurrentImage(Boolean(record.wizard_step));
  state.suspendAutoSave = false;
  syncUi();
  scheduleBaseRender();
  setStatus("Ready.");
}

async function loadNextImage() {
  captureCarryForwardGeometryPreset();
  const current = state.currentName ? `&current=${encodeURIComponent(state.currentName)}` : "";
  const res = await apiFetch(`/api/v2/next?pass=${PASS_N}${current}`);
  const data = await res.json();
  if (!data.next) {
    state.currentName = null;
    rememberCurrentImage(null);
    state.image = null;
    ctx.clearRect(0, 0, cv.width, cv.height);
    setStatus("No images left in this pass.");
    return;
  }
  await loadImage(data.next);
}

async function keepAlive() {
  if (!state.currentName) return;
  try {
    await apiFetch(
      `/api/heartbeat?image=${encodeURIComponent(state.currentName)}&pass=${PASS_N}`,
      { method: "POST" },
    );
  } catch (_) {
    // silent
  }
}

function stepIndex() {
  return STEP_ORDER.indexOf(state.wizardStep);
}

function needsBallStep() {
  if (state.ballInPlay === true) return true;
  return state.ballVisible === true;
}

function normalizeWizardProgress() {
  if (state.wizardStep === STEP.BALL && !needsBallStep()) {
    state.wizardStep = STEP.IGNORE;
  }
}

function inferWizardProgressFromState() {
  if (!state.genderCategory) {
    return {
      wizardStep: STEP.SESSION,
      sessionSubstep: SESSION_SUBSTEP.GENDER,
    };
  }
  if (state.ballInPlay === null) {
    return {
      wizardStep: STEP.SESSION,
      sessionSubstep: SESSION_SUBSTEP.BALL_IN_PLAY,
    };
  }
  if (state.ballVisible === null) {
    return {
      wizardStep: STEP.SESSION,
      sessionSubstep: SESSION_SUBSTEP.BALL_VISIBLE,
    };
  }
  if (!Array.isArray(state.focusRegion) || state.focusRegion.length < 4) {
    return {
      wizardStep: STEP.GEOMETRY,
      geometrySubstep: GEOMETRY_SUBSTEP.DISTORTION,
    };
  }
  const courtPointCount = state.geometryPolylines.reduce((sum, line) => (
    sum + (Array.isArray(line) ? line.length : 0)
  ), 0);
  if (!activeCourtQuad() && courtPointCount < 3) {
    return {
      wizardStep: STEP.GEOMETRY,
      geometrySubstep: GEOMETRY_SUBSTEP.COURT,
    };
  }
  const participantCount = state.people.filter((person) => person?.role === "participant").length;
  const playerCount = state.people.filter((person) => person?.role === "player").length;
  const refereeCount = state.people.filter((person) => person?.role === "referee").length;
  if (state.sessionType === "training") {
    if (participantCount === 0 && !state.ball && state.ignorePoints.length === 0) {
      return {
        wizardStep: STEP.PEOPLE,
        peopleSubstep: PEOPLE_SUBSTEP.PARTICIPANTS,
      };
    }
  } else {
    if (playerCount === 0 && refereeCount === 0 && !state.ball && state.ignorePoints.length === 0) {
      return {
        wizardStep: STEP.PEOPLE,
        peopleSubstep: PEOPLE_SUBSTEP.PLAYERS,
      };
    }
  }
  if (needsBallStep() && !state.ball) {
    return {
      wizardStep: STEP.BALL,
    };
  }
  return {
    wizardStep: STEP.IGNORE,
  };
}

function applyWizardProgress(progress) {
  if (!progress) return;
  const hasSavedFocusRegion = Array.isArray(state.focusRegion) && state.focusRegion.length >= 4;
  if (progress.sessionSubstep && Object.values(SESSION_SUBSTEP).includes(progress.sessionSubstep)) {
    state.sessionSubstep = progress.sessionSubstep;
  }
  if (progress.geometrySubstep && Object.values(GEOMETRY_SUBSTEP).includes(progress.geometrySubstep)) {
    state.geometrySubstep = progress.geometrySubstep === GEOMETRY_SUBSTEP.NET
      ? GEOMETRY_SUBSTEP.COURT
      : progress.geometrySubstep;
  }
  if (progress.peopleSubstep && Object.values(PEOPLE_SUBSTEP).includes(progress.peopleSubstep)) {
    state.peopleSubstep = progress.peopleSubstep;
  }
  if (progress.wizardStep && Object.values(STEP).includes(progress.wizardStep)) {
    state.wizardStep = progress.wizardStep === STEP.FOCUS && !hasSavedFocusRegion
      ? STEP.GEOMETRY
      : progress.wizardStep;
  }
  if (state.wizardStep === STEP.GEOMETRY && !hasSavedFocusRegion) {
    state.geometrySubstep = GEOMETRY_SUBSTEP.DISTORTION;
  }
  normalizeWizardProgress();
}

  function goContinueStep() {
  if (state.wizardStep === STEP.IGNORE) {
    void completeAndLoadNext();
    return;
  }
  if (state.wizardStep === STEP.SESSION) {
    if (state.sessionSubstep === SESSION_SUBSTEP.GENDER) {
      if (!state.genderCategory) {
        syncUi();
        return;
      }
      state.sessionSubstep = SESSION_SUBSTEP.BALL_IN_PLAY;
      syncUi();
      return;
    }
    if (state.sessionSubstep === SESSION_SUBSTEP.BALL_IN_PLAY) {
      if (state.ballInPlay === null) {
        syncUi();
        return;
      }
      state.sessionSubstep = SESSION_SUBSTEP.BALL_VISIBLE;
      syncUi();
      return;
    }
    if (state.sessionSubstep === SESSION_SUBSTEP.BALL_VISIBLE) {
      if (state.ballVisible === null) {
        syncUi();
        return;
      }
    }
  }
    if (state.wizardStep === STEP.GEOMETRY) {
      if (state.geometrySubstep === GEOMETRY_SUBSTEP.DISTORTION) {
        if (state.distortionPoints.length < MIN_DISTORTION_HELPER_POINTS) {
          setStatus(`Add at least ${MIN_DISTORTION_HELPER_POINTS} support points on the same rigid horizontal line.`);
          syncUi();
          render();
          return;
        }
        const fitted = estimateDistortionFromPoints(state.distortionPoints);
        if (fitted !== null) {
          state.distortionParams = fitted;
          state.distortionHelperPoints = state.distortionPoints.map((point) => [Number(point[0]), Number(point[1])]);
          const d = fitted.diagnostics;
          if (d) {
            setStatus(
              `Flatten fit: max bend ${d.before_max_abs_deviation.toFixed(2)} -> ${d.after_max_abs_deviation.toFixed(2)}, energy ${d.before_bend_energy.toFixed(1)} -> ${d.after_bend_energy.toFixed(1)}, lambda ${Number(fitted.lambda || 0).toFixed(3)}, rot ${Number(fitted.rotation_deg || 0).toFixed(2)}deg`,
            );
          }
        } else {
          setStatus("Could not fit a stable low-order distortion model from those points. Add more points along the full line.");
          syncUi();
          render();
          return;
        }
        state.focusRegion = [];
        state.focusRegionCanvasDefault = true;
        state.focusRegionCanvasPoints = null;
        ensureDefaultFocusRegion();
        state.wizardStep = STEP.FOCUS;
        state.currentPolylineIndex = null;
        state.distortionPoints = [];
        scheduleAutoSave();
        syncUi();
        scheduleBaseRender();
        return;
      }
    if (!activeCourtQuad() && geometryPointCount() < 3) {
      syncUi();
      render();
      return;
    }
    if (state.geometrySubstep === GEOMETRY_SUBSTEP.COURT) {
      state.currentPolylineIndex = null;
    }
  }
  if (state.wizardStep === STEP.FOCUS) {
    state.wizardStep = STEP.GEOMETRY;
    state.geometrySubstep = GEOMETRY_SUBSTEP.COURT;
    state.currentPolylineIndex = null;
    syncUi();
    render();
    return;
  }
  if (state.wizardStep === STEP.PEOPLE) {
    if (state.sessionType !== "training" && state.peopleSubstep === PEOPLE_SUBSTEP.PLAYERS) {
      state.peopleSubstep = PEOPLE_SUBSTEP.REFEREES;
      syncUi();
      render();
      return;
    }
  }
  const idx = stepIndex();
  if (idx >= STEP_ORDER.length - 1) return;
  let next = STEP_ORDER[idx + 1];
  if (next === STEP.BALL && !needsBallStep()) {
    next = STEP.IGNORE;
  }
  state.wizardStep = next;
  normalizeWizardProgress();
  syncUi();
  render();
}

async function completeAndLoadNext() {
  btnContinueStep.disabled = true;
  btnCompleteNext.disabled = true;
  try {
    captureCarryForwardGeometryPreset();
    if (state.isDirty) {
      const result = await saveLabel(true);
      if (result?.next_image) await loadImage(result.next_image);
      else await loadNextImage();
    } else {
      await skipImage();
    }
  } finally {
    btnContinueStep.disabled = false;
    btnCompleteNext.disabled = false;
  }
}

function bindEvents() {
  btnContinueStep.addEventListener("click", goContinueStep);
  btnCompleteNext.addEventListener("click", async () => {
    await completeAndLoadNext();
  });
  btnSkipStep.addEventListener("click", skipCurrentStep);

  canvasWrap.addEventListener("mousedown", onPointerDown);
  window.addEventListener("mousemove", onPointerMove);
  window.addEventListener("mouseup", onPointerUp);
  cv.addEventListener("mouseleave", onPointerLeave);
  canvasWrap.addEventListener("contextmenu", onCanvasContextMenu);
  window.addEventListener("resize", scheduleBaseRender);
  window.addEventListener("keydown", (ev) => {
    if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "z") {
      ev.preventDefault();
      undo();
      return;
    }
    if (ev.key === "Escape" && state.drag) {
      state.drag = null;
      render();
    }
  });

  stepBallPanel.innerHTML = "";
  stepIgnorePanel.innerHTML = "";

  setInterval(keepAlive, 3 * 60 * 1000);
}

async function boot() {
  bindEvents();
  clearRememberedWizardProgress();
  syncUi();
  const remembered = getRememberedImage();
  if (remembered) {
    try {
      await loadImage(remembered);
      return;
    } catch (_) {
      rememberCurrentImage(null);
    }
  }
  await loadNextImage();
}

boot().catch((err) => {
  console.error(err);
  setStatus(`Labeler failed to start: ${err.message}`);
});
