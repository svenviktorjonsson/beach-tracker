const cv = document.getElementById("cv");
const ctx = cv.getContext("2d");
const canvasWrap = document.getElementById("canvasWrap");
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

const COURT_SLOTS = ["left", "far", "right", "near"];
const NET_SLOTS = ["left_antenna", "net", "right_antenna"];
const ALL_SLOTS = [...COURT_SLOTS, ...NET_SLOTS];
const SLOT_LABELS = {
  left: "Left",
  far: "Far",
  right: "Right",
  near: "Near",
  left_antenna: "Left antenna",
  net: "Net",
  right_antenna: "Right antenna",
};
const STEP = {
  SESSION: "session",
  GEOMETRY: "geometry",
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
  LINES: "lines",
};
const PASS_N = 1;
const DISTORTION_GUIDE_COLOR = "#79c0ff";

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
  currentSlot: COURT_SLOTS[0],
  sessionType: null,
  genderCategory: null,
  ballInPlay: null,
  ballVisible: null,
  distortionParams: { k1: 0, k2: 0, k3: 0, rotation_deg: 0 },
  lines: {},
  people: [],
  ball: null,
  ignorePoints: [],
  inferred: null,
  history: [],
  drag: null,
  distortionPoints: [],
  hoverPoint: null,
  suggestionTimer: null,
};

state.rawCtx = state.rawCanvas.getContext("2d");
state.renderCtx = state.renderCanvas.getContext("2d");

function setStatus(msg) {
  if (statusEl) statusEl.textContent = msg || "";
}

function setInstruction(msg) {
  if (instructionBanner) instructionBanner.textContent = msg || "";
}

async function apiFetch(url, options) {
  const res = await fetch(url, { credentials: "same-origin", ...options });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res;
}

function deepClone(obj) {
  return JSON.parse(JSON.stringify(obj));
}

function pushHistory() {
  state.history.push(
    deepClone({
      wizardStep: state.wizardStep,
      sessionSubstep: state.sessionSubstep,
      geometrySubstep: state.geometrySubstep,
      currentSlot: state.currentSlot,
      sessionType: state.sessionType,
      genderCategory: state.genderCategory,
      ballInPlay: state.ballInPlay,
      ballVisible: state.ballVisible,
      distortionParams: state.distortionParams,
      lines: state.lines,
      people: state.people,
      ball: state.ball,
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
  const { k1, k2, k3 } = state.distortionParams;
  return Math.max(Math.abs(k1 || 0), Math.abs(k2 || 0), Math.abs(k3 || 0));
}

function rotationRadians() {
  return 0;
}

function linePointStatus(slot) {
  const line = state.lines[slot];
  if (!line) return "empty";
  if (line.skipped) return "skipped";
  return "done";
}

function nextUnresolvedSlot() {
  return ALL_SLOTS.find((slot) => linePointStatus(slot) === "empty") || ALL_SLOTS[ALL_SLOTS.length - 1];
}

function allGeometryResolved() {
  return ALL_SLOTS.every((slot) => linePointStatus(slot) !== "empty");
}

function placedCourtLineCount() {
  return COURT_SLOTS.filter((slot) => linePointStatus(slot) === "done").length;
}

function setSessionType(value) {
  state.sessionType = value;
  if (value === "training") {
    state.people = state.people.map((p) => ({ ...p, role: "participant" }));
  }
  syncUi();
}

function setGender(value) {
  state.genderCategory = value;
  state.sessionSubstep = SESSION_SUBSTEP.BALL_IN_PLAY;
  syncUi();
  render();
}

function setBallInPlay(value) {
  state.ballInPlay = value;
  if (value === false) {
    state.ballVisible = null;
    state.sessionSubstep = SESSION_SUBSTEP.BALL_VISIBLE;
  } else {
    state.ballVisible = null;
    state.wizardStep = STEP.GEOMETRY;
    state.geometrySubstep = GEOMETRY_SUBSTEP.DISTORTION;
  }
  syncUi();
  render();
}

function setBallVisible(value) {
  state.ballVisible = value;
  state.wizardStep = STEP.GEOMETRY;
  state.geometrySubstep = GEOMETRY_SUBSTEP.DISTORTION;
  syncUi();
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
  if (state.sessionType === "training") return "participant";
  const active = personRoleButtons.querySelector(".is-active");
  return active?.dataset.value || "player";
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
    if (state.ballInPlay === false && state.ballVisible === null) return false;
    return true;
  }
  if (step === STEP.GEOMETRY) {
    if (state.geometrySubstep === GEOMETRY_SUBSTEP.DISTORTION) {
      return true;
    }
    return placedCourtLineCount() >= 3 || allGeometryResolved();
  }
  return true;
}

function syncUi() {
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
    const courtDone = COURT_SLOTS.filter((slot) => linePointStatus(slot) === "done").length;
    const courtSkipped = COURT_SLOTS.filter((slot) => linePointStatus(slot) === "skipped").length;
    if (courtDone > 0 || courtSkipped > 0) {
      summaryItems.push({
        text: `Court lines ${courtDone}/4`,
        tone: "default",
      });
    }
    if (courtSkipped > 0) {
      summaryItems.push({
        text: `${courtSkipped} skipped`,
        tone: "warn",
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

  if (state.sessionType === "training") {
    buildChoiceButtons(personRoleButtons, ["participant"], (v) => "Participant", () => {});
    updateChoiceButtons(personRoleButtons, "participant");
  } else {
    if (personRoleButtons.children.length !== 2) {
      buildChoiceButtons(personRoleButtons, ["player", "referee"], (v) => (
        v === "player" ? "Player" : "Referee"
      ), (value) => updateChoiceButtons(personRoleButtons, value));
    }
    if (!personRoleButtons.querySelector(".is-active")) {
      updateChoiceButtons(personRoleButtons, "player");
    }
  }

  if (distortionHint) {
    distortionHint.textContent = state.geometrySubstep === GEOMETRY_SUBSTEP.DISTORTION
      ? `Click 4 points along one bent line (${state.distortionPoints.length}/4)`
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
    setInstruction("One question at a time.");
  } else if (state.wizardStep === STEP.GEOMETRY) {
    if (state.geometrySubstep === GEOMETRY_SUBSTEP.DISTORTION) {
      wizardStepTitle.textContent = "Adjust distortion";
      setInstruction("Click 4 points along one rigid bent line. The correction is fitted automatically after the 4th point.");
    } else {
      wizardStepTitle.textContent = `Place ${SLOT_LABELS[state.currentSlot]} court line`;
      setInstruction(`Click the orthogonal point from the blue origin to the ${SLOT_LABELS[state.currentSlot].toLowerCase()} court line, then drag it until the rendered line matches. Right-click or use "Skip line" if it is not visible.`);
    }
  } else if (state.wizardStep === STEP.PEOPLE) {
    wizardStepTitle.textContent = "Draw people";
    setInstruction("Drag boxes for people in the frame. Training uses participants only. Game uses players and referees.");
  } else if (state.wizardStep === STEP.BALL) {
    wizardStepTitle.textContent = "Draw the active ball";
    setInstruction("Drag one tight box around the active ball. The labeler stores center and radius from that box.");
  } else {
    wizardStepTitle.textContent = "Mark ignore balls";
    setInstruction("Click red-cross ignore points for balls that are definitely not the active ball. Then use Complete & next.");
  }

  const stepIndex = STEP_ORDER.indexOf(state.wizardStep);
  btnContinueStep.disabled = !canContinueFromStep(state.wizardStep) || state.wizardStep === STEP.IGNORE;
  const showContinue =
    (state.wizardStep === STEP.GEOMETRY && state.geometrySubstep === GEOMETRY_SUBSTEP.LINES) ||
    state.wizardStep === STEP.PEOPLE ||
    state.wizardStep === STEP.BALL;
  btnContinueStep.classList.toggle("hidden", !showContinue);
  btnSkipStep.disabled = false;
  btnSkipStep.classList.toggle("hidden", false);
  btnSkipStep.textContent =
    state.wizardStep === STEP.GEOMETRY && state.geometrySubstep === GEOMETRY_SUBSTEP.LINES
      ? "Skip line"
      : "Skip step";
  btnCompleteNext.disabled = !state.currentName;
  btnCompleteNext.classList.toggle("hidden", state.wizardStep === STEP.GEOMETRY && state.geometrySubstep === GEOMETRY_SUBSTEP.DISTORTION);
}

function getDisplaySize() {
  const outerW = Math.max(320, canvasWrap.clientWidth);
  const outerH = Math.max(320, canvasWrap.clientHeight);
  const distortionPad = 18 + Math.round(distortionStrength() * 52);
  const maxW = Math.max(220, outerW - distortionPad * 2);
  const maxH = Math.max(220, outerH - distortionPad * 2);
  const rawScale = Math.min(maxW / state.imageW, maxH / state.imageH);
  const drawW = Math.max(1, Math.round(state.imageW * rawScale));
  const drawH = Math.max(1, Math.round(state.imageH * rawScale));
  const overscan = 1 + distortionStrength() * 0.35;
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
  const srcCx = size.drawW / 2;
  const srcCy = size.drawH / 2;
  const workCx = size.workW / 2;
  const workCy = size.workH / 2;
  const { k1, k2, k3 } = state.distortionParams;
  const correctedCanvas = document.createElement("canvas");
  correctedCanvas.width = size.workW;
  correctedCanvas.height = size.workH;
  const params = { k1, k2, k3 };
  const correctedCtx = correctedCanvas.getContext("2d");
  const correctedImage = correctedCtx.createImageData(size.workW, size.workH);
  const out = correctedImage.data;
  for (let y = 0; y < size.workH; y += 1) {
    for (let x = 0; x < size.workW; x += 1) {
      const warped = distortPointWithParams(x, y, size.workW, size.workH, params);
      const sx = srcCx + (warped[0] - workCx);
      const sy = srcCy + (warped[1] - workCy);
      const di = (y * size.workW + x) * 4;
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
  correctedCtx.putImageData(correctedImage, 0, 0);
  const correctedData = correctedCtx.getImageData(0, 0, size.workW, size.workH);
  const rowLeft = new Array(size.workH).fill(-1);
  const rowRight = new Array(size.workH).fill(-1);
  for (let y = 0; y < size.workH; y += 1) {
    let left = -1;
    let right = -1;
    for (let x = 0; x < size.workW; x += 1) {
      const alpha = correctedData.data[(y * size.workW + x) * 4 + 3];
      if (alpha > 0) {
        if (left === -1) left = x;
        right = x;
      }
    }
    rowLeft[y] = left;
    rowRight[y] = right;
  }
  let best = null;
  for (let top = 0; top < size.workH; top += 1) {
    if (rowLeft[top] === -1) continue;
    let leftBound = rowLeft[top];
    let rightBound = rowRight[top];
    for (let bottom = top; bottom < size.workH; bottom += 1) {
      if (rowLeft[bottom] === -1) break;
      leftBound = Math.max(leftBound, rowLeft[bottom]);
      rightBound = Math.min(rightBound, rowRight[bottom]);
      if (rightBound < leftBound) break;
      const width = rightBound - leftBound + 1;
      const height = bottom - top + 1;
      const area = width * height;
      if (!best || area > best.area) {
        best = { x: leftBound, y: top, w: width, h: height, area };
      }
    }
  }
  if (!best) {
    best = { x: 0, y: 0, w: size.workW, h: size.workH, area: size.workW * size.workH };
  }
  const cropX = best.x;
  const cropY = best.y;
  const cropW = best.w;
  const cropH = best.h;
  const displayScale = Math.min(size.maxW / cropW, size.maxH / cropH);
  const displayW = Math.max(1, Math.round(cropW * displayScale));
  const displayH = Math.max(1, Math.round(cropH * displayScale));
  cv.width = displayW;
  cv.height = displayH;
  state.renderCanvas.width = displayW;
  state.renderCanvas.height = displayH;

  state.renderCtx.clearRect(0, 0, displayW, displayH);
  state.renderCtx.drawImage(
    correctedCanvas,
    cropX,
    cropY,
    cropW,
    cropH,
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
    bboxW: size.workW,
    bboxH: size.workH,
    cropX,
    cropY,
    cropW,
    cropH,
    displayScale,
    offsetX: 0,
    offsetY: 0,
  };
}

function imageToCanvasPoint(x, y) {
  const meta = state.renderMeta;
  if (!meta) return [0, 0];
  const rawX = x * meta.rawScaleX;
  const rawY = y * meta.rawScaleY;
  const pseudoDistortedX = meta.workW / 2 + (rawX - meta.drawW / 2);
  const pseudoDistortedY = meta.workH / 2 + (rawY - meta.drawH / 2);
  const corrected = undistortPointWithParams(
    pseudoDistortedX,
    pseudoDistortedY,
    meta.workW,
    meta.workH,
    state.distortionParams,
  );
  const dx = corrected[0] - meta.workW / 2;
  const dy = corrected[1] - meta.workH / 2;
  const angle = rotationRadians();
  const rotX = meta.bboxW / 2 + dx * Math.cos(angle) - dy * Math.sin(angle);
  const rotY = meta.bboxH / 2 + dx * Math.sin(angle) + dy * Math.cos(angle);
  return [
    meta.offsetX + (rotX - meta.cropX) * meta.displayScale,
    meta.offsetY + (rotY - meta.cropY) * meta.displayScale,
  ];
}

function canvasToImagePoint(clientX, clientY) {
  const meta = state.renderMeta;
  if (!meta) return [0, 0];
  const rect = cv.getBoundingClientRect();
  const x = ((clientX - rect.left) / rect.width) * cv.width;
  const y = ((clientY - rect.top) / rect.height) * cv.height;
  const rotX = meta.cropX + (x - meta.offsetX) / meta.displayScale;
  const rotY = meta.cropY + (y - meta.offsetY) / meta.displayScale;
  const dx = rotX - meta.bboxW / 2;
  const dy = rotY - meta.bboxH / 2;
  const angle = -rotationRadians();
  const correctedX = meta.workW / 2 + dx * Math.cos(angle) - dy * Math.sin(angle);
  const correctedY = meta.workH / 2 + dx * Math.sin(angle) + dy * Math.cos(angle);
  const pseudoDistorted = distortPointWithParams(
    correctedX,
    correctedY,
    meta.workW,
    meta.workH,
    state.distortionParams,
  );
  const rawX = meta.drawW / 2 + (pseudoDistorted[0] - meta.workW / 2);
  const rawY = meta.drawH / 2 + (pseudoDistorted[1] - meta.workH / 2);
  return [
    rawX / meta.rawScaleX,
    rawY / meta.rawScaleY,
  ];
}

function drawLineFromOrthPoint(point, color, isActive) {
  const [ox, oy] = origin();
  const nx = point.x - ox;
  const ny = point.y - oy;
  const mag = Math.hypot(nx, ny);
  const [px, py] = imageToCanvasPoint(point.x, point.y);
  ctx.save();
  if (mag > 0.0001) {
    const dx = ny / mag;
    const dy = -nx / mag;
    const length = Math.max(state.imageW, state.imageH) * 2;
    const [c1x, c1y] = imageToCanvasPoint(point.x - dx * length, point.y - dy * length);
    const [c2x, c2y] = imageToCanvasPoint(point.x + dx * length, point.y + dy * length);
    ctx.strokeStyle = color;
    ctx.lineWidth = isActive ? 3 : 2;
    ctx.beginPath();
    ctx.moveTo(c1x, c1y);
    ctx.lineTo(c2x, c2y);
    ctx.stroke();
  }
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(px, py, isActive ? 7 : 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawHoverLine(point) {
  const [ox, oy] = origin();
  const nx = point.x - ox;
  const ny = point.y - oy;
  const mag = Math.hypot(nx, ny);
  if (mag <= 0.0001) return;
  const dx = ny / mag;
  const dy = -nx / mag;
  const length = Math.max(state.imageW, state.imageH) * 2;
  const [c1x, c1y] = imageToCanvasPoint(point.x - dx * length, point.y - dy * length);
  const [c2x, c2y] = imageToCanvasPoint(point.x + dx * length, point.y + dy * length);
  const [px, py] = imageToCanvasPoint(point.x, point.y);
  ctx.save();
  ctx.strokeStyle = "rgba(121, 192, 255, 0.95)";
  ctx.lineWidth = 2;
  ctx.setLineDash([10, 7]);
  ctx.beginPath();
  ctx.moveTo(c1x, c1y);
  ctx.lineTo(c2x, c2y);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(121, 192, 255, 0.95)";
  ctx.beginPath();
  ctx.arc(px, py, 4.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawOriginMarker() {
  const [ox, oy] = imageToCanvasPoint(...origin());
  ctx.save();
  ctx.strokeStyle = "#79c0ff";
  ctx.fillStyle = "#79c0ff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(ox - 10, oy);
  ctx.lineTo(ox + 10, oy);
  ctx.moveTo(ox, oy - 10);
  ctx.lineTo(ox, oy + 10);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(ox, oy, 3.5, 0, Math.PI * 2);
  ctx.fill();
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
  const [x, y] = imageToCanvasPoint(rect.x, rect.y);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  if (dashed) ctx.setLineDash([8, 5]);
  ctx.strokeRect(x, y, rect.w * state.viewScale, rect.h * state.viewScale);
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

function render() {
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!state.image) return;
  ctx.drawImage(state.renderCanvas, 0, 0);
  if (state.wizardStep === STEP.GEOMETRY && state.geometrySubstep === GEOMETRY_SUBSTEP.DISTORTION) {
    if (state.distortionPoints.length > 1) {
      drawSuggestedLine({
        points: state.distortionPoints.map((point) => imageToCanvasPoint(point[0], point[1])),
      });
    }
    state.distortionPoints.forEach((point, index) => {
      const [x, y] = imageToCanvasPoint(point[0], point[1]);
      ctx.save();
      ctx.fillStyle = DISTORTION_GUIDE_COLOR;
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#0d1117";
      ctx.font = "700 13px Segoe UI";
      ctx.fillText(String(index + 1), x + 8, y - 8);
      ctx.restore();
    });
  }
  for (const slot of ALL_SLOTS) {
    const line = state.lines[slot];
    if (!line || line.skipped || line.x == null || line.y == null) continue;
    const color = COURT_SLOTS.includes(slot) ? "#58a6ff" : "#3fb950";
    drawLineFromOrthPoint(line, color, state.wizardStep === STEP.GEOMETRY && slot === state.currentSlot);
  }

  state.people.forEach((person) => {
    const color = person.role === "referee" ? "#f2cc60" : "#a371f7";
    drawRect(person, color);
  });
  if (state.ball) drawRect(state.ball, "#f85149");
  state.ignorePoints.forEach((point) => drawCross(point, "#f85149"));

  if (state.drag?.type === "rect") drawRect(state.drag.rect, "#ffffff", true);
}

function distortPointWithParams(x, y, width, height, params) {
  const cx = width / 2;
  const cy = height / 2;
  const norm = Math.max(width, height) / 2;
  const dx = (x - cx) / norm;
  const dy = (y - cy) / norm;
  const r2 = dx * dx + dy * dy;
  const factor = 1 + params.k1 * r2 + params.k2 * r2 * r2 + params.k3 * r2 * r2 * r2;
  return [cx + dx * factor * norm, cy + dy * factor * norm];
}

function undistortPointWithParams(x, y, width, height, params) {
  const cx = width / 2;
  const cy = height / 2;
  const norm = Math.max(width, height) / 2;
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
  return { error: error / n, theta };
}

function estimateDistortionFromPoints(points) {
  if (points.length < 4) return null;
  const score = (params) => {
    const undistorted = points.map((point) => (
      undistortPointWithParams(point[0], point[1], state.imageW, state.imageH, params)
    ));
    return { ...fitLineStats(undistorted), undistorted };
  };
  const clampParams = (params) => ({
    k1: Math.max(-0.45, Math.min(0.45, params.k1)),
    k2: 0,
    k3: 0,
  });
  let current = { k1: 0, k2: 0, k3: 0 };
  let currentStats = score(current);
  let currentScore = currentStats.error;
  for (const step of [0.2, 0.08, 0.03, 0.01, 0.004, 0.0015]) {
    let improved = true;
    while (improved) {
      improved = false;
      for (const dir of [-1, 1]) {
        const candidate = clampParams({ ...current, k1: current.k1 + dir * step });
        const candidateStats = score(candidate);
        const candidateScore = candidateStats.error;
        if (candidateScore < currentScore) {
          current = candidate;
          currentScore = candidateScore;
          currentStats = candidateStats;
          improved = true;
        }
      }
    }
  }
  return { ...clampParams(current), rotation_deg: 0 };
}

function clampRect(rect) {
  let { x, y, w, h } = rect;
  if (w < 0) {
    x += w;
    w *= -1;
  }
  if (h < 0) {
    y += h;
    h *= -1;
  }
  x = Math.max(0, Math.min(x, state.imageW));
  y = Math.max(0, Math.min(y, state.imageH));
  w = Math.max(1, Math.min(w, state.imageW - x));
  h = Math.max(1, Math.min(h, state.imageH - y));
  return { x, y, w, h };
}

function hitGeometryPoint(ix, iy) {
  const radius = 18 / state.viewScale;
  for (const slot of ALL_SLOTS) {
    const line = state.lines[slot];
    if (!line || line.skipped || line.x == null || line.y == null) continue;
    if (Math.hypot(line.x - ix, line.y - iy) <= radius) return slot;
  }
  return null;
}

function onPointerDown(ev) {
  if (!state.image) return;
  if (ev.button !== 0) return;
  const [ix, iy] = canvasToImagePoint(ev.clientX, ev.clientY);

  if (state.wizardStep === STEP.GEOMETRY) {
    if (state.geometrySubstep === GEOMETRY_SUBSTEP.DISTORTION) {
      pushHistory();
      state.distortionPoints.push([ix, iy]);
      if (state.distortionPoints.length >= 4) {
        const fitted = estimateDistortionFromPoints(state.distortionPoints);
        if (fitted !== null) {
          state.distortionParams = fitted;
        }
        state.geometrySubstep = GEOMETRY_SUBSTEP.LINES;
        state.distortionPoints = [];
        syncUi();
        scheduleBaseRender();
        return;
      }
      syncUi();
      render();
      return;
    }
    if (state.geometrySubstep !== GEOMETRY_SUBSTEP.LINES) {
      return;
    }
    const hitSlot = hitGeometryPoint(ix, iy);
    if (hitSlot) {
      state.currentSlot = hitSlot;
      state.drag = { type: "line", slot: hitSlot };
      syncUi();
      render();
      return;
    }
    pushHistory();
    state.lines[state.currentSlot] = { x: ix, y: iy, skipped: false };
    if (linePointStatus(state.currentSlot) !== "empty") {
      state.currentSlot = nextUnresolvedSlot();
    }
    syncUi();
    render();
    return;
  }

  if (state.wizardStep === STEP.PEOPLE || state.wizardStep === STEP.BALL) {
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
    render();
  }
}

function onPointerMove(ev) {
  const [ix, iy] = canvasToImagePoint(ev.clientX, ev.clientY);
  if (
    state.image &&
    state.wizardStep === STEP.GEOMETRY &&
    state.geometrySubstep === GEOMETRY_SUBSTEP.LINES &&
    !state.drag
  ) {
    state.hoverPoint = [ix, iy];
    render();
  }
  if (!state.image || !state.drag) return;
  if (state.drag.type === "line") {
    state.lines[state.drag.slot] = { x: ix, y: iy, skipped: false };
    render();
    return;
  }
  state.drag.rect = clampRect({
    x: state.drag.startX,
    y: state.drag.startY,
    w: ix - state.drag.startX,
    h: iy - state.drag.startY,
  });
  render();
}

function onPointerUp() {
  if (!state.drag) return;
  if (state.drag.type === "line") {
    state.drag = null;
    syncUi();
    render();
    return;
  }
  const rect = clampRect(state.drag.rect);
  state.drag = null;
  if (rect.w < 3 || rect.h < 3) {
    render();
    return;
  }
  pushHistory();
  if (state.wizardStep === STEP.PEOPLE) {
    state.people.push({ ...rect, role: currentPersonRole() });
  } else if (state.wizardStep === STEP.BALL) {
    state.ball = {
      ...rect,
      center_x: rect.x + rect.w / 2,
      center_y: rect.y + rect.h / 2,
      radius: Math.max(rect.w, rect.h) / 2,
    };
  }
  syncUi();
  render();
}

function onPointerLeave() {
  if (state.hoverPoint) {
    state.hoverPoint = null;
    render();
  }
}

function skipCurrentLine() {
  if (state.wizardStep !== STEP.GEOMETRY || state.geometrySubstep !== GEOMETRY_SUBSTEP.LINES) return;
  pushHistory();
  state.lines[state.currentSlot] = { x: null, y: null, skipped: true };
  state.currentSlot = nextUnresolvedSlot();
  syncUi();
  render();
}

function skipCurrentStep() {
  if (state.wizardStep === STEP.GEOMETRY) {
    if (state.geometrySubstep === GEOMETRY_SUBSTEP.DISTORTION) {
      state.geometrySubstep = GEOMETRY_SUBSTEP.LINES;
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
    goContinueStep();
    return;
  }
  if (state.wizardStep === STEP.IGNORE) {
    state.ignorePoints = [];
    syncUi();
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
  if (state.wizardStep !== STEP.GEOMETRY || state.geometrySubstep !== GEOMETRY_SUBSTEP.LINES) return;
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
  const payload = {
    image: state.currentName,
    session_type: state.sessionType,
    gender_category: state.genderCategory,
    ball_in_play: state.ballInPlay,
    ball_visible: state.ballVisible,
    distortion_k: Number(state.distortionParams.k1 || 0),
    distortion_params: {
      k1: Number(state.distortionParams.k1 || 0),
      k2: Number(state.distortionParams.k2 || 0),
      k3: Number(state.distortionParams.k3 || 0),
      rotation_deg: Number(state.distortionParams.rotation_deg || 0),
    },
    complete: !!complete,
    lines: state.lines,
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
  setStatus(complete ? "Saved and completed." : "Draft saved.");
  return data;
}

async function skipImage() {
  if (!state.currentName) return;
  const res = await apiFetch(
    `/api/v2/skip-image?image=${encodeURIComponent(state.currentName)}&pass=${PASS_N}`,
    { method: "POST" },
  );
  const data = await res.json();
  if (data.next_image) {
    await loadImage(data.next_image);
  } else {
    setStatus("No more images in this pass.");
  }
}

function applyRecord(record) {
  state.sessionType = record.session_type || record.inferred?.session_type || "training";
  state.genderCategory = record.gender_category || null;
  state.ballInPlay = typeof record.ball_in_play === "boolean" ? record.ball_in_play : null;
  state.ballVisible = typeof record.ball_visible === "boolean" ? record.ball_visible : null;
  state.sessionSubstep = !state.genderCategory
    ? SESSION_SUBSTEP.GENDER
    : state.ballInPlay === null
      ? SESSION_SUBSTEP.BALL_IN_PLAY
      : state.ballInPlay === false && state.ballVisible === null
        ? SESSION_SUBSTEP.BALL_VISIBLE
        : SESSION_SUBSTEP.GENDER;
  state.distortionParams = {
    k1: Number(record.distortion_params?.k1 ?? record.distortion_k ?? 0),
    k2: Number(record.distortion_params?.k2 ?? 0),
    k3: Number(record.distortion_params?.k3 ?? 0),
    rotation_deg: Number(record.distortion_params?.rotation_deg ?? 0),
  };
  state.lines = record.lines || {};
  state.people = Array.isArray(record.people) ? record.people : [];
  state.ball = record.ball ? { ...record.ball } : null;
  state.ignorePoints = Array.isArray(record.ignore_points) ? record.ignore_points : [];
  state.inferred = record.inferred || null;
  state.currentSlot = nextUnresolvedSlot();
  state.wizardStep = STEP.SESSION;
  state.geometrySubstep = GEOMETRY_SUBSTEP.DISTORTION;
  state.history = [];
  state.distortionPoints = [];
  state.hoverPoint = null;
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
  const [record, image] = await Promise.all([
    apiFetch(`/api/v2/label/${encodeURIComponent(name)}?pass=${PASS_N}`).then((r) => r.json()),
    loadImageElement(name),
  ]);
  state.currentName = name;
  state.image = image;
  state.imageW = image.naturalWidth;
  state.imageH = image.naturalHeight;
  applyRecord(record);
  syncUi();
  scheduleBaseRender();
  setStatus("Ready.");
}

async function loadNextImage() {
  const current = state.currentName ? `&current=${encodeURIComponent(state.currentName)}` : "";
  const res = await apiFetch(`/api/v2/next?pass=${PASS_N}${current}`);
  const data = await res.json();
  if (!data.next) {
    state.currentName = null;
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

function goContinueStep() {
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
      if (state.ballInPlay === false) {
        state.sessionSubstep = SESSION_SUBSTEP.BALL_VISIBLE;
        syncUi();
        return;
      }
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
      state.geometrySubstep = GEOMETRY_SUBSTEP.LINES;
      syncUi();
      render();
      return;
    }
    if (!allGeometryResolved() && placedCourtLineCount() < 3) {
      state.currentSlot = nextUnresolvedSlot();
      syncUi();
      render();
      return;
    }
  }
  const idx = stepIndex();
  if (idx >= STEP_ORDER.length - 1) return;
  let next = STEP_ORDER[idx + 1];
  if (state.wizardStep === STEP.PEOPLE && !needsBallStep()) {
    next = STEP.IGNORE;
  }
  state.wizardStep = next;
  syncUi();
  render();
}

function bindEvents() {
  buildChoiceButtons(personRoleButtons, ["player", "referee"], (v) => (
    v === "player" ? "Player" : "Referee"
  ), (value) => updateChoiceButtons(personRoleButtons, value));
  btnContinueStep.addEventListener("click", goContinueStep);
  btnCompleteNext.addEventListener("click", async () => {
    btnCompleteNext.disabled = true;
    try {
      const result = await saveLabel(true);
      if (result?.next_image) await loadImage(result.next_image);
      else await loadNextImage();
    } finally {
      btnCompleteNext.disabled = false;
    }
  });
  btnSkipStep.addEventListener("click", skipCurrentStep);

  cv.addEventListener("mousedown", onPointerDown);
  window.addEventListener("mousemove", onPointerMove);
  window.addEventListener("mouseup", onPointerUp);
  cv.addEventListener("mouseleave", onPointerLeave);
  cv.addEventListener("contextmenu", onCanvasContextMenu);
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

  stepBallPanel.innerHTML = '<p class="wizard-copy">Draw the active ball on the canvas, then continue.</p>';
  stepIgnorePanel.innerHTML = '<p class="wizard-copy">Click ignore points on the canvas for non-active balls. When finished, use Complete & next.</p>';

  setInterval(keepAlive, 3 * 60 * 1000);
}

async function boot() {
  bindEvents();
  syncUi();
  await loadNextImage();
}

boot().catch((err) => {
  console.error(err);
  setStatus(`Labeler failed to start: ${err.message}`);
});
