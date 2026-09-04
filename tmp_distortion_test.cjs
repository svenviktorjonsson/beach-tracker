const DISTORTION_PREVIEW_POINTS = 5; const MIN_DISTORTION_HELPER_POINTS = 4; const state = { imageW: 1280, imageH: 720 };
function averagePoint(points) {
  if (!Array.isArray(points) || points.length === 0) return [0, 0];
  const sum = points.reduce((acc, point) => [acc[0] + point[0], acc[1] + point[1]], [0, 0]);
  return [sum[0] / points.length, sum[1] / points.length];
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
function sampleHelperCurve(points, samplesPerSegment = 10) {
  if (!Array.isArray(points) || points.length < 2) return Array.isArray(points) ? points.slice() : [];
  if (points.length === 2) {
    const out = [];
    for (let i = 0; i <= samplesPerSegment; i += 1) {
      const t = i / samplesPerSegment;
      out.push([
        points[0][0] * (1 - t) + points[1][0] * t,
        points[0][1] * (1 - t) + points[1][1] * t,
      ]);
    }
    return out;
  }
  const catmull = (p0, p1, p2, p3, t) => {
    const t2 = t * t;
    const t3 = t2 * t;
    return 0.5 * (
      (2 * p1)
      + (-p0 + p2) * t
      + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
      + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    );
  };
  const out = [];
  for (let seg = 0; seg < points.length - 1; seg += 1) {
    const p0 = points[Math.max(0, seg - 1)];
    const p1 = points[seg];
    const p2 = points[seg + 1];
    const p3 = points[Math.min(points.length - 1, seg + 2)];
    const start = seg === 0 ? 0 : 1;
    for (let i = start; i <= samplesPerSegment; i += 1) {
      const t = i / samplesPerSegment;
      out.push([
        catmull(p0[0], p1[0], p2[0], p3[0], t),
        catmull(p0[1], p1[1], p2[1], p3[1], t),
      ]);
    }
  }
  return out;
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
function fitCircleArcPreview(points) {
  const frame = helperCurveLocalFrame(points);
  if (!frame || frame.locals.length < 3) return null;
  let sX = 0;
  let sY = 0;
  let sXX = 0;
  let sYY = 0;
  let sXY = 0;
  let sXXX = 0;
  let sYYY = 0;
  let sXXY = 0;
  let sXYY = 0;
  for (const point of frame.locals) {
    const x = point.x;
    const y = point.y;
    const xx = x * x;
    const yy = y * y;
    sX += x;
    sY += y;
    sXX += xx;
    sYY += yy;
    sXY += x * y;
    sXXX += xx * x;
    sYYY += yy * y;
    sXXY += xx * y;
    sXYY += x * yy;
  }
  const n = frame.locals.length;
  const sol = solveLinear3x3(
    [
      [sXX, sXY, sX],
      [sXY, sYY, sY],
      [sX, sY, n],
    ],
    [
      -(sXXX + sXYY),
      -(sXXY + sYYY),
      -(sXX + sYY),
    ],
  );
  if (!sol) return null;
  const [d, e, f] = sol;
  const cx = -d / 2;
  const cy = -e / 2;
  const radiusSq = cx * cx + cy * cy - f;
  if (!Number.isFinite(radiusSq) || radiusSq <= 1) return null;
  const branchSse = [1, -1].map((branch) => {
    let sse = 0;
    for (const point of frame.locals) {
      const inside = radiusSq - (point.x - cx) * (point.x - cx);
      if (inside <= 0) return Number.POSITIVE_INFINITY;
      const yHat = cy + branch * Math.sqrt(inside);
      const err = point.y - yHat;
      sse += err * err;
    }
    return sse;
  });
  const branch = branchSse[0] <= branchSse[1] ? 1 : -1;
  const span = helperCurveSpanForImage(frame, state.imageW, state.imageH)
    || {
      minX: Math.min(...frame.locals.map((point) => point.x)),
      maxX: Math.max(...frame.locals.map((point) => point.x)),
    };
  const steps = Math.max(48, frame.locals.length * 10);
  const samples = [];
  for (let i = 0; i <= steps; i += 1) {
    const x = span.minX + ((span.maxX - span.minX) * i) / steps;
    const inside = radiusSq - (x - cx) * (x - cx);
    if (inside <= 0) continue;
    const y = cy + branch * Math.sqrt(inside);
    samples.push([
      frame.mid[0] + frame.dir[0] * x + frame.normal[0] * y,
      frame.mid[1] + frame.dir[1] * x + frame.normal[1] * y,
    ]);
  }
  return {
    kind: "circle arc",
    sse: Math.min(...branchSse),
    samples,
  };
}
function fitRationalArcPreview(points) {
  const frame = helperCurveLocalFrame(points);
  if (!frame || frame.locals.length < 3) return null;
  const span = Math.max(...frame.locals.map((point) => Math.abs(point.x))) || 1;
  const candidates = [0, 0.15, 0.35, 0.7, 1.25, 2, 3.5, 5, 8, 12];
  let best = null;
  for (const b of candidates) {
    let s00 = 0;
    let s01 = 0;
    let s11 = 0;
    let t0 = 0;
    let t1 = 0;
    const basis = [];
    for (const point of frame.locals) {
      const xn = point.x / span;
      const g = 1 / (1 + b * xn * xn);
      basis.push(g);
      s00 += 1;
      s01 += g;
      s11 += g * g;
      t0 += point.y;
      t1 += point.y * g;
    }
    const sol = solveLinear2x2(s00, s01, s11, t0, t1);
    if (!sol) continue;
    const [c, a] = sol;
    let sse = 0;
    for (let i = 0; i < frame.locals.length; i += 1) {
      const err = frame.locals[i].y - (c + a * basis[i]);
      sse += err * err;
    }
    if (!best || sse < best.sse) {
      best = { b, c, a, sse };
    }
  }
  if (!best) return null;
  const fullSpan = helperCurveSpanForImage(frame, state.imageW, state.imageH)
    || {
      minX: Math.min(...frame.locals.map((point) => point.x)),
      maxX: Math.max(...frame.locals.map((point) => point.x)),
    };
  const steps = Math.max(48, frame.locals.length * 10);
  const samples = [];
  for (let i = 0; i <= steps; i += 1) {
    const x = fullSpan.minX + ((fullSpan.maxX - fullSpan.minX) * i) / steps;
    const xn = x / span;
    const y = best.c + best.a / (1 + best.b * xn * xn);
    samples.push([
      frame.mid[0] + frame.dir[0] * x + frame.normal[0] * y,
      frame.mid[1] + frame.dir[1] * x + frame.normal[1] * y,
    ]);
  }
  return {
    kind: "a/(1+br^2)",
    sse: best.sse,
    samples,
  };
}
function bestDistortionHelperPreview(points) {
  if (!Array.isArray(points) || points.length < DISTORTION_PREVIEW_POINTS) return null;
  const fits = [fitCircleArcPreview(points), fitRationalArcPreview(points)].filter(Boolean);
  if (fits.length === 0) return null;
  fits.sort((a, b) => a.sse - b.sse);
  return fits[0];
}
function distortPointWithParams(x, y, width, height, params) {
  const cx = width / 2 + Number(params.cx_offset || 0);
  const cy = height / 2 + Number(params.cy_offset || 0);
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
  const cx = width / 2 + Number(params.cx_offset || 0);
  const cy = height / 2 + Number(params.cy_offset || 0);
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
function estimateDistortionFromPoints(points) {
  if (points.length < MIN_DISTORTION_HELPER_POINTS) return null;
  const preview = bestDistortionHelperPreview(points);
  const orderedPoints = sortHelperPointsByProjection(points);
  const helperSamples = (preview?.samples && preview.samples.length >= 12)
    ? preview.samples
    : sampleHelperCurve(orderedPoints, 12);
  const baselineBend = sampledLineBendStats(helperSamples);
  const lineImprovesMonotonically = (candidatePoints) => {
    const candidateBend = sampledLineBendStats(candidatePoints);
    if (candidateBend.maxAbsDeviation > baselineBend.maxAbsDeviation + 0.25) return false;
    if (candidateBend.bendEnergy > baselineBend.bendEnergy + 0.5) return false;
    for (let i = 0; i < candidateBend.absDeviations.length; i += 1) {
      const base = baselineBend.absDeviations[i] ?? baselineBend.maxAbsDeviation;
      if (candidateBend.absDeviations[i] > base + 0.25) return false;
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
      + 0.05 * Math.abs(cxOffsetNorm)
      + 0.05 * Math.abs(cyOffsetNorm)
      + 0.2 * (cxOffsetNorm * cxOffsetNorm + cyOffsetNorm * cyOffsetNorm);
    return (
      stats.error * 0.15
      + stats.maxAbsDeviation * 2.8
      + stats.bendEnergy * 1.35
      + complexityPenalty
      + edgeStretchPenalty * 20.0
    );
  };
  const score = (params) => {
    const undistorted = helperSamples.map((point) => (
      undistortPointWithParams(point[0], point[1], state.imageW, state.imageH, params)
    ));
    if (!lineImprovesMonotonically(undistorted)) {
      return { error: Number.POSITIVE_INFINITY, theta: 0, undistorted, objective: Number.POSITIVE_INFINITY };
    }
    const stats = fitLineStats(undistorted);
    const bendStats = sampledLineBendStats(undistorted);
    const mergedStats = { ...stats, ...bendStats, undistorted };
    return { ...mergedStats, objective: regularizedScore(params, mergedStats) };
  };
  const clampParams = (params) => ({
    model: "division",
    lambda: Math.max(-0.28, Math.min(0.28, Number(params.lambda || 0))),
    cx_offset: Math.max(-0.18 * state.imageW, Math.min(0.18 * state.imageW, Number(params.cx_offset || 0))),
    cy_offset: Math.max(-0.18 * state.imageH, Math.min(0.18 * state.imageH, Number(params.cy_offset || 0))),
    k1: 0,
    k2: 0,
    k3: 0,
  });
  let current = clampParams({ lambda: 0, cx_offset: 0, cy_offset: 0 });
  let currentScore = score(current).objective;
  const coarseLambdaGrid = [-0.26, -0.2, -0.15, -0.1, -0.06, -0.03, 0, 0.03, 0.06, 0.1, 0.15, 0.2, 0.26];
  const coarseCxGrid = [-0.12, -0.06, 0, 0.06, 0.12].map((step) => step * state.imageW);
  const coarseCyGrid = [-0.12, -0.06, 0, 0.06, 0.12].map((step) => step * state.imageH);
  for (const lambda of coarseLambdaGrid) {
    for (const cxOffset of coarseCxGrid) {
      for (const cyOffset of coarseCyGrid) {
        const candidate = clampParams({ lambda, cx_offset: cxOffset, cy_offset: cyOffset });
        const candidateScore = score(candidate).objective;
        if (candidateScore < currentScore) {
          current = candidate;
          currentScore = candidateScore;
        }
      }
    }
  }
  const lambdaSteps = [0.08, 0.03, 0.012, 0.005, 0.002];
  const cxSteps = [0.08, 0.04, 0.02, 0.01].map((step) => step * state.imageW);
  const cySteps = [0.08, 0.04, 0.02, 0.01].map((step) => step * state.imageH);
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
          if (candidateScore < currentScore) {
            current = candidate;
            currentScore = candidateScore;
            improved = true;
          }
        }
      }
    }
  }
  const finalStats = score(current);
  const first = finalStats.undistorted[0];
  const last = finalStats.undistorted[finalStats.undistorted.length - 1];
  let rotationDeg = -(Math.atan2(last[1] - first[1], last[0] - first[0]) * 180 / Math.PI);
  while (rotationDeg <= -90) rotationDeg += 180;
  while (rotationDeg > 90) rotationDeg -= 180;
  return { ...current, rotation_deg: rotationDeg };
}

function makeSyntheticLine() {
  const pts = [];
  const cx = state.imageW / 2;
  const cy = state.imageH / 2;
  const norm = Math.max(state.imageW, state.imageH) / 2;
  const width = 900;
  const lambda = 0.22;
  for (let i = 0; i < 8; i++) {
    const ux = cx - width/2 + (width * i) / 7;
    const uy = cy - 180;
    const dx = (ux - cx) / norm;
    const dy = (uy - cy) / norm;
    const r2 = dx*dx + dy*dy;
    const factor = 1 + lambda * r2;
    pts.push([cx + dx * factor * norm, cy + dy * factor * norm]);
  }
  return pts;
}
const pts = makeSyntheticLine();
const preview = bestDistortionHelperPreview(pts);
const fitPts = preview && preview.samples && preview.samples.length >= 12 ? preview.samples : sampleHelperCurve(sortHelperPointsByProjection(pts), 12);
const before = sampledLineBendStats(fitPts);
const params = estimateDistortionFromPoints(pts);
const afterPts = fitPts.map((p) => undistortPointWithParams(p[0], p[1], state.imageW, state.imageH, params));
const after = sampledLineBendStats(afterPts);
console.log(JSON.stringify({ params, before: { max: before.maxAbsDeviation, bend: before.bendEnergy }, after: { max: after.maxAbsDeviation, bend: after.bendEnergy } }, null, 2));
