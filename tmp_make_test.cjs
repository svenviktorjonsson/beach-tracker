const fs = require('fs');
const path = 'static/app.js';
const src = fs.readFileSync(path, 'utf8');
function extract(name) {
  const marker = `function ${name}(`;
  const start = src.indexOf(marker);
  if (start < 0) throw new Error(`Missing ${name}`);
  let i = src.indexOf('{', start);
  let depth = 0;
  for (; i < src.length; i++) {
    const ch = src[i];
    if (ch === '{') depth++;
    else if (ch === '}') {
      depth--;
      if (depth === 0) return src.slice(start, i + 1);
    }
  }
  throw new Error(`Unclosed ${name}`);
}
const names = [
  'averagePoint','fitLineStats','sampledLineBendStats','sampleHelperCurve','helperCurveLocalFrame','helperCurveSpanForImage','solveLinear3x3','solveLinear2x2','fitCircleArcPreview','fitRationalArcPreview','bestDistortionHelperPreview','distortPointWithParams','undistortPointWithParams','sortHelperPointsByProjection','estimateDistortionFromPoints'
];
let code = 'const DISTORTION_PREVIEW_POINTS = 5; const MIN_DISTORTION_HELPER_POINTS = 4; const state = { imageW: 1280, imageH: 720 };\n';
for (const n of names) code += extract(n) + '\n';
code += `
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
`;
fs.writeFileSync('tmp_distortion_test.cjs', code);
console.log('written');
