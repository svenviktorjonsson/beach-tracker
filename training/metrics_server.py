"""HTTP dashboard for live training loss (stdlib only; open the printed URL in a browser)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class TrainingMetrics:
    """Thread-safe loss history for the dashboard and /metrics.json."""

    def __init__(self, human_baseline: dict[str, Any] | None = None) -> None:
        self._lock = threading.Lock()
        self._step = 0
        self.human_baseline = human_baseline
        self.epochs: list[int] = []
        self.loss_det_epoch: list[float] = []
        self.loss_seg_epoch: list[float] = []
        self.steps: list[int] = []
        self.loss_det_step: list[float] = []
        self.loss_seg_step: list[float] = []

    def record_batch(self, loss_det: float, loss_seg: float) -> None:
        with self._lock:
            self._step += 1
            self.steps.append(self._step)
            self.loss_det_step.append(loss_det)
            self.loss_seg_step.append(loss_seg)

    def record_epoch(self, epoch: int, loss_det: float, loss_seg: float) -> None:
        with self._lock:
            self.epochs.append(epoch)
            self.loss_det_epoch.append(loss_det)
            self.loss_seg_epoch.append(loss_seg)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "epochs": list(self.epochs),
                "loss_det_epoch": list(self.loss_det_epoch),
                "loss_seg_epoch": list(self.loss_seg_epoch),
                "steps": list(self.steps),
                "loss_det_step": list(self.loss_det_step),
                "loss_seg_step": list(self.loss_seg_step),
                "human_baseline": self.human_baseline,
            }


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Beach training — loss</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root { color-scheme: dark light; font-family: system-ui, sans-serif; }
    body { margin: 0; padding: 1rem 1.25rem; max-width: 1100px; margin-inline: auto; }
    h1 { font-size: 1.15rem; font-weight: 600; margin: 0 0 0.5rem; }
    p { color: #666; font-size: 0.9rem; margin: 0 0 0.75rem; }
    .note { font-size: 0.82rem; color: #555; line-height: 1.45; margin-bottom: 1rem; }
    .grid { display: grid; gap: 1.25rem; }
    @media (min-width: 720px) { .grid { grid-template-columns: 1fr 1fr; } }
    .card { border: 1px solid #ccc4; border-radius: 8px; padding: 0.75rem; background: #fff2; }
    .card h2 { font-size: 0.95rem; margin: 0 0 0.5rem; }
    canvas { max-height: 320px; }
  </style>
</head>
<body>
  <h1>Training vs human repeatability</h1>
  <p>Single scale <strong>0–1</strong>: dashed lines are inter-annotator distance (pass 1 vs pass 2), from <code>agreement.py</code> (lower = humans agree more). Solid lines are training losses <strong>divided by the max loss on that chart</strong> so they sit in the same numeric range — useful to see whether normalized loss approaches human disagreement, not a literal equality of loss vs chamfer.</p>
  <p class="note" id="hbNote"></p>
  <div class="grid">
    <div class="card"><h2>Batch</h2><canvas id="chartBatch"></canvas></div>
    <div class="card"><h2>Epoch (mean)</h2><canvas id="chartEpoch"></canvas></div>
  </div>
  <script>
    function fillConst(n, v) { const a = []; for (let i = 0; i < n; i++) a.push(v); return a; }
    function clamp01(x) { return Math.max(0, Math.min(1, x)); }
    function maxOf(arr) {
      if (!arr || !arr.length) return 1e-9;
      let m = 1e-9;
      for (let i = 0; i < arr.length; i++) if (arr[i] > m) m = arr[i];
      return m;
    }
    function normByMax(arr, mx) {
      const d = Math.max(mx, 1e-9);
      return arr.map(v => v / d);
    }
    function humanLineDatasets(h, n) {
      if (!h || !h.pairs) return [];
      const out = [];
      const line = (label, v, color, dash) => ({
        label, data: fillConst(n, v), borderColor: color, borderDash: dash,
        pointRadius: 0, borderWidth: 2, tension: 0
      });
      out.push(line('human det μ (r1↔r2)', h.det_mean, '#ea580c', [6, 4]));
      if (h.det_std > 1e-9) out.push(line('human det μ+σ', clamp01(h.det_mean + h.det_std), '#fb923c', [2, 4]));
      out.push(line('human seg μ (court+net+ex)', h.seg_mean, '#7c3aed', [6, 4]));
      if (h.seg_std > 1e-9) out.push(line('human seg μ+σ', clamp01(h.seg_mean + h.seg_std), '#a78bfa', [2, 4]));
      out.push(line('human net μ (polylines÷diag)', h.net_mean, '#0d9488', [8, 2]));
      if (h.net_std > 1e-9) out.push(line('human net μ+σ', clamp01(h.net_mean + h.net_std), '#2dd4bf', [2, 4]));
      return out;
    }
    function lossDatasets(detNorm, segNorm, batch) {
      return [
        { label: batch ? 'loss_det ÷ max (batch)' : 'loss_det ÷ max (epoch)', data: detNorm,
          borderColor: '#2563eb', tension: batch ? 0.1 : 0.15, pointRadius: batch ? 0 : 2 },
        { label: batch ? 'loss_seg ÷ max (batch)' : 'loss_seg ÷ max (epoch)', data: segNorm,
          borderColor: '#16a34a', tension: batch ? 0.1 : 0.15, pointRadius: batch ? 0 : 2 }
      ];
    }
    function yScale01() {
      return {
        min: 0,
        max: 1,
        title: { display: true, text: 'Scale 0–1 (human); loss normalized by chart max' }
      };
    }
    const batchCtx = document.getElementById('chartBatch');
    const epochCtx = document.getElementById('chartEpoch');
    const chartBatch = new Chart(batchCtx, {
      type: 'line',
      data: { labels: [], datasets: [] },
      options: {
        responsive: true,
        animation: { duration: 0 },
        interaction: { mode: 'index', intersect: false },
        scales: { x: { title: { display: true, text: 'Step' } }, y: yScale01() }
      }
    });
    const chartEpoch = new Chart(epochCtx, {
      type: 'line',
      data: { labels: [], datasets: [] },
      options: {
        responsive: true,
        animation: { duration: 200 },
        interaction: { mode: 'index', intersect: false },
        scales: { x: { title: { display: true, text: 'Epoch' } }, y: yScale01() }
      }
    });
    async function pull() {
      try {
        const r = await fetch('/metrics.json', { cache: 'no-store' });
        const m = await r.json();
        const h = m.human_baseline;
        const note = document.getElementById('hbNote');
        if (h && h.pairs) {
          note.textContent = 'Human baselines from ' + h.pairs + ' image pair(s) with both _r1 and _r2: '
            + 'det μ=' + h.det_mean.toFixed(4) + ' σ=' + h.det_std.toFixed(4) + '; '
            + 'seg μ=' + h.seg_mean.toFixed(4) + ' σ=' + h.seg_std.toFixed(4) + '; '
            + 'net μ=' + h.net_mean.toFixed(4) + ' σ=' + h.net_std.toFixed(4) + '. '
            + (h.note || '');
        } else {
          note.textContent = 'No human reference lines yet: add pairs of annotations (same stem, _r1.json and _r2.json) under data/annotations to compute inter-annotator spread.';
        }
        const ns = m.steps.length;
        const ne = m.epochs.length;
        const mxB = Math.max(maxOf(m.loss_det_step), maxOf(m.loss_seg_step), 1e-9);
        const mxE = Math.max(maxOf(m.loss_det_epoch), maxOf(m.loss_seg_epoch), 1e-9);
        chartBatch.data.labels = m.steps;
        chartBatch.data.datasets = [
          ...humanLineDatasets(h, ns),
          ...lossDatasets(normByMax(m.loss_det_step, mxB), normByMax(m.loss_seg_step, mxB), true)
        ];
        chartBatch.options.scales = { x: { title: { display: true, text: 'Step' } }, y: yScale01() };
        chartBatch.update('none');
        chartEpoch.data.labels = m.epochs;
        chartEpoch.data.datasets = [
          ...humanLineDatasets(h, ne),
          ...lossDatasets(normByMax(m.loss_det_epoch, mxE), normByMax(m.loss_seg_epoch, mxE), false)
        ];
        chartEpoch.options.scales = { x: { title: { display: true, text: 'Epoch' } }, y: yScale01() };
        chartEpoch.update();
      } catch (e) { /* ignore between polls */ }
    }
    pull();
    setInterval(pull, 1000);
  </script>
</body>
</html>
"""


def _make_handler(metrics: TrainingMetrics):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                body = _DASHBOARD_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/metrics.json":
                payload = json.dumps(metrics.snapshot()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            else:
                self.send_error(404)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def start_metrics_server(metrics: TrainingMetrics, host: str, port: int) -> ThreadingHTTPServer:
    """Start a daemon threaded HTTP server; returns the server (caller may ignore)."""
    handler = _make_handler(metrics)
    server = ThreadingHTTPServer((host, port), handler, bind_and_activate=False)
    server.daemon_threads = True
    server.allow_reuse_address = True
    server.server_bind()
    server.server_activate()

    def _run() -> None:
        server.serve_forever()

    t = threading.Thread(target=_run, name="metrics-http", daemon=True)
    t.start()
    return server
