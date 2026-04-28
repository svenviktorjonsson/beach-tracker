# Handover: neural network training (beach volleyball labeler)

This document is for an LLM or developer **starting work on ML models** for this repo. The interactive labeler lives under `beach-algo/`; the parent repo is `beach-tracker`.

## Goal

Train models that **predict label structure from images** (and eventually video frames), aligned with how humans annotate in the browser. The product owner wants **CSV as the canonical tabular export** for training (one file per label *type*, long format: one row per instance where applicable). JSON in `data/annotations/` is the **current** source of truth from the tool; **your job** is to define stable **CSV schemas + export/import** and **training code** (likely PyTorch).

## What is being labeled (schema)

Images are PNG/JPEG under `data/images/` (Docker: `/data/images/`). Annotations are JSON per image stem, dual-pass:

- `data/annotations/<stem>_r1.json`, `<stem>_r2.json` — two annotator passes.
- When agreement passes: `data/annotations/<stem>_merged.json` (see `agreement.py`).
- Legacy: `<stem>.json` may exist (treated as pass 1).

**Fields in saved JSON** (see `server.py` `AnnotationPayload` / `BallItem`):

| Layer | Content |
|-------|---------|
| `image`, `image_width`, `image_height`, `complete`, `pass` | Metadata |
| `exclusion_zones` | List of polygons: `{ "points": [[x,y], ...] }` — **global ignore** regions (crowd, sky). |
| `court_polylines` | `{ "points": [...], "scope": "main" \| "adjacent" }` — court lines; adjacent = side court / not primary. |
| `net_polylines` | Same shape; scope **main** vs **adjacent** (our net vs other net). |
| `balls` | Ellipse: `cx`, `cy`, `rx`, `ry`, `angle` (radians), `scope`: **`in_play`** vs **`other`** (spare ball / not rally ball). Legacy `r` only → circle (`rx=ry=r`, `angle=0`). |
| `people` | Axis-aligned box: `x`, `y`, `w`, `h`, `role`: **`player`**, **`referee`**, **`other`**. |

**Quality / agreement** (for understanding which labels are “trusted”, not necessarily for v1 training):

- `data/agreement_calibration.json` — threshold from first N pairs.
- `data/label_quality.json` — pair failures, `excluded_from_training` list.

**Streams list (YouTube sources):**

- `data/streams_export.csv` (or repo copy next to app): columns `name`, `youtube_url`, `date`, `court`, `duration_minutes`.
- Frames are **pre-downloaded** with `pull_youtube_frames.py` or `streams_youtube.py` into `data/images/` (requires **yt-dlp** + **ffmpeg**). The labeler does not download from YouTube.

## Data splits and future data (owner intent)

- **Current labeled exports** (`data/annotations/`, merged where applicable) are **valid training data** unless an image is marked excluded (see `label_quality.json` / HARD pile in `agreement.py`).
- **More training data** will be added over time; training code should assume **growing manifests** and stable CSV schemas (version bumps if columns change).
- **Additional verification / hold-out data** may arrive later — reserve the ability to define **explicit train vs val vs test** splits (e.g. by stem list or CSV column) rather than assuming a single static folder is “all train”.

## Product constraints (from discussions)

1. **CSV-first for training tables:** separate files per type (e.g. `balls.csv`, `people.csv`, polylines as point rows or WKT) unless a single denormalized file is explicitly chosen.
2. **Partial labels are valid:** an image may have only balls, or only court, etc.; **empty** = negative for that type unless you add an explicit manifest.
3. **Scope / roles matter:** `main` vs `adjacent`, ball `in_play` vs `other`, person `role` — treat as **classes or attributes** in the detector/segmenter.
4. **Exclusion zones** are **spatial ignore**, not per-class “negative ball” semantics.
5. **Model size:** prefer **small / efficient** models (e.g. YOLO nano/small class) trainable **locally**; weights are **MB-scale**; training RAM is dominated by batch/resolution, not checkpoint size.
6. **Joint vs separate training:** **Balls + people** often one **multi-class detector**; **court/net** and **exclusion** are usually **separate** heads or models (different geometry/loss).

## Code map (read first)

| Path | Role |
|------|------|
| `beach-algo/server.py` | Pydantic schemas, `/api/annotation`, `/api/image` |
| `beach-algo/pull_youtube_frames.py` | Batch YouTube → `data/images/` (offline labeling) |
| `beach-algo/agreement.py` | Distance between two label JSONs, merge when similar; `merge_balls` uses ellipse params |
| `beach-algo/streams_youtube.py` | YouTube download + random frame PNGs → `data/images/` |
| `beach-algo/static/app.js` | Client labeling UI (ellipse balls, polylines, etc.) |
| `beach-algo/docker-compose.yml` | `docker compose up --build` → `http://localhost:8080` |

## Suggested next steps (implementation order)

1. **Export script:** JSON (`_merged.json` or `_r1` + rules) → **CSV files** per type with stable column names + `image_path` / `stem` keys.
2. **Training dataset class:** PyTorch `Dataset` reading CSV + images; map ellipses to **boxes** or **5-D ellipse head** later; polylines to masks or keypoints.
3. **Model v1:** small object detector for **ball** (multi-class: in_play vs other) + **person** (roles); train jointly; log metrics.
4. **Optional v2:** Court/net as segmentation or polyline; exclusion as mask or ignore during loss.
5. **Checkpoints:** `torch.save({ "model_state_dict", "optimizer_state_dict", "config", "label_schema_version" }, path)`.

## Environment

- **Docker:** labeler image includes **ffmpeg** and **yt-dlp** (`requirements.txt`). Host: `beach-algo/data` → `/data`.
- **Local run:** `run_labeler.ps1` runs `docker compose up --build -d` and opens the browser (see script).

## Open decisions for the NN workstream

- Whether **merged** JSON only vs either pass for training.
- Exact **CSV column names** and **polyline encoding** (multi-row vs WKT).
- **Video** vs **single-frame** model first; inference FPS target.

## PyTorch training (implemented)

**Location:** `beach-algo/training/`.

**Canonical training tables (CSV):** one file per label type, long format for polylines. Stable column names are defined in `training/csv_schema.py`. Export from the labeler JSON on disk:

```text
set BEACH_DATA_DIR=.\data
python -m training.export_json_to_csv
```

Writes `data/training_csv/` (override with `--out-dir`): `images_manifest.csv`, `balls.csv`, `people.csv`, `court_polylines.csv`, `net_polylines.csv`, `exclusion_zones.csv`.

**Training reads either:**

- **`--train-source csv`** — images from `data/images/` plus labels rebuilt from the CSV tables (recommended; matches the CSV-first product constraint).
- **`--train-source json`** (default) — reads `data/annotations/` directly (prefers `*_merged.json`, then `*_r1.json`, then legacy `*.json`) without an export step.

**Model:** `BeachVolleyballMultiTask` — Faster R-CNN MobileNetV3-FPN (balls + people with scopes/roles) + LR-ASPP segmentation (court/net polylines + exclusion polygons rasterized to a class mask). Class names are in `training/schema.py`.

**Dependencies:** install labeler deps, then `pip install -r requirements-train.txt` (or install `torch`, `torchvision`, `numpy`, `Pillow`, `opencv-python-headless` per your CUDA/CPU build).

**Run (from `beach-algo/`):**

```text
set BEACH_DATA_DIR=.\data
python -m training.export_json_to_csv
python -m training.train --train-source csv --epochs 10 --batch-size 2 --out checkpoints\beach_multitask.pt
```

JSON-only (no export): `python -m training.train --train-source json ...`

Use `--no-pretrained` for random initialization (smoke tests without downloading COCO/VOC weights). Checkpoint includes `model_state_dict`, `optimizer_state_dict`, and `config` with `label_schema_version`, `train_source`, and `csv_dir` when applicable.

---

*End of handover. Extend this file when you lock schemas and training commands.*
