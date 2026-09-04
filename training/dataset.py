"""PyTorch Dataset from labeler JSON + images under `data/`."""

from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .geometry import ellipse_axis_aligned_box
from .model import MAX_COURT_PTS, MAX_NET_PTS
from .schema import DETECTION_CLASS_NAMES

_DET_LABEL = {name: i + 1 for i, name in enumerate(DETECTION_CLASS_NAMES)}

# In-process LRU cache of packed samples (resize + tensor targets) to avoid re-reading PNG/JSON
# every epoch — big win on slow disks / cloud sync folders. Set BEACH_TRAIN_PACK_CACHE_MAX=0 to disable.
_PACKED_CACHE_MAX = int(os.environ.get("BEACH_TRAIN_PACK_CACHE_MAX", "256"))
_PACKED_CACHE: OrderedDict[tuple, dict[str, Any]] = OrderedDict()


def _packed_cache_key(
    stem: str,
    image_size: int,
    ann_path: Path,
    img_path: Path,
) -> tuple | None:
    try:
        am = ann_path.stat().st_mtime_ns
        im = img_path.stat().st_mtime_ns
    except OSError:
        return None
    return (stem, image_size, am, im)


def _clone_packed_sample(s: dict[str, Any]) -> dict[str, Any]:
    return {
        "stem": s["stem"],
        "image": s["image"].clone(),
        "boxes": s["boxes"].clone(),
        "labels": s["labels"].clone(),
        "court_pts": s["court_pts"].clone(),
        "court_mask": s["court_mask"].clone(),
        "net_pts": s["net_pts"].clone(),
        "net_mask": s["net_mask"].clone(),
        "ball_center": s["ball_center"].clone(),
        "ball_radius": s["ball_radius"].clone(),
        "ball_size_labeled": s["ball_size_labeled"].clone(),
        "ball_visible": s["ball_visible"].clone(),
        "orig_size": s["orig_size"].clone(),
        "distortion_params": s["distortion_params"].clone(),
    }


def _packed_cache_get(key: tuple) -> dict[str, Any] | None:
    if _PACKED_CACHE_MAX <= 0 or key not in _PACKED_CACHE:
        return None
    _PACKED_CACHE.move_to_end(key)
    return _clone_packed_sample(_PACKED_CACHE[key])


def _packed_cache_put(key: tuple, sample: dict[str, Any]) -> None:
    if _PACKED_CACHE_MAX <= 0:
        return
    _PACKED_CACHE[key] = _clone_packed_sample(sample)
    _PACKED_CACHE.move_to_end(key)
    while len(_PACKED_CACHE) > _PACKED_CACHE_MAX:
        _PACKED_CACHE.popitem(last=False)


def _data_root() -> Path:
    return Path(os.environ.get("BEACH_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))


def _load_quality_excluded(data_root: Path) -> set[str]:
    q = data_root / "label_quality.json"
    if not q.is_file():
        return set()
    data = json.loads(q.read_text(encoding="utf-8"))
    ex = data.get("excluded_from_training") or []
    return set(ex) if isinstance(ex, list) else set()


def _pick_annotation(ann_dir: Path, stem: str) -> Path | None:
    for name in (f"{stem}_merged.json", f"{stem}_r1.json", f"{stem}.json"):
        p = ann_dir / name
        if p.is_file():
            return p
    return None


def _ball_label(scope: str) -> int:
    s = (scope or "in_play").lower()
    if s == "other":
        return _DET_LABEL["ball_other"]
    return _DET_LABEL["ball_in_play"]


def _person_label(role: str) -> int:
    r = (role or "player").lower()
    if r == "referee":
        return _DET_LABEL["referee"]
    if r == "other":
        return _DET_LABEL["other_person"]
    return _DET_LABEL["player"]


def _boxes_and_labels_from_ann(
    ann: dict[str, Any], w: int, h: int
) -> tuple[torch.Tensor, torch.Tensor]:
    boxes: list[list[float]] = []
    labels: list[int] = []

    for b in ann.get("balls") or []:
        if "w" in b and "x" in b:
            x1, y1 = float(b["x"]), float(b["y"])
            x2, y2 = x1 + float(b["w"]), y1 + float(b["h"])
        else:
            cx, cy = float(b["cx"]), float(b["cy"])
            if "rx" not in b and "r" in b:
                r = float(b["r"])
                rx, ry, ang = r, r, 0.0
            else:
                rx = float(b.get("rx", b.get("r", 1.0)))
                ry = float(b.get("ry", b.get("r", 1.0)))
                ang = float(b.get("angle", 0.0))
            x1, y1, x2, y2 = ellipse_axis_aligned_box(cx, cy, rx, ry, ang)
        x1 = max(0.0, min(float(w - 1), x1))
        y1 = max(0.0, min(float(h - 1), y1))
        x2 = max(0.0, min(float(w - 1), x2))
        y2 = max(0.0, min(float(h - 1), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(_ball_label(b.get("scope", "in_play")))

    for p in ann.get("people") or []:
        x, y = float(p["x"]), float(p["y"])
        ww, hh = float(p["w"]), float(p["h"])
        x1, y1, x2, y2 = x, y, x + ww, y + hh
        x1 = max(0.0, min(float(w - 1), x1))
        y1 = max(0.0, min(float(h - 1), y1))
        x2 = max(0.0, min(float(w - 1), x2))
        y2 = max(0.0, min(float(h - 1), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(_person_label(p.get("role", "player")))

    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)


def _extract_keypoints(
    ann: dict[str, Any],
    w: int,
    h: int,
    max_court: int = MAX_COURT_PTS,
    max_net: int = MAX_NET_PTS,
) -> dict[str, torch.Tensor]:
    """Extract court orthopoints and net anchor points as normalised [0,1] coords."""

    def _fit_line_orthopoint(points: list[list[float]]) -> list[float] | None:
        if len(points) < 2:
            return None
        n = float(len(points))
        cx = sum(float(p[0]) for p in points) / n
        cy = sum(float(p[1]) for p in points) / n
        sxx = 0.0
        sxy = 0.0
        syy = 0.0
        for px, py in points:
            dx = float(px) - cx
            dy = float(py) - cy
            sxx += dx * dx
            sxy += dx * dy
            syy += dy * dy
        theta = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
        dx = math.cos(theta)
        dy = math.sin(theta)
        nx = -dy
        ny = dx
        norm = math.hypot(nx, ny) or 1.0
        nx /= norm
        ny /= norm
        rho = cx * nx + cy * ny
        if rho < 0:
            rho = -rho
            nx = -nx
            ny = -ny
        ox = nx * rho
        oy = ny * rho
        return [ox / float(w), oy / float(h)]

    def _extract_court_orthopoints() -> tuple[torch.Tensor, torch.Tensor]:
        pts: list[list[float]] = []
        for pl in ann.get("court_polylines") or []:
            scope = (pl.get("scope") or "main").lower() if isinstance(pl, dict) else "main"
            if scope != "main":
                continue
            raw_points = pl.get("points") if isinstance(pl, dict) else pl
            if not isinstance(raw_points, list):
                continue
            line_points = [
                [float(p[0]), float(p[1])]
                for p in raw_points
                if isinstance(p, (list, tuple)) and len(p) >= 2
            ]
            fit = _fit_line_orthopoint(line_points)
            if fit is not None:
                pts.append(fit)
            if len(pts) >= max_court:
                break
        n = len(pts)
        while len(pts) < max_court:
            pts.append([0.0, 0.0])
        coords = torch.tensor(pts[:max_court], dtype=torch.float32)
        mask = torch.zeros(max_court, dtype=torch.bool)
        mask[:n] = True
        return coords, mask

    def _gather(polylines_key: str, max_pts: int) -> tuple[torch.Tensor, torch.Tensor]:
        pts: list[list[float]] = []
        for pl in ann.get(polylines_key) or []:
            scope = (pl.get("scope") or "main").lower() if isinstance(pl, dict) else "main"
            if scope != "main":
                continue
            raw_points = pl.get("points") if isinstance(pl, dict) else pl
            if not isinstance(raw_points, list):
                continue
            for p in raw_points:
                pts.append([float(p[0]) / w, float(p[1]) / h])
                if len(pts) >= max_pts:
                    break
            if pts:
                break  # use first main polyline only
        n = len(pts)
        while len(pts) < max_pts:
            pts.append([0.0, 0.0])
        coords = torch.tensor(pts[:max_pts], dtype=torch.float32)
        mask = torch.zeros(max_pts, dtype=torch.bool)
        mask[:n] = True
        return coords, mask

    court_pts, court_mask = _extract_court_orthopoints()
    net_pts, net_mask = _gather("net_polylines", max_net)
    return {
        "court_pts": court_pts,
        "court_mask": court_mask,
        "net_pts": net_pts,
        "net_mask": net_mask,
    }


def _extract_ball(
    ann: dict[str, Any],
    w: int,
    h: int,
) -> dict[str, torch.Tensor]:
    """Extract active ball hint as normalised [0,1] targets.

    Center is supervised from the label. Radius is only supervised when the
    annotation carries an explicit size label rather than a center-only hint.
    """
    default = {
        "center": torch.zeros(2, dtype=torch.float32),
        "radius": torch.zeros(1, dtype=torch.float32),
        "size_labeled": torch.tensor(False),
        "visible": torch.tensor(False),
    }

    balls = ann.get("balls") or []
    active = None
    for b in balls:
        scope = (b.get("scope") or "in_play").lower()
        if scope == "in_play":
            active = b
            break
    if active is None and balls:
        # Fallback to the first available ball if no explicit in-play label exists.
        active = balls[0]
    if active is None:
        return default

    if "center_x" in active and "center_y" in active and "radius" in active:
        cx = float(active["center_x"])
        cy = float(active["center_y"])
        radius = float(active["radius"])
        ww = float(active.get("w", 2.0 * radius))
        hh = float(active.get("h", 2.0 * radius))
        size_labeled = ww > 3.0 and hh > 3.0 and radius > 1.5
        visible = True
    else:
        if "w" in active and "x" in active:
            x = float(active["x"])
            y = float(active["y"])
            ww = float(active["w"])
            hh = float(active["h"])
        else:
            # Keep compatibility with ellipse exports.
            cx = float(active["cx"])
            cy = float(active["cy"])
            if "rx" not in active and "r" in active:
                r = float(active["r"])
                rx = ry = r
            else:
                rx = float(active.get("rx", active.get("r", 1.0)))
                ry = float(active.get("ry", active.get("r", 1.0)))
            ww = 2.0 * rx
            hh = 2.0 * ry
            x = cx - 0.5 * ww
            y = cy - 0.5 * hh
            visible = True
        size_labeled = ww > 3.0 and hh > 3.0
        cx = x + 0.5 * ww
        cy = y + 0.5 * hh
        radius = 0.5 * (ww + hh) / 2.0
        visible = ww > 0 and hh > 0

    cx = max(0.0, min(float(w), cx))
    cy = max(0.0, min(float(h), cy))
    radius = max(0.0, float(radius))

    # Normalised for  [0,1] in resized square pipeline.
    center = torch.tensor(
        [cx / float(w), cy / float(h)], dtype=torch.float32
    )
    ww_px = float(active.get("w", 2.0 * radius))
    hh_px = float(active.get("h", 2.0 * radius))
    radius_n = torch.tensor([0.25 * (ww_px / float(w) + hh_px / float(h))], dtype=torch.float32)
    return {
        "center": center,
        "radius": radius_n,
        "size_labeled": torch.tensor(bool(size_labeled)),
        "visible": torch.tensor(bool(visible)),
    }


def pack_training_sample(
    image: Image.Image,
    ann: dict[str, Any],
    stem: str,
    image_size: int,
    line_thickness: int,
) -> dict[str, Any]:
    """Resize image and scale boxes; extract keypoints for geometry targets."""
    w, h = image.size
    boxes, labels = _boxes_and_labels_from_ann(ann, w, h)
    kp = _extract_keypoints(ann, w, h)
    ball = _extract_ball(ann, w, h)

    tw = th = image_size
    image_r = image.resize((tw, th), Image.BILINEAR)

    sx = tw / w
    sy = th / h
    if boxes.numel() > 0:
        boxes = boxes.clone()
        boxes[:, 0] *= sx
        boxes[:, 1] *= sy
        boxes[:, 2] *= sx
        boxes[:, 3] *= sy

    img_t = torch.from_numpy(np.array(image_r)).permute(2, 0, 1).float() / 255.0
    dist = ann.get("distortion_params") or {}
    distortion_params = torch.tensor(
        [
            float(dist.get("k1", ann.get("distortion_k", 0.0) or 0.0)),
            float(dist.get("k2", 0.0)),
            float(dist.get("k3", 0.0)),
            float(dist.get("rotation_deg", 0.0)),
        ],
        dtype=torch.float32,
    )

    return {
        "stem": stem,
        "image": img_t,
        "boxes": boxes,
        "labels": labels,
        "court_pts": kp["court_pts"],
        "court_mask": kp["court_mask"],
        "net_pts": kp["net_pts"],
        "net_mask": kp["net_mask"],
        "ball_center": ball["center"],
        "ball_radius": ball["radius"],
        "ball_size_labeled": ball["size_labeled"],
        "ball_visible": ball["visible"],
        "orig_size": torch.tensor([h, w]),
        "distortion_params": distortion_params,
    }


def _resolve_image_file(images_dir: Path, stem: str, image_filename: str | None) -> Path:
    if image_filename:
        p = images_dir / Path(image_filename).name
        if p.is_file():
            return p
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        alt = images_dir / f"{stem}{ext}"
        if alt.is_file():
            return alt
    raise FileNotFoundError(
        f"No image for stem {stem!r} under {images_dir} (image_filename={image_filename!r})"
    )


class BeachAnnotationDataset(Dataset):
    """
    Loads images from `images/` and annotations from `annotations/`
    (merged > r1 > legacy). Returns resized tensors for detection + segmentation.
    """

    def __init__(
        self,
        data_root: Path | None = None,
        image_size: int = 640,
        line_thickness: int = 3,
    ) -> None:
        self.data_root = Path(data_root) if data_root is not None else _data_root()
        self.images_dir = self.data_root / "images"
        self.ann_dir = self.data_root / "annotations"
        self.image_size = image_size
        self.line_thickness = line_thickness
        self._excluded = _load_quality_excluded(self.data_root)

        stems: list[str] = []
        if self.ann_dir.is_dir():
            seen = set()
            for p in sorted(self.ann_dir.iterdir()):
                if p.suffix.lower() != ".json":
                    continue
                raw = p.stem
                for suffix in ("_merged", "_r1"):
                    if raw.endswith(suffix):
                        raw = raw[: -len(suffix)]
                        break
                if raw in seen:
                    continue
                seen.add(raw)
                img_file = self.images_dir / f"{raw}.png"
                if not img_file.is_file():
                    for ext in (".jpg", ".jpeg", ".webp", ".bmp"):
                        alt = self.images_dir / f"{raw}{ext}"
                        if alt.is_file():
                            img_file = alt
                            break
                if not img_file.is_file():
                    continue
                if img_file.name in self._excluded:
                    continue
                stems.append(raw)

        self._stems = stems

    def __len__(self) -> int:
        return len(self._stems)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        stem = self._stems[idx]
        ann_path = _pick_annotation(self.ann_dir, stem)
        assert ann_path is not None

        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        img_name = ann.get("image") or f"{stem}.png"
        img_path = self.images_dir / Path(img_name).name
        if not img_path.is_file():
            for ext in (".png", ".jpg", ".jpeg"):
                alt = self.images_dir / f"{stem}{ext}"
                if alt.is_file():
                    img_path = alt
                    break

        ck = _packed_cache_key(stem, self.image_size, ann_path, img_path)
        if ck is not None:
            hit = _packed_cache_get(ck)
            if hit is not None:
                return hit

        image = Image.open(img_path).convert("RGB")
        sample = pack_training_sample(
            image, ann, stem, self.image_size, self.line_thickness
        )
        if ck is not None:
            _packed_cache_put(ck, sample)
        return sample


class BeachCSVDataset(Dataset):
    """
    Loads the same tensors as `BeachAnnotationDataset` from exported CSV tables
    under ``csv_dir`` (see `export_json_to_csv`). Uses `images_manifest.csv` for
    frame order and filenames when present.
    """

    def __init__(
        self,
        csv_dir: Path,
        data_root: Path | None = None,
        image_size: int = 640,
        line_thickness: int = 3,
    ) -> None:
        from .csv_annotation import build_ann_index, manifest_by_stem, manifest_stems_in_order

        self.csv_dir = Path(csv_dir)
        self.data_root = Path(data_root) if data_root is not None else _data_root()
        self.images_dir = self.data_root / "images"
        self.image_size = image_size
        self.line_thickness = line_thickness

        self._ann_index = build_ann_index(self.csv_dir)
        self._stems = manifest_stems_in_order(self.csv_dir)
        self._manifest = manifest_by_stem(self.csv_dir)
        self._excluded = _load_quality_excluded(self.data_root)
        self._stems = [s for s in self._stems if s not in self._excluded]

    def __len__(self) -> int:
        return len(self._stems)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        stem = self._stems[idx]
        ann = self._ann_index.get(stem)
        if ann is None:
            ann = {
                "balls": [],
                "people": [],
                "court_polylines": [],
                "net_polylines": [],
                "exclusion_zones": [],
                "distortion_params": {"k1": 0.0, "k2": 0.0, "k3": 0.0, "rotation_deg": 0.0},
            }

        row = self._manifest.get(stem)
        image_filename = row.get("image_filename") if row else None
        img_path = _resolve_image_file(self.images_dir, stem, image_filename)
        image = Image.open(img_path).convert("RGB")
        return pack_training_sample(
            image, ann, stem, self.image_size, self.line_thickness
        )
