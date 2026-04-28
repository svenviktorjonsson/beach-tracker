"""PyTorch Dataset from labeler JSON + images under `data/`."""

from __future__ import annotations

import json
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
        "orig_size": s["orig_size"].clone(),
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
    """Extract court/net polyline vertices as normalised [0,1] coords + validity masks."""

    def _gather(polylines_key: str, max_pts: int) -> tuple[torch.Tensor, torch.Tensor]:
        pts: list[list[float]] = []
        for pl in ann.get(polylines_key) or []:
            if (pl.get("scope") or "main").lower() != "main":
                continue
            for p in pl["points"]:
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

    court_pts, court_mask = _gather("court_polylines", max_court)
    net_pts, net_mask = _gather("net_polylines", max_net)
    return {
        "court_pts": court_pts,
        "court_mask": court_mask,
        "net_pts": net_pts,
        "net_mask": net_mask,
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

    return {
        "stem": stem,
        "image": img_t,
        "boxes": boxes,
        "labels": labels,
        "court_pts": kp["court_pts"],
        "court_mask": kp["court_mask"],
        "net_pts": kp["net_pts"],
        "net_mask": kp["net_mask"],
        "orig_size": torch.tensor([h, w]),
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
            }

        row = self._manifest.get(stem)
        image_filename = row.get("image_filename") if row else None
        img_path = _resolve_image_file(self.images_dir, stem, image_filename)
        image = Image.open(img_path).convert("RGB")
        return pack_training_sample(
            image, ann, stem, self.image_size, self.line_thickness
        )
