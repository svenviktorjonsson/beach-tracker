"""Ellipse → box, polyline / polygon → label masks."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def ellipse_axis_aligned_box(
    cx: float, cy: float, rx: float, ry: float, angle: float
) -> tuple[float, float, float, float]:
    """Return (x1, y1, x2, y2) AABB for rotated ellipse."""
    if rx <= 0 or ry <= 0:
        return cx, cy, cx, cy
    t = np.linspace(0, 2 * math.pi, 33, dtype=np.float64)
    ca, sa = math.cos(angle), math.sin(angle)
    xl = rx * np.cos(t)
    yl = ry * np.sin(t)
    x = cx + xl * ca - yl * sa
    y = cy + xl * sa + yl * ca
    return float(x.min()), float(y.min()), float(x.max()), float(y.max())


def _scope_key(prefix: str, scope: str) -> str:
    s = (scope or "main").lower()
    if s == "adjacent":
        return f"{prefix}_adjacent"
    return f"{prefix}_main"


def rasterize_annotation(
    ann: dict[str, Any],
    width: int,
    height: int,
    line_thickness: int = 3,
    seg_class_index: dict[str, int] | None = None,
) -> np.ndarray:
    """
    Build [H, W] uint8 mask of segmentation class ids (0 = background).

    Later classes overwrite earlier where they overlap (exclusion drawn last).
    """
    if seg_class_index is None:
        from .schema import SEGMENTATION_CLASS_NAMES

        seg_class_index = {n: i + 1 for i, n in enumerate(SEGMENTATION_CLASS_NAMES)}

    mask = np.zeros((height, width), dtype=np.uint8)

    def draw_polyline(points: list[list[float]], class_id: int) -> None:
        if len(points) < 2:
            return
        pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(mask, [pts], isClosed=False, color=int(class_id), thickness=line_thickness)

    for pl in ann.get("court_polylines") or []:
        pts = pl.get("points") or []
        key = _scope_key("court", pl.get("scope", "main"))
        cid = seg_class_index.get(key)
        if cid is not None:
            draw_polyline(pts, cid)

    for pl in ann.get("net_polylines") or []:
        pts = pl.get("points") or []
        key = _scope_key("net", pl.get("scope", "main"))
        cid = seg_class_index.get(key)
        if cid is not None:
            draw_polyline(pts, cid)

    for zone in ann.get("exclusion_zones") or []:
        pts = zone.get("points") or []
        if len(pts) < 3:
            continue
        cid = seg_class_index.get("exclusion")
        if cid is None:
            continue
        arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [arr], int(cid))

    return mask
