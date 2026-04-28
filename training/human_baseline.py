"""
Inter-annotator distance (r1 vs r2) for dashboard reference lines.

Uses the same component metrics and weights as `annotation_distance` in agreement.py,
split into a detection blend (balls + people) and a segmentation blend (court + net
+ exclusion). Values are in ~[0, 1] (lower = humans agree more). Not the same units as
PyTorch loss, but comparable as a repeatability ceiling for the labels you train on.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from agreement import (
    balls_distance,
    exclusion_distance,
    people_distance,
    polyline_list_distance,
    _diag,
)

W_COURT, W_NET, W_EX = 0.28, 0.28, 0.12
W_BALL, W_PEO = 0.18, 0.14
W_SEG = W_COURT + W_NET + W_EX
W_DET = W_BALL + W_PEO


def _dims(
    ann: dict[str, Any], images_dir: Path, stem: str
) -> tuple[float, float] | None:
    w, h = ann.get("image_width"), ann.get("image_height")
    if w is not None and h is not None:
        return float(w), float(h)
    name = ann.get("image") or f"{stem}.png"
    p = images_dir / Path(name).name
    if p.is_file():
        from PIL import Image

        with Image.open(p) as im:
            return float(im.size[0]), float(im.size[1])
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        alt = images_dir / f"{stem}{ext}"
        if alt.is_file():
            from PIL import Image

            with Image.open(alt) as im:
                return float(im.size[0]), float(im.size[1])
    return None


def _pair_metrics(
    a: dict[str, Any], b: dict[str, Any], img_w: float, img_h: float
) -> dict[str, float]:
    d = _diag(img_w, img_h)
    c_court = polyline_list_distance(
        a.get("court_polylines") or [], b.get("court_polylines") or []
    )
    c_net = polyline_list_distance(
        a.get("net_polylines") or [], b.get("net_polylines") or []
    )
    c_ex = exclusion_distance(
        a.get("exclusion_zones") or [], b.get("exclusion_zones") or [], d
    )
    c_ball = balls_distance(a.get("balls") or [], b.get("balls") or [], d)
    c_peo = people_distance(a.get("people") or [], b.get("people") or [], d)

    court_norm = c_court / d
    net_norm = c_net / d

    det_proxy = (W_BALL * c_ball + W_PEO * c_peo) / W_DET
    seg_proxy = (W_COURT * court_norm + W_NET * net_norm + W_EX * c_ex) / W_SEG

    return {
        "det": float(det_proxy),
        "seg": float(seg_proxy),
        "net": float(net_norm),
    }


def compute_human_baseline(data_root: Path) -> dict[str, Any] | None:
    """
    Scan annotations for stems with both `_r1.json` and `_r2.json`, compute
    per-pair det/seg/net proxy distances, return mean/std for chart reference lines.
    """
    data_root = data_root.resolve()
    ann_dir = data_root / "annotations"
    images_dir = data_root / "images"
    if not ann_dir.is_dir():
        return None

    dets: list[float] = []
    segs: list[float] = []
    nets: list[float] = []

    for p in sorted(ann_dir.glob("*_r1.json")):
        stem = p.name[: -len("_r1.json")]
        r2 = ann_dir / f"{stem}_r2.json"
        if not r2.is_file():
            continue
        a = json.loads(p.read_text(encoding="utf-8"))
        b = json.loads(r2.read_text(encoding="utf-8"))
        dims = _dims(a, images_dir, stem) or _dims(b, images_dir, stem)
        if dims is None:
            continue
        img_w, img_h = dims
        m = _pair_metrics(a, b, img_w, img_h)
        dets.append(m["det"])
        segs.append(m["seg"])
        nets.append(m["net"])

    if not dets:
        return None

    def _ms(values: list[float]) -> tuple[float, float]:
        mu = float(statistics.mean(values))
        if len(values) < 2:
            return mu, 0.0
        return mu, float(statistics.pstdev(values))

    det_m, det_s = _ms(dets)
    seg_m, seg_s = _ms(segs)
    net_m, net_s = _ms(nets)

    return {
        "pairs": len(dets),
        "det_mean": det_m,
        "det_std": det_s,
        "seg_mean": seg_m,
        "seg_std": seg_s,
        "net_mean": net_m,
        "net_std": net_s,
        "note": (
            "Human r1↔r2 repeatability (agreement.py), ~0–1, lower is better. "
            "Det = balls+people blend; seg = court+net+exclusion; net = net polylines only (÷ diagonal). "
            "Dashboard plots training loss ÷ max(loss) on that chart so curves share the 0–1 axis with these lines."
        ),
    }
