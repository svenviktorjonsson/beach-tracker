"""
Distance between two label JSONs (same image size) + merge when similar.
Calibration: first N=10 pair distances set mean, std; threshold = mean + 3*std.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

CALIBRATION_N = 10
SIGMA_MULTIPLIER = 3.0


def _diag(w: float, h: float) -> float:
    d = math.hypot(w, h)
    return d if d > 1e-6 else 1.0


def _dist2(p: list[float], q: list[float]) -> float:
    return math.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1]))


def chamfer_point_sets(a: list[list[float]], b: list[list[float]]) -> float:
    if not a or not b:
        return 0.0
    s1 = sum(min(_dist2(p, q) for q in b) for p in a) / len(a)
    s2 = sum(min(_dist2(q, p) for p in b) for q in b) / len(b)
    return (s1 + s2) * 0.5


def polyline_list_distance(
    polys_a: list[dict[str, Any]], polys_b: list[dict[str, Any]]
) -> float:
    """Symmetric chamfer between two sets of polylines (matched by length desc)."""
    la = sorted(
        [p for p in polys_a if p.get("points")],
        key=lambda x: len(x["points"]),
        reverse=True,
    )
    lb = sorted(
        [p for p in polys_b if p.get("points")],
        key=lambda x: len(x["points"]),
        reverse=True,
    )
    if not la and not lb:
        return 0.0
    if not la or not lb:
        return 1.0
    n = max(len(la), len(lb))
    total = 0.0
    count = 0
    for i in range(n):
        pa = la[i]["points"] if i < len(la) else la[-1]["points"]
        pb = lb[i]["points"] if i < len(lb) else lb[-1]["points"]
        total += chamfer_point_sets(pa, pb)
        count += 1
    return total / max(count, 1)


def _ball_ellipse_params(b: dict[str, Any]) -> tuple[float, float, float, float, float]:
    cx = float(b["cx"])
    cy = float(b["cy"])
    if "rx" not in b and "r" in b:
        r = float(b["r"])
        return cx, cy, r, r, float(b.get("angle", 0.0))
    rx = float(b["rx"]) if "rx" in b else float(b.get("r", 1.0))
    ry = float(b["ry"]) if "ry" in b else float(b.get("r", 1.0))
    ang = float(b.get("angle", 0.0))
    return cx, cy, rx, ry, ang


def _ball_box_dict(b: dict[str, Any]) -> dict[str, float]:
    """Axis-aligned box for agreement / merge: from JSON box or legacy ellipse."""
    from training.geometry import ellipse_axis_aligned_box

    if "w" in b and "x" in b:
        return {
            "x": float(b["x"]),
            "y": float(b["y"]),
            "w": float(b["w"]),
            "h": float(b["h"]),
        }
    acx, acy, arx, ary, aa = _ball_ellipse_params(b)
    x1, y1, x2, y2 = ellipse_axis_aligned_box(acx, acy, arx, ary, aa)
    return {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}


def _angle_diff_norm(a: float, b: float) -> float:
    d = abs(a - b) % (2 * math.pi)
    if d > math.pi:
        d = 2 * math.pi - d
    return d / math.pi


def balls_distance(
    ba: list[dict[str, Any]], bb: list[dict[str, Any]], diag: float
) -> float:
    """Compare ball annotations as axis-aligned boxes (legacy ellipse JSON is converted)."""
    if not ba and not bb:
        return 0.0
    if not ba or not bb:
        return 0.5
    ua = list(ba)
    ub = list(bb)
    s = 0.0
    nmatch = 0
    for _ in range(max(len(ua), len(ub))):
        if not ua or not ub:
            s += 0.5
            continue
        best = (-1, -1, -1.0)
        for i, a in enumerate(ua):
            abox = _ball_box_dict(a)
            for j, b in enumerate(ub):
                bbox = _ball_box_dict(b)
                ii = iou(abox, bbox)
                if ii > best[2]:
                    best = (i, j, ii)
        i, j, _ = best
        a, b = ua.pop(i), ub.pop(j)
        abox = _ball_box_dict(a)
        bbox = _ball_box_dict(b)
        cx_a = abox["x"] + abox["w"] * 0.5
        cy_a = abox["y"] + abox["h"] * 0.5
        cx_b = bbox["x"] + bbox["w"] * 0.5
        cy_b = bbox["y"] + bbox["h"] * 0.5
        s += _dist2([cx_a, cy_a], [cx_b, cy_b]) / diag
        if (a.get("scope") or "in_play") != (b.get("scope") or "in_play"):
            s += 0.15
        nmatch += 1
    return min(1.0, s / max(nmatch, 1))


def iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax2, ay2 = float(a["x"]) + float(a["w"]), float(a["y"]) + float(a["h"])
    bx2, by2 = float(b["x"]) + float(b["w"]), float(b["y"]) + float(b["h"])
    ix1, iy1 = max(float(a["x"]), float(b["x"])), max(float(a["y"]), float(b["y"]))
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = float(a["w"]) * float(a["h"])
    bb = float(b["w"]) * float(b["h"])
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def people_distance(
    pa: list[dict[str, Any]], pb: list[dict[str, Any]], diag: float
) -> float:
    if not pa and not pb:
        return 0.0
    if not pa or not pb:
        return 0.5
    ua = list(pa)
    ub = list(pb)
    s = 0.0
    nmatch = 0
    for _ in range(max(len(ua), len(ub))):
        if not ua or not ub:
            s += 0.5
            continue
        best = (-1, -1, -1.0)
        for i, a in enumerate(ua):
            for j, b in enumerate(ub):
                iou_ab = iou(a, b)
                if iou_ab > best[2]:
                    best = (i, j, iou_ab)
        i, j, _ = best
        a, b = ua.pop(i), ub.pop(j)
        cx_a = float(a["x"]) + float(a["w"]) * 0.5
        cy_a = float(a["y"]) + float(a["h"]) * 0.5
        cx_b = float(b["x"]) + float(b["w"]) * 0.5
        cy_b = float(b["y"]) + float(b["h"]) * 0.5
        s += _dist2([cx_a, cy_a], [cx_b, cy_b]) / diag
        if (a.get("role") or "player") != (b.get("role") or "player"):
            s += 0.15
        nmatch += 1
    return min(1.0, s / max(nmatch, 1))


def exclusion_distance(
    za: list[dict[str, Any]], zb: list[dict[str, Any]], diag: float
) -> float:
    if not za and not zb:
        return 0.0
    if not za or not zb:
        return 0.4
    return polyline_list_distance(za, zb) / diag


def metadata_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Penalty in [0, ~0.25] for disagreeing on frame context (session, phase, tags)."""
    pen = 0.0
    sa, sb = a.get("session_type"), b.get("session_type")
    if sa or sb:
        if not sa or not sb:
            pen += 0.03
        elif sa != sb:
            pen += 0.07
    ga, gb = a.get("game_phase"), b.get("game_phase")
    if ga or gb:
        if not ga or not gb:
            pen += 0.03
        elif ga != gb:
            pen += 0.07
    ta = set(a.get("action_tags") or [])
    tb = set(b.get("action_tags") or [])
    if ta or tb:
        union = ta | tb
        if union:
            inter = ta & tb
            pen += 0.06 * (1.0 - len(inter) / len(union))
    return min(0.25, pen)


def annotation_distance(
    a: dict[str, Any], b: dict[str, Any], img_w: float, img_h: float
) -> float:
    """Scalar in ~[0, 1]: lower = more similar."""
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
    c_lq = chamfer_point_sets(
        a.get("label_quad") or [], b.get("label_quad") or []
    )
    c_ep = chamfer_point_sets(
        a.get("exclusion_points") or [], b.get("exclusion_points") or []
    )
    geo = (
        0.26 * (c_court / d)
        + 0.26 * (c_net / d)
        + 0.11 * c_ex
        + 0.16 * c_ball
        + 0.13 * c_peo
        + 0.04 * (c_lq / d)
        + 0.04 * (c_ep / d)
    )
    meta = metadata_distance(a, b)
    return min(1.0, geo * 0.9 + meta * 0.4)


def _avg_points(pa: list, pb: list) -> list[list[float]]:
    n = min(len(pa), len(pb))
    return [
        [(float(pa[i][0]) + float(pb[i][0])) * 0.5, (float(pa[i][1]) + float(pb[i][1])) * 0.5]
        for i in range(n)
    ]


def merge_polylines(
    la: list[dict[str, Any]], lb: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    a = sorted(
        [p for p in la if p.get("points")],
        key=lambda x: len(x["points"]),
        reverse=True,
    )
    b = sorted(
        [p for p in lb if p.get("points")],
        key=lambda x: len(x["points"]),
        reverse=True,
    )
    if not a and not b:
        return []
    if not a:
        return [
            {"points": [list(x) for x in p["points"]], "scope": p.get("scope") or "main"}
            for p in b
        ]
    if not b:
        return [
            {"points": [list(x) for x in p["points"]], "scope": p.get("scope") or "main"}
            for p in a
        ]
    out = []
    n = max(len(a), len(b))
    for i in range(n):
        pa = a[i]["points"] if i < len(a) else a[-1]["points"]
        pb = b[i]["points"] if i < len(b) else b[-1]["points"]
        scope = (a[i].get("scope") if i < len(a) else a[-1].get("scope")) or "main"
        if i < len(b):
            sb = b[i].get("scope") if i < len(b) else b[-1].get("scope")
            if sb == "adjacent":
                scope = sb
        out.append({"points": _avg_points(pa, pb), "scope": scope})
    return out


def merge_balls(ba: list[dict[str, Any]], bb: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _sort_key(x: dict[str, Any]) -> tuple[float, float]:
        d = _ball_box_dict(x)
        return (d["x"] + d["w"] * 0.5, d["y"] + d["h"] * 0.5)

    ua = sorted(ba, key=_sort_key)
    ub = sorted(bb, key=_sort_key)
    out = []
    n = max(len(ua), len(ub))
    for i in range(n):
        a = ua[i] if i < len(ua) else ua[-1]
        b = ub[i] if i < len(ub) else ub[-1]
        ab = _ball_box_dict(a)
        bb_ = _ball_box_dict(b)
        out.append(
            {
                "x": (ab["x"] + bb_["x"]) * 0.5,
                "y": (ab["y"] + bb_["y"]) * 0.5,
                "w": (ab["w"] + bb_["w"]) * 0.5,
                "h": (ab["h"] + bb_["h"]) * 0.5,
                "scope": a.get("scope") or b.get("scope") or "in_play",
            }
        )
    return out


def merge_people(pa: list[dict[str, Any]], pb: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ua = list(pa)
    ub = list(pb)
    out = []
    for _ in range(max(len(ua), len(ub))):
        if not ua or not ub:
            break
        best = (0, 0, -1.0)
        for i, a in enumerate(ua):
            for j, b in enumerate(ub):
                ii = iou(a, b)
                if ii > best[2]:
                    best = (i, j, ii)
        i, j, _ = best
        a, b = ua.pop(i), ub.pop(j)
        out.append(
            {
                "x": (float(a["x"]) + float(b["x"])) * 0.5,
                "y": (float(a["y"]) + float(b["y"])) * 0.5,
                "w": (float(a["w"]) + float(b["w"])) * 0.5,
                "h": (float(a["h"]) + float(b["h"])) * 0.5,
                "role": a.get("role") or b.get("role") or "player",
            }
        )
    return out


def merge_exclusion_zones(
    za: list[dict[str, Any]], zb: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    a = sorted(
        [z for z in za if z.get("points")],
        key=lambda x: len(x["points"]),
        reverse=True,
    )
    b = sorted(
        [z for z in zb if z.get("points")],
        key=lambda x: len(x["points"]),
        reverse=True,
    )
    if not a and not b:
        return []
    if not a:
        return [{"points": [list(x) for x in z["points"]]} for z in b]
    if not b:
        return [{"points": [list(x) for x in z["points"]]} for z in a]
    out = []
    n = max(len(a), len(b))
    for i in range(n):
        pa = a[i]["points"] if i < len(a) else a[-1]["points"]
        pb = b[i]["points"] if i < len(b) else b[-1]["points"]
        out.append({"points": _avg_points(pa, pb)})
    return out


def _merge_str_field(
    a: dict[str, Any], b: dict[str, Any], key: str
) -> str | None:
    va, vb = a.get(key), b.get(key)
    if va is None and vb is None:
        return None
    if va is None:
        return str(vb) if vb is not None else None
    if vb is None:
        return str(va)
    return str(va) if str(va) == str(vb) else "ambiguous"


def _merge_action_tags(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    ta = set(a.get("action_tags") or [])
    tb = set(b.get("action_tags") or [])
    return sorted(ta & tb)


def merge_label_quad(
    qa: list[list[float]] | None, qb: list[list[float]] | None
) -> list[list[float]] | None:
    if not qa and not qb:
        return None
    if not qa:
        return [list(map(float, p)) for p in qb]
    if not qb:
        return [list(map(float, p)) for p in qa]
    if len(qa) == 4 and len(qb) == 4:
        return _avg_points(qa, qb)
    return [list(map(float, p)) for p in qa]


def merge_point_lists(
    a: list[list[float]], b: list[list[float]]
) -> list[list[float]]:
    if not a and not b:
        return []
    if not a:
        return [list(map(float, p)) for p in b]
    if not b:
        return [list(map(float, p)) for p in a]
    ua = sorted(a, key=lambda p: (float(p[0]), float(p[1])))
    ub = sorted(b, key=lambda p: (float(p[0]), float(p[1])))
    n = max(len(ua), len(ub))
    out: list[list[float]] = []
    for i in range(n):
        pa = ua[i] if i < len(ua) else ua[-1]
        pb = ub[i] if i < len(ub) else ub[-1]
        out.append(
            [
                (float(pa[0]) + float(pb[0])) * 0.5,
                (float(pa[1]) + float(pb[1])) * 0.5,
            ]
        )
    return out


def merge_annotations(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {
        "image": a.get("image") or b.get("image"),
        "exclusion_zones": merge_exclusion_zones(
            a.get("exclusion_zones") or [], b.get("exclusion_zones") or []
        ),
        "exclusion_points": merge_point_lists(
            a.get("exclusion_points") or [], b.get("exclusion_points") or []
        ),
        "label_quad": merge_label_quad(
            a.get("label_quad"), b.get("label_quad")
        ),
        "court_polylines": merge_polylines(
            a.get("court_polylines") or [], b.get("court_polylines") or []
        ),
        "net_polylines": merge_polylines(
            a.get("net_polylines") or [], b.get("net_polylines") or []
        ),
        "balls": merge_balls(a.get("balls") or [], b.get("balls") or []),
        "people": merge_people(a.get("people") or [], b.get("people") or []),
        "notes": (a.get("notes") or "") + " | " + (b.get("notes") or ""),
        "merged_from_passes": True,
        "session_type": _merge_str_field(a, b, "session_type"),
        "game_phase": _merge_str_field(a, b, "game_phase"),
        "gender_category": _merge_str_field(a, b, "gender_category"),
        "action_tags": _merge_action_tags(a, b),
    }


def load_calibration(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "distances": [],
            "mean": None,
            "std": None,
            "threshold": None,
            "paired_stems": [],
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("paired_stems", [])
    return data


def save_calibration(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def update_calibration_with_distance(
    cal: dict[str, Any], distance: float
) -> dict[str, Any]:
    dists = list(cal.get("distances") or [])
    if len(dists) >= CALIBRATION_N:
        return cal
    dists.append(distance)
    mean = sum(dists) / len(dists)
    var = sum((x - mean) ** 2 for x in dists) / len(dists)
    std = math.sqrt(var) if len(dists) > 1 else 0.0
    threshold = None
    if len(dists) == CALIBRATION_N:
        mean = sum(dists) / CALIBRATION_N
        var = sum((x - mean) ** 2 for x in dists) / CALIBRATION_N
        std = math.sqrt(var)
        threshold = mean + SIGMA_MULTIPLIER * std
    return {
        "distances": dists,
        "mean": mean,
        "std": std,
        "threshold": threshold,
        "calibration_complete": len(dists) >= CALIBRATION_N,
    }


def passes_agreement(distance: float, cal: dict[str, Any]) -> bool | None:
    """None if not enough calibration yet; True/False otherwise."""
    thr = cal.get("threshold")
    if thr is None:
        return None
    return distance <= float(thr)
