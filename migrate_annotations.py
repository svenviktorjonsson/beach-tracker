from __future__ import annotations

import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
ANN_DIR = ROOT / "data" / "annotations"
BACKUP_DIR = ROOT / "data" / f"annotations_backup_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
MERGED_SUFFIXES = ("_merged.json", "_needs_review.json")
MIN_BALL_BOX = 6.0
CLUSTER_DIST = 40.0


def _annotation_files() -> list[Path]:
    return sorted(
        p
        for p in ANN_DIR.glob("*.json")
        if p.is_file() and not p.name.endswith(MERGED_SUFFIXES)
    )


def _ellipse_to_box(ball: dict[str, Any]) -> dict[str, float | str]:
    if ball.get("x") is not None and ball.get("w") is not None:
        return {
            "x": float(ball["x"]),
            "y": float(ball["y"]),
            "w": max(MIN_BALL_BOX, float(ball["w"])),
            "h": max(MIN_BALL_BOX, float(ball["h"])),
            "scope": "other" if ball.get("scope") == "other" else "in_play",
        }

    cx = float(ball.get("cx", 0.0))
    cy = float(ball.get("cy", 0.0))
    if "rx" not in ball and "r" in ball:
        rx = float(ball["r"])
        ry = rx
        angle = 0.0
    else:
        rx = float(ball.get("rx", ball.get("r", 1.0)))
        ry = float(ball.get("ry", ball.get("r", 1.0)))
        angle = float(ball.get("angle", 0.0))

    ca = math.cos(angle)
    sa = math.sin(angle)
    xs: list[float] = []
    ys: list[float] = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            xl = sx * rx
            yl = sy * ry
            xs.append(cx + xl * ca - yl * sa)
            ys.append(cy + xl * sa + yl * ca)

    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)
    return {
        "x": float(x1),
        "y": float(y1),
        "w": max(MIN_BALL_BOX, float(x2 - x1)),
        "h": max(MIN_BALL_BOX, float(y2 - y1)),
        "scope": "other" if ball.get("scope") == "other" else "in_play",
    }


def _cluster_points(points: list[list[float]]) -> list[list[list[float]]]:
    remaining = [[float(p[0]), float(p[1])] for p in points if len(p) >= 2]
    clusters: list[list[list[float]]] = []
    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        changed = True
        while changed:
            changed = False
            keep: list[list[float]] = []
            for pt in remaining:
                if any(math.dist(pt, c) <= CLUSTER_DIST for c in cluster):
                    cluster.append(pt)
                    changed = True
                else:
                    keep.append(pt)
            remaining = keep
        clusters.append(cluster)
    return clusters


def _box_from_cluster(cluster: list[list[float]]) -> dict[str, float | str]:
    xs = [float(p[0]) for p in cluster]
    ys = [float(p[1]) for p in cluster]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    span_x = max_x - min_x
    span_y = max_y - min_y
    pad_x = max(2.0, min(8.0, span_x * 0.35 + 2.0))
    pad_y = max(2.0, min(8.0, span_y * 0.35 + 2.0))
    w = max(MIN_BALL_BOX, span_x + 2.0 * pad_x)
    h = max(MIN_BALL_BOX, span_y + 2.0 * pad_y)
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    return {
        "x": cx - w / 2.0,
        "y": cy - h / 2.0,
        "w": w,
        "h": h,
        "scope": "in_play",
    }


def _center_from_cluster(cluster: list[list[float]]) -> list[float]:
    xs = [float(p[0]) for p in cluster]
    ys = [float(p[1]) for p in cluster]
    return [sum(xs) / len(xs), sum(ys) / len(ys)]


def _canonicalize(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    stats = {
        "ball_boxes_normalized": 0,
        "click_clusters_to_boxes": 0,
        "click_clusters_to_inactive": 0,
        "other_scope_to_inactive": 0,
        "legacy_ball_promoted": 0,
    }

    out = dict(data)
    out.setdefault("action_tags", [])
    out.setdefault("court_polylines", [])
    out.setdefault("net_polylines", [])
    out.setdefault("people", [])
    out.setdefault("notes", "")
    out.setdefault("step_skip_counts", {})
    out.setdefault("source_stream_name", None)
    out.setdefault("source_group_label", None)
    out.setdefault("source_row_index", None)
    out.setdefault("gender_category", None)
    out.setdefault("exclusion_zones", [])
    out.setdefault("exclusion_points", [])
    out.setdefault("inactive_ball_points", [])

    if "ball" in out and out["ball"] and not out.get("balls"):
        out["balls"] = [out["ball"]]
        stats["legacy_ball_promoted"] += 1
    out.pop("ball", None)

    normalized_balls: list[dict[str, float | str]] = []
    inactive_points = [
        [float(p[0]), float(p[1])]
        for p in (out.get("inactive_ball_points") or [])
        if isinstance(p, list) and len(p) >= 2
    ]
    for raw_ball in out.get("balls") or []:
        if not isinstance(raw_ball, dict):
            continue
        box = _ellipse_to_box(raw_ball)
        stats["ball_boxes_normalized"] += 1
        if box.get("scope") == "other":
            inactive_points.append(
                [float(box["x"]) + float(box["w"]) / 2.0, float(box["y"]) + float(box["h"]) / 2.0]
            )
            stats["other_scope_to_inactive"] += 1
        else:
            box["scope"] = "in_play"
            normalized_balls.append(box)

    exclusion_points = [
        [float(p[0]), float(p[1])]
        for p in (out.get("exclusion_points") or [])
        if isinstance(p, list) and len(p) >= 2
    ]

    ball_flag = out.get("ball_in_view")
    if ball_flag is None:
        ball_flag = out.get("ball_in_play")

    if exclusion_points and not normalized_balls and not inactive_points:
        clusters = _cluster_points(exclusion_points)
        if ball_flag is False:
            inactive_points.extend(_center_from_cluster(c) for c in clusters)
            out["exclusion_points"] = []
            stats["click_clusters_to_inactive"] += len(clusters)
        else:
            normalized_balls.extend(_box_from_cluster(c) for c in clusters)
            out["exclusion_points"] = []
            stats["click_clusters_to_boxes"] += len(clusters)
            if out.get("ball_in_view") is None:
                out["ball_in_view"] = True
            if out.get("ball_in_play") is None:
                out["ball_in_play"] = True

    out["balls"] = normalized_balls
    out["inactive_ball_points"] = inactive_points

    if out.get("ball_in_view") is None:
        if normalized_balls or inactive_points:
            out["ball_in_view"] = True
    if out.get("ball_in_play") is None:
        out["ball_in_play"] = out.get("ball_in_view")

    if out.get("ball_in_view") is False and normalized_balls:
        out["ball_in_view"] = True
        out["ball_in_play"] = True

    ordered = {
        "image": out.get("image"),
        "pass": out.get("pass"),
        "complete": bool(out.get("complete")),
        "image_width": out.get("image_width"),
        "image_height": out.get("image_height"),
        "exclusion_zones": out.get("exclusion_zones") or [],
        "exclusion_points": out.get("exclusion_points") or [],
        "inactive_ball_points": out.get("inactive_ball_points") or [],
        "label_quad": out.get("label_quad"),
        "court_polylines": out.get("court_polylines") or [],
        "net_polylines": out.get("net_polylines") or [],
        "balls": out.get("balls") or [],
        "people": out.get("people") or [],
        "notes": out.get("notes") or "",
        "session_type": out.get("session_type"),
        "game_phase": out.get("game_phase") if out.get("session_type") == "game" else None,
        "ball_in_view": out.get("ball_in_view"),
        "ball_in_play": out.get("ball_in_view") if out.get("ball_in_view") is not None else out.get("ball_in_play"),
        "source_stream_name": out.get("source_stream_name"),
        "source_group_label": out.get("source_group_label"),
        "source_row_index": out.get("source_row_index"),
        "gender_category": out.get("gender_category"),
        "action_tags": out.get("action_tags") or [],
        "step_skip_counts": out.get("step_skip_counts") or {},
    }
    return ordered, stats


def main() -> None:
    files = _annotation_files()
    if not files:
        print("No annotation files found.")
        return

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "files_updated": 0,
        "ball_boxes_normalized": 0,
        "click_clusters_to_boxes": 0,
        "click_clusters_to_inactive": 0,
        "other_scope_to_inactive": 0,
        "legacy_ball_promoted": 0,
    }

    for path in files:
        original = path.read_text(encoding="utf-8")
        data = json.loads(original)
        normalized, stats = _canonicalize(data)
        rendered = json.dumps(normalized, indent=2, ensure_ascii=False) + "\n"
        if rendered != original:
            shutil.copy2(path, BACKUP_DIR / path.name)
            path.write_text(rendered, encoding="utf-8")
            summary["files_updated"] += 1
            for key, value in stats.items():
                summary[key] += value

    print(f"Backup dir: {BACKUP_DIR}")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
