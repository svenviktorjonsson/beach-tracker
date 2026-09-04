"""Build annotation dicts from exported CSV rows (same semantics as labeler JSON)."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .csv_schema import MANIFEST_COLUMNS


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


def _group_polyline_rows(
    rows: Iterable[dict[str, str]],
    scope_key: str = "scope",
) -> list[dict[str, Any]]:
    """Group long-format rows into ScopedPolyline-like dicts."""
    by_key: dict[tuple[str, str, int], list[tuple[int, float, float]]] = defaultdict(list)
    for row in rows:
        stem = row["stem"]
        scope = row.get(scope_key, "main")
        pid = int(row["polyline_id"])
        pidx = int(row["point_index"])
        x, y = float(row["x"]), float(row["y"])
        by_key[(stem, scope, pid)].append((pidx, x, y))

    out: list[dict[str, Any]] = []
    for (stem, scope, _pid), pts in by_key.items():
        pts.sort(key=lambda t: t[0])
        points = [[p[1], p[2]] for p in pts]
        out.append({"stem": stem, "scope": scope, "points": points})
    return out


def _group_exclusion_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], list[tuple[int, float, float]]] = defaultdict(list)
    for row in rows:
        stem = row["stem"]
        zid = int(row["zone_id"])
        pidx = int(row["point_index"])
        x, y = float(row["x"]), float(row["y"])
        by_key[(stem, zid)].append((pidx, x, y))

    out: list[dict[str, Any]] = []
    for (_stem, _zid), pts in by_key.items():
        pts.sort(key=lambda t: t[0])
        points = [[p[1], p[2]] for p in pts]
        out.append({"points": points})
    return out


def load_manifest(csv_dir: Path) -> list[dict[str, str]]:
    path = csv_dir / "images_manifest.csv"
    rows = _read_rows(path)
    if not rows:
        return []
    # Validate headers loosely
    if rows and set(rows[0].keys()) != set(MANIFEST_COLUMNS):
        pass
    return rows


def _manifest_distortion(row: dict[str, str] | None) -> dict[str, float]:
    if not row:
        return {"k1": 0.0, "k2": 0.0, "k3": 0.0, "rotation_deg": 0.0}
    return {
        "k1": float(row.get("distortion_k1") or 0.0),
        "k2": float(row.get("distortion_k2") or 0.0),
        "k3": float(row.get("distortion_k3") or 0.0),
        "rotation_deg": float(row.get("rotation_deg") or 0.0),
    }


def build_ann_index(csv_dir: Path) -> dict[str, dict[str, Any]]:
    """
    Load all CSV tables under `csv_dir` and return stem -> annotation dict
    compatible with `geometry.rasterize_annotation` and `dataset._boxes_and_labels_from_ann`.
    """
    balls = _read_rows(csv_dir / "balls.csv")
    people = _read_rows(csv_dir / "people.csv")
    court = _read_rows(csv_dir / "court_polylines.csv")
    net = _read_rows(csv_dir / "net_polylines.csv")
    excl = _read_rows(csv_dir / "exclusion_zones.csv")

    stems: set[str] = set()
    for t in (balls, people, court, net, excl):
        for row in t:
            stems.add(row["stem"])

    manifest = load_manifest(csv_dir)
    manifest_index = {row["stem"]: row for row in manifest}
    for row in manifest:
        stems.add(row["stem"])

    by_stem_balls: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in balls:
        stem_k = row["stem"]
        if "w" in row and row.get("w") not in (None, ""):
            by_stem_balls[stem_k].append(
                {
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "w": float(row["w"]),
                    "h": float(row["h"]),
                    "scope": row.get("scope", "in_play"),
                }
            )
        else:
            by_stem_balls[stem_k].append(
                {
                    "cx": float(row["cx"]),
                    "cy": float(row["cy"]),
                    "rx": float(row["rx"]),
                    "ry": float(row["ry"]),
                    "angle": float(row.get("angle", 0)),
                    "scope": row.get("scope", "in_play"),
                }
            )

    by_stem_people: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in people:
        by_stem_people[row["stem"]].append(
            {
                "x": float(row["x"]),
                "y": float(row["y"]),
                "w": float(row["w"]),
                "h": float(row["h"]),
                "role": row.get("role", "player"),
            }
        )

    court_by_stem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pl in _group_polyline_rows(court):
        stem = pl.pop("stem")
        court_by_stem[stem].append({"scope": pl["scope"], "points": pl["points"]})

    net_by_stem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pl in _group_polyline_rows(net):
        stem = pl.pop("stem")
        net_by_stem[stem].append({"scope": pl["scope"], "points": pl["points"]})

    excl_by_stem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ex_rows_by_stem: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in excl:
        ex_rows_by_stem[row["stem"]].append(row)
    for stem, erows in ex_rows_by_stem.items():
        for z in _group_exclusion_rows(erows):
            excl_by_stem[stem].append(z)

    index: dict[str, dict[str, Any]] = {}
    for stem in stems:
        index[stem] = {
            "balls": by_stem_balls.get(stem, []),
            "people": by_stem_people.get(stem, []),
            "court_polylines": court_by_stem.get(stem, []),
            "net_polylines": net_by_stem.get(stem, []),
            "exclusion_zones": excl_by_stem.get(stem, []),
            "distortion_params": _manifest_distortion(manifest_index.get(stem)),
        }
    return index


def manifest_stems_in_order(csv_dir: Path) -> list[str]:
    """Preferred ordering from images_manifest.csv; fallback sorted stems."""
    m = load_manifest(csv_dir)
    if m:
        return [r["stem"] for r in m]
    idx = build_ann_index(csv_dir)
    return sorted(idx.keys())


def manifest_by_stem(csv_dir: Path) -> dict[str, dict[str, str]]:
    return {r["stem"]: r for r in load_manifest(csv_dir)}
