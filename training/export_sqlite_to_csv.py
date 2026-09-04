"""Export SQLite labeler v2 data to canonical CSV training tables."""

from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from .csv_schema import (
    BALLS_COLUMNS,
    COURT_POLYLINE_COLUMNS,
    EXCLUSION_COLUMNS,
    MANIFEST_COLUMNS,
    NET_POLYLINE_COLUMNS,
    PEOPLE_COLUMNS,
)


def _db_path(data_root: Path) -> Path:
    return Path(data_root) / "labeler_v2.sqlite3"


def _images_dir(data_root: Path) -> Path:
    return Path(data_root) / "images"


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _fit_line_orthopoint(points: list[tuple[float, float]]) -> tuple[float, float, float, float] | None:
    if len(points) < 2:
        return None
    n = float(len(points))
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    sxx = 0.0
    sxy = 0.0
    syy = 0.0
    for px, py in points:
        dx = px - cx
        dy = py - cy
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
    theta_mod = math.atan2(dy, dx) % math.pi
    return ox, oy, theta_mod, rho


def _court_orthopoints(conn: sqlite3.Connection, image_name: str, pass_n: int) -> list[list[float]]:
    rows = conn.execute(
        """
        SELECT line_idx, point_idx, x, y
        FROM geometry_polyline_points
        WHERE image_name = ? AND pass_n = ?
        ORDER BY line_idx, point_idx
        """,
        (image_name, pass_n),
    ).fetchall()
    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["line_idx"])].append((float(row["x"]), float(row["y"])))
    raw_segments: list[tuple[float, float, float, float]] = []
    for pts in grouped.values():
        if len(pts) < 2:
            continue
        for idx in range(len(pts) - 1):
            fit = _fit_line_orthopoint([pts[idx], pts[idx + 1]])
            if fit is not None:
                raw_segments.append(fit)
    if not raw_segments:
        return []
    raw_segments.sort(key=lambda item: (round(item[2], 4), round(item[3], 4)))
    merged: list[dict[str, float]] = []
    theta_eps = math.radians(10.0)
    rho_eps = 40.0
    for x, y, theta, rho in raw_segments:
        bucket = None
        for existing in merged:
            d_theta = abs(theta - existing["theta"])
            d_theta = min(d_theta, abs(math.pi - d_theta))
            if d_theta <= theta_eps and abs(rho - existing["rho"]) <= rho_eps:
                bucket = existing
                break
        if bucket is None:
            merged.append({"x": x, "y": y, "theta": theta, "rho": rho, "n": 1.0})
        else:
            n = bucket["n"] + 1.0
            bucket["x"] = (bucket["x"] * bucket["n"] + x) / n
            bucket["y"] = (bucket["y"] * bucket["n"] + y) / n
            bucket["theta"] = (bucket["theta"] * bucket["n"] + theta) / n
            bucket["rho"] = (bucket["rho"] * bucket["n"] + rho) / n
            bucket["n"] = n
    merged.sort(key=lambda item: (round(item["theta"], 4), round(item["rho"], 4)))
    return [[item["x"], item["y"]] for item in merged[:4]]


def _net_points_from_row(row: sqlite3.Row | None) -> list[list[float]]:
    if not row or not row["net_points"]:
        return []
    try:
        parsed = json.loads(row["net_points"])
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    out: list[list[float]] = []
    for point in parsed:
        if isinstance(point, list) and len(point) >= 2:
            out.append([float(point[0]), float(point[1])])
    return out


def _usable_row_flags(conn: sqlite3.Connection, row: sqlite3.Row, pass_n: int) -> dict[str, bool]:
    image_name = str(row["image_name"])
    court_points = _court_orthopoints(conn, image_name, pass_n)
    people_n = conn.execute(
        "SELECT COUNT(*) AS n FROM people_boxes WHERE image_name = ? AND pass_n = ?",
        (image_name, pass_n),
    ).fetchone()
    ball_n = conn.execute(
        "SELECT COUNT(*) AS n FROM ball_labels WHERE image_name = ? AND pass_n = ?",
        (image_name, pass_n),
    ).fetchone()
    net_points = _net_points_from_row(row)
    return {
        "court": len(court_points) >= 3,
        "net": len(net_points) >= 5,
        "ball": row["ball_visible"] is not None,
        "people": (people_n["n"] if people_n else 0) > 0 or row["session_type"] is not None,
    }


def sqlite_training_counts(
    data_root: Path,
    *,
    pass_n: int = 1,
    complete_only: bool = True,
) -> dict[str, int]:
    db_path = _db_path(data_root)
    counts = {"court": 0, "net": 0, "ball": 0, "people": 0, "total": 0}
    if not db_path.is_file():
        return counts
    with _connect(db_path) as conn:
        where = "pass_n = ?"
        params: list[Any] = [pass_n]
        rows = conn.execute(
            f"""
            SELECT image_name, session_type, ball_visible, net_points
            FROM labels
            WHERE {where}
            """,
            params,
        ).fetchall()
        for row in rows:
            flags = _usable_row_flags(conn, row, pass_n)
            if not any(flags.values()):
                continue
            counts["total"] += 1
            if flags["court"]:
                counts["court"] += 1
            if flags["net"]:
                counts["net"] += 1
            if flags["ball"]:
                counts["ball"] += 1
            if flags["people"]:
                counts["people"] += 1
    return counts


def export_sqlite_to_csv(
    data_root: Path,
    out_dir: Path,
    *,
    pass_n: int = 1,
    complete_only: bool = True,
) -> dict[str, int]:
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    db_path = _db_path(data_root)
    images_dir = _images_dir(data_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.is_file():
        raise FileNotFoundError(f"SQLite label database not found: {db_path}")

    paths = {
        "manifest": out_dir / "images_manifest.csv",
        "balls": out_dir / "balls.csv",
        "people": out_dir / "people.csv",
        "court": out_dir / "court_polylines.csv",
        "net": out_dir / "net_polylines.csv",
        "exclusion": out_dir / "exclusion_zones.csv",
    }

    counts = {"court": 0, "net": 0, "ball": 0, "people": 0, "total": 0}

    with _connect(db_path) as conn, \
        paths["manifest"].open("w", newline="", encoding="utf-8") as f_man, \
        paths["balls"].open("w", newline="", encoding="utf-8") as f_b, \
        paths["people"].open("w", newline="", encoding="utf-8") as f_p, \
        paths["court"].open("w", newline="", encoding="utf-8") as f_c, \
        paths["net"].open("w", newline="", encoding="utf-8") as f_n, \
        paths["exclusion"].open("w", newline="", encoding="utf-8") as f_e:

        w_man = csv.DictWriter(f_man, fieldnames=list(MANIFEST_COLUMNS))
        w_b = csv.DictWriter(f_b, fieldnames=list(BALLS_COLUMNS))
        w_p = csv.DictWriter(f_p, fieldnames=list(PEOPLE_COLUMNS))
        w_c = csv.DictWriter(f_c, fieldnames=list(COURT_POLYLINE_COLUMNS))
        w_n = csv.DictWriter(f_n, fieldnames=list(NET_POLYLINE_COLUMNS))
        w_e = csv.DictWriter(f_e, fieldnames=list(EXCLUSION_COLUMNS))
        for writer in (w_man, w_b, w_p, w_c, w_n, w_e):
            writer.writeheader()

        where = "pass_n = ?"
        params: list[Any] = [pass_n]
        label_rows = conn.execute(
            f"""
            SELECT image_name, session_type, ball_visible, distortion_params, net_points
            FROM labels
            WHERE {where}
            ORDER BY updated_at, image_name
            """,
            params,
        ).fetchall()

        for row in label_rows:
            flags = _usable_row_flags(conn, row, pass_n)
            if not any(flags.values()):
                continue
            image_name = str(row["image_name"])
            img_path = images_dir / image_name
            if not img_path.is_file():
                continue
            try:
                with Image.open(img_path) as img:
                    image_width, image_height = img.size
            except OSError:
                continue

            stem = img_path.stem
            counts["total"] += 1
            distortion = {"k1": 0.0, "k2": 0.0, "k3": 0.0, "rotation_deg": 0.0}
            if row["distortion_params"]:
                try:
                    parsed = json.loads(row["distortion_params"])
                    if isinstance(parsed, dict):
                        distortion["k1"] = float(parsed.get("k1", 0.0) or 0.0)
                        distortion["k2"] = float(parsed.get("k2", 0.0) or 0.0)
                        distortion["k3"] = float(parsed.get("k3", 0.0) or 0.0)
                        distortion["rotation_deg"] = float(parsed.get("rotation_deg", 0.0) or 0.0)
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass

            w_man.writerow({
                "stem": stem,
                "image_filename": image_name,
                "image_width": image_width,
                "image_height": image_height,
                "annotation_source": "sqlite_v2",
                "distortion_k1": distortion["k1"],
                "distortion_k2": distortion["k2"],
                "distortion_k3": distortion["k3"],
                "rotation_deg": distortion["rotation_deg"],
            })

            court_points = _court_orthopoints(conn, image_name, pass_n)
            if flags["court"]:
                counts["court"] += 1
                for point_index, (x, y) in enumerate(court_points):
                    w_c.writerow({
                        "stem": stem,
                        "image_path": image_name,
                        "scope": "main",
                        "polyline_id": 0,
                        "point_index": point_index,
                        "x": x,
                        "y": y,
                    })

            net_points = _net_points_from_row(row)
            if flags["net"]:
                counts["net"] += 1
            for point_index, (x, y) in enumerate(net_points):
                w_n.writerow({
                    "stem": stem,
                    "image_path": image_name,
                    "scope": "main",
                    "polyline_id": 0,
                    "point_index": point_index,
                    "x": x,
                    "y": y,
                })

            people_rows = conn.execute(
                """
                SELECT x, y, w, h, role
                FROM people_boxes
                WHERE image_name = ? AND pass_n = ?
                ORDER BY id
                """,
                (image_name, pass_n),
            ).fetchall()
            if flags["people"]:
                counts["people"] += 1
            for person in people_rows:
                w_p.writerow({
                    "stem": stem,
                    "image_path": image_name,
                    "x": float(person["x"]),
                    "y": float(person["y"]),
                    "w": float(person["w"]),
                    "h": float(person["h"]),
                    "role": str(person["role"] or "player"),
                })

            ball_row = conn.execute(
                """
                SELECT x, y, w, h
                FROM ball_labels
                WHERE image_name = ? AND pass_n = ?
                """,
                (image_name, pass_n),
            ).fetchone()
            if flags["ball"]:
                counts["ball"] += 1
            if ball_row:
                w_b.writerow({
                    "stem": stem,
                    "image_path": image_name,
                    "x": float(ball_row["x"]),
                    "y": float(ball_row["y"]),
                    "w": float(ball_row["w"]),
                    "h": float(ball_row["h"]),
                    "scope": "in_play",
                })

            ignore_rows = conn.execute(
                """
                SELECT idx, x, y
                FROM ignore_points
                WHERE image_name = ? AND pass_n = ?
                ORDER BY idx
                """,
                (image_name, pass_n),
            ).fetchall()
            for ignore_row in ignore_rows:
                w_e.writerow({
                    "stem": stem,
                    "image_path": image_name,
                    "zone_id": int(ignore_row["idx"]),
                    "point_index": 0,
                    "x": float(ignore_row["x"]),
                    "y": float(ignore_row["y"]),
                })

    return counts
