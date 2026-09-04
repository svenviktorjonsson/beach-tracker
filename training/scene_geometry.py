from __future__ import annotations
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from training.camera_solver import (
    BALL_DIAMETER_M,
    _rotation_matrix_from_euler,
    build_camera_model_seed,
    refine_camera_model_from_court_lines,
)
from training.export_sqlite_to_csv import _court_orthopoints


BEACH_COURT_WIDTH_M = 8.0
BEACH_COURT_LENGTH_M = 16.0


def load_analysis_court_orthopoints(conn: Any, image_name: str, pass_n: int = 1) -> list[list[float]]:
    return _court_orthopoints(conn, image_name, pass_n)


def _line_intersection_2d(
    a0: list[float],
    a1: list[float],
    b0: list[float],
    b1: list[float],
) -> list[float] | None:
    x1, y1 = float(a0[0]), float(a0[1])
    x2, y2 = float(a1[0]), float(a1[1])
    x3, y3 = float(b0[0]), float(b0[1])
    x4, y4 = float(b1[0]), float(b1[1])
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return [float(px), float(py)]


def infer_court_quad_from_support_polyline(points: list[list[float]]) -> list[list[float]] | None:
    if len(points) < 4:
        return None
    pts = [[float(p[0]), float(p[1])] for p in points[:4]]
    c0 = _line_intersection_2d(pts[3], pts[0], pts[0], pts[1])
    c1 = _line_intersection_2d(pts[0], pts[1], pts[1], pts[2])
    c2 = _line_intersection_2d(pts[1], pts[2], pts[2], pts[3])
    c3 = _line_intersection_2d(pts[2], pts[3], pts[3], pts[0])
    if any(c is None for c in [c0, c1, c2, c3]):
        return None
    return [c0, c1, c2, c3]


def order_court_points_for_grid(points: list[list[float]]) -> list[list[float]] | None:
    inferred = infer_court_quad_from_support_polyline(points)
    if inferred is None:
        if len(points) < 4:
            return None
        pts = [[float(p[0]), float(p[1])] for p in points[:4]]
    else:
        pts = inferred
    pts = sorted(pts, key=lambda p: p[1])
    top = sorted(pts[:2], key=lambda p: p[0])
    bottom = sorted(pts[2:], key=lambda p: p[0])
    return [bottom[0], top[0], top[1], bottom[1]]


def solve_analysis_homography(
    world_pts: list[list[float]],
    image_pts: list[list[float]],
) -> list[list[float]] | None:
    if len(world_pts) < 4 or len(image_pts) < 4:
        return None
    a_rows = []
    b_rows = []
    for (x, y), (u, v) in zip(world_pts[:4], image_pts[:4], strict=False):
        a_rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        b_rows.append(u)
        a_rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        b_rows.append(v)
    a = np.asarray(a_rows, dtype=np.float64)
    b = np.asarray(b_rows, dtype=np.float64)
    try:
        h = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return None
    return np.asarray(
        [
            [h[0], h[1], h[2]],
            [h[3], h[4], h[5]],
            [h[6], h[7], 1.0],
        ],
        dtype=np.float64,
    ).tolist()


def project_world_point_h(world_xy: list[float], h: list[list[float]]) -> list[float] | None:
    x, y = float(world_xy[0]), float(world_xy[1])
    den = h[2][0] * x + h[2][1] * y + h[2][2]
    if abs(den) < 1e-9:
        return None
    u = (h[0][0] * x + h[0][1] * y + h[0][2]) / den
    v = (h[1][0] * x + h[1][1] * y + h[1][2]) / den
    return [float(u), float(v)]


def project_world_point_camera(world_xyz: list[float], camera_model: dict | None) -> list[float] | None:
    if not camera_model:
        return None
    intr = camera_model.get("intrinsics") or {}
    pose = camera_model.get("pose") or {}
    fx = float(intr.get("fx", 0.0) or 0.0)
    fy = float(intr.get("fy", 0.0) or 0.0)
    cx = float(intr.get("cx", 0.0) or 0.0)
    cy = float(intr.get("cy", 0.0) or 0.0)
    if fx <= 0.0 or fy <= 0.0:
        return None
    r = _rotation_matrix_from_euler(
        float(pose.get("yaw_deg", 0.0) or 0.0),
        float(pose.get("pitch_deg", 0.0) or 0.0),
        float(pose.get("roll_deg", 0.0) or 0.0),
    )
    t = np.asarray(
        [
            float(pose.get("tx_m", 0.0) or 0.0),
            float(pose.get("ty_m", 0.0) or 0.0),
            float(pose.get("tz_m", 0.0) or 0.0),
        ],
        dtype=np.float64,
    )
    world = np.asarray(world_xyz, dtype=np.float64)
    cam = r @ world + t
    if cam[2] <= 1e-6:
        return None
    return [
        float(fx * (cam[0] / cam[2]) + cx),
        float(fy * (cam[1] / cam[2]) + cy),
    ]


def project_court_grid(camera_model: dict | None) -> list[list[list[float]]]:
    if not camera_model:
        return []
    grid: list[list[list[float]]] = []
    intr = camera_model.get("intrinsics") or {}
    pose = camera_model.get("pose") or {}
    if intr and pose:
        for x in range(0, 9):
            p0 = project_world_point_camera([float(x), 0.0, 0.0], camera_model)
            p1 = project_world_point_camera([float(x), BEACH_COURT_LENGTH_M, 0.0], camera_model)
            if p0 and p1:
                grid.append([p0, p1])
        for y in range(0, 17):
            p0 = project_world_point_camera([0.0, float(y), 0.0], camera_model)
            p1 = project_world_point_camera([BEACH_COURT_WIDTH_M, float(y), 0.0], camera_model)
            if p0 and p1:
                grid.append([p0, p1])
        if grid:
            return grid
    pts = camera_model.get("court_orthopoints") or []
    ordered = order_court_points_for_grid(pts)
    if ordered is None:
        return []
    world_quad = [
        [0.0, 0.0],
        [0.0, BEACH_COURT_LENGTH_M],
        [BEACH_COURT_WIDTH_M, BEACH_COURT_LENGTH_M],
        [BEACH_COURT_WIDTH_M, 0.0],
    ]
    h = solve_analysis_homography(world_quad, ordered)
    if h is None:
        return []
    for x in range(0, 9):
        p0 = project_world_point_h([float(x), 0.0], h)
        p1 = project_world_point_h([float(x), BEACH_COURT_LENGTH_M], h)
        if p0 and p1:
            grid.append([p0, p1])
    for y in range(0, 17):
        p0 = project_world_point_h([0.0, float(y)], h)
        p1 = project_world_point_h([BEACH_COURT_WIDTH_M, float(y)], h)
        if p0 and p1:
            grid.append([p0, p1])
    return grid


def build_camera_model_segments(camera_model: dict | None) -> list[dict[str, list[list[float]] | str]]:
    if not camera_model:
        return []
    segment_defs = [
        ([0.0, 0.0, 0.0], [BEACH_COURT_WIDTH_M, 0.0, 0.0], "court"),
        ([BEACH_COURT_WIDTH_M, 0.0, 0.0], [BEACH_COURT_WIDTH_M, BEACH_COURT_LENGTH_M, 0.0], "court"),
        ([BEACH_COURT_WIDTH_M, BEACH_COURT_LENGTH_M, 0.0], [0.0, BEACH_COURT_LENGTH_M, 0.0], "court"),
        ([0.0, BEACH_COURT_LENGTH_M, 0.0], [0.0, 0.0, 0.0], "court"),
        ([0.0, 0.5 * BEACH_COURT_LENGTH_M, 0.0], [BEACH_COURT_WIDTH_M, 0.5 * BEACH_COURT_LENGTH_M, 0.0], "court"),
    ]
    net_height_m = float(camera_model.get("net_height_m", 2.35) or 2.35)
    segment_defs.extend([
        ([0.0, 0.5 * BEACH_COURT_LENGTH_M, 0.0], [0.0, 0.5 * BEACH_COURT_LENGTH_M, net_height_m], "net"),
        ([BEACH_COURT_WIDTH_M, 0.5 * BEACH_COURT_LENGTH_M, 0.0], [BEACH_COURT_WIDTH_M, 0.5 * BEACH_COURT_LENGTH_M, net_height_m], "net"),
        ([0.0, 0.5 * BEACH_COURT_LENGTH_M, net_height_m], [BEACH_COURT_WIDTH_M, 0.5 * BEACH_COURT_LENGTH_M, net_height_m], "net"),
    ])
    segments: list[dict[str, list[list[float]] | str]] = []
    for a, b, role in segment_defs:
        pa = project_world_point_camera(a, camera_model)
        pb = project_world_point_camera(b, camera_model)
        if pa and pb:
            segments.append({"a": pa, "b": pb, "role": role})
    return segments


def build_world_projections(ball_world: dict | None, camera_model: dict | None) -> dict[str, Any]:
    pose = (camera_model or {}).get("pose") or {}
    net_height_m = float((camera_model or {}).get("net_height_m", 2.35) or 2.35)
    return {
        "top": {
            "court_outline": [[0.0, 0.0], [0.0, BEACH_COURT_LENGTH_M], [BEACH_COURT_WIDTH_M, BEACH_COURT_LENGTH_M], [BEACH_COURT_WIDTH_M, 0.0]],
            "midline": [[0.0, 0.5 * BEACH_COURT_LENGTH_M], [BEACH_COURT_WIDTH_M, 0.5 * BEACH_COURT_LENGTH_M]],
            "ball": (
                [float(ball_world["x_m"]), float(ball_world["y_m"])]
                if ball_world and "x_m" in ball_world and "y_m" in ball_world
                else None
            ),
            "camera": (
                [float(pose.get("tx_m", 0.0) or 0.0), float(pose.get("ty_m", 0.0) or 0.0)]
                if camera_model
                else None
            ),
        },
        "side": {
            "ground": [[0.0, 0.0], [BEACH_COURT_LENGTH_M, 0.0]],
            "net": [[0.5 * BEACH_COURT_LENGTH_M, 0.0], [0.5 * BEACH_COURT_LENGTH_M, net_height_m]],
            "ball": (
                [float(ball_world["y_m"]), float(ball_world["z_m"])]
                if ball_world and "y_m" in ball_world and "z_m" in ball_world
                else None
            ),
            "camera": (
                [float(pose.get("ty_m", 0.0) or 0.0), float(pose.get("tz_m", 0.0) or 0.0)]
                if camera_model
                else None
            ),
        },
    }


def estimate_ball_world(ball: dict | None, camera_model: dict | None) -> dict | None:
    if not ball or not camera_model:
        return None
    intr = camera_model.get("intrinsics") or {}
    pose = camera_model.get("pose") or {}
    fx = float(intr.get("fx", 0.0) or 0.0)
    fy = float(intr.get("fy", 0.0) or 0.0)
    cx = float(intr.get("cx", 0.0) or 0.0)
    cy = float(intr.get("cy", 0.0) or 0.0)
    if fx <= 0.0 or fy <= 0.0:
        return None
    radius_px = float(ball.get("radius") or ball.get("radius_prior") or 0.0)
    if radius_px <= 0.0:
        return None
    u = float(ball.get("center_x", 0.0))
    v = float(ball.get("center_y", 0.0))
    z_cam = max(0.05, fx * (BALL_DIAMETER_M * 0.5) / radius_px)
    ray = np.asarray([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
    cam_point = ray * z_cam
    r = _rotation_matrix_from_euler(
        float(pose.get("yaw_deg", 0.0) or 0.0),
        float(pose.get("pitch_deg", 0.0) or 0.0),
        float(pose.get("roll_deg", 0.0) or 0.0),
    )
    t = np.asarray(
        [
            float(pose.get("tx_m", 0.0) or 0.0),
            float(pose.get("ty_m", 0.0) or 0.0),
            float(pose.get("tz_m", 0.0) or 0.0),
        ],
        dtype=np.float64,
    )
    world = r.T @ (cam_point - t)
    return {
        "x_m": float(world[0]),
        "y_m": float(world[1]),
        "z_m": float(world[2]),
        "depth_m": float(z_cam),
        "radius_px": radius_px,
    }


def build_projection_artifact(
    *,
    seed_camera_model: dict | None,
    refined_camera_model: dict | None,
    ball: dict | None,
) -> dict[str, Any]:
    def _step_artifact(camera_model: dict | None) -> dict[str, Any]:
        ball_world = estimate_ball_world(ball, camera_model)
        return {
            "camera_model": camera_model,
            "model_segments": build_camera_model_segments(camera_model),
            "court_grid": project_court_grid(camera_model),
            "ball_world": ball_world,
            "world_views": build_world_projections(ball_world, camera_model),
        }

    seed_artifact = _step_artifact(seed_camera_model)
    refined_artifact = _step_artifact(refined_camera_model)
    solver_steps = []
    if refined_camera_model and isinstance(refined_camera_model.get("solver_path"), list):
        for step in refined_camera_model.get("solver_path") or []:
            step_camera = {
                "intrinsics": dict(step.get("intrinsics") or refined_camera_model.get("intrinsics") or {}),
                "distortion": dict(refined_camera_model.get("distortion") or {}),
                "pose": dict(step.get("pose") or refined_camera_model.get("pose") or {}),
                "court_orthopoints": [list(map(float, p)) for p in (step.get("court_orthopoints") or refined_camera_model.get("court_orthopoints") or [])],
                "net_height_m": float(step.get("net_height_m", refined_camera_model.get("net_height_m", 2.35)) or 2.35),
                "objective": dict(step.get("objective") or {}),
            }
            solver_steps.append(_step_artifact(step_camera))
    return {
        "seed": seed_artifact,
        "refined": refined_artifact,
        "solver_steps": solver_steps,
    }


def build_label_initialized_analysis(
    *,
    image_path: Path,
    label_record: dict,
    fallback_court_points: list[list[float]],
) -> dict[str, Any]:
    with Image.open(image_path) as img:
        image_width, image_height = img.size
    raw_polylines = label_record.get("geometry_polylines") or []
    primary_court_points = raw_polylines[0] if raw_polylines and len(raw_polylines[0]) >= 4 else []
    court_points = primary_court_points or fallback_court_points
    net_points = label_record.get("net_points") or []
    ball = label_record.get("ball") or None
    distortion_params = label_record.get("distortion_params") or {}
    seed = build_camera_model_seed(
        image_width=image_width,
        image_height=image_height,
        court_points=[],
        court_scores=None,
        net_points=net_points,
        net_scores=None,
        ball_point=ball,
        distortion_params=distortion_params,
        gender_category=label_record.get("gender_category"),
    )
    seed["court_orthopoints"] = [[float(p[0]), float(p[1])] for p in court_points]
    if len(court_points) >= 4:
        seed["notes"] = list(seed.get("notes") or []) + [
            "Initial trainer seed is rough and does not yet use the labeled court quadrilateral.",
            "The later camera fit step should move the red court toward the refined blue court geometry.",
        ]
    seed["focus_region"] = label_record.get("focus_region") or []
    seed["court_polylines"] = raw_polylines
    refined = refine_camera_model_from_court_lines(
        seed_model=seed,
        image_width=image_width,
        image_height=image_height,
        court_points=court_points,
        net_points=net_points,
        distortion_params=distortion_params,
    )
    projection_artifact = build_projection_artifact(
        seed_camera_model=seed,
        refined_camera_model=refined,
        ball=ball,
    )
    return {
        "source": "labels",
        "court_points": court_points,
        "court_target_points": primary_court_points,
        "net_points": net_points,
        "ball": ball,
        "camera_model_seed": seed,
        "camera_model": refined,
        "court_grid": projection_artifact["refined"]["court_grid"],
        "ball_world": projection_artifact["refined"]["ball_world"],
        "projection_artifact": projection_artifact,
    }
