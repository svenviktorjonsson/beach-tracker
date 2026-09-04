"""Lightweight geometric solvers for coarse volleyball scene proposals.

This module is intentionally heuristic-first:
- the NN proposes rough observations
- these solvers turn them into consistent geometric seeds
- later refiners can use the same output as warm starts
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any

import numpy as np


BEACH_COURT_WIDTH_M = 8.0
BEACH_COURT_LENGTH_M = 16.0
BALL_DIAMETER_M = 0.21
ANTENNA_ABOVE_NET_M = 0.8

_FRAME_RE = re.compile(r"_r(\d+)_")


@dataclass
class TemporalGeometryState:
    frame_index: int | None
    court_points: list[list[float]] = field(default_factory=list)
    net_points: list[list[float]] = field(default_factory=list)
    ball_center: list[float] | None = None
    ball_velocity: list[float] | None = None


def parse_frame_index(name: str | None) -> int | None:
    if not name:
        return None
    match = _FRAME_RE.search(str(name))
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _distance(a: list[float], b: list[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _blend_point(current: list[float], predicted: list[float], alpha: float) -> list[float]:
    return [
        float(current[0]) * (1.0 - alpha) + float(predicted[0]) * alpha,
        float(current[1]) * (1.0 - alpha) + float(predicted[1]) * alpha,
    ]


def _adjacent_frames(prev: TemporalGeometryState | None, frame_index: int | None) -> bool:
    if prev is None or prev.frame_index is None or frame_index is None:
        return False
    return 0 < (frame_index - prev.frame_index) <= 3


def smooth_geometry_with_temporal_prior(
    *,
    frame_index: int | None,
    court_points: list[list[float]],
    net_points: list[list[float]],
    ball_center: list[float] | None,
    prev_state: TemporalGeometryState | None,
) -> tuple[list[list[float]], list[list[float]], list[float] | None]:
    """Blend current rough proposals with a simple constant-velocity prior."""
    if not _adjacent_frames(prev_state, frame_index):
        return court_points, net_points, ball_center

    smoothed_court = court_points
    if prev_state and len(prev_state.court_points) == len(court_points) and court_points:
        smoothed_court = [
            _blend_point(cur, prev, 0.2)
            for cur, prev in zip(court_points, prev_state.court_points, strict=False)
        ]

    smoothed_net = net_points
    if prev_state and len(prev_state.net_points) == len(net_points) and net_points:
        smoothed_net = [
            _blend_point(cur, prev, 0.18)
            for cur, prev in zip(net_points, prev_state.net_points, strict=False)
        ]

    smoothed_ball = ball_center
    if (
        prev_state
        and ball_center is not None
        and prev_state.ball_center is not None
        and prev_state.ball_velocity is not None
    ):
        predicted_ball = [
            prev_state.ball_center[0] + prev_state.ball_velocity[0],
            prev_state.ball_center[1] + prev_state.ball_velocity[1],
        ]
        smoothed_ball = _blend_point(ball_center, predicted_ball, 0.35)

    return smoothed_court, smoothed_net, smoothed_ball


def update_temporal_state(
    *,
    frame_index: int | None,
    court_points: list[list[float]],
    net_points: list[list[float]],
    ball_center: list[float] | None,
    prev_state: TemporalGeometryState | None,
) -> TemporalGeometryState:
    velocity = None
    if prev_state and prev_state.ball_center is not None and ball_center is not None:
        velocity = [
            float(ball_center[0]) - float(prev_state.ball_center[0]),
            float(ball_center[1]) - float(prev_state.ball_center[1]),
        ]
    return TemporalGeometryState(
        frame_index=frame_index,
        court_points=[list(p) for p in court_points],
        net_points=[list(p) for p in net_points],
        ball_center=list(ball_center) if ball_center is not None else None,
        ball_velocity=velocity,
    )


def estimate_ball_radius_prior(
    *,
    image_width: int,
    image_height: int,
    net_points: list[list[float]],
) -> float:
    """Estimate a ball-radius prior from stable scene geometry.

    Uses the antenna base span when available. This keeps radius deterministic
    and decoupled from hand-drawn box size.
    """
    if len(net_points) >= 4:
        left_base = net_points[1]
        right_base = net_points[3]
        span_px = max(8.0, _distance(left_base, right_base))
        return max(1.0, 0.5 * span_px * (BALL_DIAMETER_M / BEACH_COURT_WIDTH_M))
    return max(1.0, 0.012 * min(float(image_width), float(image_height)))


def _current_net_height_m(gender_category: str | None) -> float:
    if gender_category == "men":
        return 2.43
    if gender_category == "mixed":
        return 2.35
    return 2.24


def _net_height_candidates_m() -> list[float]:
    return [2.20, 2.25, 2.30, 2.35, 2.40, 2.45, 2.50]


def _line_angle_degrees(a: list[float], b: list[float]) -> float:
    return math.degrees(math.atan2(float(b[1]) - float(a[1]), float(b[0]) - float(a[0])))


def _wrap_degrees(value: float) -> float:
    out = float(value)
    while out <= -180.0:
        out += 360.0
    while out > 180.0:
        out -= 360.0
    return out


def build_camera_model_seed(
    *,
    image_width: int,
    image_height: int,
    court_points: list[list[float]],
    court_scores: list[float] | None,
    net_points: list[list[float]],
    net_scores: list[float] | None,
    ball_point: dict[str, Any] | None,
    distortion_params: dict[str, float] | None = None,
    gender_category: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic camera-model seed from coarse scene proposals."""
    distortion_params = distortion_params or {}
    image_center_x = 0.5 * float(image_width)
    image_center_y = 0.5 * float(image_height)
    max_dim = max(float(image_width), float(image_height), 1.0)
    fov_degrees = 60.0
    f_guess = (0.5 * max_dim) / math.tan(math.radians(fov_degrees) * 0.5)

    sources = ["coarse_nn_proposals"]
    notes = [
        "Seed built from coarse proposal geometry and lightweight deterministic heuristics.",
        "Use as a warm start for later full camera optimization, not as a final calibration.",
    ]
    roll_deg = float(distortion_params.get("rotation_deg", 0.0) or 0.0)
    yaw_deg = 0.0
    pitch_deg = 0.0
    tz_m = 12.0
    confidence = 0.18

    if len(net_points) >= 5:
        sources.append("net_landmarks")
        left_vertical = _line_angle_degrees(net_points[1], net_points[0])
        right_vertical = _line_angle_degrees(net_points[3], net_points[4])
        avg_vertical = _wrap_degrees((left_vertical + right_vertical) * 0.5)
        roll_deg = _wrap_degrees(avg_vertical - 90.0)
        net_center_x = sum(float(p[0]) for p in net_points) / len(net_points)
        net_center_y = sum(float(p[1]) for p in net_points) / len(net_points)
        yaw_deg = ((net_center_x - image_center_x) / max(float(image_width), 1.0)) * 40.0
        pitch_deg = ((image_center_y - net_center_y) / max(float(image_height), 1.0)) * 30.0
        span_px = max(16.0, _distance(net_points[1], net_points[3]))
        tz_m = max(4.0, (f_guess * BEACH_COURT_WIDTH_M) / span_px)
        confidence = 0.42
        notes.append("Net landmarks provided the strongest pose seed.")

    if court_points:
        sources.append("court_orthopoints")
        confidence = max(confidence, 0.3 if len(court_points) >= 2 else 0.24)
        notes.append("Court observations are stored as ortholine anchor points, not raw traced lines.")

    if ball_point:
        sources.append("ball_hint")

    return {
        "version": "pinhole-radtan-v1alpha",
        "estimate_status": "initial_guess",
        "estimate_sources": sources,
        "confidence": min(0.95, confidence),
        "intrinsics": {
            "fx": f_guess,
            "fy": f_guess,
            "cx": image_center_x,
            "cy": image_center_y,
            "skew": 0.0,
            "image_width": float(image_width),
            "image_height": float(image_height),
        },
        "distortion": {
            "k1": float(distortion_params.get("k1", 0.0) or 0.0),
            "k2": float(distortion_params.get("k2", 0.0) or 0.0),
            "k3": float(distortion_params.get("k3", 0.0) or 0.0),
            "p1": 0.0,
            "p2": 0.0,
            "rotation_deg": float(distortion_params.get("rotation_deg", 0.0) or 0.0),
        },
        "pose": {
            "tx_m": 0.0,
            "ty_m": -0.5 * BEACH_COURT_LENGTH_M,
            "tz_m": tz_m,
            "yaw_deg": yaw_deg,
            "pitch_deg": pitch_deg,
            "roll_deg": roll_deg,
        },
        "net_anchors": [
            {
                "label": label,
                "image": [float(point[0]), float(point[1])],
                "world": world,
            }
            for label, point, world in zip(
                ["left_antenna_top", "left_antenna_base", "net_center_top", "right_antenna_base", "right_antenna_top"],
                net_points[:5],
                [
                    [0.0, 0.5 * BEACH_COURT_LENGTH_M, _current_net_height_m(gender_category) + ANTENNA_ABOVE_NET_M],
                    [0.0, 0.5 * BEACH_COURT_LENGTH_M, 0.0],
                    [0.5 * BEACH_COURT_WIDTH_M, 0.5 * BEACH_COURT_LENGTH_M, _current_net_height_m(gender_category)],
                    [BEACH_COURT_WIDTH_M, 0.5 * BEACH_COURT_LENGTH_M, 0.0],
                    [BEACH_COURT_WIDTH_M, 0.5 * BEACH_COURT_LENGTH_M, _current_net_height_m(gender_category) + ANTENNA_ABOVE_NET_M],
                ],
                strict=False,
            )
        ],
        "court_orthopoints": [[float(p[0]), float(p[1])] for p in court_points],
        "notes": notes,
    }


def _line_intersection(
    a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray
) -> np.ndarray | None:
    da = a1 - a0
    db = b1 - b0
    det = da[0] * db[1] - da[1] * db[0]
    if abs(det) < 1e-9:
        return None
    rhs = b0 - a0
    t = (rhs[0] * db[1] - rhs[1] * db[0]) / det
    return a0 + t * da


def _infer_court_quad_from_support_polyline(points: list[list[float]]) -> list[list[float]] | None:
    if len(points) < 4:
        return None
    pts = np.asarray(points[:4], dtype=np.float64)
    l0 = (pts[0], pts[1])
    l1 = (pts[1], pts[2])
    l2 = (pts[2], pts[3])
    l3 = (pts[3], pts[0])
    c0 = _line_intersection(*l3, *l0)
    c1 = _line_intersection(*l0, *l1)
    c2 = _line_intersection(*l1, *l2)
    c3 = _line_intersection(*l2, *l3)
    if any(c is None for c in [c0, c1, c2, c3]):
        return None
    return [c0.tolist(), c1.tolist(), c2.tolist(), c3.tolist()]


def _line_intersection_from_point_dir(
    point_a: np.ndarray,
    dir_a: np.ndarray,
    point_b: np.ndarray,
    dir_b: np.ndarray,
) -> np.ndarray | None:
    mat = np.asarray(
        [
            [float(dir_a[0]), -float(dir_b[0])],
            [float(dir_a[1]), -float(dir_b[1])],
        ],
        dtype=np.float64,
    )
    rhs = np.asarray(
        [
            float(point_b[0] - point_a[0]),
            float(point_b[1] - point_a[1]),
        ],
        dtype=np.float64,
    )
    try:
        t = np.linalg.solve(mat, rhs)
    except np.linalg.LinAlgError:
        return None
    return point_a + dir_a * float(t[0])


def _support_line_geometry(points: list[list[float]]) -> dict[str, np.ndarray] | None:
    if len(points) < 4:
        return None
    p0 = np.asarray(points[0], dtype=np.float64)
    p1 = np.asarray(points[1], dtype=np.float64)
    p2 = np.asarray(points[2], dtype=np.float64)
    p3 = np.asarray(points[3], dtype=np.float64)
    left_dir = p1 - p0
    far_dir = p2 - p1
    right_dir = p2 - p3
    if float(np.linalg.norm(left_dir)) < 1e-6 or float(np.linalg.norm(far_dir)) < 1e-6 or float(np.linalg.norm(right_dir)) < 1e-6:
        return None
    far_left = _line_intersection_from_point_dir(p0, left_dir, p1, far_dir)
    far_right = _line_intersection_from_point_dir(p3, right_dir, p2, far_dir)
    if far_left is None or far_right is None:
        return None
    return {
        "p0": p0,
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "left_dir": left_dir,
        "far_dir": far_dir,
        "right_dir": right_dir,
        "far_left": far_left,
        "far_right": far_right,
    }


def _order_court_quad(points: list[list[float]]) -> list[list[float]] | None:
    if len(points) < 4:
        return None
    inferred = _infer_court_quad_from_support_polyline(points)
    pts = np.asarray(inferred if inferred is not None else points[:4], dtype=np.float64)
    order_y = np.argsort(pts[:, 1])
    top = pts[order_y[:2]]
    bottom = pts[order_y[2:]]
    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]
    top_left, top_right = top[0], top[1]
    bottom_left, bottom_right = bottom[0], bottom[1]
    # near-left origin, then far-left, far-right, near-right
    return [
        bottom_left.tolist(),
        top_left.tolist(),
        top_right.tolist(),
        bottom_right.tolist(),
    ]


def _court_world_quad() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0],
            [0.0, BEACH_COURT_LENGTH_M],
            [BEACH_COURT_WIDTH_M, BEACH_COURT_LENGTH_M],
            [BEACH_COURT_WIDTH_M, 0.0],
        ],
        dtype=np.float64,
    )


def _rotation_matrix_from_euler(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = np.asarray([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64)
    ry = np.asarray([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    return ry @ rx @ rz


def _euler_from_rotation_matrix(r: np.ndarray) -> tuple[float, float, float]:
    r = np.asarray(r, dtype=np.float64)
    yaw = math.degrees(math.atan2(r[0, 2], r[2, 2]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -r[1, 2]))))
    roll = math.degrees(math.atan2(r[1, 0], r[1, 1]))
    return yaw, pitch, roll


def _solve_homography(world_xy: np.ndarray, image_xy: np.ndarray) -> np.ndarray | None:
    if world_xy.shape[0] < 4 or image_xy.shape[0] < 4:
        return None
    a_rows = []
    b_rows = []
    for (x, y), (u, v) in zip(world_xy[:4], image_xy[:4], strict=False):
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
    )


def _project_homography(h: np.ndarray, world_xy: np.ndarray) -> np.ndarray:
    homog = np.concatenate([world_xy, np.ones((world_xy.shape[0], 1), dtype=np.float64)], axis=1)
    proj = (h @ homog.T).T
    den = proj[:, 2:3]
    den = np.where(np.abs(den) < 1e-9, np.where(den < 0, -1e-9, 1e-9), den)
    proj /= den
    return proj[:, :2]


def _decompose_planar_homography(h: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        k_inv = np.linalg.inv(k)
    except np.linalg.LinAlgError:
        return None
    b = k_inv @ h
    b1 = b[:, 0]
    b2 = b[:, 1]
    b3 = b[:, 2]
    norm = 1.0 / max(np.linalg.norm(b1), 1e-9)
    r1 = b1 * norm
    r2 = b2 * norm
    t = b3 * norm
    r3 = np.cross(r1, r2)
    r = np.stack([r1, r2, r3], axis=1)
    u, _, vh = np.linalg.svd(r)
    r = u @ vh
    if np.linalg.det(r) < 0:
        r[:, 2] *= -1.0
        t *= -1.0
    return r, t


def _polyline_mean_segment_error(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    n = 0
    for idx in range(1, len(points) - 1):
        ax = float(points[idx][0]) - float(points[idx - 1][0])
        ay = float(points[idx][1]) - float(points[idx - 1][1])
        bx = float(points[idx + 1][0]) - float(points[idx][0])
        by = float(points[idx + 1][1]) - float(points[idx][1])
        denom = max(math.hypot(ax, ay) * math.hypot(bx, by), 1e-9)
        total += abs(ax * by - ay * bx) / denom
        n += 1
    return total / max(n, 1)


def _point_line_distance(point: np.ndarray, line_a: np.ndarray, line_b: np.ndarray) -> float:
    direction = line_b - line_a
    denom = float(np.linalg.norm(direction))
    if denom < 1e-9:
        return float(np.linalg.norm(point - line_a))
    rel = point - line_a
    return abs(float(rel[0] * direction[1] - rel[1] * direction[0])) / denom


def _segment_fraction(point: np.ndarray, line_a: np.ndarray, line_b: np.ndarray) -> float:
    direction = line_b - line_a
    denom = float(np.dot(direction, direction))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(point - line_a, direction) / denom)


def _project_world_point_camera(world_xyz: list[float], intrinsics: dict[str, float], pose: dict[str, float]) -> list[float] | None:
    fx = float(intrinsics.get("fx", 0.0) or 0.0)
    fy = float(intrinsics.get("fy", 0.0) or 0.0)
    cx = float(intrinsics.get("cx", 0.0) or 0.0)
    cy = float(intrinsics.get("cy", 0.0) or 0.0)
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
    p = np.asarray(world_xyz, dtype=np.float64)
    cam = r @ p + t
    z = float(cam[2])
    if z <= 1e-6:
        return None
    return [
        fx * float(cam[0]) / z + cx,
        fy * float(cam[1]) / z + cy,
    ]


def _build_net_world_points(net_height_m: float) -> list[list[float]]:
    mid_y = 0.5 * BEACH_COURT_LENGTH_M
    return [
        [0.0, mid_y, net_height_m + ANTENNA_ABOVE_NET_M],
        [0.0, mid_y, 0.0],
        [0.5 * BEACH_COURT_WIDTH_M, mid_y, net_height_m],
        [BEACH_COURT_WIDTH_M, mid_y, 0.0],
        [BEACH_COURT_WIDTH_M, mid_y, net_height_m + ANTENNA_ABOVE_NET_M],
    ]


def _camera_fit_objective(
    image_quad: np.ndarray,
    support_points: list[list[float]],
    h: np.ndarray,
    intrinsics: dict[str, float],
    pose: dict[str, float],
    net_points: list[list[float]],
    net_height_m: float,
    left_scale: float,
    right_scale: float,
) -> dict[str, float]:
    near_left, far_left, far_right, near_right = image_quad
    support_err = 0.0
    support_position_penalty = 0.0
    preferred_near_frac = 0.55
    max_far_frac = 0.92
    if len(support_points) >= 1:
        p = np.asarray(support_points[0], dtype=np.float64)
        support_err += _point_line_distance(p, near_left, far_left)
        t = _segment_fraction(p, near_left, far_left)
        support_position_penalty += (t - preferred_near_frac) ** 2
    if len(support_points) >= 2:
        p = np.asarray(support_points[1], dtype=np.float64)
        support_err += 0.5 * (
            _point_line_distance(p, near_left, far_left)
            + _point_line_distance(p, far_left, far_right)
        )
    if len(support_points) >= 3:
        p = np.asarray(support_points[2], dtype=np.float64)
        support_err += 0.5 * (
            _point_line_distance(p, far_left, far_right)
            + _point_line_distance(p, far_right, near_right)
        )
    if len(support_points) >= 4:
        p = np.asarray(support_points[3], dtype=np.float64)
        support_err += _point_line_distance(p, far_right, near_right)
        t = _segment_fraction(p, near_right, far_right)
        support_position_penalty += (t - preferred_near_frac) ** 2
    support_err /= max(len(support_points), 1)

    world_mid = np.asarray([[0.0, 0.5 * BEACH_COURT_LENGTH_M], [BEACH_COURT_WIDTH_M, 0.5 * BEACH_COURT_LENGTH_M]], dtype=np.float64)
    midline_img = _project_homography(h, world_mid)
    net_base_err = 0.0
    net_top_err = 0.0
    if len(net_points) >= 4 and midline_img.shape[0] == 2:
        net_base_err = 0.5 * (
            _point_line_distance(np.asarray(net_points[1], dtype=np.float64), midline_img[0], midline_img[1])
            + _point_line_distance(np.asarray(net_points[3], dtype=np.float64), midline_img[0], midline_img[1])
        )
    if len(net_points) >= 5:
        predicted = []
        for world in _build_net_world_points(net_height_m):
            proj = _project_world_point_camera(world, intrinsics, pose)
            if proj is None:
                net_top_err = 1e6
                break
            predicted.append(proj)
        if predicted:
            pred_arr = np.asarray(predicted, dtype=np.float64)
            obs_arr = np.asarray(net_points[:5], dtype=np.float64)
            net_top_err = float(np.sqrt(np.mean(np.sum((pred_arr - obs_arr) ** 2, axis=1))))

    extension_penalty = 0.18 * ((left_scale - 1.0) ** 2 + (right_scale - 1.0) ** 2)
    symmetry_penalty = 0.15 * ((left_scale - right_scale) ** 2)
    total = (
        support_err * 2.8
        + net_base_err * 1.4
        + net_top_err * 0.7
        + support_position_penalty * 180.0
        + extension_penalty
        + symmetry_penalty
    )
    return {
        "total": float(total),
        "support_err": float(support_err),
        "net_base_err": float(net_base_err),
        "net_top_err": float(net_top_err),
        "support_position_penalty": float(support_position_penalty),
        "extension_penalty": float(extension_penalty),
        "symmetry_penalty": float(symmetry_penalty),
    }


def _evaluate_camera_candidate(
    *,
    support_geom: dict[str, np.ndarray],
    world_quad: np.ndarray,
    k: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    image_width: int,
    image_height: int,
    court_points: list[list[float]],
    net_points: list[list[float]],
    left_scale: float,
    right_scale: float,
    net_height_m: float,
) -> dict[str, Any] | None:
    if left_scale < 1.4 or right_scale < 1.4 or net_height_m < 2.2 or net_height_m > 2.5:
        return None
    far_left = support_geom["far_left"]
    far_right = support_geom["far_right"]
    p0 = support_geom["p0"]
    p3 = support_geom["p3"]
    near_left = far_left + (p0 - far_left) * left_scale
    near_right = far_right + (p3 - far_right) * right_scale
    candidate_quad = np.asarray([near_left, far_left, far_right, near_right], dtype=np.float64)
    h = _solve_homography(world_quad, candidate_quad)
    if h is None:
        return None
    pose = _decompose_planar_homography(h, k)
    if pose is None:
        return None
    r, t = pose
    yaw_deg, pitch_deg, roll_deg = _euler_from_rotation_matrix(r)
    pose_dict = {
        "tx_m": float(t[0]),
        "ty_m": float(t[1]),
        "tz_m": float(abs(t[2])),
        "yaw_deg": float(yaw_deg),
        "pitch_deg": float(pitch_deg),
        "roll_deg": float(roll_deg),
    }
    intrinsics_dict = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "skew": 0.0,
        "image_width": float(image_width),
        "image_height": float(image_height),
    }
    objective = _camera_fit_objective(
        candidate_quad,
        court_points,
        h,
        intrinsics_dict,
        pose_dict,
        net_points,
        net_height_m,
        float(left_scale),
        float(right_scale),
    )
    return {
        "image_quad": candidate_quad.copy(),
        "h": h,
        "pose": pose_dict,
        "intrinsics": intrinsics_dict,
        "objective": objective,
        "net_height_m": float(net_height_m),
        "left_scale": float(left_scale),
        "right_scale": float(right_scale),
    }


def refine_camera_model_from_court_lines(
    *,
    seed_model: dict[str, Any],
    image_width: int,
    image_height: int,
    court_points: list[list[float]],
    net_points: list[list[float]],
    distortion_params: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Refine the camera model from final court-line observations.

    The court observations are treated as measurements in the already
    warp-corrected working view. Distortion remains part of the model, but the
    homography/pose solve happens on the corrected-view geometry where straight
    world lines should already be straight.
    """
    refined = dict(seed_model)
    refined["distortion"] = dict(seed_model.get("distortion") or {})
    if distortion_params:
      refined["distortion"].update({
          "k1": float(distortion_params.get("k1", refined["distortion"].get("k1", 0.0)) or 0.0),
          "k2": float(distortion_params.get("k2", refined["distortion"].get("k2", 0.0)) or 0.0),
          "k3": float(distortion_params.get("k3", refined["distortion"].get("k3", 0.0)) or 0.0),
          "rotation_deg": float(distortion_params.get("rotation_deg", refined["distortion"].get("rotation_deg", 0.0)) or 0.0),
      })
    refined["focus_region"] = list(seed_model.get("focus_region") or [])
    refined["court_polylines"] = [list(map(list, seed_model.get("court_polylines", [court_points])[0]))] if court_points else list(seed_model.get("court_polylines") or [])
    notes = list(seed_model.get("notes") or [])
    sources = list(seed_model.get("estimate_sources") or [])
    support_geom = _support_line_geometry(court_points)
    if support_geom is None:
        notes.append("Full camera refinement skipped: need four usable court support points.")
        refined["notes"] = notes
        refined["estimate_sources"] = sources
        return refined

    world_quad = _court_world_quad()
    intr = seed_model.get("intrinsics") or {}
    fx = float(intr.get("fx", 0.0) or 0.0)
    fy = float(intr.get("fy", 0.0) or 0.0)
    cx = float(intr.get("cx", 0.5 * image_width) or 0.5 * image_width)
    cy = float(intr.get("cy", 0.5 * image_height) or 0.5 * image_height)
    if fx <= 0.0 or fy <= 0.0:
        max_dim = max(float(image_width), float(image_height), 1.0)
        f_guess = (0.5 * max_dim) / math.tan(math.radians(60.0) * 0.5)
        fx = fy = f_guess
    k = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    best: dict[str, Any] | None = None
    solver_path: list[dict[str, Any]] = []
    for left_scale in np.linspace(1.4, 8.0, 15):
        for right_scale in np.linspace(1.4, 8.0, 15):
            for net_height_m in _net_height_candidates_m():
                candidate = _evaluate_camera_candidate(
                    support_geom=support_geom,
                    world_quad=world_quad,
                    k=k,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    image_width=image_width,
                    image_height=image_height,
                    court_points=court_points,
                    net_points=net_points,
                    left_scale=float(left_scale),
                    right_scale=float(right_scale),
                    net_height_m=float(net_height_m),
                )
                if candidate is None:
                    continue
                if best is None or candidate["objective"]["total"] < best["objective"]["total"]:
                    best = candidate
                    solver_path.append({
                        "intrinsics": dict(candidate["intrinsics"]),
                        "pose": dict(candidate["pose"]),
                        "court_orthopoints": [[float(p[0]), float(p[1])] for p in candidate["image_quad"].tolist()],
                        "net_height_m": float(candidate["net_height_m"]),
                        "objective": dict(candidate["objective"]),
                    })

    if best is not None:
        for scale_step, net_step in [(0.5, 0.04), (0.2, 0.02), (0.08, 0.01), (0.03, 0.01)]:
            improved = True
            while improved:
                improved = False
                current = best
                candidates: list[dict[str, Any]] = []
                for dl in (-scale_step, 0.0, scale_step):
                    for dr in (-scale_step, 0.0, scale_step):
                        for dh in (-net_step, 0.0, net_step):
                            if dl == 0.0 and dr == 0.0 and dh == 0.0:
                                continue
                            cand = _evaluate_camera_candidate(
                                support_geom=support_geom,
                                world_quad=world_quad,
                                k=k,
                                fx=fx,
                                fy=fy,
                                cx=cx,
                                cy=cy,
                                image_width=image_width,
                                image_height=image_height,
                                court_points=court_points,
                                net_points=net_points,
                                left_scale=current["left_scale"] + dl,
                                right_scale=current["right_scale"] + dr,
                                net_height_m=current["net_height_m"] + dh,
                            )
                            if cand is not None:
                                candidates.append(cand)
                improving = [cand for cand in candidates if cand["objective"]["total"] + 1e-9 < best["objective"]["total"]]
                if improving:
                    cand = min(improving, key=lambda c: c["objective"]["total"])
                    best = cand
                    solver_path.append({
                        "intrinsics": dict(cand["intrinsics"]),
                        "pose": dict(cand["pose"]),
                        "court_orthopoints": [[float(p[0]), float(p[1])] for p in cand["image_quad"].tolist()],
                        "net_height_m": float(cand["net_height_m"]),
                        "objective": dict(cand["objective"]),
                    })
                    improved = True

    if best is None:
        notes.append("Full camera refinement skipped: joint court/net solve failed.")
        refined["notes"] = notes
        refined["estimate_sources"] = sources
        return refined

    if len(solver_path) > 24:
        keep_idx = np.linspace(0, len(solver_path) - 1, 24).round().astype(int)
        solver_path = [solver_path[int(i)] for i in keep_idx]

    h = best["h"]
    image_quad = np.asarray(best["image_quad"], dtype=np.float64)
    reproj = _project_homography(h, world_quad)
    reproj_err = float(np.sqrt(np.mean(np.sum((reproj - image_quad) ** 2, axis=1))))
    poly_err = _polyline_mean_segment_error(court_points)
    objective = best["objective"]

    sources.extend(["refined_court_lines", "net_constrained_camera_fit", "corrected_view_homography"])
    notes.append("Court-line camera refinement solved on the already warp-corrected working view.")
    notes.append(
        "Joint fit used court support lines plus antenna/net constraints to extend the court toward the camera."
    )
    notes.append(
        f"Homography reprojection RMSE: {reproj_err:.3f}px; polyline bend error: {poly_err:.4f}; "
        f"support err {objective['support_err']:.3f}px; net base err {objective['net_base_err']:.3f}px; "
        f"net top err {objective['net_top_err']:.3f}px; support position penalty {objective['support_position_penalty']:.4f}."
    )
    notes.append(f"Inferred net height: {best['net_height_m']:.3f}m within prior range 2.20m-2.50m.")

    refined["estimate_status"] = "refined"
    refined["estimate_sources"] = list(dict.fromkeys(sources))
    refined["confidence"] = min(
        0.98,
        max(
            float(seed_model.get("confidence", 0.0) or 0.0),
            0.78 if objective["total"] < 16.0 else 0.62,
        ),
    )
    refined["intrinsics"] = best["intrinsics"]
    refined["pose"] = best["pose"]
    refined["court_polylines"] = [[list(map(float, p)) for p in court_points]]
    refined["court_orthopoints"] = [[float(p[0]), float(p[1])] for p in image_quad.tolist()]
    refined["reprojection_error_px"] = reproj_err
    refined["support_line_error_px"] = objective["support_err"]
    refined["net_base_error_px"] = objective["net_base_err"]
    refined["net_top_error_px"] = objective["net_top_err"]
    refined["net_height_m"] = best["net_height_m"]
    refined["left_scale"] = best["left_scale"]
    refined["right_scale"] = best["right_scale"]
    refined["solver_path"] = solver_path
    refined["notes"] = notes
    return refined
