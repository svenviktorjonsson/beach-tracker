"""Shared rules: court/net point counts, wizard skip counts, and queue completion.

step_skip_counts (2 = layer skipped in UI) affects *wizard* resolution only via
court_layer_resolved / … — not pass_done_for_queue. The work queue drops a frame
only when there is real geometry (or complete=True), so skip presses alone
cannot “finish” an image for the queue without actual labels.
"""

from __future__ import annotations

from typing import Any

COURT_MIN_PTS = 3
COURT_MAX_PTS = 8
NET_MIN_PTS = 4
NET_MAX_PTS = 5
SKIP_DONE = 1


def _skip_count(data: dict[str, Any], key: str) -> int:
    raw = data.get("step_skip_counts") or {}
    if not isinstance(raw, dict):
        return 0
    v = raw.get(key)
    return int(v) if isinstance(v, int) and v >= 0 else 0


def court_polyline_valid(p: dict[str, Any]) -> bool:
    pts = p.get("points") or []
    n = len(pts)
    return COURT_MIN_PTS <= n <= COURT_MAX_PTS


def net_polyline_valid(p: dict[str, Any]) -> bool:
    pts = p.get("points") or []
    return NET_MIN_PTS <= len(pts) <= NET_MAX_PTS


def filter_court_polylines(polys: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in polys:
        if not isinstance(p, dict):
            continue
        if court_polyline_valid(p):
            out.append(p)
    return out


def filter_net_polylines(polys: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in polys:
        if not isinstance(p, dict):
            continue
        if net_polyline_valid(p):
            out.append(p)
    return out


def _court_polylines_list(data: dict[str, Any]) -> list[dict[str, Any]]:
    cp = data.get("court_polylines")
    if isinstance(cp, list) and cp:
        return [p for p in cp if isinstance(p, dict)]
    lines = data.get("lines")
    if isinstance(lines, list):
        legacy: list[dict[str, Any]] = []
        for l in lines:
            if not isinstance(l, dict):
                continue
            pts = l.get("points") or []
            legacy.append(
                {
                    "points": pts,
                    "scope": "adjacent" if l.get("label") == "adjacent" else "main",
                }
            )
        return legacy
    return []


def _net_polylines_list(data: dict[str, Any]) -> list[dict[str, Any]]:
    np = data.get("net_polylines")
    if not isinstance(np, list):
        return []
    return [p for p in np if isinstance(p, dict)]


def court_layer_resolved(data: dict[str, Any]) -> bool:
    if _skip_count(data, "court") >= SKIP_DONE:
        return True
    polys = _court_polylines_list(data)
    if not polys:
        return False
    return all(court_polyline_valid(p) for p in polys)


def net_layer_resolved(data: dict[str, Any]) -> bool:
    if _skip_count(data, "net") >= SKIP_DONE:
        return True
    polys = _net_polylines_list(data)
    return any(net_polyline_valid(p) for p in polys)


def people_layer_resolved(data: dict[str, Any]) -> bool:
    if _skip_count(data, "people") >= SKIP_DONE:
        return True
    people = data.get("people")
    return isinstance(people, list) and len(people) > 0


def ball_labeling_applies(data: dict[str, Any]) -> bool:
    """Whether the frame should ask for ball geometry."""
    biv = data.get("ball_in_view")
    if biv is None:
        biv = data.get("ball_in_play")
    return biv is not False


def ball_layer_resolved(data: dict[str, Any]) -> bool:
    if not ball_labeling_applies(data):
        return True
    if _skip_count(data, "ball") >= SKIP_DONE:
        return True
    balls = data.get("balls")
    if isinstance(balls, list) and len(balls) > 0:
        return True
    b = data.get("ball")
    return isinstance(b, dict) and isinstance(b.get("cx"), (int, float))


def has_session_and_crop(data: dict[str, Any]) -> bool:
    st = (data.get("session_type") or "").strip().lower()
    if st not in ("training", "game"):
        return False
    lq = data.get("label_quad")
    return isinstance(lq, list) and len(lq) == 4


def court_layer_has_valid_geometry(data: dict[str, Any]) -> bool:
    """Queue completion only — ignores step_skip_counts."""
    polys = _court_polylines_list(data)
    if not polys:
        return False
    return all(court_polyline_valid(p) for p in polys)


def net_layer_has_valid_geometry(data: dict[str, Any]) -> bool:
    polys = _net_polylines_list(data)
    return any(net_polyline_valid(p) for p in polys)


def people_layer_has_valid_geometry(data: dict[str, Any]) -> bool:
    people = data.get("people")
    return isinstance(people, list) and len(people) > 0


def ball_layer_has_valid_geometry(data: dict[str, Any]) -> bool:
    if not ball_labeling_applies(data):
        return True
    balls = data.get("balls")
    if isinstance(balls, list) and len(balls) > 0:
        return True
    b = data.get("ball")
    return isinstance(b, dict) and isinstance(b.get("cx"), (int, float))


def pass_done_for_queue(data: dict[str, Any]) -> bool:
    """True = omit frame from /api/next (complete flag or all real geometry present).

    Does not use step_skip_counts — skips only advance the wizard client-side.
    """
    if bool(data.get("complete")):
        return True
    if not has_session_and_crop(data):
        return False
    return (
        court_layer_has_valid_geometry(data)
        and net_layer_has_valid_geometry(data)
        and people_layer_has_valid_geometry(data)
        and ball_layer_has_valid_geometry(data)
    )


def pass_is_partial(data: dict[str, Any]) -> bool:
    """Started an annotation file but pass not done."""
    if pass_done_for_queue(data):
        return False
    # any saved signal
    if has_session_and_crop(data):
        return True
    st = (data.get("session_type") or "").strip().lower()
    if st in ("training", "game"):
        return True
    if _court_polylines_list(data) or _net_polylines_list(data):
        return True
    return False
