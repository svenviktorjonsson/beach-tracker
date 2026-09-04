from __future__ import annotations

import base64
import bisect
import json
import os
import random
import re
import secrets
import socket
import sqlite3
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from fastapi import Cookie, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request

from agreement import (
    annotation_distance,
    load_calibration,
    merge_annotations,
    passes_agreement,
    save_calibration,
    update_calibration_with_distance,
)
from streams_youtube import read_streams_csv, resolve_streams_csv
from training.scene_geometry import (
    build_projection_artifact,
    build_label_initialized_analysis,
    estimate_ball_world,
    load_analysis_court_orthopoints,
    project_court_grid,
)

_training_manager = None
def _get_training_manager():
    global _training_manager
    if _training_manager is None:
        try:
            from training.manager import TrainingManager
            _training_manager = TrainingManager(
                data_root=DATA_DIR,
                checkpoint_dir=DATA_DIR / "checkpoints",
            )
        except ImportError:
            return None
    return _training_manager

from labeling_rules import (
    filter_court_polylines,
    filter_net_polylines,
    pass_done_for_queue,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
IMAGES_SUB = "images"
CLIPS_SUB = "clips"
ANN_SUB = "annotations"
CLIP_EVENTS_SUB = "clips"  # under annotations/
CAL_FILE = DATA_DIR / "agreement_calibration.json"
QUALITY_FILE = DATA_DIR / "label_quality.json"
IMAGE_INDEX_FILE = DATA_DIR / "labeler_image_index.json"
LABELER_V2_DB_FILE = DATA_DIR / "labeler_v2.sqlite3"
LABELER_HOST_PORT = int(os.environ.get("LABELER_HOST_PORT", "8081"))
TRAINER_BASIC_AUTH_USER = os.environ.get("TRAINER_BASIC_AUTH_USER", "trainer")
TRAINER_BASIC_AUTH_PASS = os.environ.get("TRAINER_BASIC_AUTH_PASS", "")
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ALLOWED_CLIP_EXT = {".mp4", ".webm", ".mkv"}
FRAME_STEM_CLIP_RE = re.compile(r"^(.+)_f\d+_[\d.]+s$", re.IGNORECASE)
FRAME_TIME_IN_STEM_RE = re.compile(r"_f\d+_([\d.]+)s$", re.IGNORECASE)
FRAME_INDEX_IN_STEM_RE = re.compile(r"_f(\d+)_", re.IGNORECASE)
ROW_INDEX_IN_STEM_RE = re.compile(r"_r(\d+)(?:_|$)", re.IGNORECASE)


def _detect_lan_ip() -> str | None:
    sock: socket.socket | None = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("10.255.255.255", 1))
        ip = sock.getsockname()[0]
        if ip and not ip.startswith("127."):
            return ip
    except OSError:
        return None
    finally:
        if sock is not None:
            sock.close()
    return None


def _labeler_fallback_origins() -> list[str]:
    origins: list[str] = []
    lan_ip = _detect_lan_ip()
    if lan_ip:
        origins.append(f"http://{lan_ip}:{LABELER_HOST_PORT}")
    origins.append(f"http://127.0.0.1:{LABELER_HOST_PORT}")
    origins.append(f"http://localhost:{LABELER_HOST_PORT}")
    return origins


def _render_labeler_html() -> str:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    injected = (
        "<script>"
        f"window.__LABELER_FALLBACK_ORIGINS__ = {json.dumps(_labeler_fallback_origins())};"
        "</script>"
    )
    return html.replace("</head>", f"  {injected}\n</head>", 1)


def _is_labelable_frame(filename: str) -> bool:
    """Exclude second-frame sidecars (*_next.png) and legacy *_diff.png from the label queue."""
    lo = filename.lower()
    if lo.endswith("_diff.png") or lo.endswith("_next.png"):
        return False
    return True


def _images_dir() -> Path:
    return DATA_DIR / IMAGES_SUB


def _clips_dir() -> Path:
    return DATA_DIR / CLIPS_SUB


def _ann_dir() -> Path:
    return DATA_DIR / ANN_SUB


def _clip_events_dir() -> Path:
    return _ann_dir() / CLIP_EVENTS_SUB


def _ensure_dirs() -> None:
    _images_dir().mkdir(parents=True, exist_ok=True)
    _clips_dir().mkdir(parents=True, exist_ok=True)
    _ann_dir().mkdir(parents=True, exist_ok=True)
    _clip_events_dir().mkdir(parents=True, exist_ok=True)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_connect() -> sqlite3.Connection:
    LABELER_V2_DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(LABELER_V2_DB_FILE, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.OperationalError:
        # OneDrive/Windows-mounted volumes inside Docker can be flaky with WAL.
        conn.execute("PRAGMA journal_mode = DELETE")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def _frame_index_from_name(name: str) -> int | None:
    m = FRAME_INDEX_IN_STEM_RE.search(_stem(name))
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


_labelable_names_cache: list[str] | None = None
_labelable_names_cache_mtime_ns: int | None = None
_image_index_names_cache: list[str] | None = None
_image_index_cache_mtime_ns: int | None = None
_annotation_state_cache: dict[tuple[int, str], bool] | None = None
_annotation_state_mtime_ns: int | None = None

# (pass 1|2, stem) -> complete flag from annotation JSON (None = no file / not started).
# Invalidated when data/images listing changes; updated on POST /api/annotation.
_pass_completion_cache: dict[tuple[int, str], bool | None] = {}

# Per pass: (partially labeled sorted names, not-started sorted names). Built from completion cache.
_work_queues: dict[int, tuple[list[str], list[str]]] = {}
# data/images mtime when _work_queues[pass] was last built for that pass.
_work_queue_images_mtime: dict[int, int | None] = {}
# label_quality.json mtime when queues were built (exclusions affect who needs work).
_work_queue_quality_mtime: dict[int, int | None] = {}

_quality_cache: dict | None = None
_quality_mtime_ns: int | None = None
_streams_rows_cache: list[dict[str, str]] | None = None
_streams_rows_mtime_ns: int | None = None
_labeler_v2_catalog_mtime_ns: int | None = None
_labeler_v2_catalog_lock = threading.Lock()

# Per pass randomization state for /api/next selection.
# Keeps short memory to avoid ping-pong between the same few frames.
_recent_next_images: dict[int, deque[str]] = {}
_last_next_video_key: dict[int, str | None] = {}
RECENT_NEXT_MEMORY = 24

# ---------------------------------------------------------------------------
# Claim / lock system for parallel labeling.
#
# Each image may be labeled at most twice (pass 1 + pass 2).  We allow up to
# MAX_CLAIMS_PER_IMAGE concurrent claims so that two people can work on the
# same image simultaneously (one per pass) but a third person is steered away.
#
# Claims are keyed by (image_safe_name, pass_n) → (client_id, timestamp).
# A claim expires after CLAIM_TTL_SECONDS of inactivity (heartbeat / save
# refreshes it).  This prevents stale locks when someone closes their browser.
# ---------------------------------------------------------------------------
MAX_CLAIMS_PER_IMAGE = 2
CLAIM_TTL_SECONDS = 10 * 60  # 10 minutes

# (safe_image_name, pass_n) → (client_id, monotonic timestamp of last touch)
_claims: dict[tuple[str, int], tuple[str, float]] = {}
_last_v2_served: dict[tuple[str, int], str] = {}
_train_analysis_cache: dict[tuple[str, str | None, bool], dict[str, Any]] = {}


def _now_mono() -> float:
    return time.monotonic()


def _expire_stale_claims() -> None:
    cutoff = _now_mono() - CLAIM_TTL_SECONDS
    stale = [k for k, (_, ts) in _claims.items() if ts < cutoff]
    for k in stale:
        del _claims[k]


def _active_claims_for_image(safe: str) -> list[tuple[int, str]]:
    """Return [(pass_n, client_id)] for all live claims on this image."""
    _expire_stale_claims()
    out: list[tuple[int, str]] = []
    for (img, pn), (cid, _) in _claims.items():
        if img == safe:
            out.append((pn, cid))
    return out


def _claim_image(safe: str, pass_n: int, client_id: str) -> bool:
    """Try to claim an image+pass slot.  Returns True if granted."""
    _expire_stale_claims()
    key = (safe, pass_n)
    existing = _claims.get(key)
    if existing is not None:
        if existing[0] == client_id:
            _claims[key] = (client_id, _now_mono())
            return True
        return False
    active = _active_claims_for_image(safe)
    if len(active) >= MAX_CLAIMS_PER_IMAGE:
        return False
    _claims[key] = (client_id, _now_mono())
    return True


def _release_claim(safe: str, pass_n: int, client_id: str) -> None:
    key = (safe, pass_n)
    existing = _claims.get(key)
    if existing is not None and existing[0] == client_id:
        del _claims[key]


def _touch_claim(safe: str, pass_n: int, client_id: str) -> None:
    key = (safe, pass_n)
    existing = _claims.get(key)
    if existing is not None and existing[0] == client_id:
        _claims[key] = (client_id, _now_mono())


def _image_available_for_client(safe: str, client_id: str) -> bool:
    """True if this client can still get this image (already holds a claim or room left)."""
    active = _active_claims_for_image(safe)
    for _, cid in active:
        if cid == client_id:
            return True
    return len(active) < MAX_CLAIMS_PER_IMAGE


def _images_dir_mtime_ns() -> int | None:
    d = _images_dir()
    try:
        return d.stat().st_mtime_ns
    except OSError:
        return None


def _ann_dir_mtime_ns() -> int | None:
    d = _ann_dir()
    try:
        return d.stat().st_mtime_ns
    except OSError:
        return None


def _invalidate_pass_completion_cache() -> None:
    _pass_completion_cache.clear()
    _work_queues.clear()
    _work_queue_images_mtime.clear()
    _work_queue_quality_mtime.clear()


def _annotation_key_from_path(path: Path) -> tuple[int, str] | None:
    name = path.name
    if not name.endswith(".json"):
        return None
    stem = path.stem
    if stem.endswith("_merged") or stem.endswith("_needs_review"):
        return None
    if stem.endswith("_r1"):
        return (1, stem[:-3])
    if stem.endswith("_r2"):
        return (2, stem[:-3])
    return (1, stem)


def _load_annotation_state_cache() -> dict[tuple[int, str], bool]:
    global _annotation_state_cache, _annotation_state_mtime_ns
    d = _ann_dir()
    mtime = _ann_dir_mtime_ns()
    if _annotation_state_cache is not None and mtime == _annotation_state_mtime_ns:
        return _annotation_state_cache
    out: dict[tuple[int, str], bool] = {}
    if d.is_dir():
        files: list[tuple[Path, tuple[int, str]]] = []
        explicit_r1 = {
            key[1]
            for p in d.iterdir()
            if p.is_file()
            for key in [(_annotation_key_from_path(p))]
            if key is not None and key[0] == 1 and p.stem.endswith("_r1")
        }
        for p in sorted(d.iterdir()):
            if not p.is_file():
                continue
            key = _annotation_key_from_path(p)
            if key is not None:
                files.append((p, key))
        for p, key in files:
            pass_n, stem = key
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            # Prefer explicit r1 files over legacy stem.json when both exist.
            if pass_n == 1 and stem in explicit_r1 and not p.stem.endswith("_r1"):
                continue
            out[key] = bool(pass_done_for_queue(data))
    _annotation_state_cache = out
    _annotation_state_mtime_ns = mtime
    return out


def _invalidate_quality_cache() -> None:
    global _quality_cache, _quality_mtime_ns
    _quality_cache = None
    _quality_mtime_ns = None


def _cached_pass_completion(pass_n: int, stem: str) -> bool | None:
    """None = no annotation file; True = pass done (complete or rules satisfied); False = needs work."""
    key = (pass_n, stem)
    if key in _pass_completion_cache:
        return _pass_completion_cache[key]
    ann_state = _load_annotation_state_cache()
    if key not in ann_state:
        _pass_completion_cache[key] = None
        return None
    v = ann_state[key]
    _pass_completion_cache[key] = v
    return v


def _set_pass_completion_cache(pass_n: int, stem: str, value: bool | None) -> None:
    _pass_completion_cache[(pass_n, stem)] = value


def _video_key_for_image(name: str) -> str:
    """Best-effort video grouping key from a frame filename."""
    st = _stem(name)
    base = _stem_base_from_frame_stem(st)
    if base:
        return base
    return st


def _prune_next_selection_state(pass_n: int, allowed_names: set[str]) -> None:
    """Drop stale entries from recent-memory state when queues change."""
    dq = _recent_next_images.get(pass_n)
    if dq is not None:
        _recent_next_images[pass_n] = deque(
            (n for n in dq if n in allowed_names),
            maxlen=RECENT_NEXT_MEMORY,
        )
    last_key = _last_next_video_key.get(pass_n)
    if last_key and not any(_video_key_for_image(n) == last_key for n in allowed_names):
        _last_next_video_key[pass_n] = None


def _warm_pass_completion_for_all_stems(pass_n: int) -> None:
    """Populate completion cache for every labelable image (one-time per images-dir generation)."""
    for name in _list_labelable_names_cached():
        _cached_pass_completion(pass_n, _stem(name))


def _quality_file_mtime_ns() -> int | None:
    if not QUALITY_FILE.is_file():
        return None
    try:
        return QUALITY_FILE.stat().st_mtime_ns
    except OSError:
        return None


def _rebuild_work_queues(pass_n: int) -> tuple[list[str], list[str]]:
    """Recompute partial / not-started lists from cache + exclusions (no disk if cache warm)."""
    quality = _load_quality()
    excluded = set(quality.get("excluded_from_training") or [])
    partial: list[str] = []
    rest: list[str] = []
    for name in sorted(_list_labelable_names_cached()):
        if name in excluded:
            continue
        st = _stem(name)
        c = _cached_pass_completion(pass_n, st)
        if c is None:
            rest.append(name)
        elif c is True:
            continue
        else:
            partial.append(name)
    q = (partial, rest)
    _work_queues[pass_n] = q
    _work_queue_images_mtime[pass_n] = _images_dir_mtime_ns()
    _work_queue_quality_mtime[pass_n] = _quality_file_mtime_ns()
    _prune_next_selection_state(pass_n, set(partial) | set(rest))
    return q


def _ensure_work_queues(pass_n: int) -> tuple[list[str], list[str]]:
    """Warm all stems once, then build queues when images dir or label_quality changed."""
    im = _images_dir_mtime_ns()
    qm = _quality_file_mtime_ns()
    if (
        pass_n in _work_queues
        and _work_queue_images_mtime.get(pass_n) == im
        and _work_queue_quality_mtime.get(pass_n) == qm
    ):
        return _work_queues[pass_n]
    _warm_pass_completion_for_all_stems(pass_n)
    return _rebuild_work_queues(pass_n)


def _pick_next_from_queues(
    pass_n: int,
    partial: list[str],
    rest: list[str],
    current_safe: str | None,
    client_id: str | None = None,
) -> str | None:
    """Pick next frame randomly while avoiding same-video and quick repeats.

    When *client_id* is set, images that already have MAX_CLAIMS_PER_IMAGE
    active claims (from other clients) are filtered out so parallel labelers
    don't collide.
    """
    pool = partial + rest
    if not pool:
        return None

    candidates = list(pool)

    if client_id:
        available = [n for n in candidates if _image_available_for_client(n, client_id)]
        if available:
            candidates = available

    if current_safe and len(candidates) > 1:
        non_current = [n for n in candidates if n != current_safe]
        if non_current:
            candidates = non_current

    avoid_video_key: str | None = None
    if current_safe:
        avoid_video_key = _video_key_for_image(current_safe)
    if not avoid_video_key:
        avoid_video_key = _last_next_video_key.get(pass_n)

    if avoid_video_key and len(candidates) > 1:
        non_same_video = [n for n in candidates if _video_key_for_image(n) != avoid_video_key]
        if non_same_video:
            candidates = non_same_video

    recent = _recent_next_images.setdefault(pass_n, deque(maxlen=RECENT_NEXT_MEMORY))
    recent_set = set(recent)
    if len(candidates) > 1 and recent_set:
        non_recent = [n for n in candidates if n not in recent_set]
        if non_recent:
            candidates = non_recent

    chosen = random.choice(candidates)
    recent.append(chosen)
    _last_next_video_key[pass_n] = _video_key_for_image(chosen)
    return chosen


def _patch_work_queue_after_save(pass_n: int, safe: str, stem: str) -> None:
    """Update one pass queue after a save without scanning every image (fast path)."""
    if pass_n not in _work_queues:
        _rebuild_work_queues(pass_n)
        return
    quality = _load_quality()
    excluded = set(quality.get("excluded_from_training") or [])
    partial = [x for x in _work_queues[pass_n][0] if x != safe]
    rest = [x for x in _work_queues[pass_n][1] if x != safe]
    if safe in excluded:
        _work_queues[pass_n] = (partial, rest)
        return
    c = _pass_completion_cache.get((pass_n, stem))
    if c is None:
        bisect.insort(rest, safe)
    elif c is False:
        bisect.insort(partial, safe)
    _work_queues[pass_n] = (partial, rest)


def _list_labelable_names_cached() -> list[str]:
    """Labelable PNG names under images/ (unsorted); cached until data/images mtime changes."""
    global _labelable_names_cache, _labelable_names_cache_mtime_ns
    d = _images_dir()
    mtime = _images_dir_mtime_ns()
    if _labelable_names_cache is not None and mtime == _labelable_names_cache_mtime_ns:
        return _labelable_names_cache
    _invalidate_pass_completion_cache()
    names = _load_labelable_names_from_index(mtime)
    if names is None:
        names = []
        for p in d.iterdir():
            if (
                p.is_file()
                and p.suffix.lower() in ALLOWED_EXT
                and _is_labelable_frame(p.name)
            ):
                names.append(p.name)
    _labelable_names_cache = names
    _labelable_names_cache_mtime_ns = mtime
    return names


def _load_labelable_names_from_index(images_mtime_ns: int | None) -> list[str] | None:
    global _image_index_names_cache, _image_index_cache_mtime_ns
    if not IMAGE_INDEX_FILE.is_file():
        return None
    try:
        idx_mtime = IMAGE_INDEX_FILE.stat().st_mtime_ns
    except OSError:
        return None
    if _image_index_names_cache is not None and idx_mtime == _image_index_cache_mtime_ns:
        return _image_index_names_cache
    try:
        data = json.loads(IMAGE_INDEX_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    raw_names = data.get("labelable_images")
    if not isinstance(raw_names, list):
        return None
    names = [
        str(name)
        for name in raw_names
        if isinstance(name, str)
        and Path(name).suffix.lower() in ALLOWED_EXT
        and _is_labelable_frame(name)
    ]
    _image_index_names_cache = names
    _image_index_cache_mtime_ns = idx_mtime
    return names


def _has_any_labelable_png() -> bool:
    _ensure_dirs()
    for p in _images_dir().iterdir():
        if (
            p.is_file()
            and p.suffix.lower() in ALLOWED_EXT
            and _is_labelable_frame(p.name)
        ):
            return True
    return False


def _safe_basename_media(
    name: str, allowed_suffixes: set[str], err_label: str = "filename"
) -> str:
    """Single path component only; allows Unicode (e.g. Swedish letters in PNG names)."""
    base = os.path.basename(name)
    if not base or ".." in base or "/" in name or "\\" in name:
        raise HTTPException(400, f"Invalid {err_label}")
    if any(ord(c) < 32 for c in base):
        raise HTTPException(400, f"Invalid {err_label}")
    suf = Path(base).suffix.lower()
    if suf not in allowed_suffixes:
        raise HTTPException(400, "Unsupported file type")
    return base


def _safe_clip_filename(name: str) -> str:
    return _safe_basename_media(name, ALLOWED_CLIP_EXT, "clip filename")


def _stem_base_from_frame_stem(stem: str) -> str | None:
    m = FRAME_STEM_CLIP_RE.match(stem)
    return m.group(1) if m else None


def _parse_clip_time_from_frame_stem(stem: str) -> float | None:
    m = FRAME_TIME_IN_STEM_RE.search(stem)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _find_clip_for_stem_base(stem_base: str) -> str | None:
    """Match pull_youtube_clips naming: ``{stem_base}_t{start}s_{len}s.mp4``."""
    d = _clips_dir()
    if not d.is_dir():
        return None
    for p in sorted(d.glob(f"{stem_base}_t*.mp4")):
        if p.is_file():
            return p.name
    return None


def _path_clip_events(stem: str) -> Path:
    return _clip_events_dir() / f"{stem}_events.json"


def _safe_filename(name: str) -> str:
    return _safe_basename_media(name, ALLOWED_EXT, "filename")


def _stem(safe: str) -> str:
    return Path(safe).stem


def _path_r1(stem: str) -> Path:
    return _ann_dir() / f"{stem}_r1.json"


def _path_r2(stem: str) -> Path:
    return _ann_dir() / f"{stem}_r2.json"


def _path_legacy(stem: str) -> Path:
    return _ann_dir() / f"{stem}.json"


def _path_merged(stem: str) -> Path:
    return _ann_dir() / f"{stem}_merged.json"


def _path_review(stem: str) -> Path:
    return _ann_dir() / f"{stem}_needs_review.json"


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_quality() -> dict:
    global _quality_cache, _quality_mtime_ns
    _ensure_dirs()
    if not QUALITY_FILE.is_file():
        _invalidate_quality_cache()
        return {"pair_failures": {}, "excluded_from_training": []}
    try:
        mtime = QUALITY_FILE.stat().st_mtime_ns
    except OSError:
        mtime = None
    if _quality_cache is not None and mtime == _quality_mtime_ns:
        return _quality_cache
    data = json.loads(QUALITY_FILE.read_text(encoding="utf-8"))
    data.setdefault("pair_failures", {})
    data.setdefault("excluded_from_training", [])
    if not isinstance(data["excluded_from_training"], list):
        data["excluded_from_training"] = []
    _quality_cache = data
    _quality_mtime_ns = mtime
    return data


def _load_stream_rows_cached() -> list[dict[str, str]] | None:
    global _streams_rows_cache, _streams_rows_mtime_ns
    p = resolve_streams_csv()
    if p is None or not p.is_file():
        _streams_rows_cache = None
        _streams_rows_mtime_ns = None
        return None
    try:
        mtime = p.stat().st_mtime_ns
    except OSError:
        mtime = None
    if _streams_rows_cache is not None and mtime == _streams_rows_mtime_ns:
        return _streams_rows_cache
    rows = read_streams_csv(p)
    _streams_rows_cache = rows
    _streams_rows_mtime_ns = mtime
    return rows


def _row_index_from_image_name(name: str) -> int | None:
    m = ROW_INDEX_IN_STEM_RE.search(_stem(name))
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _image_slug_prefix(name: str) -> str | None:
    st = _stem(name)
    m = re.match(r"^(.*)_yt[a-zA-Z0-9_-]{11}_r\d+_f\d+_[\d.]+s$", st, flags=re.IGNORECASE)
    if not m:
        return None
    p = (m.group(1) or "").strip("_")
    return p or None


def _infer_session_type_from_stream_name(stream_name: str) -> Literal["training", "game"] | None:
    n = (stream_name or "").strip().lower()
    if not n:
        return None
    if any(t in n for t in ("tränare", "tranare", "träning", "traning", "training", "coach")):
        return "training"
    if " vs " in f" {n} " or "_vs_" in n or "vs_" in n or "_vs" in n:
        return "game"
    if any(
        t in n
        for t in (
            "gruppspel",
            "kval",
            "kvart",
            "semi",
            "final",
            "åttondel",
            "attondel",
            "sextondel",
            "round of",
            "playoff",
            "slutspel",
        )
    ):
        return "game"
    # Most named streams (e.g. team/group names) are not explicit match stages.
    return "training"


def _infer_image_session_type(name: str) -> tuple[Literal["training", "game"] | None, str | None]:
    rows = _load_stream_rows_cached()
    name_guess = _infer_session_type_from_stream_name(name)
    if not rows:
        return name_guess, None
    idx = _row_index_from_image_name(name)
    if idx is None or idx < 0 or idx >= len(rows):
        return name_guess, None
    r = rows[idx]
    stream_name = (r.get("name") or "").strip() or None
    if not stream_name:
        return name_guess, None
    stream_guess = _infer_session_type_from_stream_name(stream_name)
    if stream_guess == "training":
        return "training", stream_name
    if stream_guess == "game" or name_guess == "game":
        return "game", stream_name
    return stream_guess or name_guess, stream_name


def _init_labeler_v2_db() -> None:
    _ensure_dirs()
    with _db_connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS images (
                name TEXT PRIMARY KEY,
                neighbor_prev TEXT,
                neighbor_next TEXT,
                inferred_session_type TEXT,
                inferred_stream_name TEXT,
                source_group_label TEXT,
                source_row_index INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS labels (
                image_name TEXT NOT NULL,
                pass_n INTEGER NOT NULL,
                session_type TEXT,
                gender_category TEXT,
                game_phase TEXT,
                wizard_step TEXT,
                session_substep TEXT,
                geometry_substep TEXT,
                people_substep TEXT,
                distortion_k REAL NOT NULL DEFAULT 0.0,
                distortion_params TEXT,
                distortion_helper_points TEXT,
                focus_region TEXT,
                camera_model TEXT,
                net_points TEXT,
                net_polylines TEXT,
                complete INTEGER NOT NULL DEFAULT 0,
                skipped INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (image_name, pass_n),
                FOREIGN KEY (image_name) REFERENCES images(name) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS geometry_lines (
                image_name TEXT NOT NULL,
                pass_n INTEGER NOT NULL,
                slot TEXT NOT NULL,
                x REAL,
                y REAL,
                skipped INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (image_name, pass_n, slot),
                FOREIGN KEY (image_name, pass_n)
                    REFERENCES labels(image_name, pass_n) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS geometry_polyline_points (
                image_name TEXT NOT NULL,
                pass_n INTEGER NOT NULL,
                line_idx INTEGER NOT NULL,
                point_idx INTEGER NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (image_name, pass_n, line_idx, point_idx),
                FOREIGN KEY (image_name, pass_n)
                    REFERENCES labels(image_name, pass_n) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS people_boxes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_name TEXT NOT NULL,
                pass_n INTEGER NOT NULL,
                role TEXT NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                w REAL NOT NULL,
                h REAL NOT NULL,
                FOREIGN KEY (image_name, pass_n)
                    REFERENCES labels(image_name, pass_n) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS ball_labels (
                image_name TEXT NOT NULL,
                pass_n INTEGER NOT NULL,
                center_x REAL NOT NULL,
                center_y REAL NOT NULL,
                radius REAL NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                w REAL NOT NULL,
                h REAL NOT NULL,
                PRIMARY KEY (image_name, pass_n),
                FOREIGN KEY (image_name, pass_n)
                    REFERENCES labels(image_name, pass_n) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS ignore_points (
                image_name TEXT NOT NULL,
                pass_n INTEGER NOT NULL,
                idx INTEGER NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                PRIMARY KEY (image_name, pass_n, idx),
                FOREIGN KEY (image_name, pass_n)
                    REFERENCES labels(image_name, pass_n) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_labels_pass_status
                ON labels(pass_n, complete, skipped);
            CREATE INDEX IF NOT EXISTS idx_people_boxes_image_pass
                ON people_boxes(image_name, pass_n);
            CREATE INDEX IF NOT EXISTS idx_ignore_points_image_pass
                ON ignore_points(image_name, pass_n);
            CREATE INDEX IF NOT EXISTS idx_geometry_polyline_points_image_pass
                ON geometry_polyline_points(image_name, pass_n, line_idx, point_idx);
            """
        )
        for ddl in (
            "ALTER TABLE labels ADD COLUMN ball_in_play INTEGER",
            "ALTER TABLE labels ADD COLUMN ball_visible INTEGER",
            "ALTER TABLE labels ADD COLUMN distortion_params TEXT",
            "ALTER TABLE labels ADD COLUMN distortion_helper_points TEXT",
            "ALTER TABLE labels ADD COLUMN focus_region TEXT",
            "ALTER TABLE labels ADD COLUMN camera_model TEXT",
            "ALTER TABLE labels ADD COLUMN net_points TEXT",
            "ALTER TABLE labels ADD COLUMN net_polylines TEXT",
            "ALTER TABLE labels ADD COLUMN wizard_step TEXT",
            "ALTER TABLE labels ADD COLUMN session_substep TEXT",
            "ALTER TABLE labels ADD COLUMN geometry_substep TEXT",
            "ALTER TABLE labels ADD COLUMN people_substep TEXT",
        ):
            try:
                conn.execute(ddl)
            except sqlite3.OperationalError:
                pass


def _sync_labeler_v2_image_catalog() -> None:
    global _labeler_v2_catalog_mtime_ns
    current_mtime = _images_dir_mtime_ns()
    if _labeler_v2_catalog_mtime_ns == current_mtime:
        return
    with _labeler_v2_catalog_lock:
        current_mtime = _images_dir_mtime_ns()
        if _labeler_v2_catalog_mtime_ns == current_mtime:
            return

        names = sorted(_list_labelable_names_cached())
        now = _utcnow_iso()
        grouped: dict[str, list[str]] = {}
        for name in names:
            base = _stem_base_from_frame_stem(_stem(name)) or _stem(name)
            grouped.setdefault(base, []).append(name)

        neighbor_map: dict[str, tuple[str | None, str | None]] = {}
        for items in grouped.values():
            ordered = sorted(
                items,
                key=lambda n: (
                    _frame_index_from_name(n) if _frame_index_from_name(n) is not None else 999999,
                    _parse_clip_time_from_frame_stem(_stem(n))
                    if _parse_clip_time_from_frame_stem(_stem(n)) is not None
                    else 1e18,
                    n,
                ),
            )
            for idx, name in enumerate(ordered):
                prev_name = ordered[idx - 1] if idx > 0 else None
                next_name = ordered[idx + 1] if idx + 1 < len(ordered) else None
                neighbor_map[name] = (prev_name, next_name)

        with _db_connect() as conn:
            existing = {
                str(row["name"])
                for row in conn.execute("SELECT name FROM images").fetchall()
            }
            removed = existing.difference(names)
            for safe in removed:
                conn.execute("DELETE FROM images WHERE name = ?", (safe,))
            for name in names:
                session_type, stream_name = _infer_image_session_type(name)
                row_index = _row_index_from_image_name(name)
                group_label = _image_slug_prefix(name)
                prev_name, next_name = neighbor_map.get(name, (None, None))
                conn.execute(
                    """
                    INSERT INTO images (
                        name, neighbor_prev, neighbor_next, inferred_session_type,
                        inferred_stream_name, source_group_label, source_row_index,
                        created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET
                        neighbor_prev = excluded.neighbor_prev,
                        neighbor_next = excluded.neighbor_next,
                        inferred_session_type = excluded.inferred_session_type,
                        inferred_stream_name = excluded.inferred_stream_name,
                        source_group_label = excluded.source_group_label,
                        source_row_index = excluded.source_row_index,
                        updated_at = excluded.updated_at
                    """,
                    (
                        name,
                        prev_name,
                        next_name,
                        session_type,
                        stream_name,
                        group_label,
                        row_index,
                        now,
                        now,
                    ),
                )
        _labeler_v2_catalog_mtime_ns = current_mtime


def _neighbor_names_for_image(name: str) -> tuple[str | None, str | None]:
    names = _list_labelable_names_cached()
    base = _stem_base_from_frame_stem(_stem(name)) or _stem(name)
    grouped = [
        item
        for item in names
        if (_stem_base_from_frame_stem(_stem(item)) or _stem(item)) == base
    ]
    ordered = sorted(
        grouped,
        key=lambda n: (
            _frame_index_from_name(n) if _frame_index_from_name(n) is not None else 999999,
            _parse_clip_time_from_frame_stem(_stem(n))
            if _parse_clip_time_from_frame_stem(_stem(n)) is not None
            else 1e18,
            n,
        ),
    )
    try:
        idx = ordered.index(name)
    except ValueError:
        return None, None
    prev_name = ordered[idx - 1] if idx > 0 else None
    next_name = ordered[idx + 1] if idx + 1 < len(ordered) else None
    return prev_name, next_name


def _ensure_v2_image_row(name: str) -> None:
    safe = _safe_filename(name)
    session_type, stream_name = _infer_image_session_type(safe)
    row_index = _row_index_from_image_name(safe)
    group_label = _image_slug_prefix(safe)
    prev_name, next_name = _neighbor_names_for_image(safe)
    now = _utcnow_iso()
    with _db_connect() as conn:
        conn.execute(
            """
            INSERT INTO images (
                name, neighbor_prev, neighbor_next, inferred_session_type,
                inferred_stream_name, source_group_label, source_row_index,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                neighbor_prev = excluded.neighbor_prev,
                neighbor_next = excluded.neighbor_next,
                inferred_session_type = excluded.inferred_session_type,
                inferred_stream_name = excluded.inferred_stream_name,
                source_group_label = excluded.source_group_label,
                source_row_index = excluded.source_row_index,
                updated_at = excluded.updated_at
            """,
            (
                safe,
                prev_name,
                next_name,
                session_type,
                stream_name,
                group_label,
                row_index,
                now,
                now,
            ),
        )


def _save_quality(data: dict) -> None:
    global _quality_cache, _quality_mtime_ns
    QUALITY_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _quality_cache = data
    try:
        _quality_mtime_ns = QUALITY_FILE.stat().st_mtime_ns
    except OSError:
        _quality_mtime_ns = None


class ScopedPolyline(BaseModel):
    points: list[list[float]] = Field(default_factory=list)
    scope: str = "main"


class ExclusionZone(BaseModel):
    points: list[list[float]] = Field(default_factory=list)


class BallItem(BaseModel):
    """Axis-aligned box in image coords (matches detector training). Legacy ellipse JSON is coerced to a box."""

    x: float
    y: float
    w: float
    h: float
    scope: str = "in_play"

    @model_validator(mode="before")
    @classmethod
    def _legacy_ellipse_to_box(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        if d.get("w") is not None and d.get("x") is not None:
            return d
        if "cx" not in d or "cy" not in d:
            return d
        from training.geometry import ellipse_axis_aligned_box

        cx, cy = float(d["cx"]), float(d["cy"])
        if "rx" not in d and "r" in d:
            r = float(d["r"])
            rx = ry = r
        else:
            rx = float(d.get("rx", d.get("r", 1.0)))
            ry = float(d.get("ry", d.get("r", 1.0)))
        ang = float(d.get("angle", 0.0))
        x1, y1, x2, y2 = ellipse_axis_aligned_box(cx, cy, rx, ry, ang)
        d["x"] = x1
        d["y"] = y1
        d["w"] = max(0.0, x2 - x1)
        d["h"] = max(0.0, y2 - y1)
        d.setdefault("scope", "in_play")
        return d


class PersonItem(BaseModel):
    x: float
    y: float
    w: float
    h: float
    role: str = "player"


ACTION_TAG_VALUES = frozenset(
    {"finger_set", "bump_set", "reception", "serve"}
)


class AnnotationPayload(BaseModel):
    image: str
    exclusion_zones: list[ExclusionZone] = Field(default_factory=list)
    exclusion_points: list[list[float]] = Field(
        default_factory=list,
        description="Click-only ball false-positive markers (no polygon).",
    )
    inactive_ball_points: list[list[float]] = Field(
        default_factory=list,
        description="Click-only markers for visible balls that are not in active play.",
    )
    label_quad: list[list[float]] | None = Field(
        default=None,
        description="Four image corners [x,y] in order around the labeled region (training ROI).",
    )
    court_polylines: list[ScopedPolyline] = Field(default_factory=list)
    net_polylines: list[ScopedPolyline] = Field(default_factory=list)
    balls: list[BallItem] = Field(default_factory=list)
    people: list[PersonItem] = Field(default_factory=list)
    notes: str = ""
    complete: bool = False
    image_width: float | None = None
    image_height: float | None = None
    session_type: Literal["training", "game"] | None = None
    game_phase: Literal["between_balls", "during_play"] | None = None
    ball_in_view: bool | None = None
    ball_in_play: bool | None = None  # legacy; prefer ball_in_view
    source_stream_name: str | None = None
    source_group_label: str | None = None
    source_row_index: int | None = None
    gender_category: Literal["women", "men", "mixed"] | None = None
    action_tags: list[str] = Field(default_factory=list)
    step_skip_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Skip-step presses per layer; 2 on a layer = treat layer as done without data.",
    )

    @model_validator(mode="before")
    @classmethod
    def _filter_action_tags(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        tags = d.get("action_tags")
        if isinstance(tags, list):
            d["action_tags"] = sorted(
                {str(t) for t in tags if str(t) in ACTION_TAG_VALUES}
            )
        return d

    @model_validator(mode="after")
    def _normalize_context(self) -> AnnotationPayload:
        """Game phase only applies when session is game."""
        if self.session_type != "game":
            return self.model_copy(update={"game_phase": None})
        return self


class V2BallLabel(BaseModel):
    center_x: float
    center_y: float
    radius: float = Field(gt=0)
    x: float
    y: float
    w: float = Field(gt=0)
    h: float = Field(gt=0)


class V2PersonBox(BaseModel):
    x: float
    y: float
    w: float
    h: float
    role: Literal["player", "referee", "participant"]


class V2DistortionParams(BaseModel):
    model: str = "division"
    lambda_: float = Field(default=0.0, alias="lambda")
    cx_offset: float = 0.0
    cy_offset: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    rotation_deg: float = 0.0

    model_config = {
        "populate_by_name": True,
    }


class V2CameraIntrinsics(BaseModel):
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    skew: float = 0.0
    image_width: float = 0.0
    image_height: float = 0.0


class V2CameraDistortion(BaseModel):
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    rotation_deg: float = 0.0


class V2CameraPose(BaseModel):
    tx_m: float | None = None
    ty_m: float | None = None
    tz_m: float | None = None
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0


class V2CameraAnchor(BaseModel):
    label: str
    image: list[float] = Field(default_factory=list)
    world: list[float] = Field(default_factory=list)


class V2CameraModel(BaseModel):
    version: str = "pinhole-radtan-v1alpha"
    estimate_status: Literal["empty", "initial_guess", "refined"] = "initial_guess"
    estimate_sources: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    intrinsics: V2CameraIntrinsics = Field(default_factory=V2CameraIntrinsics)
    distortion: V2CameraDistortion = Field(default_factory=V2CameraDistortion)
    pose: V2CameraPose = Field(default_factory=V2CameraPose)
    net_anchors: list[V2CameraAnchor] = Field(default_factory=list)
    focus_region: list[list[float]] = Field(default_factory=list)
    court_polylines: list[list[list[float]]] = Field(default_factory=list)
    court_center_hint: list[float] | None = None
    court_model_quad: list[list[float]] = Field(default_factory=list)
    center_axis_points: list[list[float]] = Field(default_factory=list)
    center_axis_locked: bool = False
    coordinate_system: dict[str, Any] = Field(default_factory=dict)
    center_line_t: float = 0.5
    net_height_m: float = 2.35
    left_net_dx_px: float = 0.0
    right_net_dx_px: float = 0.0
    left_antenna_dx_px: float = 0.0
    right_antenna_dx_px: float = 0.0
    court_support_polyline: list[list[float]] = Field(default_factory=list)
    court_quad_image: list[list[float]] = Field(default_factory=list)
    court_model_segments: list[dict[str, Any]] = Field(default_factory=list)
    court_homography: list[list[float]] | None = None
    notes: list[str] = Field(default_factory=list)


class V2LabelPayload(BaseModel):
    image: str
    session_type: Literal["training", "game"] | None = None
    gender_category: Literal["women", "men", "mixed"] | None = None
    ball_in_play: bool | None = None
    ball_visible: bool | None = None
    wizard_step: str | None = None
    session_substep: str | None = None
    geometry_substep: str | None = None
    people_substep: str | None = None
    distortion_k: float = 0.0
    distortion_params: V2DistortionParams | None = None
    distortion_helper_points: list[list[float]] = Field(default_factory=list)
    complete: bool = False
    geometry_polylines: list[list[list[float]]] = Field(default_factory=list)
    net_points: list[list[float]] = Field(default_factory=list)
    net_polylines: list[list[list[float]]] = Field(default_factory=list)
    focus_region: list[list[float]] = Field(default_factory=list)
    camera_model: V2CameraModel | None = None
    people: list[V2PersonBox] = Field(default_factory=list)
    ball: V2BallLabel | None = None
    ignore_points: list[list[float]] = Field(default_factory=list)


class ClipMarker(BaseModel):
    """Timestamp in clip-local seconds (same timeline as the MP4)."""

    t_sec: float = Field(ge=0)
    kind: Literal[
        "point_start",
        "whistle",
        "finger_set",
        "bump_set",
        "reception",
        "serve",
        "other",
    ] = "point_start"
    note: str = ""


class ClipEventsPayload(BaseModel):
    clip: str
    session_type: Literal["training", "game"] | None = None
    markers: list[ClipMarker] = Field(default_factory=list)


app = FastAPI()


def _trainer_auth_enabled() -> bool:
    return bool(TRAINER_BASIC_AUTH_PASS)


def _trainer_auth_failed() -> Response:
    return Response(
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="Trainer"'},
    )


def _trainer_authorized(request: Request) -> bool:
    client_host = (request.client.host if request.client else "") or ""
    host_header = (request.headers.get("host", "") or "").split(":")[0].lower()
    request_host = (request.url.hostname or "").lower() if request.url.hostname else ""
    if (
        client_host in {"127.0.0.1", "::1"}
        or host_header in {"localhost", "127.0.0.1", "::1", "[::1]"}
        or request_host in {"localhost", "127.0.0.1", "::1"}
    ):
        return True
    if not _trainer_auth_enabled():
        return True
    header = request.headers.get("Authorization", "")
    if not header.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(header[6:], validate=True).decode("utf-8")
    except Exception:
        return False
    username, sep, password = decoded.partition(":")
    if not sep:
        return False
    return (
        secrets.compare_digest(username, TRAINER_BASIC_AUTH_USER)
        and secrets.compare_digest(password, TRAINER_BASIC_AUTH_PASS)
    )


@app.middleware("http")
async def trainer_auth_middleware(request: Request, call_next):
    path = request.url.path
    if path == "/train" or path.startswith("/api/train"):
        if not _trainer_authorized(request):
            return _trainer_auth_failed()
    return await call_next(request)


@app.middleware("http")
async def no_store_api_responses(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/api/"):
        # PNGs are static per filename; allow browser cache so repeat loads are instant.
        if path == "/api/image" or path.startswith("/api/image/"):
            response.headers["Cache-Control"] = "private, max-age=86400"
        else:
            response.headers["Cache-Control"] = "no-store"
    return response


@app.on_event("startup")
def startup() -> None:
    _ensure_dirs()
    _init_labeler_v2_db()
    def _warm():
        if _list_labelable_names_cached():
            _ensure_work_queues(1)
            _ensure_work_queues(2)
    threading.Thread(target=_warm, daemon=True).start()


@app.get("/train")
def train_page() -> Response:
    return FileResponse(STATIC_DIR / "train.html")


@app.get("/")
def home_page() -> Response:
    resp = HTMLResponse(_render_labeler_html())
    resp.set_cookie(
        "labeler_id",
        str(uuid.uuid4()),
        max_age=365 * 86400,
        httponly=False,
        samesite="lax",
    )
    return resp


@app.get("/labeler-style.css")
def labeler_style() -> Response:
    resp = FileResponse(STATIC_DIR / "style.css", media_type="text/css; charset=utf-8")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/labeler-app.js")
def labeler_app() -> Response:
    resp = FileResponse(
        STATIC_DIR / "app.js",
        media_type="application/javascript; charset=utf-8",
    )
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/labeler")
def labeler_page(labeler_id: str | None = Cookie(default=None)) -> Response:
    resp = HTMLResponse(_render_labeler_html())
    if not labeler_id:
        resp.set_cookie(
            "labeler_id",
            str(uuid.uuid4()),
            max_age=365 * 86400,
            httponly=False,
            samesite="lax",
        )
    return resp


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "data_dir": str(DATA_DIR)}


@app.get("/api/calibration")
def get_calibration() -> dict:
    return load_calibration(CAL_FILE)


def _image_meta(name: str) -> dict:
    safe = name
    st = _stem(safe)
    r1 = _read_json(_path_r1(st))
    r2 = _read_json(_path_r2(st))
    leg = _read_json(_path_legacy(st))
    if r1 is None and leg is not None:
        r1 = leg
    merged = _path_merged(st).is_file()
    review = _path_review(st).is_file()
    r1_done = bool(r1 and r1.get("complete"))
    r2_done = bool(r2 and r2.get("complete"))
    both = r1 is not None and r2 is not None and r1_done and r2_done
    q = _load_quality()
    ex = set(q.get("excluded_from_training") or [])
    fails = q.get("pair_failures") or {}
    return {
        "name": name,
        "has_r1": r1 is not None,
        "has_r2": r2 is not None,
        "r1_complete": r1_done,
        "r2_complete": r2_done,
        "r1_partial": r1 is not None and not r1_done,
        "r2_partial": r2 is not None and not r2_done,
        "has_merged": merged,
        "needs_review": review,
        "both_complete": both,
        "excluded": name in ex,
        "pair_failures": int(fails.get(name, 0)),
    }


@app.get("/api/images/has-labelable")
def api_has_labelable_png() -> dict:
    """Fast check: any labelable PNG exists (stops at first match; no full sort)."""
    return {"has": _has_any_labelable_png()}


@app.get("/api/images")
def list_images(
    details: bool = Query(
        False,
        description="If true, include per-image metadata (slow on large folders).",
    ),
) -> dict:
    _ensure_dirs()
    names = sorted(_list_labelable_names_cached())
    out: dict[str, Any] = {"images": names}
    if details:
        out["details"] = [_image_meta(n) for n in names]
    return out


@app.get("/api/streams/info")
def streams_info() -> dict:
    """Locate streams_export.csv (if present) and report row count."""
    p = resolve_streams_csv()
    if p is None or not p.is_file():
        return {"streams_csv": None, "rows": 0}
    rows = read_streams_csv(p)
    return {"streams_csv": str(p.resolve()), "rows": len(rows)}


@app.get("/api/image-context/{name}")
def image_context(name: str) -> dict:
    """Best-effort context for auto defaults (e.g. training vs game)."""
    safe = _safe_filename(name)
    session_type, stream_name = _infer_image_session_type(safe)
    row_index = _row_index_from_image_name(safe)
    slug = _image_slug_prefix(safe)
    return {
        "image": safe,
        "session_type": session_type,
        "stream_name": stream_name,
        "source_row_index": row_index,
        "source_group_label": slug,
    }


def _image_file_response(raw_name: str) -> FileResponse:
    safe = _safe_filename(raw_name)
    path = _images_dir() / safe
    if not path.is_file():
        raise HTTPException(404, "Image not found")
    return FileResponse(path)


@app.get("/api/image")
def get_image_query(
    file: str = Query(..., description="Filename under data/images (Unicode ok)"),
) -> FileResponse:
    """Preferred: ``/api/image?file=…`` avoids URL-path issues with ä, ö, etc."""
    return _image_file_response(file)


@app.get("/api/image/{name:path}")
def get_image_path(name: str) -> FileResponse:
    """Legacy path form; query form is preferred for non-ASCII names."""
    return _image_file_response(name)


@app.get("/api/clips")
def list_clips() -> dict:
    """MP4/WebM/MKV under data/clips (merged A+V from pull_youtube_clips).

    Used for a future step-2 UI: video + model predictions + human box review.
    The main labeler is static (frames) only.
    """
    _ensure_dirs()
    clips: list[dict[str, Any]] = []
    for p in sorted(_clips_dir().iterdir()):
        if p.is_file() and p.suffix.lower() in ALLOWED_CLIP_EXT:
            clips.append(
                {
                    "name": p.name,
                    "stem": p.stem,
                    "size": p.stat().st_size,
                }
            )
    return {"clips": clips}


@app.get("/api/clip/{name}")
def get_clip_file(name: str) -> FileResponse:
    """Stream clip with Range support (seek/scrub + audio in sync)."""
    safe = _safe_clip_filename(name)
    path = _clips_dir() / safe
    if not path.is_file():
        raise HTTPException(404, "Clip not found")
    suf = path.suffix.lower()
    media = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mkv": "video/x-matroska",
    }.get(suf, "application/octet-stream")
    return FileResponse(path, media_type=media, filename=safe)


@app.get("/api/clip-for-image/{name}")
def clip_for_image(name: str) -> dict:
    """Resolve data/clips/*.mp4 for frames from pull_youtube_clips naming."""
    safe = _safe_filename(name)
    st = _stem(safe)
    base = _stem_base_from_frame_stem(st)
    if not base:
        return {"clip": None, "clip_time_sec": None}
    clip = _find_clip_for_stem_base(base)
    if not clip:
        return {"clip": None, "clip_time_sec": None}
    t = _parse_clip_time_from_frame_stem(st)
    return {"clip": clip, "clip_time_sec": t}


@app.get("/api/clip-events/{name}")
def get_clip_events(name: str) -> dict:
    safe = _safe_clip_filename(name)
    st = _stem(safe)
    path = _path_clip_events(st)
    if not path.is_file():
        return {
            "clip": safe,
            "markers": [],
            "session_type": None,
            "updated_at": None,
        }
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/api/clip-events")
def save_clip_events(payload: ClipEventsPayload) -> dict:
    safe = _safe_clip_filename(payload.clip)
    clip_path = _clips_dir() / safe
    if not clip_path.is_file():
        raise HTTPException(404, "Clip not found under data/clips")
    st = _stem(safe)
    out_path = _path_clip_events(st)
    data = {
        "clip": safe,
        "session_type": payload.session_type,
        "markers": [m.model_dump() for m in payload.markers],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"saved": out_path.name, "markers": len(payload.markers)}


@app.get("/api/annotation/{name}")
def get_annotation(
    name: str,
    pass_n: int = Query(1, alias="pass", ge=1, le=2),
    labeler_id: str | None = Cookie(default=None),
) -> Response:
    safe = _safe_filename(name)
    st = _stem(safe)
    if labeler_id:
        _claim_image(safe, pass_n, labeler_id)
    if pass_n == 1:
        path = _path_r1(st)
        if not path.is_file():
            leg = _path_legacy(st)
            if leg.is_file():
                return JSONResponse(content=json.loads(leg.read_text(encoding="utf-8")))
            return Response(content="null", media_type="application/json")
    else:
        path = _path_r2(st)
        if not path.is_file():
            return Response(content="null", media_type="application/json")
    return JSONResponse(content=json.loads(path.read_text(encoding="utf-8")))


def _load_v2_label_record(safe: str, pass_n: int) -> dict:
    with _db_connect() as conn:
        image_row = conn.execute(
            """
            SELECT
                name,
                neighbor_prev,
                neighbor_next,
                inferred_session_type,
                inferred_stream_name,
                source_group_label,
                source_row_index
            FROM images
            WHERE name = ?
            """,
            (safe,),
        ).fetchone()
        if image_row is None:
            raise HTTPException(404, "Image not found in catalog")

        label_row = conn.execute(
            """
            SELECT session_type, gender_category, ball_in_play, ball_visible, wizard_step, session_substep, geometry_substep, people_substep, distortion_k, complete, skipped
                 , focus_region
                 , camera_model
                 , net_points
                 , net_polylines
                 , distortion_params
                 , distortion_helper_points
            FROM labels
            WHERE image_name = ? AND pass_n = ?
            """,
            (safe, pass_n),
        ).fetchone()
        line_rows = conn.execute(
            """
            SELECT slot, x, y, skipped
            FROM geometry_lines
            WHERE image_name = ? AND pass_n = ?
            """,
            (safe, pass_n),
        ).fetchall()
        polyline_rows = conn.execute(
            """
            SELECT line_idx, point_idx, x, y
            FROM geometry_polyline_points
            WHERE image_name = ? AND pass_n = ?
            ORDER BY line_idx, point_idx
            """,
            (safe, pass_n),
        ).fetchall()

    if image_row is None:
        inferred_session_type, inferred_stream_name = _infer_image_session_type(safe)
        prev_name, next_name = _neighbor_names_for_image(safe)
        image_meta = {
            "neighbor_prev": prev_name,
            "neighbor_next": next_name,
            "inferred_session_type": inferred_session_type,
            "inferred_stream_name": inferred_stream_name,
            "source_group_label": _image_slug_prefix(safe),
            "source_row_index": _row_index_from_image_name(safe),
        }
    else:
        image_meta = dict(image_row)
        people_rows = conn.execute(
            """
            SELECT x, y, w, h, role
            FROM people_boxes
            WHERE image_name = ? AND pass_n = ?
            ORDER BY id
            """,
            (safe, pass_n),
        ).fetchall()
        ball_row = conn.execute(
            """
            SELECT center_x, center_y, radius, x, y, w, h
            FROM ball_labels
            WHERE image_name = ? AND pass_n = ?
            """,
            (safe, pass_n),
        ).fetchone()
        ignore_rows = conn.execute(
            """
            SELECT x, y
            FROM ignore_points
            WHERE image_name = ? AND pass_n = ?
            ORDER BY idx
            """,
            (safe, pass_n),
        ).fetchall()

    geometry_polylines: list[list[list[float]]] = []
    line_map: dict[int, list[list[float]]] = {}
    for row in polyline_rows:
        line_idx = int(row["line_idx"])
        line_map.setdefault(line_idx, []).append([float(row["x"]), float(row["y"])])
    if line_map:
        geometry_polylines = [line_map[idx] for idx in sorted(line_map)]
    else:
        legacy_lines: dict[str, dict[str, float | bool | None]] = {}
        for row in line_rows:
            legacy_lines[str(row["slot"])] = {
                "x": row["x"],
                "y": row["y"],
                "skipped": bool(row["skipped"]),
            }
        ordered_legacy = []
        for slot in ("left", "far", "right", "near", "left_antenna", "net", "right_antenna"):
            line = legacy_lines.get(slot)
            if not line or line.get("skipped") or line.get("x") is None or line.get("y") is None:
                continue
            ordered_legacy.append([[float(line["x"]), float(line["y"])]])
        geometry_polylines = ordered_legacy

    distortion_params = {"model": "division", "lambda": 0.0, "cx_offset": 0.0, "cy_offset": 0.0, "k1": 0.0, "k2": 0.0, "k3": 0.0, "rotation_deg": 0.0}
    if label_row:
        raw_params = label_row["distortion_params"]
        if raw_params:
            try:
                parsed = json.loads(raw_params)
                distortion_params = {
                    "model": str(parsed.get("model", "division")),
                    "lambda": float(parsed.get("lambda", 0.0)),
                    "cx_offset": float(parsed.get("cx_offset", 0.0)),
                    "cy_offset": float(parsed.get("cy_offset", 0.0)),
                    "k1": float(parsed.get("k1", 0.0)),
                    "k2": float(parsed.get("k2", 0.0)),
                    "k3": float(parsed.get("k3", 0.0)),
                    "rotation_deg": float(parsed.get("rotation_deg", 0.0)),
                }
            except (TypeError, ValueError, json.JSONDecodeError):
                distortion_params = {"model": "division", "lambda": 0.0, "cx_offset": 0.0, "cy_offset": 0.0, "k1": float(label_row["distortion_k"]), "k2": 0.0, "k3": 0.0, "rotation_deg": 0.0}
        else:
            distortion_params = {"model": "division", "lambda": 0.0, "cx_offset": 0.0, "cy_offset": 0.0, "k1": float(label_row["distortion_k"]), "k2": 0.0, "k3": 0.0, "rotation_deg": 0.0}

    return {
        "image": safe,
        "pass": pass_n,
        "complete": bool(label_row["complete"]) if label_row else False,
        "skipped": bool(label_row["skipped"]) if label_row else False,
        "distortion_k": float(label_row["distortion_k"]) if label_row else 0.0,
        "distortion_params": distortion_params,
        "distortion_helper_points": json.loads(label_row["distortion_helper_points"]) if label_row and label_row["distortion_helper_points"] else [],
        "session_type": label_row["session_type"] if label_row else None,
        "gender_category": label_row["gender_category"] if label_row else None,
        "ball_in_play": bool(label_row["ball_in_play"]) if label_row and label_row["ball_in_play"] is not None else None,
        "ball_visible": bool(label_row["ball_visible"]) if label_row and label_row["ball_visible"] is not None else None,
        "wizard_step": label_row["wizard_step"] if label_row else None,
        "session_substep": label_row["session_substep"] if label_row else None,
        "geometry_substep": label_row["geometry_substep"] if label_row else None,
        "people_substep": label_row["people_substep"] if label_row else None,
        "focus_region": json.loads(label_row["focus_region"]) if label_row and label_row["focus_region"] else [],
        "camera_model": json.loads(label_row["camera_model"]) if label_row and label_row["camera_model"] else None,
        "geometry_polylines": geometry_polylines,
        "net_points": (
            json.loads(label_row["net_points"])
            if label_row and label_row["net_points"]
            else (
                [point for line in json.loads(label_row["net_polylines"]) for point in line]
                if label_row and label_row["net_polylines"]
                else []
            )
        ),
        "net_polylines": json.loads(label_row["net_polylines"]) if label_row and label_row["net_polylines"] else [],
        "people": [dict(row) for row in people_rows],
        "ball": dict(ball_row) if ball_row else None,
        "ignore_points": [[float(row["x"]), float(row["y"])] for row in ignore_rows],
        "neighbors": {
            "prev": image_meta["neighbor_prev"],
            "next": image_meta["neighbor_next"],
        },
        "inferred": {
            "session_type": image_meta["inferred_session_type"],
            "stream_name": image_meta["inferred_stream_name"],
            "source_group_label": image_meta["source_group_label"],
            "source_row_index": image_meta["source_row_index"],
        },
    }


def _upsert_v2_label(payload: V2LabelPayload, pass_n: int) -> None:
    safe = _safe_filename(payload.image)
    img_path = _images_dir() / safe
    if not img_path.is_file():
        raise HTTPException(404, "Image not found; save the file under data/images first")

    clean_polylines: list[list[tuple[float, float]]] = []
    for line in payload.geometry_polylines:
        if not isinstance(line, list):
            continue
        clean_line: list[tuple[float, float]] = []
        for point in line:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                clean_line.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError):
                continue
        if clean_line:
            clean_polylines.append(clean_line)
    clean_net_points: list[tuple[float, float]] = []
    if payload.net_points:
        for point in payload.net_points:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                clean_net_points.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError):
                continue
    else:
        for line in payload.net_polylines:
            if not isinstance(line, list):
                continue
            for point in line:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                try:
                    clean_net_points.append((float(point[0]), float(point[1])))
                except (TypeError, ValueError):
                    continue

    now = _utcnow_iso()
    with _db_connect() as conn:
        conn.execute(
            """
            INSERT INTO labels (
                image_name, pass_n, session_type, gender_category, ball_in_play, ball_visible,
                wizard_step, session_substep, geometry_substep, people_substep,
                distortion_k, distortion_params, distortion_helper_points, focus_region, camera_model, net_points, net_polylines, complete, skipped, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            ON CONFLICT(image_name, pass_n) DO UPDATE SET
                session_type = excluded.session_type,
                gender_category = excluded.gender_category,
                ball_in_play = excluded.ball_in_play,
                ball_visible = excluded.ball_visible,
                wizard_step = excluded.wizard_step,
                session_substep = excluded.session_substep,
                geometry_substep = excluded.geometry_substep,
                people_substep = excluded.people_substep,
                distortion_k = excluded.distortion_k,
                distortion_params = excluded.distortion_params,
                distortion_helper_points = excluded.distortion_helper_points,
                focus_region = excluded.focus_region,
                camera_model = excluded.camera_model,
                net_points = excluded.net_points,
                net_polylines = excluded.net_polylines,
                complete = excluded.complete,
                skipped = 0,
                updated_at = excluded.updated_at
            """,
            (
                safe,
                pass_n,
                payload.session_type,
                payload.gender_category,
                1 if payload.ball_in_play is True else 0 if payload.ball_in_play is False else None,
                1 if payload.ball_visible is True else 0 if payload.ball_visible is False else None,
                payload.wizard_step,
                payload.session_substep,
                payload.geometry_substep,
                payload.people_substep,
                float(payload.distortion_k),
                json.dumps((payload.distortion_params.model_dump(by_alias=True) if payload.distortion_params else {"k1": float(payload.distortion_k), "k2": 0.0, "k3": 0.0})),
                json.dumps(payload.distortion_helper_points),
                json.dumps(payload.focus_region),
                json.dumps(payload.camera_model.model_dump() if payload.camera_model else None),
                json.dumps([[x, y] for x, y in clean_net_points]),
                json.dumps(payload.net_polylines),
                1 if payload.complete else 0,
                now,
            ),
        )
        conn.execute(
            "DELETE FROM geometry_lines WHERE image_name = ? AND pass_n = ?",
            (safe, pass_n),
        )
        conn.execute(
            "DELETE FROM geometry_polyline_points WHERE image_name = ? AND pass_n = ?",
            (safe, pass_n),
        )
        for line_idx, line in enumerate(clean_polylines):
            for point_idx, point in enumerate(line):
                conn.execute(
                    """
                    INSERT INTO geometry_polyline_points (
                        image_name, pass_n, line_idx, point_idx, x, y, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (safe, pass_n, line_idx, point_idx, point[0], point[1], now),
                )

        conn.execute(
            "DELETE FROM people_boxes WHERE image_name = ? AND pass_n = ?",
            (safe, pass_n),
        )
        for person in payload.people:
            conn.execute(
                """
                INSERT INTO people_boxes (image_name, pass_n, role, x, y, w, h)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    safe,
                    pass_n,
                    person.role,
                    float(person.x),
                    float(person.y),
                    float(person.w),
                    float(person.h),
                ),
            )

        conn.execute(
            "DELETE FROM ball_labels WHERE image_name = ? AND pass_n = ?",
            (safe, pass_n),
        )
        if payload.ball is not None:
            conn.execute(
                """
                INSERT INTO ball_labels (
                    image_name, pass_n, center_x, center_y, radius, x, y, w, h
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    safe,
                    pass_n,
                    float(payload.ball.center_x),
                    float(payload.ball.center_y),
                    float(payload.ball.radius),
                    float(payload.ball.x),
                    float(payload.ball.y),
                    float(payload.ball.w),
                    float(payload.ball.h),
                ),
            )

        conn.execute(
            "DELETE FROM ignore_points WHERE image_name = ? AND pass_n = ?",
            (safe, pass_n),
        )
        for idx, point in enumerate(payload.ignore_points):
            if not isinstance(point, list | tuple) or len(point) < 2:
                continue
            conn.execute(
                """
                INSERT INTO ignore_points (image_name, pass_n, idx, x, y)
                VALUES (?, ?, ?, ?, ?)
                """,
                (safe, pass_n, idx, float(point[0]), float(point[1])),
            )


def _load_v2_label_status(pass_n: int) -> dict[str, tuple[bool, bool]]:
    with _db_connect() as conn:
        rows = conn.execute(
            """
            SELECT image_name, complete, skipped
            FROM labels
            WHERE pass_n = ?
            """,
            (pass_n,),
        ).fetchall()
    return {
        str(row["image_name"]): (bool(row["complete"]), bool(row["skipped"]))
        for row in rows
    }


def _choose_next_from_pool(
    pool: list[str],
    current_safe: str | None,
    client_id: str | None,
) -> str | None:
    if client_id:
        available = [n for n in pool if _image_available_for_client(n, client_id)]
        if available:
            pool = available
    if not pool:
        return None
    candidates = [name for name in pool if name != current_safe] or pool
    return random.choice(candidates)


def _remember_v2_served(client_id: str | None, pass_n: int, image_name: str | None) -> None:
    if not client_id or not image_name:
        return
    _last_v2_served[(client_id, pass_n)] = image_name


def _next_v2_image_for_pass(
    pass_n: int,
    current_safe: str | None = None,
    client_id: str | None = None,
) -> str | None:
    names = sorted(_list_labelable_names_cached())
    if not names:
        return None
    status = _load_v2_label_status(pass_n)
    partial = [
        name
        for name in names
        if name in status and not status[name][0] and not status[name][1]
    ]
    untouched = [name for name in names if name not in status]
    return _choose_next_from_pool(partial, current_safe, client_id) or _choose_next_from_pool(
        untouched, current_safe, client_id
    )


@app.get("/api/v2/label/{name}")
def get_label_v2(
    name: str,
    pass_n: int = Query(1, alias="pass", ge=1, le=2),
    labeler_id: str | None = Cookie(default=None),
) -> Response:
    safe = _safe_filename(name)
    if labeler_id:
        _claim_image(safe, pass_n, labeler_id)
    return JSONResponse(content=_load_v2_label_record(safe, pass_n))


@app.post("/api/v2/label")
def save_label_v2(
    payload: V2LabelPayload,
    pass_n: int = Query(1, alias="pass", ge=1, le=2),
    labeler_id: str | None = Cookie(default=None),
) -> dict:
    safe = _safe_filename(payload.image)
    _upsert_v2_label(payload, pass_n)
    if labeler_id and payload.complete:
        _release_claim(safe, pass_n, labeler_id)
    next_name = _next_v2_image_for_pass(pass_n, safe, client_id=labeler_id)
    if next_name and labeler_id:
        _claim_image(next_name, pass_n, labeler_id)
        _remember_v2_served(labeler_id, pass_n, next_name)
    return {
        "saved": True,
        "image": safe,
        "complete": payload.complete,
        "next_image": next_name,
    }


@app.get("/api/v2/next")
def api_next_v2(
    pass_n: int = Query(1, alias="pass", ge=1, le=2),
    current: str | None = None,
    labeler_id: str | None = Cookie(default=None),
) -> Response:
    _ensure_dirs()
    cid = labeler_id or str(uuid.uuid4())
    cur: str | None = None
    if current:
        try:
            cur = _safe_filename(current)
        except HTTPException:
            cur = None
    if cur is None:
        cur = _last_v2_served.get((cid, pass_n))
    nxt = _next_v2_image_for_pass(pass_n, cur, client_id=cid)
    if nxt:
        _claim_image(nxt, pass_n, cid)
        _remember_v2_served(cid, pass_n, nxt)
    resp = JSONResponse(content={"next": nxt})
    if not labeler_id:
        resp.set_cookie("labeler_id", cid, max_age=365 * 86400, httponly=False, samesite="lax")
    return resp


@app.post("/api/v2/skip-image")
def skip_image_v2(
    image: str = Query(...),
    pass_n: int = Query(1, alias="pass", ge=1, le=2),
    labeler_id: str | None = Cookie(default=None),
) -> dict:
    safe = _safe_filename(image)
    if labeler_id:
        _release_claim(safe, pass_n, labeler_id)
    with _db_connect() as conn:
        conn.execute(
            """
            INSERT INTO labels (
                image_name, pass_n, session_type, gender_category, ball_in_play, ball_visible,
                distortion_k, complete, skipped, updated_at
            )
            VALUES (?, ?, NULL, NULL, NULL, NULL, 0.0, 0, 1, ?)
            ON CONFLICT(image_name, pass_n) DO UPDATE SET
                complete = 0,
                skipped = 1,
                updated_at = excluded.updated_at
            """,
            (safe, pass_n, _utcnow_iso()),
        )
    next_name = _next_v2_image_for_pass(pass_n, safe, client_id=labeler_id)
    if next_name and labeler_id:
        _claim_image(next_name, pass_n, labeler_id)
        _remember_v2_served(labeler_id, pass_n, next_name)
    return {"skipped": safe, "next_image": next_name}


def _try_compare_and_merge(stem: str, img_name: str) -> dict:
    r1 = _read_json(_path_r1(stem)) or _read_json(_path_legacy(stem))
    r2 = _read_json(_path_r2(stem))
    out: dict = {}
    if not r1 or not r2 or not r1.get("complete") or not r2.get("complete"):
        return out

    w = float(r1.get("image_width") or r2.get("image_width") or 1920)
    h = float(r1.get("image_height") or r2.get("image_height") or 1080)

    d1 = {k: r1[k] for k in r1 if k not in ("complete", "pass", "notes")}
    d2 = {k: r2[k] for k in r2 if k not in ("complete", "pass", "notes")}
    d1["image"] = img_name
    d2["image"] = img_name
    dist = annotation_distance(d1, d2, w, h)
    out["compare_distance"] = dist

    cal = load_calibration(CAL_FILE)
    paired = set(cal.get("paired_stems") or [])
    if stem not in paired:
        if len(cal.get("distances") or []) < 10:
            cal = update_calibration_with_distance(cal, dist)
        paired.add(stem)
        cal["paired_stems"] = list(paired)
        save_calibration(CAL_FILE, cal)
    else:
        cal = load_calibration(CAL_FILE)

    out["calibration"] = {
        "n": len(cal.get("distances") or []),
        "threshold": cal.get("threshold"),
        "mean": cal.get("mean"),
    }

    agree = passes_agreement(dist, cal)
    out["agreement"] = agree
    if agree is True:
        merged = merge_annotations(r1, r2)
        merged["complete"] = True
        merged["merged_from"] = ["r1", "r2"]
        merged["image_width"] = w
        merged["image_height"] = h
        _path_merged(stem).write_text(json.dumps(merged, indent=2), encoding="utf-8")
        if _path_review(stem).is_file():
            _path_review(stem).unlink()
        q = _load_quality()
        fn = dict(q.get("pair_failures") or {})
        ex = set(q.get("excluded_from_training") or [])
        fn.pop(img_name, None)
        ex.discard(img_name)
        q["pair_failures"] = fn
        q["excluded_from_training"] = sorted(ex)
        _save_quality(q)
        out["merged"] = True
    elif agree is False:
        _path_review(stem).write_text(
            json.dumps(
                {
                    "image": img_name,
                    "distance": dist,
                    "threshold": cal.get("threshold"),
                    "reason": "distance_above_threshold",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        out["merged"] = False
        out["needs_review"] = True
        q = _load_quality()
        fn = dict(q.get("pair_failures") or {})
        ex = set(q.get("excluded_from_training") or [])
        if img_name not in ex:
            fn[img_name] = fn.get(img_name, 0) + 1
            if fn[img_name] >= 2:
                ex.add(img_name)
            q["pair_failures"] = fn
            q["excluded_from_training"] = sorted(ex)
            _save_quality(q)
        q = _load_quality()
        out["pair_failures"] = int(q.get("pair_failures", {}).get(img_name, 0))
        out["excluded_from_training"] = img_name in (q.get("excluded_from_training") or [])
    else:
        out["merged"] = None
        out["needs_calibration"] = True

    return out


@app.post("/api/annotation")
def save_annotation(
    payload: AnnotationPayload,
    pass_n: int = Query(1, alias="pass", ge=1, le=2),
    labeler_id: str | None = Cookie(default=None),
) -> dict:
    safe = _safe_filename(payload.image)
    img_path = _images_dir() / safe
    if not img_path.is_file():
        raise HTTPException(404, "Image not found; save the file under data/images first")

    st = _stem(safe)
    iw = payload.image_width
    ih = payload.image_height
    data = {
        "image": safe,
        "pass": pass_n,
        "complete": payload.complete,
        "image_width": iw,
        "image_height": ih,
        "exclusion_zones": [z.model_dump() for z in payload.exclusion_zones],
        "exclusion_points": [list(map(float, p)) for p in (payload.exclusion_points or [])],
        "inactive_ball_points": [
            list(map(float, p)) for p in (payload.inactive_ball_points or [])
        ],
        "label_quad": payload.label_quad,
        "court_polylines": filter_court_polylines(
            [p.model_dump() for p in payload.court_polylines]
        ),
        "net_polylines": filter_net_polylines(
            [p.model_dump() for p in payload.net_polylines]
        ),
        "balls": [b.model_dump() for b in payload.balls],
        "people": [p.model_dump() for p in payload.people],
        "notes": payload.notes or "",
        "session_type": payload.session_type,
        "game_phase": payload.game_phase
        if payload.session_type == "game"
        else None,
        "ball_in_view": payload.ball_in_view if payload.ball_in_view is not None else payload.ball_in_play,
        "ball_in_play": payload.ball_in_view if payload.ball_in_view is not None else payload.ball_in_play,
        "source_stream_name": payload.source_stream_name,
        "source_group_label": payload.source_group_label,
        "source_row_index": payload.source_row_index,
        "gender_category": payload.gender_category,
        "action_tags": list(payload.action_tags or []),
        "step_skip_counts": dict(payload.step_skip_counts or {}),
    }
    out = _path_r1(st) if pass_n == 1 else _path_r2(st)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _set_pass_completion_cache(
        pass_n, st, True if pass_done_for_queue(data) else False
    )

    qm_before = _quality_file_mtime_ns()
    cmp = _try_compare_and_merge(st, safe)
    qm_after = _quality_file_mtime_ns()
    if qm_before != qm_after:
        # Rare: merge/review updated label_quality.json — rebuild both passes.
        _rebuild_work_queues(1)
        _rebuild_work_queues(2)
    else:
        _patch_work_queue_after_save(pass_n, safe, st)

    if labeler_id and payload.complete:
        _release_claim(safe, pass_n, labeler_id)

    next_name = _next_image_for_pass(pass_n, safe, client_id=labeler_id)
    if next_name and labeler_id:
        _claim_image(next_name, pass_n, labeler_id)

    return {
        "saved": out.name,
        "complete": payload.complete,
        "pass_done": bool(pass_done_for_queue(data)),
        "next_image": next_name,
        **cmp,
    }


@app.get("/api/next")
def api_next(
    pass_n: int = Query(1, alias="pass", ge=1, le=2),
    current: str | None = None,
    labeler_id: str | None = Cookie(default=None),
) -> Response:
    _ensure_dirs()
    cid = labeler_id or str(uuid.uuid4())
    cur: str | None = None
    if current:
        try:
            cur = _safe_filename(current)
        except HTTPException:
            cur = None
    nxt = _next_image_for_pass(pass_n, cur, client_id=cid)
    if nxt:
        _claim_image(nxt, pass_n, cid)
    resp = JSONResponse(content={"next": nxt})
    if not labeler_id:
        resp.set_cookie("labeler_id", cid, max_age=365 * 86400, httponly=False, samesite="lax")
    return resp


def _next_image_for_pass(
    pass_n: int,
    current_safe: str | None = None,
    client_id: str | None = None,
) -> str | None:
    """Next labelable frame: partial work first, then not-started (sorted by filename).

    Work queues are built after startup and refreshed after each save; /api/next only
    indexes two short lists in RAM (no per-request directory walk or JSON reads).
    """
    _ensure_dirs()
    if not _list_labelable_names_cached():
        return None
    partial, rest = _ensure_work_queues(pass_n)
    return _pick_next_from_queues(pass_n, partial, rest, current_safe, client_id)


@app.post("/api/skip-image")
def skip_image(
    image: str = Query(...),
    pass_n: int = Query(1, alias="pass", ge=1, le=2),
    labeler_id: str | None = Cookie(default=None),
) -> dict:
    """Record that a labeler skipped this image.  Two skips → auto-exclude."""
    safe = _safe_filename(image)
    if labeler_id:
        _release_claim(safe, pass_n, labeler_id)

    q = _load_quality()
    skip_counts: dict[str, int] = dict(q.get("image_skip_counts") or {})
    skip_counts[safe] = skip_counts.get(safe, 0) + 1
    q["image_skip_counts"] = skip_counts

    excluded = False
    if skip_counts[safe] >= 2:
        ex = set(q.get("excluded_from_training") or [])
        if safe not in ex:
            ex.add(safe)
            q["excluded_from_training"] = sorted(ex)
            excluded = True
            _invalidate_pass_completion_cache()

    _save_quality(q)
    next_name = _next_image_for_pass(pass_n, safe, client_id=labeler_id)
    if next_name and labeler_id:
        _claim_image(next_name, pass_n, labeler_id)
    return {
        "skipped": safe,
        "skip_count": skip_counts[safe],
        "excluded_from_training": excluded,
        "next_image": next_name,
    }


@app.post("/api/heartbeat")
def heartbeat(
    image: str = Query(...),
    pass_n: int = Query(1, alias="pass", ge=1, le=2),
    labeler_id: str | None = Cookie(default=None),
) -> dict:
    """Keep a claim alive while the user is actively labeling."""
    if labeler_id:
        safe = _safe_filename(image)
        _touch_claim(safe, pass_n, labeler_id)
    return {"ok": True}


@app.get("/api/claims")
def get_claims() -> dict:
    """Debug / dashboard: show active claims across all images."""
    _expire_stale_claims()
    by_image: dict[str, list[dict]] = {}
    for (img, pn), (cid, ts) in _claims.items():
        by_image.setdefault(img, []).append({
            "pass": pn,
            "client": cid[:8],
            "age_s": round(_now_mono() - ts),
        })
    return {
        "active_claims": len(_claims),
        "images": by_image,
    }


@app.get("/api/label-quality")
def label_quality() -> dict:
    _ensure_dirs()
    return _load_quality()


@app.get("/api/train/status")
def train_status():
    mgr = _get_training_manager()
    if mgr is None:
        return {"error": "Training dependencies not installed (torch)"}
    return mgr.status()


@app.post("/api/train/start")
def train_start(
    epochs: int = Query(50),
    batch_size: int = Query(1),
    lr: float = Query(1e-4),
    image_size: int = Query(320),
    model_profile: str = Query("fast"),
    targets: str = Query("court,net,ball"),
    resume: bool = Query(False),
    augment: bool = Query(True),
):
    mgr = _get_training_manager()
    if mgr is None:
        raise HTTPException(500, "Training dependencies not installed")
    tgt_list = [t.strip() for t in targets.split(",") if t.strip()]
    return mgr.start(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        image_size=image_size,
        model_profile=model_profile,
        targets=tgt_list,
        resume=resume,
        augment=augment,
    )


@app.post("/api/train/pause")
def train_pause():
    mgr = _get_training_manager()
    if mgr is None:
        raise HTTPException(500, "Training dependencies not installed")
    return mgr.pause()


@app.post("/api/train/reset")
def train_reset():
    mgr = _get_training_manager()
    if mgr is None:
        raise HTTPException(500, "Training dependencies not installed")
    return mgr.reset()


@app.get("/api/train/dataset-counts")
def train_dataset_counts():
    """Count usable SQLite-backed training labels."""
    from training.export_sqlite_to_csv import sqlite_training_counts

    _ensure_dirs()
    return sqlite_training_counts(DATA_DIR, pass_n=1, complete_only=False)


@app.get("/api/train/losses")
def train_losses():
    mgr = _get_training_manager()
    if mgr is None:
        raise HTTPException(500, "Training dependencies not installed")

    def _loss_row(r):
        return {
            "step": r.step,
            "epoch": r.epoch,
            "batch_in_epoch": r.batch_in_epoch,
            "total_batches": r.total_batches,
            "images_processed": getattr(r, "images_processed", 0),
            "dataset_size": getattr(r, "dataset_size", 0),
            "loss_court": float(getattr(r, "loss_court", 0.0)),
            "loss_net": float(getattr(r, "loss_net", 0.0)),
            "loss_people": float(getattr(r, "loss_people", 0.0)),
            "loss_ball": float(getattr(r, "loss_ball", 0.0)),
        }

    return {
        "losses": [_loss_row(r) for r in mgr.loss_history],
        "epochs": mgr.epoch_history,
        "targets": mgr.config.get("targets", []),
    }


@app.get("/api/train/stream")
async def train_stream(request: Request):
    """SSE endpoint for real-time loss updates.

    _get_training_manager() imports PyTorch — must run in a worker thread, not on the asyncio
    loop, or the whole server freezes on first connection while torch loads.
    """
    from starlette.responses import StreamingResponse
    import asyncio
    import queue

    mgr = await run_in_threadpool(_get_training_manager)
    if mgr is None:
        raise HTTPException(500, "Training dependencies not installed")

    q = mgr.subscribe()

    async def event_gen():
        try:
            yield f"data: {json.dumps(mgr.status(), default=str)}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    evt = q.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.25)
                    continue
                try:
                    yield f"data: {json.dumps(evt, default=str)}\n\n"
                except (TypeError, ValueError):
                    await asyncio.sleep(0.25)
        finally:
            mgr.unsubscribe(q)

    return StreamingResponse(event_gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.post("/api/train/test")
def train_test(image: str = Query(None)):
    """Run inference on a specific or random image."""
    mgr = _get_training_manager()
    if mgr is None:
        raise HTTPException(503, detail="Training dependencies not installed")
    _ensure_dirs()
    try:
        if image:
            safe = _safe_filename(image)
        else:
            names = _list_labelable_names_cached()
            if not names:
                raise HTTPException(404, detail="No images found")
            safe = random.choice(names)
        img_path = IMAGES_DIR / safe
        if not img_path.exists():
            raise HTTPException(404, detail=f"Image not found: {safe}")
        img_sz = 256
        try:
            img_sz = int(mgr.config.get("image_size", 256) or 256)
        except (TypeError, ValueError):
            img_sz = 256
        result = mgr.predict(img_path, image_size=img_sz)
        if isinstance(result, dict) and result.get("error"):
            raise HTTPException(503, detail=result["error"])
        result["image"] = safe
        return result
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, detail=f"Inference failed: {e}") from e


def _analysis_court_orthopoints(image_name: str, pass_n: int = 1) -> list[list[float]]:
    with _db_connect() as conn:
        return load_analysis_court_orthopoints(conn, image_name, pass_n)


@app.get("/api/train/labeled-images")
def train_labeled_images():
    _ensure_dirs()
    items: list[dict[str, Any]] = []
    with _db_connect() as conn:
        rows = conn.execute(
            """
            SELECT image_name, wizard_step, complete
            FROM labels
            WHERE pass_n = 1
            ORDER BY image_name
            """
        ).fetchall()
        for row in rows:
            image_name = str(row["image_name"])
            try:
                label_record = _load_v2_label_record(image_name, pass_n=1)
                raw_polylines = label_record.get("geometry_polylines") or []
                court_points = raw_polylines[0] if raw_polylines and raw_polylines[0] else _analysis_court_orthopoints(image_name, pass_n=1)
            except Exception:
                court_points = []
            ball_n = conn.execute(
                "SELECT COUNT(*) AS n FROM ball_labels WHERE image_name = ? AND pass_n = 1",
                (image_name,),
            ).fetchone()
            if not court_points and not (ball_n and int(ball_n["n"] or 0) > 0):
                continue
            items.append({
                "image": image_name,
                "complete": bool(row["complete"]),
                "wizard_step": row["wizard_step"],
                "court_points": len(court_points),
                "ball_labels": int(ball_n["n"] or 0) if ball_n else 0,
            })
    return {"items": items}


@app.get("/api/train/analyze-image")
def train_analyze_image(image: str, include_nn: bool = True):
    _ensure_dirs()
    safe = _safe_filename(image)
    img_path = _images_dir() / safe
    if not img_path.exists():
        raise HTTPException(404, detail=f"Image not found: {safe}")
    label_record = _load_v2_label_record(safe, pass_n=1)
    cache_key = (safe, label_record.get("updated_at"), bool(include_nn))
    cached = _train_analysis_cache.get(cache_key)
    if cached is not None:
        return cached
    label_analysis = build_label_initialized_analysis(
        image_path=img_path,
        label_record=label_record,
        fallback_court_points=_analysis_court_orthopoints(safe, pass_n=1),
    )
    nn_analysis = None
    if include_nn:
        mgr = _get_training_manager()
        if mgr is not None:
            img_sz = 256
            try:
                img_sz = int(mgr.config.get("image_size", 256) or 256)
            except (TypeError, ValueError):
                img_sz = 256
            nn_result = mgr.predict(img_path, image_size=img_sz)
            if isinstance(nn_result, dict) and not nn_result.get("error"):
                nn_camera = nn_result.get("camera_model")
                nn_analysis = {
                    "source": "nn",
                    "court_points": nn_result.get("court_points") or [],
                    "net_points": nn_result.get("net_points") or [],
                    "ball": nn_result.get("ball"),
                    "camera_model": nn_camera,
                    "court_grid": project_court_grid(nn_camera),
                    "ball_world": estimate_ball_world(nn_result.get("ball"), nn_camera),
                    "projection_artifact": build_projection_artifact(
                        seed_camera_model=nn_camera,
                        refined_camera_model=nn_camera,
                        ball=nn_result.get("ball"),
                    ),
                    "coarse_proposals": nn_result.get("coarse_proposals") or {},
                }
    result = {
        "image": safe,
        "image_url": f"/api/image?file={quote(safe)}",
        "label_record": label_record,
        "label_analysis": label_analysis,
        "nn_analysis": nn_analysis,
    }
    _train_analysis_cache[cache_key] = result
    if len(_train_analysis_cache) > 64:
        oldest_key = next(iter(_train_analysis_cache))
        _train_analysis_cache.pop(oldest_key, None)
    return result


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
