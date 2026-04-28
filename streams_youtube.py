"""
Download YouTube videos with yt-dlp and extract random PNG frames with ffmpeg.
Used by server.py and the CLI batch importer.
"""

from __future__ import annotations

import csv
import os
import random
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


YOUTUBE_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?v=|embed/|shorts/)|youtu\.be/)([a-zA-Z0-9_-]{11})"
)


def parse_youtube_id(url: str) -> str:
    m = YOUTUBE_ID_RE.search(url)
    if m:
        return m.group(1)
    raise ValueError(f"Could not parse YouTube video id from: {url[:120]}")


def which_yt_dlp() -> str | None:
    return shutil.which("yt-dlp") or shutil.which("yt-dlp.exe")


def which_ffmpeg() -> str | None:
    return shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")


def which_ffprobe() -> str | None:
    return shutil.which("ffprobe") or shutil.which("ffprobe.exe")


def _env_flag(name: str) -> bool:
    v = (os.environ.get(name) or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def resolve_ytdlp_cookies_file(
    explicit: Path | str | bool | None,
) -> Path | None:
    """Path to a Netscape-format cookies file for yt-dlp, or None.

    - ``explicit=False``: no cookies (e.g. CLI ``--no-cookies``).
    - ``explicit`` is a path: that file must exist and is used (env skip does not apply).
    - ``explicit is None``: unless ``YTDLP_SKIP_COOKIES`` is set, reads env
      ``YTDLP_COOKIES`` or ``YT_DLP_COOKIES`` (non-empty path must exist).

    **When to skip cookies:** Public videos usually work **without** cookies (yt-dlp
    uses the android client). With ``--cookies``, several clients are disabled and
    extraction often needs the **web** client plus JS / n-challenge handling (see
    yt-dlp EJS wiki). If you see only storyboards / "format not available", try
    ``YTDLP_SKIP_COOKIES=1`` or ``--no-cookies`` before investing in EJS setup.
    """
    if explicit is False:
        return None
    if explicit is not None:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise FileNotFoundError(
                f"yt-dlp cookies file not found: {p}\n"
                "Export from your browser (Netscape format). See:\n"
                "https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp"
            )
        return p.resolve()
    if _env_flag("YTDLP_SKIP_COOKIES"):
        return None
    env = (os.environ.get("YTDLP_COOKIES") or os.environ.get("YT_DLP_COOKIES") or "").strip()
    if not env:
        return None
    p = Path(env).expanduser()
    if not p.is_file():
        raise FileNotFoundError(
            f"YTDLP_COOKIES / YT_DLP_COOKIES file not found: {p}"
        )
    return p.resolve()


def _ytdlp_cookie_args(cookies_file: Path | None) -> list[str]:
    if cookies_file is None:
        return []
    return ["--cookies", str(cookies_file)]


def _ytdlp_global_extra_args(cookies_file: Path | None = None) -> list[str]:
    """Extra args right after ``yt-dlp``: optional shlex ``YTDLP_EXTRA_ARGS``, YouTube client.

    **Cookies:** yt-dlp **skips** several ``player_client`` values when ``--cookies`` is set
    (e.g. **android**, **ios** — “does not support cookies”), which then yields only thumbnails.
    So when cookies are used we **do not** force a client by default; yt-dlp picks its own chain
    (often **web**, which may need Node/Deno for JS challenges — see EJS wiki).

    Without cookies we default to **android**. Override any time: ``YTDLP_YOUTUBE_PLAYER_CLIENT``.
    """
    out: list[str] = []
    raw = os.environ.get("YTDLP_EXTRA_ARGS", "").strip()
    if raw:
        out.extend(shlex.split(raw))
    raw_client = os.environ.get("YTDLP_YOUTUBE_PLAYER_CLIENT")
    if raw_client is None:
        if cookies_file is None:
            out.extend(["--extractor-args", "youtube:player_client=android"])
        # else: no default --extractor-args; forced clients + cookies often break Shorts/long
    else:
        client = raw_client.strip()
        if client and client.lower() not in ("none", "off", "disabled", "0"):
            out.extend(["--extractor-args", f"youtube:player_client={client}"])
    return out


def _ytdlp_probe_pace_args() -> list[str]:
    """Pacing for short yt-dlp calls (duration, stream URL). Defaults favor not getting blocked."""
    sr = os.environ.get("YTDLP_PROBE_SLEEP_REQUESTS", "1").strip()
    try:
        v = float(sr)
    except ValueError:
        v = 1.0
    if v <= 0:
        return []
    return ["--sleep-requests", str(v)]


def _ytdlp_modest_download_args() -> list[str]:
    """Slow, polite full-file downloads — fewer parallel fragments, pauses between requests.

    Env overrides (defaults are conservative to reduce 429 / “bot” blocks):
    ``YTDLP_CONCURRENT_FRAGMENTS`` (default 2), ``YTDLP_SLEEP_REQUESTS`` (default 2),
    ``YTDLP_SLEEP_INTERVAL`` seconds before each download (default 5; 0 disables).
    """
    raw = os.environ.get("YTDLP_CONCURRENT_FRAGMENTS", "2").strip()
    try:
        n = max(1, min(int(raw), 16))
    except ValueError:
        n = 2
    sr = os.environ.get("YTDLP_SLEEP_REQUESTS", "2").strip()
    try:
        float(sr)
    except ValueError:
        sr = "2"
    out = [
        "--concurrent-fragments",
        str(n),
        "--sleep-requests",
        sr,
    ]
    sit = os.environ.get("YTDLP_SLEEP_INTERVAL", "5").strip()
    try:
        si = float(sit)
    except ValueError:
        si = 5.0
    if si > 0:
        out.extend(["--sleep-interval", str(si)])
    return out


def _yt_dlp_duration_sec(
    url: str,
    timeout: int = 120,
    *,
    cookies_file: Path | None = None,
) -> float:
    """Duration without downloading the full video (metadata only)."""
    r = subprocess.run(
        [
            "yt-dlp",
            *_ytdlp_global_extra_args(cookies_file),
            *_ytdlp_cookie_args(cookies_file),
            *_ytdlp_probe_pace_args(),
            "--no-download",
            "--print",
            "%(duration)s",
            "--no-playlist",
            url,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"yt-dlp duration failed: {(r.stderr or r.stdout)[:600]}"
        )
    line = (r.stdout or "").strip().splitlines()[0].strip()
    if not line or line.lower() in ("na", "none"):
        raise RuntimeError("Could not read duration (live or unavailable?)")
    return float(line)


def _yt_dlp_best_video_url(
    url: str,
    max_height: int = 480,
    timeout: int = 120,
    *,
    cookies_file: Path | None = None,
) -> str:
    """Direct HTTPS URL to the smallest usable video-only stream (no full file download)."""
    fmt = (
        f"bestvideo[height<={max_height}]/"
        f"bestvideo[height<=720]/"
        "bestvideo/"
        "best[height<=480]/worst"
    )
    r = subprocess.run(
        [
            "yt-dlp",
            *_ytdlp_global_extra_args(cookies_file),
            *_ytdlp_cookie_args(cookies_file),
            *_ytdlp_probe_pace_args(),
            "-f",
            fmt,
            "-g",
            "--no-playlist",
            url,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"yt-dlp -g failed: {(r.stderr or r.stdout)[:600]}"
        )
    lines = [x.strip() for x in (r.stdout or "").splitlines() if x.strip()]
    if not lines:
        raise RuntimeError("No stream URL from yt-dlp -g")
    return lines[0]


def _ffmpeg_frame_from_url(
    video_url: str,
    t_sec: float,
    out_path: Path,
    timeout: int = 600,
) -> None:
    """Grab one frame at t_sec from a stream URL (no local video file)."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{t_sec:.4f}",
        "-i",
        video_url,
        "-vframes",
        "1",
        "-q:v",
        "2",
        str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(
            f"ffmpeg frame grab failed: {(r.stderr or r.stdout)[-800:]}"
        )
    if not out_path.is_file() or out_path.stat().st_size < 100:
        raise RuntimeError("ffmpeg produced empty or missing PNG")


def _random_seek_times(
    duration: float,
    n: int,
    *,
    max_frames: int = 5,
    pair_gap_sec: float = 0.0,
    margin_pct: float = 0.10,
) -> list[float]:
    """Random ``t`` values in the middle portion of the video.

    Skips the first and last *margin_pct* (default 10%) of the video to avoid
    pre-/post-game footage.  If ``pair_gap_sec`` > 0, ensures ``t + gap`` fits.
    """
    n = max(1, min(int(n), max_frames))
    pct_margin = max(0.0, min(0.45, float(margin_pct)))
    gap = max(0.0, float(pair_gap_sec))
    abs_margin_lo = max(0.1, duration * pct_margin)
    abs_margin_hi = max(0.1, duration * pct_margin) + gap
    if duration <= 0.2:
        return [min(duration * 0.5, max(0.0, duration - abs_margin_hi - 0.01))] * n
    lo = abs_margin_lo
    hi = max(lo + 0.05, duration - abs_margin_hi)
    if hi <= lo:
        return [duration * 0.5] * n
    times: set[float] = set()
    guard = 0
    while len(times) < n and guard < n * 50:
        times.add(random.uniform(lo, hi))
        guard += 1
    while len(times) < n:
        times.add((lo + hi) / 2)
    return sorted(times)


def _video_duration_sec(path: Path) -> float:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr[:500] if r.stderr else r.stdout}")
    return float(r.stdout.strip())


def _download_youtube_to_dir(
    url: str,
    out_dir: Path,
    max_height: int = 480,
    *,
    cookies_file: Path | None = None,
) -> Path:
    """Full download fallback — uses smaller resolution + parallel fragments when possible."""
    out_dir.mkdir(parents=True, exist_ok=True)
    template = str(out_dir / "%(id)s.%(ext)s")
    ck = _ytdlp_cookie_args(cookies_file)
    fmt = (
        f"bv*[height<={max_height}]+ba/b[height<={max_height}]/"
        "bv*[height<=720]+ba/b[height<=720]/b"
    )
    modest = _ytdlp_modest_download_args()
    globx = _ytdlp_global_extra_args()
    cmd = [
        "yt-dlp",
        *globx,
        *ck,
        *modest,
        "-f",
        fmt,
        "--merge-output-format",
        "mp4",
        "-o",
        template,
        "--no-playlist",
        url,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        cmd2 = [
            "yt-dlp",
            *globx,
            *ck,
            *modest,
            "-f",
            f"best[height<={max_height}][ext=mp4]/best[height<={max_height}]/best[ext=mp4]/best",
            "-o",
            template,
            "--no-playlist",
            url,
        ]
        r2 = subprocess.run(cmd2, capture_output=True, text=True)
        if r2.returncode != 0:
            err = f"{r.stderr or r.stdout}\n---\n{r2.stderr or r2.stdout}"
            hint = ""
            low = err.lower()
            if "sign in" in low or "not a bot" in low or "429" in low:
                hint = (
                    "\n\nIf YouTube blocked the request: use a cookies file "
                    "(--cookies or env YTDLP_COOKIES). See pull_youtube_frames.py --help."
                )
            raise RuntimeError(
                "yt-dlp download failed (see stderr below)." + hint + f"\n---\n{err}"
            )
    vid = parse_youtube_id(url)
    for pat in (f"{vid}.mp4", f"{vid}.webm", f"{vid}.mkv"):
        p = out_dir / pat
        if p.is_file():
            return p
    matches = list(out_dir.glob(f"{vid}.*"))
    if matches:
        return matches[0]
    any_v = [p for p in out_dir.iterdir() if p.is_file() and p.suffix.lower() in (".mp4", ".webm", ".mkv", ".m4v")]
    if any_v:
        return any_v[0]
    raise RuntimeError("yt-dlp finished but no video file was found in temp dir")


def extract_png_frames(
    video_path: Path,
    images_dir: Path,
    times_sec: list[float],
    stem_base: str,
) -> list[str]:
    images_dir.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    for i, t in enumerate(times_sec):
        out_name = f"{stem_base}_f{i + 1}_{t:.2f}s.png"
        out_path = images_dir / out_name
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{t:.4f}",
            "-i",
            str(video_path),
            "-vframes",
            "1",
            "-q:v",
            "2",
            str(out_path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {r.stderr[-800:] if r.stderr else r.stdout}")
        out.append(out_name)
    return out


def _random_clip_start(duration_sec: float, clip_len: float) -> float:
    clip_len = min(clip_len, duration_sec)
    if duration_sec <= clip_len + 1e-3:
        return 0.0
    hi = duration_sec - clip_len
    return random.uniform(0.0, hi)


def _ffmpeg_extract_one_frame(
    video_path: Path, t_sec: float, out_path: Path, timeout: int = 120
) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{t_sec:.4f}",
        "-i",
        str(video_path),
        "-vframes",
        "1",
        "-q:v",
        "2",
        str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(
            f"ffmpeg frame extract failed: {(r.stderr or r.stdout)[-600:]}"
        )
    if not out_path.is_file() or out_path.stat().st_size < 80:
        raise RuntimeError("ffmpeg produced empty frame")


def pair_diff_png(path_a: Path, path_b: Path, out_path: Path) -> None:
    """Grayscale |I_a - I_b| for motion / change cues (Pillow)."""
    from PIL import Image, ImageChops

    im1 = Image.open(path_a).convert("RGB")
    im2 = Image.open(path_b).convert("RGB")
    if im1.size != im2.size:
        im2 = im2.resize(im1.size, Image.Resampling.BILINEAR)
    diff = ImageChops.difference(im1, im2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    diff.convert("L").save(out_path)


def _extract_adjacent_pair_disk(
    video_path: Path,
    images_dir: Path,
    stem_base: str,
    index_1based: int,
    t: float,
    *,
    pair_gap_sec: float,
    duration: float,
) -> tuple[str, str | None]:
    """Primary RGB at ``t`` and second frame at ``t+gap`` (``*_next.png``); same geometry as primary."""
    t = max(0.0, min(t, duration - 0.05))
    primary_name = f"{stem_base}_f{index_1based}_{t:.2f}s.png"
    out_path = images_dir / primary_name
    _ffmpeg_extract_one_frame(video_path, t, out_path)
    if pair_gap_sec <= 0:
        return primary_name, None
    gap = float(pair_gap_sec)
    t2 = min(t + gap, duration - 0.04)
    next_name = f"{stem_base}_f{index_1based}_{t:.2f}s_next.png"
    next_path = images_dir / next_name
    _ffmpeg_extract_one_frame(video_path, t2, next_path)
    return primary_name, next_name


def _extract_adjacent_pair_stream(
    video_url: str,
    images_dir: Path,
    stem_base: str,
    index_1based: int,
    t: float,
    *,
    pair_gap_sec: float,
    duration: float,
) -> tuple[str, str | None]:
    t = max(0.0, min(t, duration - 0.05))
    primary_name = f"{stem_base}_f{index_1based}_{t:.2f}s.png"
    out_path = images_dir / primary_name
    _ffmpeg_frame_from_url(video_url, t, out_path)
    if pair_gap_sec <= 0:
        return primary_name, None
    gap = float(pair_gap_sec)
    t2 = min(t + gap, duration - 0.04)
    next_name = f"{stem_base}_f{index_1based}_{t:.2f}s_next.png"
    next_path = images_dir / next_name
    _ffmpeg_frame_from_url(video_url, t2, next_path)
    return primary_name, next_name


def download_youtube_clip_mp4(
    url: str,
    clips_dir: Path,
    stem_base: str,
    *,
    clip_duration_sec: float = 120.0,
    max_height: int = 480,
    timeout_sec: int = 7200,
    cookies_file: Path | str | bool | None = None,
) -> dict[str, Any]:
    """Download a contiguous segment (default ~2 min) with audio when possible.

    Uses yt-dlp ``--download-sections`` on a merged format so **video and audio
    stay in sync** in the output MP4 (for labeling whistles / point starts).
    Falls back to downloading a full temp file and trimming with ffmpeg (slow).

    Output: ``<clips_dir>/<stem_base>_t<start>s_<len>s.mp4``
    """
    if not which_yt_dlp():
        raise RuntimeError("yt-dlp not found on PATH")
    clips_dir = Path(clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)
    cookies_path = resolve_ytdlp_cookies_file(cookies_file)

    duration = _yt_dlp_duration_sec(url, cookies_file=cookies_path)
    clip_len = min(float(clip_duration_sec), duration)
    start = _random_clip_start(duration, clip_len)
    end = min(start + clip_len, duration)
    if end <= start + 0.05:
        raise RuntimeError("Video too short for a clip segment")

    safe_stem = f"{stem_base}_t{start:.1f}s_{clip_len:.0f}s"
    out_path = clips_dir / f"{safe_stem}.mp4"

    fmt = (
        f"bv*[height<={max_height}]+ba/b[height<={max_height}]/"
        "bv*[height<=720]+ba/b[height<=720]/b"
    )
    cmd = [
        "yt-dlp",
        *_ytdlp_global_extra_args(cookies_path),
        *_ytdlp_cookie_args(cookies_path),
        *_ytdlp_modest_download_args(),
        "-f",
        fmt,
        "--merge-output-format",
        "mp4",
        "--force-overwrites",
        "-o",
        str(out_path),
        "--download-sections",
        f"*{start}-{end}",
        "--no-playlist",
        url,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    if r.returncode == 0 and out_path.is_file() and out_path.stat().st_size > 1000:
        return {
            "youtube_url": url,
            "youtube_id": parse_youtube_id(url),
            "source_duration_sec": duration,
            "clip_start_sec": start,
            "clip_end_sec": end,
            "clip_path": str(out_path),
            "mode": "download_sections",
            "max_height": max_height,
        }

    with tempfile.TemporaryDirectory(prefix="yt_clip_") as tmp:
        tmp_path = Path(tmp)
        video_path = _download_youtube_to_dir(
            url, tmp_path, max_height=max_height, cookies_file=cookies_path
        )
        dur2 = _video_duration_sec(video_path)
        start_adj = min(start, max(0.0, dur2 - 0.1))
        end_adj = min(end, dur2)
        seg_len = max(0.1, end_adj - start_adj)
        cmd2 = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_adj:.4f}",
            "-i",
            str(video_path),
            "-t",
            f"{seg_len:.4f}",
            "-avoid_negative_ts",
            "make_zero",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        r2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=timeout_sec)
        if r2.returncode != 0 or not out_path.is_file():
            raise RuntimeError(
                "Could not download clip (sections + trim fallback failed).\n"
                f"sections stderr: {(r.stderr or r.stdout)[:800]}\n"
                f"trim stderr: {(r2.stderr or r2.stdout)[:800]}"
            )
        return {
            "youtube_url": url,
            "youtube_id": parse_youtube_id(url),
            "source_duration_sec": dur2,
            "clip_start_sec": start_adj,
            "clip_end_sec": start_adj + seg_len,
            "clip_path": str(out_path),
            "mode": "trim_fallback",
            "sections_error": (r.stderr or r.stdout)[:400],
            "max_height": max_height,
        }


def extract_frames_from_local_clip(
    video_path: Path,
    images_dir: Path,
    count: int,
    stem_base: str,
    *,
    max_frames: int = 40,
    diff_gap_sec: float | None = None,
) -> dict[str, Any]:
    """Sample random frames from a local video file; optional |Δframe| PNGs for motion.

    ``diff_gap_sec`` (e.g. 1/25 for one frame at 25 fps): for each sampled time ``t``,
    compares frame at ``t`` with frame at ``t + gap`` (clamped to clip duration).
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(str(video_path))
    dur = _video_duration_sec(video_path)
    times = _random_seek_times(dur, count, max_frames=max_frames)
    images_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    diff_names: list[str] = []

    for i, t in enumerate(times):
        t = max(0.0, min(t, dur - 0.05))
        name = f"{stem_base}_f{i + 1}_{t:.2f}s.png"
        out_path = images_dir / name
        _ffmpeg_extract_one_frame(video_path, t, out_path)
        names.append(name)

        if diff_gap_sec is not None and float(diff_gap_sec) > 0:
            gap = float(diff_gap_sec)
            t2 = min(t + gap, dur - 0.04)
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False
            ) as tf:
                tmp_b = Path(tf.name)
            try:
                _ffmpeg_extract_one_frame(video_path, t2, tmp_b)
                diff_name = f"{stem_base}_f{i + 1}_{t:.2f}s_diff.png"
                diff_path = images_dir / diff_name
                pair_diff_png(out_path, tmp_b, diff_path)
                diff_names.append(diff_name)
            finally:
                if tmp_b.is_file():
                    tmp_b.unlink()

    out: dict[str, Any] = {
        "clip_path": str(video_path),
        "clip_duration_sec": dur,
        "times_sec": times,
        "extracted": names,
        "stem_base": stem_base,
    }
    if diff_names:
        out["diff_extracted"] = diff_names
        out["diff_gap_sec"] = float(diff_gap_sec)
    return out


def slugify_clip_label(name: str, max_len: int = 48) -> str:
    """Safe filename segment for PNG stems (alphanumeric, underscore, hyphen)."""
    s = re.sub(r"[^\w\-]+", "_", (name or "").strip(), flags=re.UNICODE)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] if s else ""


def extract_youtube_frames(
    url: str,
    images_dir: Path,
    count: int = 3,
    row_index: int | None = None,
    clip_slug: str | None = None,
    *,
    fast: bool = True,
    max_height: int = 480,
    adjacent_pair: bool = True,
    pair_gap_sec: float = 1.0 / 25.0,
    cookies_file: Path | str | bool | None = None,
) -> dict[str, Any]:
    """Pull PNG frames from a YouTube URL.

    **adjacent_pair** (default True): for each sample time ``t``, also save the next RGB
    frame at ``t+gap`` as ``*_next.png``. Training can load (primary, next) pairs; the
    labeler only uses the primary ``*_f*_*s.png``. ``*_next.png`` is excluded from the
    labeling queue.

    **cookies_file**: Netscape-format cookies for yt-dlp (or set env ``YTDLP_COOKIES``).
    Often required when YouTube returns "Sign in" / bot checks or HTTP 429.
    Pass ``False`` or set env ``YTDLP_SKIP_COOKIES=1`` to omit cookies (typical for
    public videos; avoids web-client + n-challenge issues).

    **fast=True** (default): metadata + one small video stream URL, then ffmpeg grabs
    frames **without** downloading the full merged file — usually much faster.

    **fast=False** or if the fast path fails: download a reduced-resolution file to disk
    (still capped by ``max_height``) and extract with ffmpeg locally.
    """
    if not which_yt_dlp():
        raise RuntimeError(
            "yt-dlp not found on PATH. Install: pip install yt-dlp (and ensure the script is on PATH)."
        )
    if not which_ffmpeg() or not which_ffprobe():
        raise RuntimeError(
            "ffmpeg / ffprobe not found on PATH (needed to extract PNG frames). "
            "Install FFmpeg for Windows, add its `bin` folder to your user PATH, "
            "open a new terminal, then run `where ffmpeg`. "
            "Alternatively run pulls in Docker (image includes ffmpeg)."
        )
    cookies_path = resolve_ytdlp_cookies_file(cookies_file)
    count = max(1, min(int(count), 5))
    gap = float(pair_gap_sec) if adjacent_pair else 0.0
    vid = parse_youtube_id(url)
    row_part = f"_r{row_index:04d}" if row_index is not None else ""
    slug = slugify_clip_label(clip_slug) if clip_slug else ""
    prefix = f"{slug}_" if slug else ""
    stem_base = f"{prefix}yt{vid}{row_part}"
    images_dir.mkdir(parents=True, exist_ok=True)

    fast_error: str | None = None
    if fast:
        try:
            duration = _yt_dlp_duration_sec(url, cookies_file=cookies_path)
            times = _random_seek_times(duration, count, pair_gap_sec=gap)
            video_url = _yt_dlp_best_video_url(
                url, max_height=max_height, cookies_file=cookies_path
            )
            names: list[str] = []
            next_names: list[str] = []
            for i, t in enumerate(times):
                if gap > 0:
                    p, nx = _extract_adjacent_pair_stream(
                        video_url,
                        images_dir,
                        stem_base,
                        i + 1,
                        t,
                        pair_gap_sec=gap,
                        duration=duration,
                    )
                    names.append(p)
                    if nx:
                        next_names.append(nx)
                else:
                    out_name = f"{stem_base}_f{i + 1}_{t:.2f}s.png"
                    _ffmpeg_frame_from_url(
                        video_url, t, images_dir / out_name
                    )
                    names.append(out_name)
            out: dict[str, Any] = {
                "youtube_id": vid,
                "youtube_url": url,
                "duration_sec": duration,
                "times_sec": times,
                "extracted": names,
                "stem_base": stem_base,
                "clip_slug": slug or None,
                "mode": "stream",
                "max_height": max_height,
                "pair_gap_sec": gap,
                "adjacent_pair": gap > 0,
            }
            if next_names:
                out["next_extracted"] = next_names
            return out
        except Exception as e:
            fast_error = str(e)

    with tempfile.TemporaryDirectory(prefix="yt_dl_") as tmp:
        tmp_path = Path(tmp)
        video_path = _download_youtube_to_dir(url, tmp_path, max_height=max_height)
        duration = _video_duration_sec(video_path)
        times = _random_seek_times(duration, count, pair_gap_sec=gap)
        names: list[str] = []
        next_names: list[str] = []
        if gap > 0:
            for i, t in enumerate(times):
                p, nx = _extract_adjacent_pair_disk(
                    video_path,
                    images_dir,
                    stem_base,
                    i + 1,
                    t,
                    pair_gap_sec=gap,
                    duration=duration,
                )
                names.append(p)
                if nx:
                    next_names.append(nx)
        else:
            names = extract_png_frames(video_path, images_dir, times, stem_base)

    out = {
        "youtube_id": vid,
        "youtube_url": url,
        "duration_sec": duration,
        "times_sec": times,
        "extracted": names,
        "stem_base": stem_base,
        "clip_slug": slug or None,
        "mode": "download",
        "max_height": max_height,
        "pair_gap_sec": gap,
        "adjacent_pair": gap > 0,
    }
    if next_names:
        out["next_extracted"] = next_names
    if cookies_path is not None:
        out["ytdlp_cookies"] = str(cookies_path)
    if fast_error is not None:
        out["fast_path_error"] = fast_error
    return out


def resolve_streams_csv() -> Path | None:
    env = os.environ.get("STREAMS_CSV")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    here = Path(__file__).resolve().parent
    for p in (here / "streams_export.csv", here / "data" / "streams_export.csv"):
        if p.is_file():
            return p
    data_dir = os.environ.get("DATA_DIR", "")
    if data_dir:
        p = Path(data_dir) / "streams_export.csv"
        if p.is_file():
            return p
    return None


def read_streams_csv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: (v or "").strip() for k, v in row.items()})
    return rows


def url_from_csv_row(rows: list[dict[str, str]], data_row_0based: int) -> tuple[str, dict[str, str]]:
    if data_row_0based < 0 or data_row_0based >= len(rows):
        raise IndexError(f"CSV data row index {data_row_0based} out of range (0..{len(rows) - 1})")
    r = rows[data_row_0based]
    u = r.get("youtube_url") or ""
    if not u:
        for k, v in r.items():
            if "youtube" in k.lower() and "url" in k.lower() and v:
                u = v
                break
    if not u:
        raise ValueError("Row has no youtube_url")
    return u, r


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract random PNG frames from YouTube URLs listed in streams_export.csv"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to streams_export.csv (default: auto-detect next to this script or DATA_DIR)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data root (images go to <data-dir>/images). Default: DATA_DIR env or ./data",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First data row index (0 = first row after header)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of CSV rows to process (default 1 for testing)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Frames per video (1–5)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Single YouTube URL (ignores --csv row range if set)",
    )
    parser.add_argument(
        "--no-pair",
        action="store_true",
        help="Single PNG per sample only (no paired *_next.png)",
    )
    parser.add_argument(
        "--pair-gap",
        type=float,
        default=1.0 / 25.0,
        metavar="SEC",
        help="Time between adjacent frames (default: 1/25 s)",
    )
    cg = parser.add_mutually_exclusive_group()
    cg.add_argument(
        "--cookies",
        type=Path,
        default=None,
        metavar="FILE",
        help="Netscape cookies file for yt-dlp (fixes YouTube bot/login). Or set YTDLP_COOKIES.",
    )
    cg.add_argument(
        "--no-cookies",
        action="store_true",
        help="Do not pass cookies (or YTDLP_COOKIES). Prefer for public videos.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir or Path(os.environ.get("DATA_DIR", "data"))
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    ck: Path | bool | None
    if args.no_cookies:
        ck = False
    else:
        ck = args.cookies
    pair_kw = dict(
        adjacent_pair=not args.no_pair,
        pair_gap_sec=float(args.pair_gap),
        cookies_file=ck,
    )
    if args.url:
        out = extract_youtube_frames(
            args.url.strip(),
            images_dir,
            count=args.count,
            row_index=None,
            **pair_kw,
        )
        print("OK", out)
        return

    csv_path = args.csv or resolve_streams_csv()
    if not csv_path or not csv_path.is_file():
        raise SystemExit(
            "Could not find streams_export.csv. Place it next to streams_youtube.py, "
            "under data/, or pass --csv PATH"
        )
    rows = read_streams_csv(csv_path)
    end = min(len(rows), args.start + max(1, args.limit))
    for i in range(args.start, end):
        url, meta = url_from_csv_row(rows, i)
        name = (meta.get("name") or "")[:50]
        print(f"Row {i}: {name!r} -> {url[:60]}...")
        try:
            out = extract_youtube_frames(
                url,
                images_dir,
                count=args.count,
                row_index=i,
                clip_slug=(meta.get("name") or "").strip() or None,
                **pair_kw,
            )
            print("  extracted:", out["extracted"])
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
