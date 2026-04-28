"""
Download YouTube frames into data/images/ for offline labeling and training.

Run this separately (or on a schedule / in the background) — not from the browser.
The labeler only reads files already on disk.

Naming (PNG files under data/images/):
  Primary (labeled):  [{slug}_]yt{YOUTUBE11}_r{ROW:04d}_f{FRAME}_{seek}s.png
  Second frame (pair): same stem + ``_next.png``  (time ``seek + pair_gap_sec``, clamped to video)

  - Same YouTube video id (yt…) groups frames from one clip.
  - r{ROW} is the CSV row index (0-based) or link-list line index.
  - Optional slug prefix comes from the CSV ``name`` column when present.
  - Default pulls include both; ``pull_manifest.json`` lists ``pairs``: ``[[primary, next], …]`` per run.

Examples:
  python pull_youtube_frames.py --csv data/streams_export.csv --count 3
  python pull_youtube_frames.py --links youtube_links.txt --count 4
  python pull_youtube_frames.py --url "https://www.youtube.com/watch?v=..."

Requires yt-dlp and ffmpeg on PATH (same as the Docker labeler image).

**Adjacent pairs (default):** each sample saves primary ``*_f*_*s.png`` and ``*_next.png`` at
``t+gap`` for two-frame training; only the primary is labeled. Use ``--no-pair`` for single-frame
pulls only.

**YouTube blocks (429 / “Sign in to confirm you’re not a bot”):** pass a Netscape cookies file
with ``--cookies PATH`` or set env ``YTDLP_COOKIES`` to that path (see yt-dlp FAQ). Export cookies
from a browser where you are logged into YouTube.

**Cookies vs public videos:** With cookies, yt-dlp often uses the **web** client; android/ios are
skipped, and the **n-challenge** may require extra JS (EJS) setup. For **public** videos, prefer
``--no-cookies`` or ``YTDLP_SKIP_COOKIES=1`` so the default **android** path can list real formats.

Speed (see streams_youtube.extract_youtube_frames):
  - Default **fast** path: metadata + small video stream URL + ffmpeg frame grabs (no full file).
  - Fallback: reduced-resolution download; yt-dlp defaults to **slow, polite** settings (few
    fragments, pauses — see env ``YTDLP_*`` in ``streams_youtube.py``).
  - **--workers** (CSV): use **1** to avoid parallel hits; higher only if you accept block risk.
  - **--sleep-between-rows** (default 10s): pause between videos in sequential mode (0=off).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from streams_youtube import (
    extract_youtube_frames,
    read_streams_csv,
    resolve_streams_csv,
    url_from_csv_row,
)


def _fmt_duration(seconds: float) -> str:
    """Human-readable duration for ETA / elapsed."""
    if seconds < 0 or seconds > 86400 * 30:
        return "?"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _read_links_file(path: Path) -> list[str]:
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        u = raw.strip()
        if u and not u.startswith("#"):
            lines.append(u)
    return lines


def _pairs_from_run(run: dict) -> list[list[str]]:
    """``[[primary.png, same_stem_next.png], …]`` when ``next_extracted`` matches ``extracted``."""
    prim = run.get("extracted") or []
    nxt = run.get("next_extracted") or []
    if not nxt or len(prim) != len(nxt):
        return []
    return [[p, q] for p, q in zip(prim, nxt)]


def _runs_for_manifest(runs: list[dict]) -> list[dict]:
    out: list[dict] = []
    for r in runs:
        rr = dict(r)
        pairs = _pairs_from_run(rr)
        if pairs:
            rr["pairs"] = pairs
        out.append(rr)
    return out


def _print_run_files(out: dict) -> None:
    """Stdout: primary filenames and paired ``*_next.png`` when present."""
    print(f"  -> primary: {out['extracted']}", flush=True)
    nxt = out.get("next_extracted")
    if nxt:
        print(f"  -> next:    {nxt}", flush=True)


def _pull_options_from_args(args: argparse.Namespace) -> dict:
    d = {
        "adjacent_pair": not args.no_pair,
        "pair_gap_sec": float(args.pair_gap),
        "sleep_between_rows_sec": float(getattr(args, "sleep_between_rows", 0) or 0),
        "csv_workers": max(1, min(int(getattr(args, "workers", 1)), 8)),
    }
    if getattr(args, "no_cookies", False):
        d["cookies"] = "off"
    elif getattr(args, "cookies", None) is not None:
        d["cookies_file"] = str(Path(args.cookies).resolve())
    env_ck = (os.environ.get("YTDLP_COOKIES") or os.environ.get("YT_DLP_COOKIES") or "").strip()
    if (
        env_ck
        and "cookies_file" not in d
        and d.get("cookies") != "off"
    ):
        d["ytdlp_cookies_from_env"] = env_ck
    return d


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull PNG frames from YouTube into data/images/ (offline use)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data root (images go to <dir>/images). Default: DATA_DIR env or ./data",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="streams_export.csv (default: auto-detect like STREAMS_CSV / data/)",
    )
    parser.add_argument(
        "--links",
        type=Path,
        default=None,
        help="Text file: one YouTube URL per line (# comments allowed)",
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        metavar="URL",
        help="YouTube URL (repeatable; processed with --links)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First CSV data row index (0 = first row after header)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max CSV rows to process (default: all)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Frames per video (1–5)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Write JSON summary here (default: <data-dir>/pull_manifest.json)",
    )
    parser.add_argument(
        "--no-fast",
        action="store_true",
        help="Always download a reduced-resolution file per video (slow; for debugging)",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=480,
        metavar="PX",
        help="Max video height for streams / fallback download (default: 480)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="CSV mode only: parallel row workers (1–8). Prefer 1–2 to reduce YouTube 429/blocks.",
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
        help="Time between adjacent frames (default: 1/25 s ≈ one frame at 25 fps)",
    )
    ck = parser.add_mutually_exclusive_group()
    ck.add_argument(
        "--cookies",
        type=Path,
        default=None,
        metavar="FILE",
        help="Netscape cookies file for yt-dlp (YouTube login/bot). Or set env YTDLP_COOKIES.",
    )
    ck.add_argument(
        "--no-cookies",
        action="store_true",
        help="Ignore cookies and YTDLP_COOKIES (recommended for public videos).",
    )
    parser.add_argument(
        "--sleep-between-rows",
        type=float,
        default=10.0,
        metavar="SEC",
        help="Seconds to wait after each URL/CSV row before the next (default: 10; 0=off). "
        "Sequential mode only; reduces burst traffic and 429/blocks.",
    )
    args = parser.parse_args()
    workers = max(1, min(int(args.workers), 8))

    data_dir = args.data_dir or Path(os.environ.get("DATA_DIR", "data"))
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.manifest or (data_dir / "pull_manifest.json")
    runs: list[dict] = []
    errors: list[dict] = []

    if args.url or args.links:
        url_list: list[tuple[str, int | None, str | None]] = []
        n = 0
        for u in args.url:
            u = u.strip()
            if u:
                url_list.append((u, n, None))
                n += 1
        if args.links:
            if not args.links.is_file():
                print(f"ERROR: --links file not found: {args.links}", file=sys.stderr)
                sys.exit(1)
            for u in _read_links_file(args.links):
                url_list.append((u, n, None))
                n += 1
        if not url_list:
            print("ERROR: No URLs from --url / --links", file=sys.stderr)
            sys.exit(1)
        n_total = len(url_list)
        print(
            f"Processing {n_total} URL(s). ETA is based on average time per row so far.",
            flush=True,
        )
        if not args.no_pair:
            print(
                f"  Adjacent pairs: ON (gap={float(args.pair_gap)}s) → primary + *_next.png",
                flush=True,
            )
        if float(args.sleep_between_rows) > 0 and n_total > 1:
            print(
                f"  Pause between URLs: {float(args.sleep_between_rows)}s",
                flush=True,
            )
        t_batch = time.perf_counter()
        for j, (u, row_idx, slug) in enumerate(url_list):
            print(f"[{j + 1}/{n_total}] URL {u[:72]}…", flush=True)
            t_row = time.perf_counter()
            try:
                out = extract_youtube_frames(
                    u,
                    images_dir,
                    count=args.count,
                    row_index=row_idx,
                    clip_slug=slug,
                    fast=not args.no_fast,
                    max_height=args.max_height,
                    adjacent_pair=not args.no_pair,
                    pair_gap_sec=float(args.pair_gap),
                    cookies_file=args.cookies,
                )
                runs.append(out)
                _print_run_files(out)
            except Exception as e:
                errors.append({"url": u, "error": str(e)})
                print(f"  ERROR: {e}", file=sys.stderr)
            dt = time.perf_counter() - t_row
            done = j + 1
            eta = ((time.perf_counter() - t_batch) / done) * (n_total - done)
            print(
                f"  Row time {dt:.1f}s  ETA remaining ~{_fmt_duration(eta)}",
                flush=True,
            )
        if n_total:
            total_s = time.perf_counter() - t_batch
            print(f"Batch finished in {_fmt_duration(total_s)}.", flush=True)
        _write_manifest(
            manifest_path,
            data_dir,
            runs,
            errors,
            pull_options=_pull_options_from_args(args),
        )
        sys.exit(130 if errors else 0)

    csv_path = args.csv or resolve_streams_csv()
    if not csv_path or not csv_path.is_file():
        print(
            "ERROR: No streams_export.csv. Use --csv PATH, place data/streams_export.csv, "
            "or use --links / --url.",
            file=sys.stderr,
        )
        sys.exit(1)

    rows = read_streams_csv(csv_path)
    if args.limit is not None:
        end = min(len(rows), args.start + max(1, args.limit))
    else:
        end = len(rows)

    n_total = end - args.start
    fast = not args.no_fast
    mh = max(144, min(int(args.max_height), 1080))

    print(
        f"Processing {n_total} stream row(s) (CSV rows {args.start} … {end - 1}). "
        f"Fast stream path: {'on' if fast else 'off'}; max height {mh}px; workers={workers}.",
        flush=True,
    )
    if not args.no_pair:
        print(
            f"  Adjacent pairs: ON (gap={float(args.pair_gap)}s) → primary + *_next.png",
            flush=True,
        )
    if workers == 1 and float(args.sleep_between_rows) > 0:
        print(
            f"  Pause between rows: {float(args.sleep_between_rows)}s (sequential).",
            flush=True,
        )
    elif workers > 2 and n_total > 1:
        print(
            "  Note: workers>2 runs multiple videos in parallel — prefer --workers 1–2 to ease rate limits.",
            flush=True,
        )

    def _extract_kwargs() -> dict:
        d = {
            "fast": fast,
            "max_height": mh,
            "adjacent_pair": not args.no_pair,
            "pair_gap_sec": float(args.pair_gap),
        }
        if args.no_cookies:
            d["cookies_file"] = False
        elif args.cookies is not None:
            d["cookies_file"] = args.cookies
        return d

    t_batch = time.perf_counter()

    if workers > 1 and n_total > 1:
        lock = threading.Lock()
        done_cnt = [0]

        def _run_csv_row(i: int) -> tuple[int, dict | None, dict | None]:
            url, meta = url_from_csv_row(rows, i)
            name = (meta.get("name") or "").strip() or None
            label = f"Row {i}" + (f" ({name!r})" if name else "")
            print(f"[queue] {label} -> {url[:72]}…", flush=True)
            t_row = time.perf_counter()
            try:
                out = extract_youtube_frames(
                    url,
                    images_dir,
                    count=args.count,
                    row_index=i,
                    clip_slug=name,
                    **_extract_kwargs(),
                )
                dt = time.perf_counter() - t_row
                with lock:
                    done_cnt[0] += 1
                    k = done_cnt[0]
                    eta = ((time.perf_counter() - t_batch) / k) * (n_total - k)
                print(
                    f"[{k}/{n_total}] {label} mode={out.get('mode')} {dt:.1f}s  "
                    f"ETA ~{_fmt_duration(eta)}",
                    flush=True,
                )
                _print_run_files(out)
                return (i, out, None)
            except Exception as e:
                with lock:
                    done_cnt[0] += 1
                print(f"  ERROR row {i}: {e}", file=sys.stderr)
                return (i, None, {"row": i, "url": url, "error": str(e)})

        batch: list[tuple[int, dict | None, dict | None]] = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_run_csv_row, i) for i in range(args.start, end)]
            for fut in as_completed(futs):
                batch.append(fut.result())
        batch.sort(key=lambda x: x[0])
        for _i, out, err in batch:
            if out is not None:
                runs.append(out)
            if err is not None:
                errors.append(err)
    else:
        for j, i in enumerate(range(args.start, end)):
            url, meta = url_from_csv_row(rows, i)
            name = (meta.get("name") or "").strip() or None
            label = f"Row {i}" + (f" ({name!r})" if name else "")
            print(f"[{j + 1}/{n_total}] {label} -> {url[:80]}…", flush=True)
            t_row = time.perf_counter()
            try:
                out = extract_youtube_frames(
                    url,
                    images_dir,
                    count=args.count,
                    row_index=i,
                    clip_slug=name,
                    **_extract_kwargs(),
                )
                runs.append(out)
                print(f"  (mode={out.get('mode')})", flush=True)
                _print_run_files(out)
            except Exception as e:
                errors.append({"row": i, "url": url, "error": str(e)})
                print(f"  ERROR: {e}", file=sys.stderr)
            dt = time.perf_counter() - t_row
            done = j + 1
            eta = ((time.perf_counter() - t_batch) / done) * (n_total - done)
            print(
                f"  Row time {dt:.1f}s  ETA remaining ~{_fmt_duration(eta)}",
                flush=True,
            )
            if (
                workers == 1
                and float(args.sleep_between_rows) > 0
                and j + 1 < n_total
            ):
                time.sleep(float(args.sleep_between_rows))

    if n_total:
        total_s = time.perf_counter() - t_batch
        print(
            f"Batch finished: {n_total} row(s) in {_fmt_duration(total_s)} ({total_s / n_total:.1f}s avg/row).",
            flush=True,
        )

    _write_manifest(
        manifest_path,
        data_dir,
        runs,
        errors,
        pull_options=_pull_options_from_args(args),
    )
    sys.exit(130 if errors else 0)


def _write_manifest(
    path: Path,
    data_dir: Path,
    runs: list[dict],
    errors: list[dict],
    *,
    pull_options: dict | None = None,
) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir.resolve()),
        "images_dir": str((data_dir / "images").resolve()),
        "pull_options": pull_options or {},
        "runs": _runs_for_manifest(runs),
        "errors": errors,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {path}")


if __name__ == "__main__":
    main()
