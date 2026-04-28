"""
Download ~2-minute MP4 clips per stream, then sample PNG frames from each clip.

Workflow:
  1. ``data/clips/<stem>_t<start>s_<len>s.mp4`` — one random segment per video (audio when available).
  2. ``data/images/`` — random frames from that clip (not from the full YouTube timeline).

Optional ``--diff`` writes grayscale |frame(t) - frame(t+gap)| PNGs next to each frame
(``*_diff.png``) so models can exploit short-term motion.

Run separately from the labeler (Docker or host with yt-dlp + ffmpeg + Pillow).

Examples:
  python pull_youtube_clips.py --csv data/streams_export.csv --limit 2 --count 12 --diff
  python pull_youtube_clips.py --url "https://youtu.be/..." --count 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from streams_youtube import (
    download_youtube_clip_mp4,
    extract_frames_from_local_clip,
    parse_youtube_id,
    read_streams_csv,
    resolve_streams_csv,
    slugify_clip_label,
    url_from_csv_row,
)


def _fmt_duration(seconds: float) -> str:
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


def _write_manifest(
    path: Path,
    data_dir: Path,
    runs: list[dict],
    errors: list[dict],
) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir.resolve()),
        "clips_dir": str((data_dir / "clips").resolve()),
        "images_dir": str((data_dir / "images").resolve()),
        "runs": runs,
        "errors": errors,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download 2-minute clips to data/clips/, sample frames into data/images/."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data root (clips in <dir>/clips, frames in <dir>/images). Default DATA_DIR or ./data",
    )
    parser.add_argument("--csv", type=Path, default=None, help="streams_export.csv")
    parser.add_argument(
        "--links",
        type=Path,
        default=None,
        help="Text file: one YouTube URL per line",
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        metavar="URL",
        help="YouTube URL (repeatable)",
    )
    parser.add_argument("--start", type=int, default=0, help="First CSV row index")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max CSV rows (default: all)",
    )
    parser.add_argument(
        "--clip-duration",
        type=float,
        default=120.0,
        metavar="SEC",
        help="Target clip length in seconds (default: 120 = 2 min; capped by video length)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=12,
        help="Random frames to extract per clip (default 12)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=40,
        help="Hard cap on frame count per clip (default 40)",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=480,
        metavar="PX",
        help="Max video height for yt-dlp (default 480)",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Also save |Δframe| PNGs (*_diff.png) vs frame t+gap",
    )
    parser.add_argument(
        "--diff-gap",
        type=float,
        default=1.0 / 25.0,
        metavar="SEC",
        help="Time between paired frames for diff (default: 1/25 ≈ 25 fps)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Parallel CSV rows (1–8; clips are large — start with 1–2)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="JSON log (default: <data-dir>/pull_clips_manifest.json)",
    )
    parser.add_argument(
        "--frames-only",
        type=Path,
        default=None,
        metavar="CLIP.mp4",
        help="Skip download; extract frames (+ optional --diff) from this file only",
    )
    args = parser.parse_args()

    workers = max(1, min(int(args.workers), 8))
    data_dir = args.data_dir or Path(os.environ.get("DATA_DIR", "data"))
    clips_dir = data_dir / "clips"
    images_dir = data_dir / "images"
    clips_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.manifest or (data_dir / "pull_clips_manifest.json")
    runs: list[dict] = []
    errors: list[dict] = []

    mh = max(144, min(int(args.max_height), 1080))
    count = max(1, min(int(args.count), int(args.max_frames)))
    diff_gap = float(args.diff_gap) if args.diff else None

    def _stem(vid: str, row_index: int | None, slug: str | None) -> str:
        row_part = f"_r{row_index:04d}" if row_index is not None else ""
        pref = f"{slugify_clip_label(slug)}_" if slug else ""
        return f"{pref}yt{vid}{row_part}"

    if args.frames_only:
        vp = Path(args.frames_only)
        if not vp.is_file():
            print(f"ERROR: file not found: {vp}", file=sys.stderr)
            sys.exit(1)
        stem = vp.stem
        try:
            ex = extract_frames_from_local_clip(
                vp,
                images_dir,
                count,
                stem,
                max_frames=int(args.max_frames),
                diff_gap_sec=diff_gap,
            )
            runs.append({"clip": str(vp), "frames": ex})
            print("Extracted:", ex.get("extracted"), ex.get("diff_extracted") or "")
        except Exception as e:
            errors.append({"clip": str(vp), "error": str(e)})
            print(f"ERROR: {e}", file=sys.stderr)
        _write_manifest(manifest_path, data_dir, runs, errors)
        sys.exit(130 if errors else 0)

    def _process_url(
        url: str, row_index: int | None, slug: str | None
    ) -> dict[str, object]:
        vid = parse_youtube_id(url)
        stem_base = _stem(vid, row_index, slug)
        clip_info = download_youtube_clip_mp4(
            url,
            clips_dir,
            stem_base,
            clip_duration_sec=float(args.clip_duration),
            max_height=mh,
        )
        clip_path = Path(clip_info["clip_path"])
        frame_out = extract_frames_from_local_clip(
            clip_path,
            images_dir,
            count,
            stem_base,
            max_frames=int(args.max_frames),
            diff_gap_sec=diff_gap,
        )
        return {"clip": clip_info, "frames": frame_out}

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
                print(f"ERROR: --links not found: {args.links}", file=sys.stderr)
                sys.exit(1)
            for u in _read_links_file(args.links):
                url_list.append((u, n, None))
                n += 1
        if not url_list:
            print("ERROR: No URLs", file=sys.stderr)
            sys.exit(1)

        t0 = time.perf_counter()
        for j, (u, ri, sl) in enumerate(url_list):
            print(f"[{j + 1}/{len(url_list)}] {u[:72]}…", flush=True)
            try:
                out = _process_url(u, ri, sl)
                runs.append(out)
                fr = out["frames"]
                assert isinstance(fr, dict)
                print(f"  clip -> {out['clip'].get('clip_path')}", flush=True)
                print(f"  frames -> {fr.get('extracted')}", flush=True)
                if fr.get("diff_extracted"):
                    print(f"  diff   -> {fr['diff_extracted']}", flush=True)
            except Exception as e:
                errors.append({"url": u, "error": str(e)})
                print(f"  ERROR: {e}", file=sys.stderr)
        print(f"Done in {_fmt_duration(time.perf_counter() - t0)}.", flush=True)
        _write_manifest(manifest_path, data_dir, runs, errors)
        sys.exit(130 if errors else 0)

    csv_path = args.csv or resolve_streams_csv()
    if not csv_path or not csv_path.is_file():
        print(
            "ERROR: No streams_export.csv. Use --csv, or --url / --links, or --frames-only.",
            file=sys.stderr,
        )
        sys.exit(1)

    rows = read_streams_csv(csv_path)
    if args.limit is not None:
        end = min(len(rows), args.start + max(1, args.limit))
    else:
        end = len(rows)
    n_total = end - args.start
    print(
        f"Clips: {float(args.clip_duration):.0f}s target; {count} frames/clip; "
        f"diff={'on' if args.diff else 'off'}; workers={workers}; rows {args.start}..{end - 1}.",
        flush=True,
    )

    t_batch = time.perf_counter()

    def _run_row(i: int) -> tuple[int, dict | None, dict | None]:
        url, meta = url_from_csv_row(rows, i)
        name = (meta.get("name") or "").strip() or None
        label = f"Row {i}" + (f" ({name!r})" if name else "")
        print(f"[queue] {label} -> {url[:72]}…", flush=True)
        try:
            out = _process_url(url, i, name)
            fr = out["frames"]
            assert isinstance(fr, dict)
            print(
                f"[done] {label} -> {len(fr.get('extracted') or [])} frames",
                flush=True,
            )
            return (i, out, None)
        except Exception as e:
            print(f"  ERROR row {i}: {e}", file=sys.stderr)
            return (i, None, {"row": i, "url": url, "error": str(e)})

    if workers > 1 and n_total > 1:
        batch: list[tuple[int, dict | None, dict | None]] = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_run_row, i) for i in range(args.start, end)]
            for fut in as_completed(futs):
                batch.append(fut.result())
        batch.sort(key=lambda x: x[0])
        for _i, out, err in batch:
            if out is not None:
                runs.append(out)
            if err is not None:
                errors.append(err)
    else:
        for i in range(args.start, end):
            _i, out, err = _run_row(i)
            if out is not None:
                runs.append(out)
            if err is not None:
                errors.append(err)

    if n_total:
        total_s = time.perf_counter() - t_batch
        print(
            f"Batch finished: {n_total} row(s) in {_fmt_duration(total_s)}.",
            flush=True,
        )
    _write_manifest(manifest_path, data_dir, runs, errors)
    sys.exit(130 if errors else 0)


if __name__ == "__main__":
    main()
