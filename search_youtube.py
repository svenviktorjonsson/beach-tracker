"""
Search YouTube for beach volleyball videos and write a links file for pull_youtube_frames.py.

Usage:
  python search_youtube.py                              # default: "beach volleyball world" × 20 results
  python search_youtube.py --query "FIVB beach volley"  # custom search
  python search_youtube.py --results 50                 # more results
  python search_youtube.py --min-duration 120           # skip clips under 2 min

Then pull frames:
  python pull_youtube_frames.py --links data/youtube_search_links.txt --count 3 --no-cookies
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from streams_youtube import which_yt_dlp


def search_youtube(
    query: str,
    max_results: int = 20,
    min_duration_sec: float = 120,
) -> list[dict]:
    """Use yt-dlp flat-playlist search to find YouTube video URLs + metadata."""
    if not which_yt_dlp():
        raise RuntimeError("yt-dlp not found on PATH")

    cmd = [
        "yt-dlp",
        f"ytsearch{max_results}:{query}",
        "--flat-playlist",
        "--dump-json",
        "--no-download",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"yt-dlp search failed: {(r.stderr or r.stdout)[:800]}")

    results: list[dict] = []
    for line in r.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        duration = entry.get("duration") or 0
        if duration < min_duration_sec:
            continue

        vid = entry.get("id") or entry.get("url")
        if not vid:
            continue
        url = f"https://www.youtube.com/watch?v={vid}" if len(vid) == 11 else vid
        results.append({
            "url": url,
            "title": entry.get("title", ""),
            "duration": duration,
            "channel": entry.get("channel") or entry.get("uploader") or "",
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Search YouTube for beach volleyball videos.")
    parser.add_argument(
        "--query",
        default="beach volleyball world",
        help='Search query (default: "beach volleyball world")',
    )
    parser.add_argument(
        "--results",
        type=int,
        default=20,
        help="Max search results from YouTube (default: 20)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=120,
        metavar="SEC",
        help="Skip videos shorter than this (seconds, default: 120 = 2 min)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output links file (default: data/youtube_search_links.txt)",
    )
    args = parser.parse_args()

    out_path = args.output or Path("data/youtube_search_links.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Searching YouTube: {args.query!r} (up to {args.results} results, min {args.min_duration}s)...")
    results = search_youtube(args.query, max_results=args.results, min_duration_sec=args.min_duration)

    if not results:
        print("No videos found matching criteria.", file=sys.stderr)
        sys.exit(1)

    lines: list[str] = [f"# YouTube search: {args.query!r}  ({len(results)} results)"]
    for r in results:
        dur_m = r["duration"] / 60
        lines.append(f"# {r['title'][:80]}  [{dur_m:.0f}m]  ({r['channel'][:40]})")
        lines.append(r["url"])
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nFound {len(results)} videos. Links written to: {out_path}")
    print(f"\nNext step — pull 3 frames per video (skipping first/last 10%):")
    print(f"  python pull_youtube_frames.py --links {out_path} --count 3 --no-cookies")


if __name__ == "__main__":
    main()
