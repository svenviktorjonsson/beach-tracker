"""Export `data/annotations/*.json` to CSV tables for training (HANDOVER_NN canonical format)."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from PIL import Image

from .csv_schema import (
    BALLS_COLUMNS,
    COURT_POLYLINE_COLUMNS,
    EXCLUSION_COLUMNS,
    MANIFEST_COLUMNS,
    NET_POLYLINE_COLUMNS,
    PEOPLE_COLUMNS,
)
from .dataset import _data_root, _pick_annotation
from .geometry import ellipse_axis_aligned_box


def _annotation_source(path: Path) -> str:
    n = path.name
    if "_merged" in n:
        return "merged"
    if "_r1" in n:
        return "r1"
    if "_r2" in n:
        return "r2"
    return "legacy"


def _image_dims(ann: dict, img_path: Path) -> tuple[int, int]:
    w = ann.get("image_width")
    h = ann.get("image_height")
    if w is not None and h is not None:
        return int(float(w)), int(float(h))
    with Image.open(img_path) as im:
        return im.size


def _rel_image_path(filename: str) -> str:
    return f"images/{Path(filename).name}"


def export_json_to_csv(data_root: Path, out_dir: Path) -> None:
    data_root = data_root.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    images_dir = data_root / "images"
    ann_dir = data_root / "annotations"

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    stems: list[str] = []
    if images_dir.is_dir():
        for p in sorted(images_dir.iterdir()):
            if p.suffix.lower() in exts:
                if _pick_annotation(ann_dir, p.stem) is not None:
                    stems.append(p.stem)

    paths = {
        "manifest": out_dir / "images_manifest.csv",
        "balls": out_dir / "balls.csv",
        "people": out_dir / "people.csv",
        "court": out_dir / "court_polylines.csv",
        "net": out_dir / "net_polylines.csv",
        "exclusion": out_dir / "exclusion_zones.csv",
    }

    with (
        paths["manifest"].open("w", newline="", encoding="utf-8") as f_man,
        paths["balls"].open("w", newline="", encoding="utf-8") as f_b,
        paths["people"].open("w", newline="", encoding="utf-8") as f_p,
        paths["court"].open("w", newline="", encoding="utf-8") as f_c,
        paths["net"].open("w", newline="", encoding="utf-8") as f_n,
        paths["exclusion"].open("w", newline="", encoding="utf-8") as f_e,
    ):
        w_man = csv.DictWriter(f_man, fieldnames=list(MANIFEST_COLUMNS))
        w_b = csv.DictWriter(f_b, fieldnames=list(BALLS_COLUMNS))
        w_p = csv.DictWriter(f_p, fieldnames=list(PEOPLE_COLUMNS))
        w_c = csv.DictWriter(f_c, fieldnames=list(COURT_POLYLINE_COLUMNS))
        w_n = csv.DictWriter(f_n, fieldnames=list(NET_POLYLINE_COLUMNS))
        w_e = csv.DictWriter(f_e, fieldnames=list(EXCLUSION_COLUMNS))
        for w in (w_man, w_b, w_p, w_c, w_n, w_e):
            w.writeheader()

        for stem in stems:
            ann_path = _pick_annotation(ann_dir, stem)
            assert ann_path is not None
            ann = json.loads(ann_path.read_text(encoding="utf-8"))

            img_name = ann.get("image") or f"{stem}.png"
            img_path = images_dir / Path(img_name).name
            if not img_path.is_file():
                for ext in (".png", ".jpg", ".jpeg"):
                    alt = images_dir / f"{stem}{ext}"
                    if alt.is_file():
                        img_path = alt
                        break

            if not img_path.is_file():
                continue

            iw, ih = _image_dims(ann, img_path)
            rel = _rel_image_path(img_path.name)
            src = _annotation_source(ann_path)
            w_man.writerow(
                {
                    "stem": stem,
                    "image_filename": img_path.name,
                    "image_width": str(iw),
                    "image_height": str(ih),
                    "annotation_source": src,
                }
            )

            for b in ann.get("balls") or []:
                if "w" in b and "x" in b:
                    x_, y_, w_, h_ = float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])
                else:
                    cx, cy = float(b["cx"]), float(b["cy"])
                    if "rx" not in b and "r" in b:
                        r = float(b["r"])
                        rx = ry = r
                    else:
                        rx = float(b.get("rx", b.get("r", 1.0)))
                        ry = float(b.get("ry", b.get("r", 1.0)))
                    ang = float(b.get("angle", 0.0))
                    x1, y1, x2, y2 = ellipse_axis_aligned_box(cx, cy, rx, ry, ang)
                    x_, y_, w_, h_ = x1, y1, x2 - x1, y2 - y1
                w_b.writerow(
                    {
                        "stem": stem,
                        "image_path": rel,
                        "x": str(x_),
                        "y": str(y_),
                        "w": str(w_),
                        "h": str(h_),
                        "scope": b.get("scope", "in_play"),
                    }
                )

            for p in ann.get("people") or []:
                w_p.writerow(
                    {
                        "stem": stem,
                        "image_path": rel,
                        "x": str(p["x"]),
                        "y": str(p["y"]),
                        "w": str(p["w"]),
                        "h": str(p["h"]),
                        "role": p.get("role", "player"),
                    }
                )

            for pi, pl in enumerate(ann.get("court_polylines") or []):
                pts = pl.get("points") or []
                scope = pl.get("scope", "main")
                for pj, pt in enumerate(pts):
                    w_c.writerow(
                        {
                            "stem": stem,
                            "image_path": rel,
                            "scope": scope,
                            "polyline_id": str(pi),
                            "point_index": str(pj),
                            "x": str(pt[0]),
                            "y": str(pt[1]),
                        }
                    )

            for pi, pl in enumerate(ann.get("net_polylines") or []):
                pts = pl.get("points") or []
                scope = pl.get("scope", "main")
                for pj, pt in enumerate(pts):
                    w_n.writerow(
                        {
                            "stem": stem,
                            "image_path": rel,
                            "scope": scope,
                            "polyline_id": str(pi),
                            "point_index": str(pj),
                            "x": str(pt[0]),
                            "y": str(pt[1]),
                        }
                    )

            for zi, zone in enumerate(ann.get("exclusion_zones") or []):
                pts = zone.get("points") or []
                for pj, pt in enumerate(pts):
                    w_e.writerow(
                        {
                            "stem": stem,
                            "image_path": rel,
                            "zone_id": str(zi),
                            "point_index": str(pj),
                            "x": str(pt[0]),
                            "y": str(pt[1]),
                        }
                    )


def main() -> None:
    p = argparse.ArgumentParser(description="Export annotation JSON to training CSV tables")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Root with images/ and annotations/ (default: BEACH_DATA_DIR or beach-algo/data)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for CSV files (default: <data-dir>/training_csv)",
    )
    args = p.parse_args()

    data_root = args.data_dir if args.data_dir is not None else _data_root()
    out_dir = args.out_dir if args.out_dir is not None else data_root / "training_csv"
    export_json_to_csv(data_root, out_dir)
    print(f"Wrote CSV tables to {out_dir}")


if __name__ == "__main__":
    main()
