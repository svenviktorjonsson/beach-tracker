"""Train detection + segmentation on beach labeler JSON (see HANDOVER_NN.md)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import BeachAnnotationDataset, BeachCSVDataset
from .human_baseline import compute_human_baseline
from .metrics_server import TrainingMetrics, start_metrics_server
from .model import BeachVolleyballMultiTask
from .schema import DETECTION_CLASS_NAMES, SEGMENTATION_CLASS_NAMES


def _default_data_root() -> Path:
    return Path(os.environ.get("BEACH_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))


def collate_batch(batch: list[dict]) -> dict:
    return {
        "stem": [b["stem"] for b in batch],
        "images": [b["image"] for b in batch],
        "det_targets": [{"boxes": b["boxes"], "labels": b["labels"]} for b in batch],
        "seg_masks": torch.stack([b["seg_mask"] for b in batch], dim=0),
    }


def train_epoch(
    model: BeachVolleyballMultiTask,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    seg_loss_weight: float,
    metrics: TrainingMetrics | None = None,
) -> tuple[float, float]:
    model.train()
    total_det = 0.0
    total_seg = 0.0
    n = 0

    for batch in loader:
        images = [im.to(device) for im in batch["images"]]
        targets = [
            {k: v.to(device) for k, v in t.items()} for t in batch["det_targets"]
        ]
        seg_masks = batch["seg_masks"].to(device)

        optimizer.zero_grad(set_to_none=True)

        det_losses = model.detector(images, targets)
        loss_det = sum(det_losses.values())

        stacked = torch.stack(images, dim=0)
        logits = model.segmenter(stacked)["out"]
        logits = F.interpolate(
            logits,
            size=seg_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        loss_seg = F.cross_entropy(logits, seg_masks)

        loss = loss_det + seg_loss_weight * loss_seg
        loss.backward()
        optimizer.step()

        bd = float(loss_det.detach())
        bs = float(loss_seg.detach())
        total_det += bd
        total_seg += bs
        n += 1
        if metrics is not None:
            metrics.record_batch(bd, bs)

    if n == 0:
        return 0.0, 0.0
    return total_det / n, total_seg / n


def main() -> None:
    p = argparse.ArgumentParser(description="Train beach volleyball multi-task model")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Root with images/ and annotations/ (default: BEACH_DATA_DIR or beach-algo/data)",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--seg-loss-weight", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Random init (no COCO/VOC pretrained weights; faster for smoke tests)",
    )
    p.add_argument("--out", type=Path, default=Path("checkpoints/beach_multitask.pt"))
    p.add_argument(
        "--train-source",
        choices=("json", "csv"),
        default="json",
        help="json = read annotations/*.json directly. csv = read tables from --csv-dir "
        "(canonical tabular export; run python -m training.export_json_to_csv first).",
    )
    p.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help="Directory with images_manifest.csv, balls.csv, … (default: <data-dir>/training_csv)",
    )
    p.add_argument(
        "--metrics-host",
        type=str,
        default="127.0.0.1",
        help="Bind address for the loss dashboard (default: localhost only)",
    )
    p.add_argument(
        "--metrics-port",
        type=int,
        default=8765,
        help="HTTP port for live loss charts in the browser; set 0 to disable (default: 8765)",
    )
    args = p.parse_args()

    data_root = args.data_dir if args.data_dir is not None else _default_data_root()
    csv_dir = args.csv_dir if args.csv_dir is not None else data_root / "training_csv"

    if args.train_source == "json":
        ds = BeachAnnotationDataset(data_root=data_root, image_size=args.image_size)
        if len(ds) == 0:
            raise SystemExit(
                f"No labeled images found under {data_root / 'images'} with matching JSON in "
                f"{data_root / 'annotations'}. Add merged or r1 annotations first."
            )
    else:
        ds = BeachCSVDataset(
            csv_dir=csv_dir, data_root=data_root, image_size=args.image_size
        )
        if len(ds) == 0:
            raise SystemExit(
                f"No training rows found for CSV source. Export tables first:\n"
                f"  python -m training.export_json_to_csv --data-dir {data_root}\n"
                f"Expected files under {csv_dir} (at least images_manifest.csv)."
            )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0,
    )

    device = torch.device(args.device)
    num_det = 1 + len(DETECTION_CLASS_NAMES)
    num_seg = 1 + len(SEGMENTATION_CLASS_NAMES)
    model = BeachVolleyballMultiTask(
        num_detection_classes=num_det,
        num_segmentation_classes=num_seg,
        pretrained_detection=not args.no_pretrained,
        pretrained_segmentation=not args.no_pretrained,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    metrics: TrainingMetrics | None = None
    if args.metrics_port > 0:
        hb = compute_human_baseline(data_root)
        metrics = TrainingMetrics(human_baseline=hb)
        start_metrics_server(metrics, args.metrics_host, args.metrics_port)
        print(
            f"Loss dashboard: http://{args.metrics_host}:{args.metrics_port}/ "
            f"(JSON: /metrics.json)"
        )

    for epoch in range(1, args.epochs + 1):
        loss_det, loss_seg = train_epoch(
            model,
            loader,
            optimizer,
            device,
            args.seg_loss_weight,
            metrics=metrics,
        )
        if metrics is not None:
            metrics.record_epoch(epoch, loss_det, loss_seg)
        print(f"epoch {epoch}/{args.epochs}  loss_det={loss_det:.4f}  loss_seg={loss_seg:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "data_root": str(data_root),
                "train_source": args.train_source,
                "csv_dir": str(csv_dir) if args.train_source == "csv" else None,
                "image_size": args.image_size,
                "num_detection_classes": num_det,
                "num_segmentation_classes": num_seg,
                "detection_class_names": list(DETECTION_CLASS_NAMES),
                "segmentation_class_names": list(SEGMENTATION_CLASS_NAMES),
                "label_schema_version": 1,
            },
        },
        args.out,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
