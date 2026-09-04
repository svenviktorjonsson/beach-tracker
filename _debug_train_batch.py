"""One-shot repro for trainer geometry loss; delete after debugging."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.augment import augment_collated_batch
from training.dataset import BeachCSVDataset
from training.export_sqlite_to_csv import export_sqlite_to_csv
from training.manager import TrainingManager


def main() -> None:
    root = Path(__file__).resolve().parent / "data"
    csv_dir = root / "training_v2_csv"
    export_sqlite_to_csv(root, csv_dir, pass_n=1, complete_only=True)
    ds = BeachCSVDataset(csv_dir=csv_dir, data_root=root, image_size=320)
    if len(ds) == 0:
        print("no data")
        return

    mgr = TrainingManager(data_root=root)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=mgr._collate, num_workers=0)
    batch = next(iter(loader))
    batch = augment_collated_batch(batch, 320)
    print("ball_radius shape", batch["ball_radius"].shape)
    print(
        "boxes shapes",
        [batch["det_targets"][i]["boxes"].shape for i in range(len(batch["images"]))],
    )

    mgr._build_model(enable_detector=False, model_profile="fast")
    mgr._optimizer = torch.optim.AdamW(mgr._model.parameters(), lr=1e-4)

    images = [im.to(mgr._device) for im in batch["images"]]
    stacked = torch.stack(images, dim=0)
    geo_pred = mgr._model.geometry(stacked)
    court_pts = batch["court_pts"].to(mgr._device)
    court_mask = batch["court_mask"].to(mgr._device)
    net_pts = batch["net_pts"].to(mgr._device)
    net_mask = batch["net_mask"].to(mgr._device)
    ball_center = batch["ball_center"].to(mgr._device)
    ball_radius = batch["ball_radius"].to(mgr._device).reshape(-1)
    ball_visible = batch["ball_visible"].to(mgr._device)
    loss, *rest = mgr._geometry_loss(
        geo_pred,
        court_pts,
        court_mask,
        net_pts,
        net_mask,
        ball_center,
        ball_visible,
        ball_radius,
    )
    loss.backward()
    print("ok", float(loss), rest)


if __name__ == "__main__":
    main()
