"""Background training manager with pause/resume/reset, SSE loss streaming."""

from __future__ import annotations

import copy
import os
import threading
import time
import queue
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .schema import DETECTION_CLASS_NAMES

# PyTorch / OpenCV / torchvision are imported lazily inside training and inference paths so
# the HTTP server can construct TrainingManager and serve /train + SSE without a 30–90s import stall.


class TrainState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"


@dataclass
class LossRecord:
    step: int
    epoch: int
    batch_in_epoch: int
    total_batches: int
    images_processed: int
    dataset_size: int
    loss_court: float
    loss_net: float
    loss_people: float
    loss_ball: float
    ts: float = field(default_factory=time.time)


class TrainingManager:
    def __init__(self, data_root: Path, checkpoint_dir: Path | None = None):
        self.data_root = data_root
        self.checkpoint_dir = checkpoint_dir or data_root / "checkpoints"
        self.state = TrainState.IDLE
        self._thread: threading.Thread | None = None
        self._pause_event = threading.Event()
        self._stop_flag = False
        self._lock = threading.Lock()

        self.loss_history: list[LossRecord] = []
        self.epoch_history: list[dict] = []
        self._subscribers: list[queue.Queue] = []

        self.current_epoch = 0
        self.current_step = 0
        self.total_epochs = 0
        self.config: dict[str, Any] = {}

        self._model: Any = None
        self._optimizer: Any = None
        self._device: Any = None
        self._initial_state_dict: dict | None = None

    # ── status / pub-sub ──────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "state": self.state.value,
            "epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "step": self.current_step,
            "config": self.config,
            "loss_count": len(self.loss_history),
        }

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=500)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def _broadcast(self, event: dict) -> None:
        with self._lock:
            dead = []
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._subscribers.remove(q)

    # ── control ───────────────────────────────────────────────────────

    def start(
        self,
        epochs: int = 50,
        batch_size: int = 1,
        lr: float = 1e-4,
        image_size: int = 384,
        targets: list[str] | None = None,
        resume: bool = False,
        augment: bool = True,
        **_kw: Any,
    ) -> dict:
        with self._lock:
            if self.state == TrainState.PAUSED and resume:
                self.state = TrainState.RUNNING
                self._pause_event.set()
                self._broadcast({"type": "state", "state": "running"})
                return {"ok": True, "action": "resumed"}

            if self.state == TrainState.RUNNING:
                return {"ok": False, "error": "already running"}

        self.config = {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "image_size": image_size,
            "targets": targets or ["court", "net", "ball", "people"],
            "augment": augment,
        }

        if not resume:
            self.loss_history.clear()
            self.epoch_history.clear()
            self.current_epoch = 0
            self.current_step = 0
            if self._model is not None and self._initial_state_dict is not None:
                import torch

                self._model.load_state_dict(self._initial_state_dict)
                self._optimizer = torch.optim.AdamW(
                    self._model.parameters(), lr=lr
                )

        self.total_epochs = epochs
        self._stop_flag = False
        self._pause_event.set()
        self.state = TrainState.RUNNING

        self._broadcast(
            {
                "type": "info",
                "message": "Training started — preparing the model and aiming for the first real batch loss as quickly as possible.",
            }
        )
        self._thread = threading.Thread(target=self._train_loop, daemon=True)
        self._thread.start()
        self._broadcast({"type": "state", "state": "running"})
        return {"ok": True, "action": "started"}

    def pause(self) -> dict:
        with self._lock:
            if self.state != TrainState.RUNNING:
                return {"ok": False, "error": "not running"}
            self.state = TrainState.PAUSED
            self._pause_event.clear()
        self._broadcast({"type": "state", "state": "paused"})
        return {"ok": True}

    def reset(self) -> dict:
        with self._lock:
            if self.state == TrainState.RUNNING:
                return {"ok": False, "error": "pause first"}
            self._stop_flag = True
            self._pause_event.set()
            self.state = TrainState.IDLE
            self.loss_history.clear()
            self.epoch_history.clear()
            self.current_epoch = 0
            self.current_step = 0
            self.total_epochs = 0
            self.config = {}
            if self._model is not None and self._initial_state_dict is not None:
                self._model.load_state_dict(self._initial_state_dict)
                self._optimizer = None
        self._broadcast({"type": "state", "state": "idle"})
        self._broadcast({"type": "reset"})
        return {"ok": True}

    # ── model / checkpoint ────────────────────────────────────────────

    def _build_model(self) -> None:
        import torch
        from .model import BeachVolleyballMultiTask

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device_str)
        num_det = 1 + len(DETECTION_CLASS_NAMES)

        ckpt_path = self.checkpoint_dir / "beach_multitask.pt"
        if self._model is None:
            self._broadcast({"type": "info", "message": f"Loading model ({device_str})…"})
            self._model = BeachVolleyballMultiTask(
                num_detection_classes=num_det,
                pretrained_detection=True,
                pretrained_geometry=True,
            )
            ckpt: dict | None = None
            if ckpt_path.exists():
                self._broadcast({"type": "info", "message": "Loading checkpoint…"})
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    self._model.load_state_dict(ckpt["model_state_dict"], strict=False)
                    print(f"[train] Resumed model from {ckpt_path}")
                except Exception as e:
                    bad = ckpt_path.with_suffix(".pt.corrupt")
                    try:
                        ckpt_path.rename(bad)
                    except OSError:
                        ckpt_path.unlink(missing_ok=True)
                    msg = (
                        f"Checkpoint was corrupted ({e!s}); removed it and using fresh pretrained weights."
                    )
                    print(f"[train] {msg}")
                    self._broadcast({"type": "info", "message": msg})

            self._broadcast({"type": "info", "message": f"Moving model to {device_str}…"})
            self._model.to(self._device)

            self._initial_state_dict = copy.deepcopy(self._model.state_dict())

            self._optimizer = torch.optim.AdamW(
                self._model.parameters(), lr=self.config.get("lr", 1e-4)
            )

            if ckpt is not None and "optimizer_state_dict" in ckpt:
                try:
                    self._optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception:
                    pass
            self._broadcast({"type": "info", "message": "Model ready."})
        else:
            if self._optimizer is None:
                self._optimizer = torch.optim.AdamW(
                    self._model.parameters(), lr=self.config.get("lr", 1e-4)
                )

    def _save_checkpoint(self) -> None:
        import torch
        from .model import MAX_COURT_PTS, MAX_NET_PTS

        if self._model is None:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / "beach_multitask.pt"
        tmp = path.with_suffix(".pt.tmp")
        payload = {
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict() if self._optimizer else {},
            "config": {
                "image_size": self.config.get("image_size", 640),
                "num_detection_classes": 1 + len(DETECTION_CLASS_NAMES),
                "max_court_pts": MAX_COURT_PTS,
                "max_net_pts": MAX_NET_PTS,
                "detection_class_names": list(DETECTION_CLASS_NAMES),
                "label_schema_version": 2,
            },
            "epoch": self.current_epoch,
        }
        torch.save(payload, tmp)
        os.replace(tmp, path)

    # ── collate ───────────────────────────────────────────────────────

    def _collate(self, batch):
        import torch

        return {
            "stem": [b["stem"] for b in batch],
            "images": [b["image"] for b in batch],
            "det_targets": [{"boxes": b["boxes"], "labels": b["labels"]} for b in batch],
            "court_pts": torch.stack([b["court_pts"] for b in batch]),
            "court_mask": torch.stack([b["court_mask"] for b in batch]),
            "net_pts": torch.stack([b["net_pts"] for b in batch]),
            "net_mask": torch.stack([b["net_mask"] for b in batch]),
        }

    # ── loss helpers ──────────────────────────────────────────────────

    def _make_loss_event(self, rec: LossRecord, event_type: str = "loss") -> dict:
        return {
            "type": event_type,
            "step": rec.step,
            "epoch": rec.epoch,
            "batch_in_epoch": rec.batch_in_epoch,
            "total_batches": rec.total_batches,
            "images_processed": rec.images_processed,
            "dataset_size": rec.dataset_size,
            "loss_court": rec.loss_court,
            "loss_net": rec.loss_net,
            "loss_people": rec.loss_people,
            "loss_ball": rec.loss_ball,
        }

    @staticmethod
    def _geometry_loss(pred, gt_court_pts, gt_court_mask, gt_net_pts, gt_net_mask):
        """
        Polyline-aware geometry loss.

        The human labels are sampled polylines with variable point counts, not stable semantic
        landmarks. So instead of comparing point i to point i directly, resample both predicted
        and GT polylines densely along arc length and compare the resulting curves.
        """
        import torch
        import torch.nn.functional as F

        def _resample_polyline(points: torch.Tensor, samples: int = 48) -> torch.Tensor:
            if points.shape[0] == 1:
                return points.repeat(samples, 1)
            seg = points[1:] - points[:-1]
            seg_len = torch.sqrt(torch.clamp((seg * seg).sum(dim=1), min=1e-12))
            cum = torch.cat(
                [
                    torch.zeros(1, device=points.device, dtype=points.dtype),
                    torch.cumsum(seg_len, dim=0),
                ],
                dim=0,
            )
            total_len = cum[-1]
            if float(total_len.detach()) <= 1e-8:
                return points[:1].repeat(samples, 1)
            t = torch.linspace(
                0.0, 1.0, samples, device=points.device, dtype=points.dtype
            ) * total_len
            idx = torch.bucketize(t, cum[1:], right=False)
            idx = torch.clamp(idx, 0, points.shape[0] - 2)
            seg_start = cum[idx]
            seg_denom = torch.clamp(seg_len[idx], min=1e-8)
            alpha = ((t - seg_start) / seg_denom).unsqueeze(1)
            return points[idx] + alpha * (points[idx + 1] - points[idx])

        def _curve_loss(
            pred_pts: torch.Tensor,
            pred_vis: torch.Tensor,
            gt_pts: torch.Tensor,
            gt_mask: torch.Tensor,
        ) -> torch.Tensor:
            vis_target = gt_mask.float()
            vis_loss = F.binary_cross_entropy_with_logits(pred_vis, vis_target)
            total_gt = int(gt_mask.sum().item())
            if total_gt <= 0:
                return vis_loss
            gt_curve = _resample_polyline(gt_pts[gt_mask], samples=64)
            # Use the first N predicted points to define the predicted curve; visibility still
            # learns which slots are active, while the geometry loss becomes independent of the
            # original click count along the line.
            pred_curve = _resample_polyline(pred_pts[:total_gt], samples=64)
            d = torch.cdist(pred_curve, gt_curve, p=2)
            # Symmetric point-to-curve distance approximates line-to-line distance while staying differentiable.
            curve_loss = d.min(dim=1).values.mean() + d.min(dim=0).values.mean()
            return curve_loss + vis_loss

        batch_n = gt_court_pts.shape[0]
        court_losses = []
        net_losses = []
        for b in range(batch_n):
            court_losses.append(
                _curve_loss(
                    pred["court_pts"][b],
                    pred["court_vis"][b],
                    gt_court_pts[b],
                    gt_court_mask[b],
                )
            )
            net_losses.append(
                _curve_loss(
                    pred["net_pts"][b],
                    pred["net_vis"][b],
                    gt_net_pts[b],
                    gt_net_mask[b],
                )
            )

        court_loss = torch.stack(court_losses).mean() if court_losses else torch.tensor(0.0, device=gt_court_pts.device)
        net_loss = torch.stack(net_losses).mean() if net_losses else torch.tensor(0.0, device=gt_net_pts.device)
        total = court_loss + net_loss
        return total, float(court_loss.detach()), float(net_loss.detach())

    @staticmethod
    def _split_det_losses(det_losses, targets):
        """Allocate Faster R-CNN loss totals for dashboard lines (ball vs people).

        Torchvision returns one combined dict (rpn + ROI classifier + box reg, etc.). We do **not**
        have separate ball vs people backward paths — this only splits the scalar **sum** for
        plotting, weighted by how many **ground-truth boxes** of each kind appear in the batch.

        - If a batch has **no ball GT boxes**, the ball line gets **0** (even though RPN /
          classification still trains background vs foreground).
        - If there are **no GT boxes at all**, RPN/objectness loss is still non-zero; we split
          that total 50/50 for display so the chart does not misleadingly show zeros everywhere.
        """
        total = float(sum(det_losses.values()).detach())
        n_ball = 0
        n_people = 0
        for t in targets:
            labels = t["labels"]
            n_ball += int(((labels == 1) | (labels == 2)).sum())
            n_people += int(((labels >= 3) & (labels <= 5)).sum())
        n_total = n_ball + n_people
        if n_total == 0:
            half = total * 0.5
            return total, half, half
        return total, total * n_people / n_total, total * n_ball / n_total

    # ── training loop ─────────────────────────────────────────────────

    def _should_train_det(self) -> bool:
        tgt = self.config.get("targets", [])
        return "ball" in tgt or "people" in tgt

    def _should_train_geo(self) -> bool:
        tgt = self.config.get("targets", [])
        return "court" in tgt or "net" in tgt

    def _eval_baseline(self, loader, train_det, train_geo, total_batches, dataset_size):
        """Forward pass on first batch without gradients to get baseline loss."""
        import torch

        try:
            self._model.train()
            batch = next(iter(loader))
            images = [im.to(self._device) for im in batch["images"]]
            targets = [{k: v.to(self._device) for k, v in t.items()} for t in batch["det_targets"]]

            loss_court = 0.0
            loss_net = 0.0
            loss_people = 0.0
            loss_ball = 0.0

            if train_det:
                det_losses = self._model.detector(images, targets)
                _, loss_people, loss_ball = self._split_det_losses(det_losses, targets)

            if train_geo:
                stacked = torch.stack(images, dim=0)
                geo_pred = self._model.geometry(stacked)
                court_pts = batch["court_pts"].to(self._device)
                court_mask = batch["court_mask"].to(self._device)
                net_pts = batch["net_pts"].to(self._device)
                net_mask = batch["net_mask"].to(self._device)
                _, loss_court, loss_net = self._geometry_loss(
                    geo_pred, court_pts, court_mask, net_pts, net_mask
                )

            rec = LossRecord(
                step=0, epoch=0, batch_in_epoch=0, total_batches=total_batches,
                images_processed=0, dataset_size=dataset_size,
                loss_court=loss_court, loss_net=loss_net,
                loss_people=loss_people, loss_ball=loss_ball,
            )
            self.loss_history.append(rec)
            self._broadcast(self._make_loss_event(rec, "baseline"))
        except Exception as e:
            print(f"[train] baseline eval failed: {e}")

    def _train_loop(self) -> None:
        # Broadcast before heavy imports — cold `import torch` / dataset can take minutes.
        self._broadcast({"type": "info", "message": "Importing PyTorch and data pipeline…"})
        import torch
        from torch.utils.data import DataLoader

        from .augment import augment_collated_batch
        from .dataset import BeachAnnotationDataset

        try:
            self._broadcast({"type": "info", "message": "Initializing model (downloading weights on first run)…"})
            self._build_model()
            self._broadcast({"type": "info", "message": "Scanning labeled images…"})
            img_sz = self.config.get("image_size", 640)
            use_aug = self.config.get("augment", True)
            ds = BeachAnnotationDataset(
                data_root=self.data_root,
                image_size=img_sz,
            )
            if len(ds) == 0:
                self._broadcast({"type": "error", "message": "No labeled images found"})
                self.state = TrainState.IDLE
                return

            bs = self.config.get("batch_size", 2)
            nw = int(os.environ.get("BEACH_TRAIN_NUM_WORKERS", "0"))
            pin = torch.cuda.is_available()
            dl_common: dict[str, Any] = {
                "batch_size": bs,
                "collate_fn": self._collate,
                "pin_memory": pin,
            }
            if nw > 0:
                dl_common["num_workers"] = nw
                dl_common["persistent_workers"] = True
                dl_common["prefetch_factor"] = min(4, max(2, nw))
            else:
                dl_common["num_workers"] = 0

            loader_baseline = DataLoader(ds, shuffle=False, **dl_common)
            loader = DataLoader(ds, shuffle=True, **dl_common)
            total_batches = (len(ds) + bs - 1) // bs

            train_det = self._should_train_det()
            train_geo = self._should_train_geo()
            start_epoch = self.current_epoch + 1
            total = self.config.get("epochs", 50)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self._optimizer, T_max=total * total_batches, eta_min=1e-6,
            )

            aug_note = " (aug each step: random ±45° roll + photometric per image)" if use_aug else " (no aug)"
            self._broadcast({
                "type": "info",
                "message": f"Training on {len(ds)} images ({total_batches} batches/epoch), det={train_det}, geo={train_geo}{aug_note}",
                "dataset_size": len(ds),
                "total_batches": total_batches,
                "targets": self.config.get("targets", []),
            })

            run_bl = os.environ.get("BEACH_TRAIN_RUN_BASELINE", "").lower() in (
                "1", "true", "yes",
            )
            if self.current_step == 0 and run_bl:
                self._broadcast({"type": "info", "message": "Evaluating baseline loss…"})
                self._eval_baseline(loader_baseline, train_det, train_geo, total_batches, len(ds))
            elif self.current_step == 0:
                self._broadcast({
                    "type": "info",
                    "message": "Skipping extra baseline pass — first point will be the first training batch.",
                })

            for epoch in range(start_epoch, total + 1):
                if self._stop_flag:
                    break
                self.current_epoch = epoch
                self._model.train()
                n_batches = 0
                images_seen = 0

                for batch in loader:
                    self._pause_event.wait()
                    if self._stop_flag:
                        break

                    if use_aug:
                        batch = augment_collated_batch(batch, img_sz)

                    images = [im.to(self._device) for im in batch["images"]]
                    targets = [{k: v.to(self._device) for k, v in t.items()} for t in batch["det_targets"]]
                    batch_len = len(images)

                    self._optimizer.zero_grad(set_to_none=True)

                    loss_det_total = torch.tensor(0.0, device=self._device)
                    b_people = 0.0
                    b_ball = 0.0

                    if train_det:
                        det_losses = self._model.detector(images, targets)
                        loss_det_total = sum(det_losses.values())
                        _, b_people, b_ball = self._split_det_losses(det_losses, targets)

                    loss_geo_total = torch.tensor(0.0, device=self._device)
                    b_court = 0.0
                    b_net = 0.0

                    if train_geo:
                        stacked = torch.stack(images, dim=0)
                        geo_pred = self._model.geometry(stacked)
                        court_pts = batch["court_pts"].to(self._device)
                        court_mask = batch["court_mask"].to(self._device)
                        net_pts = batch["net_pts"].to(self._device)
                        net_mask = batch["net_mask"].to(self._device)
                        loss_geo_total, b_court, b_net = self._geometry_loss(
                            geo_pred, court_pts, court_mask, net_pts, net_mask
                        )

                    loss = loss_det_total + loss_geo_total
                    if loss.requires_grad:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=5.0)
                        self._optimizer.step()
                        scheduler.step()

                    self.current_step += 1
                    n_batches += 1
                    images_seen += batch_len

                    rec = LossRecord(
                        step=self.current_step,
                        epoch=epoch,
                        batch_in_epoch=n_batches,
                        total_batches=total_batches,
                        images_processed=images_seen,
                        dataset_size=len(ds),
                        loss_court=b_court,
                        loss_net=b_net,
                        loss_people=b_people,
                        loss_ball=b_ball,
                    )
                    self.loss_history.append(rec)
                    self._broadcast(self._make_loss_event(rec))

                if self._stop_flag:
                    break

                self.epoch_history.append({"epoch": epoch})
                self._broadcast({
                    "type": "epoch",
                    "epoch": epoch,
                    "total_epochs": total,
                })
                self._save_checkpoint()

            with self._lock:
                if not self._stop_flag:
                    self.state = TrainState.IDLE
                    self._broadcast({"type": "state", "state": "idle"})
                    self._broadcast({"type": "done", "epochs": self.current_epoch})

        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            print(f"[train] error: {tb}")
            self._broadcast({"type": "error", "message": str(exc)})
            with self._lock:
                self.state = TrainState.IDLE

    # ── inference ─────────────────────────────────────────────────────

    def predict(self, image_path: str | Path, image_size: int = 640) -> dict:
        """Run inference on a single image, returning detections + geometry keypoints."""
        import torch
        from PIL import Image
        import torchvision.transforms.functional as TF

        if self._model is None:
            self._build_model()

        self._model.eval()
        device = self._device

        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size
        img_resized = img.resize((image_size, image_size), Image.BILINEAR)
        tensor = TF.to_tensor(img_resized).to(device)

        with torch.no_grad():
            preds = self._model.detector([tensor])
            geo = self._model.geometry(tensor.unsqueeze(0))

        # ── detections ──
        pred = preds[0]
        boxes = pred["boxes"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()

        sx = orig_w / image_size
        sy = orig_h / image_size

        detections = []
        for i in range(len(boxes)):
            if scores[i] < 0.3:
                continue
            b = boxes[i]
            lbl = int(labels[i])
            cls_name = DETECTION_CLASS_NAMES[lbl - 1] if 1 <= lbl <= len(DETECTION_CLASS_NAMES) else "unknown"
            detections.append({
                "box": [float(b[0] * sx), float(b[1] * sy), float(b[2] * sx), float(b[3] * sy)],
                "label": cls_name,
                "score": float(scores[i]),
            })

        # ── geometry keypoints ──
        court_pts = geo["court_pts"][0].cpu().numpy()  # (max_court, 2) in [0,1]
        court_vis = torch.sigmoid(geo["court_vis"][0]).cpu().numpy()
        net_pts = geo["net_pts"][0].cpu().numpy()
        net_vis = torch.sigmoid(geo["net_vis"][0]).cpu().numpy()

        court_points = []
        for i in range(len(court_pts)):
            if court_vis[i] > 0.5:
                court_points.append([
                    float(court_pts[i][0] * orig_w),
                    float(court_pts[i][1] * orig_h),
                ])

        net_points = []
        for i in range(len(net_pts)):
            if net_vis[i] > 0.5:
                net_points.append([
                    float(net_pts[i][0] * orig_w),
                    float(net_pts[i][1] * orig_h),
                ])

        return {
            "image_width": orig_w,
            "image_height": orig_h,
            "detections": detections,
            "court_points": court_points,
            "net_points": net_points,
        }
