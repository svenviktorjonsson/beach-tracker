"""Random augmentations for training: fixed square output; labels follow the pixels."""

from __future__ import annotations

import math
import random

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF


def _map_points_rotate_crop_resize(
    px: float,
    py: float,
    M: np.ndarray,
    left: int,
    top: int,
    side: int,
    out_s: int,
) -> tuple[float, float]:
    """Map one pixel from input square through OpenCV-style rot+expand, center crop, resize."""
    xp = float(M[0, 0] * px + M[0, 1] * py + M[0, 2])
    yp = float(M[1, 0] * px + M[1, 1] * py + M[1, 2])
    xc = xp - left
    yc = yp - top
    xo = xc * (out_s / side)
    yo = yc * (out_s / side)
    return xo / out_s, yo / out_s


def _rotate_expand_crop_resize_square(
    image_chw: torch.Tensor,
    boxes: torch.Tensor,
    court_pts: torch.Tensor,
    court_mask: torch.Tensor,
    net_pts: torch.Tensor,
    net_mask: torch.Tensor,
    *,
    image_size: int,
    angle_deg: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    In-plane rotation (angle_deg), expanded canvas, center square crop, resize to image_size.
    Labels follow the same transform as the pixels (labels stay in normal broadcast / upright view).
    """
    S = image_size
    dev = image_chw.device
    # [H,W,C] uint8 for cv2
    hwc = (image_chw.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    h, w = hwc.shape[:2]
    assert h == w == S, "expected square input"

    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rad = math.radians(angle_deg)
    nW = int(abs(w * math.cos(rad)) + abs(h * math.sin(rad)))
    nH = int(abs(w * math.sin(rad)) + abs(h * math.cos(rad)))
    M = M.astype(np.float64)
    M[0, 2] += (nW / 2.0) - w / 2.0
    M[1, 2] += (nH / 2.0) - h / 2.0

    warped = cv2.warpAffine(
        hwc, M, (nW, nH), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0),
    )

    side = min(nW, nH)
    left = (nW - side) // 2
    top = (nH - side) // 2
    cropped = warped[top : top + side, left : left + side]
    resized = cv2.resize(cropped, (S, S), interpolation=cv2.INTER_LINEAR)

    image = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    if dev.type != "cpu":
        image = image.to(dev)

    # --- keypoints (normalised [0,1] in input square) ---
    # No in-place ops on tensors that may participate in autograd (clamp_/[:] = break backward).
    def _map_norm_pts(pts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        device, dtype = pts.device, pts.dtype
        rows: list[torch.Tensor] = []
        for i in range(pts.shape[0]):
            if not bool(mask[i].item()):
                rows.append(pts[i].detach())
                continue
            px = float(pts[i, 0].detach().cpu()) * S
            py = float(pts[i, 1].detach().cpu()) * S
            nx, ny = _map_points_rotate_crop_resize(px, py, M, left, top, side, S)
            rows.append(
                torch.tensor([nx, ny], device=device, dtype=dtype)
            )
        out = torch.stack(rows, dim=0)
        return torch.clamp(out, 0.0, 1.0)

    court_pts = _map_norm_pts(court_pts, court_mask)
    net_pts = _map_norm_pts(net_pts, net_mask)

    # --- boxes: rotate corners, crop+scale, AABB ---
    new_boxes: list[list[float]] = []
    if boxes.numel() > 0:
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i].detach().tolist()
            corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            xs: list[float] = []
            ys: list[float] = []
            for px, py in corners:
                nx, ny = _map_points_rotate_crop_resize(px, py, M, left, top, side, S)
                xs.append(nx * S)
                ys.append(ny * S)
            new_boxes.append([min(xs), min(ys), max(xs), max(ys)])
        boxes = torch.tensor(new_boxes, dtype=torch.float32, device=boxes.device)
        boxes = torch.stack(
            [
                torch.clamp(boxes[:, 0], 0.0, float(S - 1)),
                torch.clamp(boxes[:, 1], 0.0, float(S - 1)),
                torch.clamp(boxes[:, 2], 0.0, float(S - 1)),
                torch.clamp(boxes[:, 3], 0.0, float(S - 1)),
            ],
            dim=1,
        )

    return image, boxes, court_pts, court_mask, net_pts, net_mask


def apply_training_augmentations(
    image: torch.Tensor,
    boxes: torch.Tensor,
    court_pts: torch.Tensor,
    court_mask: torch.Tensor,
    net_pts: torch.Tensor,
    net_mask: torch.Tensor,
    *,
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    image: [3, H, W] in [0, 1], square
    boxes: [N, 4] xyxy in pixel coords
    *_pts: [K, 2] normalised [0, 1]

    Geometry: random roll in [-45°, 45°] only (matches typical camera tilt vs upright labels).
    Expand → center crop square → resize. No flips — same “viewing side” as your labeling.
    """
    # Typical hand-held / wide-angle tilt vs the upright frames you labeled
    angle_deg = random.uniform(-45.0, 45.0)
    image, boxes, court_pts, court_mask, net_pts, net_mask = _rotate_expand_crop_resize_square(
        image, boxes, court_pts, court_mask, net_pts, net_mask,
        image_size=image_size,
        angle_deg=angle_deg,
    )

    # --- Photometric (torchvision brightness is a multiplicative factor ≥ 0) ---
    c = random.uniform(0.85, 1.15)
    bright = max(0.1, 1.0 + random.uniform(-0.08, 0.08))
    image = TF.adjust_contrast(image, c)
    image = TF.adjust_brightness(image, bright)
    if random.random() < 0.4:
        s = random.uniform(0.9, 1.1)
        image = TF.adjust_saturation(image, s)

    if random.random() < 0.35:
        k = random.choice([3, 5])
        sigma = random.uniform(0.4, 1.6)
        image = TF.gaussian_blur(image, kernel_size=[k, k], sigma=[sigma, sigma])

    image = image.clamp(0.0, 1.0)

    if boxes.numel() > 0:
        keep = (boxes[:, 2] > boxes[:, 0] + 0.5) & (boxes[:, 3] > boxes[:, 1] + 0.5)
        boxes = boxes[keep]

    return image, boxes, court_pts, court_mask, net_pts, net_mask


def augment_collated_batch(batch: dict, image_size: int) -> dict:
    """
    Random aug per sample, called from the training loop on each batch (not in Dataset.__getitem__).
    Every optimizer step sees a fresh random angle for each image in the batch.
    """
    images = batch["images"]
    det_targets = batch["det_targets"]
    court_pts = batch["court_pts"]
    court_mask = batch["court_mask"]
    net_pts = batch["net_pts"]
    net_mask = batch["net_mask"]
    new_images: list = []
    new_det: list = []
    new_cp: list = []
    new_cm: list = []
    new_np: list = []
    new_nm: list = []
    for i in range(len(images)):
        img, bx, cp, cm, np_, nm = apply_training_augmentations(
            images[i],
            det_targets[i]["boxes"],
            court_pts[i],
            court_mask[i],
            net_pts[i],
            net_mask[i],
            image_size=image_size,
        )
        new_images.append(img)
        new_det.append({"boxes": bx, "labels": det_targets[i]["labels"]})
        new_cp.append(cp)
        new_cm.append(cm)
        new_np.append(np_)
        new_nm.append(nm)
    return {
        "stem": batch["stem"],
        "images": new_images,
        "det_targets": new_det,
        "court_pts": torch.stack(new_cp, dim=0),
        "court_mask": torch.stack(new_cm, dim=0),
        "net_pts": torch.stack(new_np, dim=0),
        "net_mask": torch.stack(new_nm, dim=0),
    }
