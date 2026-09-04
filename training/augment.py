"""Random augmentations for geometry training with aligned point/box transforms."""

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
    """Map one pixel from input square through rot + expanded crop + resize."""
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
    ball_center: torch.Tensor,
    ball_visible: torch.Tensor,
    ball_radius: torch.Tensor,
    *,
    image_size: int,
    angle_deg: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Rotate image by angle, crop centered square, and resize back to image_size."""
    S = image_size
    dev = image_chw.device

    hwc = (image_chw.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    h, w = hwc.shape[:2]
    assert h == w == S, "expected square input"

    center = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rad = math.radians(angle_deg)
    n_w = int(abs(w * math.cos(rad)) + abs(h * math.sin(rad)))
    n_h = int(abs(w * math.sin(rad)) + abs(h * math.cos(rad)))
    m = m.astype(np.float64)
    m[0, 2] += (n_w / 2.0) - (w / 2.0)
    m[1, 2] += (n_h / 2.0) - (h / 2.0)

    warped = cv2.warpAffine(
        hwc, m, (n_w, n_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
    )
    side = min(n_w, n_h)
    left = (n_w - side) // 2
    top = (n_h - side) // 2
    cropped = warped[top : top + side, left : left + side]
    resized = cv2.resize(cropped, (S, S), interpolation=cv2.INTER_LINEAR)

    image = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    if dev.type != "cpu":
        image = image.to(dev)

    def _map_norm_points(pts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out_rows: list[torch.Tensor] = []
        out_device = pts.device
        out_dtype = pts.dtype
        for i in range(pts.shape[0]):
            if not bool(mask[i].item()):
                out_rows.append(pts[i])
                continue
            x = float(pts[i, 0].detach().cpu()) * S
            y = float(pts[i, 1].detach().cpu()) * S
            nx, ny = _map_points_rotate_crop_resize(x, y, m, left, top, side, S)
            out_rows.append(torch.tensor([nx, ny], device=out_device, dtype=out_dtype))
        return torch.clamp(torch.stack(out_rows, dim=0), 0.0, 1.0)

    court_pts = _map_norm_points(court_pts, court_mask)
    net_pts = _map_norm_points(net_pts, net_mask)

    ball_center_n = torch.tensor([0.5, 0.5], device=dev, dtype=image_chw.dtype)
    # Keep shape (1,) so collated batch stays (B,1); 0-dim scalars stack to (B,) and break squeeze(1) in trainer.
    ball_radius_n = torch.zeros(1, device=dev, dtype=image_chw.dtype)
    visible = bool(ball_visible.item())
    visible_in_frame = False
    if visible:
        cx = float(ball_center[0].detach().cpu()) * S
        cy = float(ball_center[1].detach().cpu()) * S
        nx, ny = _map_points_rotate_crop_resize(cx, cy, m, left, top, side, S)
        ball_center_n = torch.tensor([nx, ny], device=dev, dtype=image_chw.dtype)
        visible_in_frame = (0.0 <= nx <= 1.0) and (0.0 <= ny <= 1.0)
        scale = float(S) / float(side)
        ball_radius_n = torch.tensor(
            [min(max(float(ball_radius) * scale, 0.0), 1.0)],
            device=dev,
            dtype=image_chw.dtype,
        )

    ball_visible_out = torch.tensor(
        bool(visible and visible_in_frame),
        device=dev,
        dtype=torch.bool,
    )
    ball_radius_n = torch.clamp(ball_radius_n, 0.0, 1.0)

    # Boxes: rotate corners -> crop -> resize -> AABB.
    boxes_out: list[list[float]] = []
    if boxes.numel() > 0:
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i].detach().tolist()
            corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            xs: list[float] = []
            ys: list[float] = []
            for px, py in corners:
                nx, ny = _map_points_rotate_crop_resize(px, py, m, left, top, side, S)
                xs.append(nx * S)
                ys.append(ny * S)
            boxes_out.append([min(xs), min(ys), max(xs), max(ys)])
        boxes = torch.tensor(boxes_out, dtype=torch.float32, device=boxes.device)
        boxes = torch.stack(
            [
                torch.clamp(boxes[:, 0], 0.0, float(S - 1)),
                torch.clamp(boxes[:, 1], 0.0, float(S - 1)),
                torch.clamp(boxes[:, 2], 0.0, float(S - 1)),
                torch.clamp(boxes[:, 3], 0.0, float(S - 1)),
            ],
            dim=1,
        )

    return (
        image,
        boxes,
        court_pts,
        court_mask,
        net_pts,
        net_mask,
        ball_center_n,
        ball_radius_n,
        ball_visible_out,
    )


def apply_training_augmentations(
    image: torch.Tensor,
    boxes: torch.Tensor,
    court_pts: torch.Tensor,
    court_mask: torch.Tensor,
    net_pts: torch.Tensor,
    net_mask: torch.Tensor,
    ball_center: torch.Tensor,
    ball_visible: torch.Tensor,
    ball_radius: torch.Tensor,
    *,
    image_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Geometric + photometric augmentations for one sample."""
    angle_deg = random.uniform(-45.0, 45.0)
    image, boxes, court_pts, court_mask, net_pts, net_mask, ball_center, ball_radius, ball_visible = (
        _rotate_expand_crop_resize_square(
            image,
            boxes,
            court_pts,
            court_mask,
            net_pts,
            net_mask,
            ball_center,
            ball_visible,
            ball_radius,
            image_size=image_size,
            angle_deg=angle_deg,
        )
    )

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

    return (
        image,
        boxes,
        court_pts,
        court_mask,
        net_pts,
        net_mask,
        ball_center,
        ball_radius,
        ball_visible,
    )


def augment_collated_batch(batch: dict, image_size: int) -> dict:
    """Apply augmentations sample-by-sample and return a fully transformed batch."""
    images = batch["images"]
    det_targets = batch["det_targets"]
    court_pts = batch["court_pts"]
    court_mask = batch["court_mask"]
    net_pts = batch["net_pts"]
    net_mask = batch["net_mask"]
    ball_center = batch["ball_center"]
    ball_radius = batch["ball_radius"]
    ball_size_labeled = batch["ball_size_labeled"]
    ball_visible = batch["ball_visible"]

    new_images: list[torch.Tensor] = []
    new_det: list[dict[str, torch.Tensor]] = []
    new_cp: list[torch.Tensor] = []
    new_cm: list[torch.Tensor] = []
    new_np: list[torch.Tensor] = []
    new_nm: list[torch.Tensor] = []
    new_bc: list[torch.Tensor] = []
    new_br: list[torch.Tensor] = []
    new_bsl: list[torch.Tensor] = []
    new_bv: list[torch.Tensor] = []

    for i in range(len(images)):
        bi = i if ball_visible.ndim > 0 else 0
        ci = i if ball_center.ndim > 1 else 0
        ri = i if ball_radius.ndim > 0 else 0
        si = i if ball_size_labeled.ndim > 0 else 0
        img, bx, cp, cm, np_, nm, bc, br, bv = apply_training_augmentations(
            images[i],
            det_targets[i]["boxes"],
            court_pts[i],
            court_mask[i],
            net_pts[i],
            net_mask[i],
            ball_center[ci],
            ball_visible[bi],
            ball_radius[ri],
            image_size=image_size,
        )
        new_images.append(img)
        new_det.append({"boxes": bx, "labels": det_targets[i]["labels"]})
        new_cp.append(cp)
        new_cm.append(cm)
        new_np.append(np_)
        new_nm.append(nm)
        new_bc.append(bc)
        new_br.append(br)
        new_bsl.append(ball_size_labeled[si])
        new_bv.append(bv)

    return {
        "stem": batch["stem"],
        "images": new_images,
        "det_targets": new_det,
        "court_pts": torch.stack(new_cp, dim=0),
        "court_mask": torch.stack(new_cm, dim=0),
        "net_pts": torch.stack(new_np, dim=0),
        "net_mask": torch.stack(new_nm, dim=0),
        "ball_center": torch.stack(new_bc, dim=0),
        "ball_radius": torch.stack(new_br, dim=0),
        "ball_size_labeled": torch.stack(new_bsl, dim=0),
        "ball_visible": torch.stack(new_bv, dim=0),
    }
