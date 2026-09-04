"""Compact multi-task models for coarse court/net/ball proposals."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

MAX_COURT_PTS = 4
MAX_NET_PTS = 5


def _replace_detection_head(model: nn.Module, num_classes: int) -> None:
    """Swap Faster R-CNN classification head to a fixed class count."""
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


class GeometryRegressor(nn.Module):
    """Predict coarse court orthopoints, net landmarks, and ball hints.

    These outputs are proposal seeds for downstream geometric solvers rather than
    final calibrated geometry.
    """

    def __init__(
        self,
        max_court_pts: int = MAX_COURT_PTS,
        max_net_pts: int = MAX_NET_PTS,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = True,
        hidden: int = 128,
        width_mult: float = 0.5,
    ) -> None:
        super().__init__()
        self.max_court_pts = max_court_pts
        self.max_net_pts = max_net_pts

        weights = (
            MobileNet_V3_Small_Weights.DEFAULT
            if pretrained_backbone and abs(width_mult - 1.0) < 1e-6
            else None
        )
        backbone = mobilenet_v3_small(weights=weights, width_mult=width_mult)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        with torch.no_grad():
            sample = torch.zeros(1, 3, 320, 320)
            feat_dim = int(self.features(sample).shape[1])
        del sample

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        # Fast first-stage head: shared projection + linear outputs for all geometry
        # targets. Keeping this compact directly targets CPU/laptop inference
        # throughput while preserving a single model path for court/net + ball.
        self._shared = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, max(32, hidden // 2)),
            nn.ReLU(inplace=True),
        )

        self._head = nn.Linear(max(32, hidden // 2), hidden // 2)

        self._head_out = nn.ReLU(inplace=True)
        self.court_coords = nn.Linear(hidden // 2, max_court_pts * 2)
        self.court_vis = nn.Linear(hidden // 2, max_court_pts)
        self.net_coords = nn.Linear(hidden // 2, max_net_pts * 2)
        self.net_vis = nn.Linear(hidden // 2, max_net_pts)
        self.ball_center = nn.Linear(hidden // 2, 2)
        self.ball_radius = nn.Linear(hidden // 2, 1)
        self.ball_vis = nn.Linear(hidden // 2, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.pool(self.features(x)).flatten(1)
        feat_latent = self._head_out(self._head(self._shared(feat)))
        return {
            "court_pts": torch.sigmoid(self.court_coords(feat_latent)).view(-1, self.max_court_pts, 2),
            "court_vis": self.court_vis(feat_latent),
            "net_pts": torch.sigmoid(self.net_coords(feat_latent)).view(-1, self.max_net_pts, 2),
            "net_vis": self.net_vis(feat_latent),
            "ball_center": torch.sigmoid(self.ball_center(feat_latent)),
            "ball_radius": torch.sigmoid(self.ball_radius(feat_latent)),
            "ball_vis": self.ball_vis(feat_latent).squeeze(-1),
        }


class BeachVolleyballMultiTask(nn.Module):
    """
    Wrapper that supports:
    - optional detection branch (kept for compatibility and people-only training)
    - fast geometry/ball regression branch for court/net/ball.
    """

    def __init__(
        self,
        *,
        num_detection_classes: int = 6,
        pretrained_detection: bool = True,
        pretrained_geometry: bool = True,
        enable_detector: bool = False,
        geometry_width_mult: float = 0.4,
        geometry_hidden: int = 128,
        freeze_geometry_backbone: bool = True,
        # Backward-compatible keyword args used by legacy scripts:
        num_segmentation_classes: int | None = None,
        pretrained_segmentation: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        self.geometry = GeometryRegressor(
            pretrained_backbone=pretrained_geometry,
            freeze_backbone=freeze_geometry_backbone,
            hidden=geometry_hidden,
            width_mult=geometry_width_mult,
        )
        self.detector = None
        if enable_detector:
            det_weights = (
                FasterRCNN_MobileV3_Large_FPN_Weights.DEFAULT
                if pretrained_detection
                else None
            )
            self.detector = fasterrcnn_mobilenet_v3_large_fpn(weights=det_weights)
            _replace_detection_head(self.detector, num_detection_classes)
        self._legacy_segmentation_ignored = (
            num_segmentation_classes is not None
            or pretrained_segmentation is not None
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Primary training/inference path is geometry + ball.
        return self.geometry(x)
