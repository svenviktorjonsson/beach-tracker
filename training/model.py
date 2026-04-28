"""Multi-task model: object detection (balls + people) + geometry keypoint regression (court/net)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

MAX_COURT_PTS = 8
MAX_NET_PTS = 5


def _replace_detection_head(model: nn.Module, num_classes: int) -> None:
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


class GeometryRegressor(nn.Module):
    """
    Predicts court and net keypoint coordinates from an image.

    Uses MobileNetV3-Small as backbone (fast, ~3ms GPU) with two FC heads
    that output normalised (0-1) xy coordinates for each keypoint plus a
    visibility logit per point so the model can learn how many points exist.
    """

    def __init__(
        self,
        max_court_pts: int = MAX_COURT_PTS,
        max_net_pts: int = MAX_NET_PTS,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.max_court_pts = max_court_pts
        self.max_net_pts = max_net_pts

        backbone = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        )
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = 576  # MobileNetV3-Small last channel dim

        self.court_coords = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Linear(256, max_court_pts * 2),
            nn.Sigmoid(),
        )
        self.court_vis = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, max_court_pts),
        )

        self.net_coords = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Linear(256, max_net_pts * 2),
            nn.Sigmoid(),
        )
        self.net_vis = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, max_net_pts),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.pool(self.features(x)).flatten(1)
        return {
            "court_pts": self.court_coords(feat).view(-1, self.max_court_pts, 2),
            "court_vis": self.court_vis(feat),
            "net_pts": self.net_coords(feat).view(-1, self.max_net_pts, 2),
            "net_vis": self.net_vis(feat),
        }


class BeachVolleyballMultiTask(nn.Module):
    """
    Joint training wrapper:
    - `detector`   : Faster R-CNN MobileNetV3-FPN for balls + people.
    - `geometry`   : GeometryRegressor for court/net keypoint regression.

    `num_detection_classes` includes background (label 0).
    """

    def __init__(
        self,
        num_detection_classes: int = 6,
        pretrained_detection: bool = True,
        pretrained_geometry: bool = True,
    ) -> None:
        super().__init__()
        det_weights = (
            FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            if pretrained_detection
            else None
        )
        self.detector = fasterrcnn_mobilenet_v3_large_fpn(weights=det_weights)
        _replace_detection_head(self.detector, num_detection_classes)

        self.geometry = GeometryRegressor(pretrained=pretrained_geometry)
