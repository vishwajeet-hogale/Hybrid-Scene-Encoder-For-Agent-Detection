from __future__ import annotations
from typing import Any, Dict

import torch
import lightning.pytorch as L

from src.models.hybrid_scene_encoder import HybridSceneEncoder
from src.models.lidar.pointpillars_encoder import PointPillarsEncoder
from src.models.lidar.pillars import PillarConfig
from src.models.heads.center_head import CenterHead
from src.models.heads.targets import GridSpec, build_targets
from src.models.heads.losses import sigmoid_focal_loss, masked_l1
from src.vis.decode import decode_centers
from src.vis.bev_plot import visualize_sample_bev


class LitPerceptionModel(L.LightningModule):
    """
    LightningModule that owns:
    - Hybrid scene encoder (LiDAR now, map/camera later)
    - Task heads (detection now, prediction/planning later)
    """
    def __init__(
        self,
        pillar_cfg: Dict[str, Any],
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ---- Encoders ----
        lidar_encoder = PointPillarsEncoder(PillarConfig(**pillar_cfg))

        encoders = torch.nn.ModuleDict({
            "lidar": lidar_encoder,
            # future: "map", "camera"
        })

        self.scene_encoder = HybridSceneEncoder(
            encoders=encoders,
            fusion=None,   # future: cross-attn / FiLM / concat
        )
        self.num_classes = 3  # vehicle, pedestrian, cyclist (matches your mapping)
        self.head = CenterHead(in_channels=128, num_classes=self.num_classes, reg_channels=8)

        # grid spec must match PointPillars cfg
        pcfg = pillar_cfg
        self.grid = GridSpec(
            x_min=pcfg["x_range"][0], x_max=pcfg["x_range"][1],
            y_min=pcfg["y_range"][0], y_max=pcfg["y_range"][1],
            voxel_size_x=pcfg["voxel_size"][0],
            voxel_size_y=pcfg["voxel_size"][1],
        )

        self.hm_weight = 1.0
        self.reg_weight = 1.0
        self.gaussian_sigma = 1.5
        # ---- Heads (future) ----
        self.detection_head = None
        # self.prediction_head = None
        # self.planning_head = None

        self.lr = lr
        self.weight_decay = weight_decay

    def transfer_batch_to_device(self, batch: Dict[str, Any], device: torch.device, dataloader_idx: int):
        batch["lidar_points"] = [p.to(device) for p in batch["lidar_points"]]
        batch["gt_boxes"] = [b.to(device) for b in batch["gt_boxes"]]
        batch["gt_labels"] = [l.to(device) for l in batch["gt_labels"]]

        if "map_polylines" in batch:
            batch["map_polylines"] = batch["map_polylines"].to(device)
        if "map_mask" in batch:
            batch["map_mask"] = batch["map_mask"].to(device)

        return batch

    def forward(self, batch: Dict[str, Any]):
        enc_out = self.scene_encoder(batch)
        return enc_out

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        enc_out = self.forward(batch)
        bev = enc_out["bev"]
        hm_logits, reg_pred = self.head(bev)
        hm_tgt, reg_tgt, reg_mask = build_targets(
            batch["gt_boxes"],
            batch["gt_labels"],
            grid=self.grid,
            num_classes=self.num_classes,
            sigma=self.gaussian_sigma,
        )
        hm_loss = sigmoid_focal_loss(hm_logits, hm_tgt)
        reg_loss = masked_l1(reg_pred, reg_tgt, reg_mask)
        loss = hm_loss + reg_loss
        B = len(batch["lidar_points"])
        self.log("train/loss", loss, prog_bar=True, batch_size=B, on_step=True)
        self.log("train/hm_loss", hm_loss, prog_bar=False, batch_size=B, on_step=True)
        self.log("train/reg_loss", reg_loss, prog_bar=False, batch_size=B, on_step=True)
        self.log("train/num_pos", reg_mask.sum(), prog_bar=False, batch_size=B, on_step=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        enc_out = self.forward(batch)
        bev = enc_out["bev"]

        hm_logits, reg_pred = self.head(bev)

        hm_tgt, reg_tgt, reg_mask = build_targets(
            batch["gt_boxes"],
            batch["gt_labels"],
            grid=self.grid,
            num_classes=self.num_classes,
            sigma=self.gaussian_sigma,
        )

        hm_loss = sigmoid_focal_loss(hm_logits, hm_tgt)
        reg_loss = masked_l1(reg_pred, reg_tgt, reg_mask)
        loss = self.hm_weight * hm_loss + self.reg_weight * reg_loss

        B = len(batch["lidar_points"])
        self.log("val/loss", loss, prog_bar=True, batch_size=B)
        self.log("val/hm_loss", hm_loss, prog_bar=False, batch_size=B)
        self.log("val/reg_loss", reg_loss, prog_bar=False, batch_size=B)
        self.log("val/num_pos", reg_mask.sum(), prog_bar=False, batch_size=B)
        if batch_idx == 0 and self.current_epoch % 1 == 0:
            decoded = decode_centers(
                hm_logits=hm_logits.detach(),
                reg=reg_pred.detach(),
                grid=self.grid,              # your GridSpec
                K=100,
                score_thresh=0.3,
            )
            # visualize first sample in batch
            b0 = 0
            visualize_sample_bev(
                lidar_points=batch["lidar_points"][b0],
                gt_boxes=batch["gt_boxes"][b0],
                pred_boxes=decoded[b0]["boxes"],
                pred_scores=decoded[b0]["scores"],
                xlim=(self.grid.x_min, self.grid.x_max),
                ylim=(self.grid.y_min, self.grid.y_max),
            )
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)