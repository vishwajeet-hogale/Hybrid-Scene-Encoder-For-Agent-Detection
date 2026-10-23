from __future__ import annotations
from typing import Any, Dict, List

import torch
import lightning.pytorch as L

from src.models.hybrid_scene_encoder import HybridSceneEncoder
from src.models.lidar.pointpillars_encoder import PointPillarsEncoder
from src.models.lidar.pillars import PillarConfig
from src.models.heads.center_head import CenterHead
from src.models.heads.targets import BoxPrior, GridSpec, build_targets
from src.models.heads.losses import sigmoid_focal_loss, masked_l1
from src.eval.center_ap import CenterDistanceAP
from src.vis.decode import circle_nms, decode_centers
from src.vis.bev_plot import visualize_sample_bev


CLASS_NAMES = ["vehicle", "pedestrian", "cyclist"]


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
        weight_decay: float = 1e-2,
        hm_weight: float = 1.0,
        reg_weight: float = 0.25,
        pct_start: float = 0.4,
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
        self.num_classes = len(CLASS_NAMES)  # vehicle, pedestrian, cyclist (matches the dataset mapping)
        self.head = CenterHead(in_channels=128, num_classes=self.num_classes, reg_channels=8)

        # grid spec must match PointPillars cfg
        pcfg = pillar_cfg
        self.grid = GridSpec(
            x_min=pcfg["x_range"][0], x_max=pcfg["x_range"][1],
            y_min=pcfg["y_range"][0], y_max=pcfg["y_range"][1],
            voxel_size_x=pcfg["voxel_size"][0],
            voxel_size_y=pcfg["voxel_size"][1],
        )

        # per-class priors for z / size normalization (buffers: follow the device, saved in checkpoints)
        self.prior = BoxPrior()
        z_mean, size_mean = self.prior.tensors("cpu")
        self.register_buffer("z_mean", z_mean)
        self.register_buffer("size_mean", size_mean)

        self.hm_weight = hm_weight
        self.reg_weight = reg_weight
        self.val_ap = CenterDistanceAP(self.grid, CLASS_NAMES, dist_thresh=2.0)
        # ---- Heads (future) ----
        self.detection_head = None
        # self.prediction_head = None
        # self.planning_head = None

        self.lr = lr
        self.weight_decay = weight_decay
        self.pct_start = pct_start

    def transfer_batch_to_device(self, batch: Dict[str, Any], device: torch.device, dataloader_idx: int):
        # lidar_points / gt_boxes / gt_labels stay on CPU: pillarization, target building and AP matching are
        # variable-size CPU work; only fixed-size tensors go to the accelerator (see PointPillarsEncoder.forward)
        if "map_polylines" in batch:
            batch["map_polylines"] = batch["map_polylines"].to(device)
        if "map_mask" in batch:
            batch["map_mask"] = batch["map_mask"].to(device)

        return batch

    def forward(self, batch: Dict[str, Any]):
        enc_out = self.scene_encoder(batch)
        return enc_out

    @torch.no_grad()
    def decode(self, hm_logits: torch.Tensor, reg: torch.Tensor, score_thresh: float = 0.3, K: int = 100) -> List[Dict[str, torch.Tensor]]:
        """Heatmap/regression maps -> per-sample {"boxes" [M,7] ego (x,y,z,w,l,h,yaw), "scores", "labels"} on CPU."""
        dets = decode_centers(hm_logits.cpu(), reg.cpu(), grid=self.grid, K=K, score_thresh=score_thresh,
                              z_mean=self.z_mean.cpu(), size_mean=self.size_mean.cpu())
        return [circle_nms(d) for d in dets]

    def _shared_step(self, batch: Dict[str, Any]):
        bev = self.forward(batch)["bev"]
        hm_logits, reg_pred = self.head(bev)
        hm_tgt, reg_tgt, reg_mask = build_targets(
            batch["gt_boxes"],
            batch["gt_labels"],
            grid=self.grid,
            num_classes=self.num_classes,
            prior=self.prior,
        )
        hm_tgt, reg_tgt, reg_mask = hm_tgt.to(self.device), reg_tgt.to(self.device), reg_mask.to(self.device)
        hm_loss = sigmoid_focal_loss(hm_logits, hm_tgt)
        reg_loss = masked_l1(reg_pred, reg_tgt, reg_mask)
        loss = self.hm_weight * hm_loss + self.reg_weight * reg_loss
        return loss, hm_loss, reg_loss, reg_mask, hm_logits, reg_pred

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        loss, hm_loss, reg_loss, reg_mask, _, _ = self._shared_step(batch)
        B = len(batch["lidar_points"])
        self.log("train/loss", loss, prog_bar=True, batch_size=B, on_step=True)
        self.log("train/hm_loss", hm_loss, prog_bar=False, batch_size=B, on_step=True)
        self.log("train/reg_loss", reg_loss, prog_bar=False, batch_size=B, on_step=True)
        self.log("train/num_pos", reg_mask.sum(), prog_bar=False, batch_size=B, on_step=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        loss, hm_loss, reg_loss, reg_mask, hm_logits, reg_pred = self._shared_step(batch)

        B = len(batch["lidar_points"])
        self.log("val/loss", loss, prog_bar=True, batch_size=B)
        self.log("val/hm_loss", hm_loss, prog_bar=False, batch_size=B)
        self.log("val/reg_loss", reg_loss, prog_bar=False, batch_size=B)
        self.log("val/num_pos", reg_mask.sum(), prog_bar=False, batch_size=B)

        # low threshold so AP sees the whole precision/recall curve
        decoded = self.decode(hm_logits.detach(), reg_pred.detach(), score_thresh=0.05, K=200)
        self.val_ap.update(decoded, batch["gt_boxes"], batch["gt_labels"])

        if batch_idx == 0 and self.current_epoch % 1 == 0:
            pass
            # visualize first sample in batch
            # b0 = 0
            # visualize_sample_bev(
            #     lidar_points=batch["lidar_points"][b0],
            #     gt_boxes=batch["gt_boxes"][b0],
            #     pred_boxes=decoded[b0]["boxes"],
            #     pred_scores=decoded[b0]["scores"],
            #     xlim=(self.grid.x_min, self.grid.x_max),
            #     ylim=(self.grid.y_min, self.grid.y_max),
            # )

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            ap = self.val_ap.compute()
            self.log("val/mAP", ap["mAP"], prog_bar=True)
            for name in CLASS_NAMES:
                self.log(f"val/AP_{name}", ap[name])
        self.val_ap.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=int(self.trainer.estimated_stepping_batches),
            pct_start=self.pct_start,
            div_factor=10.0,
            final_div_factor=100.0,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
