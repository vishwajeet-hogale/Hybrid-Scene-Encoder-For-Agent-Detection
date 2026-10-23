from __future__ import annotations
from typing import Any, Dict, List

import torch
import lightning.pytorch as L

from src.models.hybrid_scene_encoder import HybridSceneEncoder
from src.models.lidar.pointpillars_encoder import BEVBackbone, PointPillarsEncoder
from src.models.camera.lss_encoder import CameraBEVEncoder
from src.models.fusion import ConvFuser
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
    - Hybrid scene encoder (LiDAR, optionally fused with cameras; map later)
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
        use_camera: bool = False,
        depth_weight: float = 0.5,
        pretrained_lr_scale: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ---- Encoders ----
        self.use_camera = use_camera
        self.depth_weight = depth_weight
        pillar_config = PillarConfig(**pillar_cfg)
        # with cameras, the BEV backbone runs after fusion, shared by both modalities (BEVFusion layout)
        lidar_encoder = PointPillarsEncoder(pillar_config, with_backbone=not use_camera)

        encoders = torch.nn.ModuleDict({
            "lidar": lidar_encoder,
            # future: "map"
        })
        fusion, backbone = None, None
        if use_camera:
            encoders["camera"] = CameraBEVEncoder(pillar_config)
            fusion = ConvFuser(lidar_channels=64, camera_channels=encoders["camera"].out_channels)
            backbone = BEVBackbone(64, 128)

        self.scene_encoder = HybridSceneEncoder(
            encoders=encoders,
            fusion=fusion,
            backbone=backbone,
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
        self.pretrained_lr_scale = pretrained_lr_scale

    def transfer_batch_to_device(self, batch: Dict[str, Any], device: torch.device, dataloader_idx: int):
        # lidar_points / gt_boxes / gt_labels stay on CPU: pillarization, target building and AP matching are
        # variable-size CPU work; only fixed-size tensors go to the accelerator (see PointPillarsEncoder.forward)
        for k in ("cam_feats", "cam2ego", "cam_K", "cam_depth"):
            if k in batch:
                batch[k] = batch[k].to(device)
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

    def load_pretrained(self, state_dict: Dict[str, torch.Tensor]):
        """
        Load weights for fine-tuning. A LiDAR-only checkpoint can initialize a camera-fusion model: its BEV backbone
        moves from the LiDAR encoder to the shared post-fusion backbone; the camera branch and fuser keep their init
        (the fuser's zero-initialized last conv makes the model start out equal to the LiDAR-only one).
        """
        if self.use_camera:
            old, new = "scene_encoder.encoders.lidar.backbone.", "scene_encoder.backbone."
            state_dict = {(new + k[len(old):] if k.startswith(old) else k): v for k, v in state_dict.items()}
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected[:5]}")
        return missing

    def _shared_step(self, batch: Dict[str, Any]):
        outs = self.forward(batch)
        hm_logits, reg_pred = self.head(outs["bev"])
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
        depth_loss = None
        if self.use_camera:
            depth_loss = self.scene_encoder.encoders["camera"].depth_loss(outs["cam_depth_logits"], batch["cam_depth"])
            loss = loss + self.depth_weight * depth_loss
        return loss, hm_loss, reg_loss, reg_mask, hm_logits, reg_pred, depth_loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        loss, hm_loss, reg_loss, reg_mask, _, _, depth_loss = self._shared_step(batch)
        B = len(batch["lidar_points"])
        if depth_loss is not None:
            self.log("train/depth_loss", depth_loss, prog_bar=False, batch_size=B, on_step=True)
        self.log("train/loss", loss, prog_bar=True, batch_size=B, on_step=True)
        self.log("train/hm_loss", hm_loss, prog_bar=False, batch_size=B, on_step=True)
        self.log("train/reg_loss", reg_loss, prog_bar=False, batch_size=B, on_step=True)
        self.log("train/num_pos", reg_mask.sum(), prog_bar=False, batch_size=B, on_step=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        loss, hm_loss, reg_loss, reg_mask, hm_logits, reg_pred, depth_loss = self._shared_step(batch)

        B = len(batch["lidar_points"])
        if depth_loss is not None:
            self.log("val/depth_loss", depth_loss, prog_bar=False, batch_size=B)
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
        # fine-tuning a fusion model from LiDAR weights: new modules (camera branch, fuser) train at lr,
        # everything else at lr * pretrained_lr_scale so the pretrained LiDAR path isn't knocked out of its optimum
        new_prefixes = ("scene_encoder.encoders.camera.", "scene_encoder.fusion.")
        new = [p for n, p in self.named_parameters() if n.startswith(new_prefixes)]
        old = [p for n, p in self.named_parameters() if not n.startswith(new_prefixes)]
        groups = [{"params": old, "lr": self.lr * self.pretrained_lr_scale}]
        if new:
            groups.append({"params": new, "lr": self.lr})
        optimizer = torch.optim.AdamW(groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[g["lr"] for g in groups],
            total_steps=int(self.trainer.estimated_stepping_batches),
            pct_start=self.pct_start,
            div_factor=10.0,
            final_div_factor=100.0,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
