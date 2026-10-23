from __future__ import annotations
from typing import List, Sequence

import torch
import torch.nn as nn

from src.models.lidar.pillars import PillarConfig, decorate_pillar_points, pillarize


class PillarFeatureNet(nn.Module):
    """
    PFN: per-point Linear + BN + ReLU, then masked max pooling over the points of each pillar.
    Input points: [P, T, 9] (decorated, see decorate_pillar_points)
    Output:       [P, C]
    """
    def __init__(self, in_dim: int = 9, out_dim: int = 64):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.BatchNorm1d(out_dim)

    def forward(self, pillar_points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        P, T, _ = pillar_points.shape
        x = self.linear(pillar_points)                                   # [P,T,C]
        x = self.norm(x.reshape(P * T, -1)).reshape(P, T, -1)
        x = torch.relu(x)
        x = x * mask.unsqueeze(-1).to(x.dtype)       # mask padded points (ReLU output >= 0, so 0 never wins the max)
        x, _ = x.max(dim=1)                          # [P,C]
        return x


def conv_bn_relu(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class BEVBackbone(nn.Module):
    """
    Multi-scale BEV backbone (PointPillars / CenterPoint "RPN"):
      stages at stride 1, 2, 4 -> each upsampled back to stride 1 -> concat -> 1x1 fuse.
    Receptive field ~39 cells (~20 m at 0.5 m pillars) vs ~9 cells for a single-scale stack.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int = 128,
        stage_channels: Sequence[int] = (64, 128, 256),
        stage_strides: Sequence[int] = (1, 2, 2),
        stage_depths: Sequence[int] = (3, 3, 3),
        up_channels: int = 64,
    ):
        super().__init__()
        self.stages = nn.ModuleList()
        self.ups = nn.ModuleList()
        ch, total_stride = in_ch, 1
        for c, s, d in zip(stage_channels, stage_strides, stage_depths):
            layers = [conv_bn_relu(ch, c, stride=s)] + [conv_bn_relu(c, c) for _ in range(d - 1)]
            self.stages.append(nn.Sequential(*layers))
            ch, total_stride = c, total_stride * s
            # upsample this stage's output back to the input resolution
            self.ups.append(nn.Sequential(
                nn.ConvTranspose2d(c, up_channels, total_stride, stride=total_stride, bias=False),
                nn.BatchNorm2d(up_channels),
                nn.ReLU(inplace=True),
            ))
        self.fuse = nn.Sequential(
            nn.Conv2d(up_channels * len(stage_channels), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for stage, up in zip(self.stages, self.ups):
            x = stage(x)
            outs.append(up(x))
        return self.fuse(torch.cat(outs, dim=1))


class PointPillarsEncoder(nn.Module):
    """
    Input:  list of B point clouds, each [Ni,4] ego frame
    Output: BEV features [B, C, H, W]
    """
    def __init__(
        self,
        cfg: PillarConfig,
        pillar_feat_dim: int = 64,
        bev_feat_dim: int = 128,
    ):
        super().__init__()
        self.cfg = cfg
        self.pfn = PillarFeatureNet(9, pillar_feat_dim)
        self.backbone = BEVBackbone(pillar_feat_dim, bev_feat_dim)

        self.H = int((cfg.y_range[1] - cfg.y_range[0]) / cfg.voxel_size[1])
        self.W = int((cfg.x_range[1] - cfg.x_range[0]) / cfg.voxel_size[0])

    def forward(self, batch_points: List[torch.Tensor]) -> torch.Tensor:
        """
        Pillarization runs on CPU (variable-size ops) and every sample is padded to max_pillars, so all GPU
        tensors have fixed shapes. On MPS, changing shapes every step makes the kernel cache grow without bound.
        """
        device = next(self.parameters()).device
        B, P, T = len(batch_points), self.cfg.max_pillars, self.cfg.max_points_per_pillar
        HW = self.H * self.W

        pillars = torch.zeros((B, P, T, 9), dtype=torch.float32)
        masks = torch.zeros((B, P, T), dtype=torch.bool)
        flat_idx = torch.full((B, P), HW, dtype=torch.long)    # padded pillars go to an extra "trash" cell HW
        for b, pts in enumerate(batch_points):
            p, c, m = pillarize(pts.cpu(), self.cfg, self.H, self.W)        # [n,T,4], [n,2] (iy,ix), [n,T]
            n = p.shape[0]
            pillars[b, :n] = decorate_pillar_points(p, c, m, self.cfg)      # [n,T,9]
            masks[b, :n] = m
            flat_idx[b, :n] = c[:, 0] * self.W + c[:, 1]

        pillars, masks, flat_idx = pillars.to(device), masks.to(device), flat_idx.to(device)
        pillar_feats = self.pfn(pillars.reshape(B * P, T, 9), masks.reshape(B * P, T))   # [B*P, Cp]

        # scatter to BEV: one fixed-size index_put over the whole batch, then drop the trash cell
        Cp = pillar_feats.shape[1]
        flat_idx = flat_idx + torch.arange(B, device=device).unsqueeze(1) * (HW + 1)
        bev = torch.zeros((B * (HW + 1), Cp), device=device, dtype=pillar_feats.dtype)
        bev = bev.index_put((flat_idx.reshape(-1),), pillar_feats)
        # contiguous() matters: a strided BEV propagates through the convs and breaks BatchNorm backward on MPS
        bev = bev.reshape(B, HW + 1, Cp)[:, :HW].permute(0, 2, 1).reshape(B, Cp, self.H, self.W).contiguous()

        bev = self.backbone(bev)             # [B,C,H,W]
        return bev
