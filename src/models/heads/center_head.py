from __future__ import annotations
import torch
import torch.nn as nn


class CenterHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, reg_channels: int = 8):
        super().__init__()
        self.num_classes = num_classes
        self.reg_channels = reg_channels

        def head_block(out_ch):
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_ch, 1),
            )

        self.hm_head = head_block(num_classes)
        self.reg_head = head_block(reg_channels)

        # common trick: initialize heatmap bias low so initial probs ~0
        nn.init.constant_(self.hm_head[-1].bias, -2.19)  # sigmoid(-2.19) ~ 0.10

    def forward(self, bev_feat: torch.Tensor):
        hm = self.hm_head(bev_feat)
        reg = self.reg_head(bev_feat)
        return hm, reg
