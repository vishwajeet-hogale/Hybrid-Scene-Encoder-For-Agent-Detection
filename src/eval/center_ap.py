from __future__ import annotations
from typing import Dict, List, Sequence

import numpy as np
import torch

from src.models.heads.targets import GridSpec


class CenterDistanceAP:
    """
    nuScenes-style detection AP with BEV center-distance matching (no IoU):
      - per class, predictions sorted by score are greedily matched to the nearest unmatched GT within dist_thresh meters
      - AP = mean precision over recall levels 0.1..1.0 (precision 0 at unreached recall)
      - yaw_err_<class>: mean absolute heading error (degrees) of matched predictions with score >= yaw_score
        (center matching ignores orientation, like nuScenes AP; this is a simple version of its AOE metric)
    Only GT boxes inside the BEV grid are counted. Accumulate with update(), read with compute().
    """
    def __init__(self, grid: GridSpec, class_names: Sequence[str], dist_thresh: float = 2.0, yaw_score: float = 0.2):
        self.grid = grid
        self.class_names = list(class_names)
        self.dist_thresh = dist_thresh
        self.yaw_score = yaw_score
        self.reset()

    def reset(self) -> None:
        self.preds: Dict[int, List] = {c: [] for c in range(len(self.class_names))}  # (score, is_tp)
        self.num_gt: Dict[int, int] = {c: 0 for c in range(len(self.class_names))}
        self.yaw_errs: Dict[int, List[float]] = {c: [] for c in range(len(self.class_names))}

    def update(self, dets: List[Dict[str, torch.Tensor]], gt_boxes: List[torch.Tensor], gt_labels: List[torch.Tensor]) -> None:
        g = self.grid
        for det, gb, gl in zip(dets, gt_boxes, gt_labels):
            pb, ps, pl = det["boxes"].cpu().numpy(), det["scores"].cpu().numpy(), det["labels"].cpu().numpy()
            gb, gl = gb.cpu().numpy(), gl.cpu().numpy()
            in_grid = (gb[:, 0] >= g.x_min) & (gb[:, 0] < g.x_max) & (gb[:, 1] >= g.y_min) & (gb[:, 1] < g.y_max)
            gb, gl = gb[in_grid], gl[in_grid]

            for c in range(len(self.class_names)):
                gt_yaw = gb[gl == c][:, 6]
                gt_xy = gb[gl == c][:, :2]
                self.num_gt[c] += len(gt_xy)
                used = np.zeros(len(gt_xy), dtype=bool)
                idx = np.where(pl == c)[0]
                for i in idx[np.argsort(-ps[idx])]:
                    hit = False
                    if len(gt_xy):
                        d = np.linalg.norm(gt_xy - pb[i, :2], axis=1)
                        d[used] = np.inf
                        j = int(d.argmin())
                        if d[j] < self.dist_thresh:
                            used[j] = True
                            hit = True
                            if ps[i] >= self.yaw_score:
                                self.yaw_errs[c].append(abs((pb[i, 6] - gt_yaw[j] + np.pi) % (2 * np.pi) - np.pi))
                    self.preds[c].append((float(ps[i]), hit))

    def compute(self) -> Dict[str, float]:
        out = {}
        for c, name in enumerate(self.class_names):
            preds = sorted(self.preds[c], key=lambda p: -p[0])
            if self.num_gt[c] == 0 or len(preds) == 0:
                out[name] = 0.0
                continue
            tp = np.cumsum([hit for _, hit in preds])
            precision = tp / np.arange(1, len(preds) + 1)
            recall = tp / self.num_gt[c]
            out[name] = float(np.mean([precision[recall >= r].max() if (recall >= r).any() else 0.0
                                       for r in np.linspace(0.1, 1.0, 10)]))
        out["mAP"] = float(np.mean([out[n] for n in self.class_names]))
        for c, name in enumerate(self.class_names):
            out[f"yaw_err_{name}"] = float(np.degrees(np.mean(self.yaw_errs[c]))) if self.yaw_errs[c] else 0.0
        return out
