from __future__ import annotations
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion



# --- category -> class id (edit as you like) ---
# returns: 0=vehicle, 1=pedestrian, 2=cyclist
def map_nuscenes_category_to_label(category_name: str) -> Optional[int]:
    # check bicycle/motorcycle before the generic "vehicle." prefix, otherwise they'd be labeled vehicle
    if category_name.startswith("vehicle.bicycle") or category_name.startswith("vehicle.motorcycle"):
        return 2
    if category_name.startswith("vehicle."):
        # you can exclude parked vehicles later if you want
        return 0
    if category_name.startswith("human.pedestrian"):
        return 1
    return None


def transform_box_global_to_ego(box: Box, ego_pose: Dict[str, Any]) -> Box:
    """
    Transform a box from global frame into ego frame at the time of the sample_data.
    """
    box = box.copy()
    # translate
    box.translate(-np.array(ego_pose["translation"]))
    # rotate by inverse ego rotation
    box.rotate(Quaternion(ego_pose["rotation"]).inverse)
    return box


def load_lidar_points_ego(nusc: NuScenes, sd_token: str) -> torch.Tensor:
    """
    Load a LiDAR sweep and return its points in the EGO frame at the sweep timestamp: [N,4] = x,y,z,intensity.
    """
    lidar_path = nusc.get_sample_data_path(sd_token)
    pc = LidarPointCloud.from_file(lidar_path)
    pts = pc.points.T.astype(np.float32)  # [N,4] = x,y,z,intensity in LiDAR sensor frame

    # --- Get sample_data + calibrated sensor (LiDAR -> ego) ---
    sd = nusc.get("sample_data", sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

    T_lidar_to_ego = transform_matrix(
        translation=cs["translation"],
        rotation=Quaternion(cs["rotation"]),
        inverse=False,
    )

    # Apply transform to xyz
    xyz1 = np.concatenate([pts[:, :3], np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)  # [N,4]
    xyz_ego = (T_lidar_to_ego @ xyz1.T).T[:, :3]  # [N,3]

    # Re-attach intensity
    pts_ego = np.concatenate([xyz_ego, pts[:, 3:4]], axis=1)  # [N,4]
    return torch.from_numpy(pts_ego).float()


def augment_scene(points: torch.Tensor, boxes: torch.Tensor,
                  flip_prob: float = 0.5, max_rot: float = np.pi / 8, scale_range=(0.95, 1.05)):
    """
    Global augmentation applied jointly to points [N,4] and boxes [M,7] (x,y,z,w,l,h,yaw), both in ego frame:
      - left/right flip (y -> -y, yaw -> -yaw); no front/back flip since the BEV grid is not symmetric in x
      - rotation about the ego z axis by U(-max_rot, max_rot)
      - uniform scaling by U(scale_range)
    """
    points, boxes = points.clone(), boxes.clone()

    if torch.rand(1).item() < flip_prob:
        points[:, 1] = -points[:, 1]
        boxes[:, 1] = -boxes[:, 1]
        boxes[:, 6] = -boxes[:, 6]

    theta = (torch.rand(1).item() * 2 - 1) * max_rot
    c, s = float(np.cos(theta)), float(np.sin(theta))
    R = torch.tensor([[c, -s], [s, c]], dtype=points.dtype)
    points[:, :2] = points[:, :2] @ R.T
    boxes[:, :2] = boxes[:, :2] @ R.T
    boxes[:, 6] = boxes[:, 6] + theta

    scale = scale_range[0] + torch.rand(1).item() * (scale_range[1] - scale_range[0])
    points[:, :3] *= scale
    boxes[:, :6] *= scale

    boxes[:, 6] = torch.atan2(torch.sin(boxes[:, 6]), torch.cos(boxes[:, 6]))  # wrap yaw to (-pi, pi]
    return points, boxes


class NuScenesDataset(Dataset):
    def __init__(self, nusc: NuScenes, sample_tokens: List[str], split: str, bev_cfg: Dict[str, Any], map_cfg: Dict[str, Any],
                 augment: bool = False):
        super().__init__()
        self.nusc = nusc
        self.sample_tokens = sample_tokens
        self.split = split
        self.bev_cfg = bev_cfg
        self.map_cfg = map_cfg
        self.augment = augment

    def __len__(self) -> int:
        return len(self.sample_tokens)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_token = self.sample_tokens[idx]
        sample = self.nusc.get("sample", sample_token)

        # --- Load LiDAR points in EGO frame ---
        sd_token = sample["data"]["LIDAR_TOP"]
        points = load_lidar_points_ego(self.nusc, sd_token)  # [N,4]

        # --- Get ego pose at LiDAR timestamp ---
        sd = self.nusc.get("sample_data", sd_token)
        ego_pose = self.nusc.get("ego_pose", sd["ego_pose_token"])

        # --- Load GT annotations and transform to ego frame ---
        gt_boxes_list = []
        gt_labels_list = []

        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)

            label = map_nuscenes_category_to_label(ann["category_name"])
            if label is None:
                continue

            # nuScenes provides size as (w, l, h) in annotation records
            w, l, h = ann["size"]
            center = np.array(ann["translation"], dtype=np.float32)
            q = Quaternion(ann["rotation"])

            box = Box(center=center, size=np.array([w, l, h], dtype=np.float32), orientation=q)

            # Transform global -> ego
            box_ego = transform_box_global_to_ego(box, ego_pose)

            # yaw in ego frame from box orientation
            yaw = box_ego.orientation.yaw_pitch_roll[0]

            # For BEV detectors, we typically use (x,y,z,w,l,h,yaw)
            gt_boxes_list.append([box_ego.center[0], box_ego.center[1], box_ego.center[2], w, l, h, yaw])
            gt_labels_list.append(label)

        if len(gt_boxes_list) == 0:
            gt_boxes = torch.zeros((0, 7), dtype=torch.float32)
            gt_labels = torch.zeros((0,), dtype=torch.int64)
        else:
            gt_boxes = torch.tensor(gt_boxes_list, dtype=torch.float32)
            gt_labels = torch.tensor(gt_labels_list, dtype=torch.int64)

        if self.augment:
            points, gt_boxes = augment_scene(points, gt_boxes)

        # placeholders (we’ll fill later)
        bev = torch.zeros(8, 200, 200, dtype=torch.float32)
        map_polylines = torch.zeros(self.map_cfg["max_polylines"], self.map_cfg["points_per_polyline"], 2, dtype=torch.float32)
        map_mask = torch.zeros(self.map_cfg["max_polylines"], dtype=torch.bool)

        return {
            "sample_token": sample_token,
            "scene_token": sample["scene_token"],
            "lidar_points": points,     # [Ni,4]
            "gt_boxes": gt_boxes,       # [N,7] ego frame
            "gt_labels": gt_labels,     # [N]
            "bev": bev,
            "map_polylines": map_polylines,
            "map_mask": map_mask,
        }
