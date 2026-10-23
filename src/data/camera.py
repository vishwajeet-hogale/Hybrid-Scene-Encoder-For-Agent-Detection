"""
Camera inputs for LiDAR-camera fusion:
  - frozen DINOv2 ViT-S/14 image features (cached to disk per camera image)
  - camera geometry: camera -> ego (at the LiDAR timestamp) transform and feature-scale intrinsics
  - LiDAR-guided depth: LiDAR points projected into each camera's feature grid
"""
from __future__ import annotations
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion


CAMERAS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]

# 1600x900 images are resized to 448x252 (same aspect) for DINOv2; with 14 px patches that is an 18x32 grid,
# so each feature cell covers 50x50 original pixels.
IMG_W, IMG_H = 1600, 900
NET_W, NET_H = 448, 252
FEAT_H, FEAT_W = 18, 32
FEAT_STRIDE = IMG_W / FEAT_W  # 50.0
FEAT_DIM = 384


class DinoV2Extractor:
    """Frozen DINOv2 ViT-S/14: RGB images -> patch features [N, 384, 18, 32] (float16, CPU)."""
    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True).eval().to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, images_rgb: List[np.ndarray]) -> torch.Tensor:
        x = np.stack([cv2.resize(im, (NET_W, NET_H), interpolation=cv2.INTER_AREA) for im in images_rgb])
        x = torch.from_numpy(x).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        x = (x - self.mean) / self.std
        tokens = self.model.forward_features(x)["x_norm_patchtokens"]            # [N, 18*32, 384]
        return tokens.transpose(1, 2).reshape(len(images_rgb), FEAT_DIM, FEAT_H, FEAT_W).half().cpu()


class CameraFeatureStore:
    """
    Per-image DINOv2 features, cached as <cache_dir>/<camera sample_data token>.pt.
    Cache misses are computed on the fly if allow_compute (rendering sweeps); training expects a full cache.
    """
    def __init__(self, cache_dir: str, allow_compute: bool = False):
        self.cache_dir = cache_dir
        self.allow_compute = allow_compute
        self._extractor: Optional[DinoV2Extractor] = None

    def path(self, sd_token: str) -> str:
        return os.path.join(self.cache_dir, f"{sd_token}.pt")

    def get(self, nusc: NuScenes, cam_sd_tokens: List[str]) -> torch.Tensor:
        feats: List[Optional[torch.Tensor]] = []
        missing = []
        for i, tok in enumerate(cam_sd_tokens):
            p = self.path(tok)
            if os.path.exists(p):
                feats.append(torch.load(p))
            else:
                feats.append(None)
                missing.append(i)
        if missing:
            if not self.allow_compute:
                raise FileNotFoundError(f"No cached camera features in {self.cache_dir}; run cache_camera_features.py first")
            if self._extractor is None:
                self._extractor = DinoV2Extractor()
            images = [load_rgb(nusc, cam_sd_tokens[i]) for i in missing]
            for i, f in zip(missing, self._extractor(images)):
                feats[i] = f
        return torch.stack(feats)                                                 # [N_cams, 384, 18, 32] fp16


def load_rgb(nusc: NuScenes, sd_token: str) -> np.ndarray:
    path = os.path.join(nusc.dataroot, nusc.get("sample_data", sd_token)["filename"])
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def camera_to_lidar_ego(nusc: NuScenes, lidar_sd: Dict, cam_sd: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
      cam2ego: [4,4] camera frame -> ego frame at the LiDAR timestamp (camera -> ego(t_cam) -> global -> ego(t_lidar))
      K_feat:  [3,3] intrinsics in feature-grid units (pixel coordinates divided by FEAT_STRIDE)
    """
    cs_c = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    ep_c = nusc.get("ego_pose", cam_sd["ego_pose_token"])
    ep_l = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    cam2ego_c = transform_matrix(cs_c["translation"], Quaternion(cs_c["rotation"]), inverse=False)
    ego_c2global = transform_matrix(ep_c["translation"], Quaternion(ep_c["rotation"]), inverse=False)
    global2ego_l = transform_matrix(ep_l["translation"], Quaternion(ep_l["rotation"]), inverse=True)
    cam2ego = global2ego_l @ ego_c2global @ cam2ego_c
    K_feat = np.diag([1.0 / FEAT_STRIDE, 1.0 / FEAT_STRIDE, 1.0]) @ np.array(cs_c["camera_intrinsic"])
    return cam2ego.astype(np.float32), K_feat.astype(np.float32)


def lidar_depth_map(points_ego: np.ndarray, cam2ego: np.ndarray, K_feat: np.ndarray, min_depth: float = 0.5) -> np.ndarray:
    """
    Project ego-frame LiDAR points [N,>=3] into one camera's feature grid.
    Returns [FEAT_H, FEAT_W] with the nearest point's depth per cell (0 = no LiDAR point in that cell).
    """
    ego2cam = np.linalg.inv(cam2ego)
    pc = points_ego[:, :3] @ ego2cam[:3, :3].T + ego2cam[:3, 3]
    pc = pc[pc[:, 2] > min_depth]
    uv = pc @ K_feat.T
    u, v = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]
    keep = (u >= 0) & (u < FEAT_W) & (v >= 0) & (v < FEAT_H)
    cell = np.floor(v[keep]).astype(np.int64) * FEAT_W + np.floor(u[keep]).astype(np.int64)
    depth = np.full(FEAT_H * FEAT_W, np.inf, dtype=np.float32)
    np.minimum.at(depth, cell, pc[keep, 2].astype(np.float32))
    depth[np.isinf(depth)] = 0.0
    return depth.reshape(FEAT_H, FEAT_W)


def nearest_sample_data(nusc: NuScenes, first_sd_token: str, timestamp: int) -> str:
    """Walk a channel's sample_data chain from first_sd_token and return the token closest in time to timestamp."""
    tok, best, best_dt = first_sd_token, first_sd_token, float("inf")
    while tok:
        sd = nusc.get("sample_data", tok)
        dt = abs(sd["timestamp"] - timestamp)
        if dt < best_dt:
            best, best_dt = tok, dt
        elif sd["timestamp"] > timestamp:
            break
        tok = sd["next"]
    return best


def build_camera_inputs(nusc: NuScenes, lidar_sd_token: str, cam_sd_tokens: List[str], points_ego: np.ndarray,
                        store: CameraFeatureStore) -> Dict[str, torch.Tensor]:
    """All camera inputs for one LiDAR sweep (unaugmented): features, cam2ego, K_feat, sparse LiDAR depth."""
    lidar_sd = nusc.get("sample_data", lidar_sd_token)
    cam2ego, K, depth = [], [], []
    for tok in cam_sd_tokens:
        c2e, k = camera_to_lidar_ego(nusc, lidar_sd, nusc.get("sample_data", tok))
        cam2ego.append(c2e)
        K.append(k)
        depth.append(lidar_depth_map(points_ego, c2e, k))
    return {
        "cam_feats": store.get(nusc, cam_sd_tokens),             # [6, 384, 18, 32] fp16
        "cam2ego": torch.from_numpy(np.stack(cam2ego)),          # [6, 4, 4]
        "cam_K": torch.from_numpy(np.stack(K)),                  # [6, 3, 3]
        "cam_depth": torch.from_numpy(np.stack(depth)),          # [6, 18, 32]
    }
