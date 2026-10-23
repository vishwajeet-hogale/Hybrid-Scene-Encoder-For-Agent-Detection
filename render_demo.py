"""
Render a demo video/GIF of the detector on a nuScenes scene:
  left  = LiDAR BEV (ego frame) with predicted boxes
  right = CAM_FRONT with the same boxes projected into the image

Usage:
  python render_demo.py --ckpt outputs/lightning_logs/version_N/checkpoints/last.ckpt --scene scene-0103
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
from typing import Dict, List

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion

from src.data.camera import CAMERAS, CameraFeatureStore, build_camera_inputs
from src.data.nuscenes_dataset import load_lidar_points_ego, map_nuscenes_category_to_label, transform_box_global_to_ego
from src.models.lit_perception_model import LitPerceptionModel



CLASS_NAMES = ["vehicle", "pedestrian", "cyclist"]
CLASS_COLORS = [(255, 159, 28), (46, 196, 255), (255, 77, 166)]  # RGB per class
BG = (14, 15, 20)
FONT_PATHS = ["/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def load_font(size: int):
    for p in FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def scene_channel_tokens(nusc: NuScenes, scene: Dict, channel: str) -> List[str]:
    """All sample_data tokens (keyframes + sweeps) of one channel inside a scene, in time order."""
    first = nusc.get("sample", scene["first_sample_token"])
    last = nusc.get("sample", scene["last_sample_token"])
    tokens = []
    sd = nusc.get("sample_data", first["data"][channel])
    while sd["timestamp"] <= last["timestamp"]:
        tokens.append(sd["token"])
        if sd["next"] == "":
            break
        sd = nusc.get("sample_data", sd["next"])
    return tokens


def box_corners_3d(box: np.ndarray) -> np.ndarray:
    """box (x,y,z,w,l,h,yaw) -> [8,3] corners. 0-3 bottom, 4-7 top; 0,1,4,5 are the front face."""
    x, y, z, w, l, h, yaw = box
    dx = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * l / 2
    dy = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * w / 2
    dz = np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * h / 2
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([x + c * dx - s * dy, y + s * dx + c * dy, z + dz], axis=1)


def ego_lidar_to_camera(nusc: NuScenes, lidar_sd: Dict, cam_sd: Dict) -> np.ndarray:
    """4x4 transform: ego frame at the LiDAR timestamp -> camera frame (via global, at the camera timestamp)."""
    ep_l = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    ep_c = nusc.get("ego_pose", cam_sd["ego_pose_token"])
    cs_c = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    ego_l_to_global = transform_matrix(ep_l["translation"], Quaternion(ep_l["rotation"]), inverse=False)
    global_to_ego_c = transform_matrix(ep_c["translation"], Quaternion(ep_c["rotation"]), inverse=True)
    ego_c_to_cam = transform_matrix(cs_c["translation"], Quaternion(cs_c["rotation"]), inverse=True)
    return ego_c_to_cam @ global_to_ego_c @ ego_l_to_global


class BEVRenderer:
    def __init__(self, x_range, y_range, res: float = 0.1):
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.res = res
        self.H = int(round((self.x_max - self.x_min) / res))  # rows: +x (forward) points up
        self.W = int(round((self.y_max - self.y_min) / res))  # cols: +y (left) points left

    def to_px(self, xy: np.ndarray) -> np.ndarray:
        col = (self.y_max - xy[..., 1]) / self.res
        row = (self.x_max - xy[..., 0]) / self.res
        return np.stack([col, row], axis=-1)

    def _base(self, points: np.ndarray, low, high) -> np.ndarray:
        """Background + range rings + LiDAR points (brightness by height between colors low..high)."""
        img = np.full((self.H, self.W, 3), BG, dtype=np.uint8)

        # range rings every 10 m around the ego
        origin = tuple(np.round(self.to_px(np.zeros(2))).astype(int))
        for r in range(10, 80, 10):
            cv2.circle(img, origin, int(r / self.res), (38, 40, 50), 1, cv2.LINE_AA)

        keep = ((points[:, 0] >= self.x_min) & (points[:, 0] < self.x_max) &
                (points[:, 1] >= self.y_min) & (points[:, 1] < self.y_max))
        pts = points[keep]
        px = np.floor(self.to_px(pts[:, :2])).astype(int)
        t = np.clip((pts[:, 2] + 0.3) / 2.5, 0.0, 1.0)[:, None]
        low, high = np.array(low), np.array(high)
        cols = (low + t * (high - low)).astype(np.uint8)
        order = np.argsort(pts[:, 2])  # draw higher points on top
        for dr, dc in [(0, 0), (0, 1), (1, 0), (1, 1)]:  # 2x2 px splats so a single sweep doesn't look sparse
            r = np.clip(px[order, 1] + dr, 0, self.H - 1)
            c = np.clip(px[order, 0] + dc, 0, self.W - 1)
            img[r, c] = cols[order]
        return img

    def _draw_ego(self, img: np.ndarray) -> None:
        # ego vehicle (nuScenes Renault Zoe: ~4.1 m x 1.7 m, center ~1.3 m ahead of the rear axle)
        ego = np.array([1.3, 0.0, 0.0, 1.73, 4.08, 1.5, 0.0])
        ego_poly = np.round(self.to_px(box_corners_3d(ego)[:4, :2]) * 16).astype(np.int32)
        cv2.fillPoly(img, [ego_poly], (235, 235, 240), cv2.LINE_AA, shift=4)

    def render_heatmap(self, hm: np.ndarray, points: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        hm: [C, H_grid, W_grid] sigmoid heatmap indexed [c, iy, ix] (before peak NMS).
        Each pixel takes the color of its strongest class, brightness = sqrt(score) so weak activations stay visible.
        """
        img = self._base(points, (40, 44, 56), (78, 84, 100)).astype(np.float32)  # dim points for context
        # grid [c, iy, ix] -> image [c, row, col] with +x up and +y left
        hm_img = np.ascontiguousarray(hm.transpose(0, 2, 1)[:, ::-1, ::-1])
        hm_img = np.stack([cv2.resize(h, (self.W, self.H), interpolation=cv2.INTER_LINEAR) for h in hm_img])
        cls = hm_img.argmax(0)
        floor = 0.05  # background scores sit around 0.01-0.04; hide them so only real activations show
        v = np.sqrt(np.clip((hm_img.max(0) - floor) / (1 - floor), 0.0, 1.0))[..., None]
        heat = np.array(CLASS_COLORS, dtype=np.float32)[cls]
        img = np.clip(img * (1 - v) + heat * v, 0, 255).astype(np.uint8)

        # rings around the decoded peaks, to link activations to the boxes
        for box in boxes:
            c = tuple(np.round(self.to_px(box[:2]) * 16).astype(int))
            cv2.circle(img, c, 14 * 16, (245, 245, 250), 1, cv2.LINE_AA, shift=4)
        self._draw_ego(img)
        return img

    def render(self, points: np.ndarray, boxes: np.ndarray, labels: np.ndarray, gt_boxes: np.ndarray = None) -> np.ndarray:
        img = self._base(points, (80, 90, 115), (205, 228, 245))

        # predicted boxes: translucent fill + outline + heading tick
        overlay = img.copy()
        polys = []
        for box, lab in zip(boxes, labels):
            corners = self.to_px(box_corners_3d(box)[:4, :2])
            poly = np.round(corners * 16).astype(np.int32)  # 4 fractional bits for sub-pixel AA
            color = CLASS_COLORS[int(lab)]
            cv2.fillPoly(overlay, [poly], color, cv2.LINE_AA, shift=4)
            polys.append((poly, corners, color))
        img = cv2.addWeighted(overlay, 0.28, img, 0.72, 0)
        for poly, corners, color in polys:
            cv2.polylines(img, [poly], True, color, 2, cv2.LINE_AA, shift=4)
            center = corners.mean(0)
            front = corners[:2].mean(0)
            cv2.line(img, tuple(np.round(center * 16).astype(int)), tuple(np.round(front * 16).astype(int)),
                     color, 2, cv2.LINE_AA, shift=4)
        # ground truth: thin white outlines with a heading tick (drawn on top so predictions can be compared)
        for box in (gt_boxes if gt_boxes is not None else []):
            corners = self.to_px(box_corners_3d(box)[:4, :2])
            poly = np.round(corners * 16).astype(np.int32)
            cv2.polylines(img, [poly], True, (245, 245, 250), 1, cv2.LINE_AA, shift=4)
            c, f = corners.mean(0), corners[:2].mean(0)
            cv2.line(img, tuple(np.round(c * 16).astype(int)), tuple(np.round(f * 16).astype(int)), (245, 245, 250), 1, cv2.LINE_AA, shift=4)
        self._draw_ego(img)
        return img


def draw_boxes_on_camera(img: np.ndarray, boxes: np.ndarray, labels: np.ndarray,
                         T_ego_to_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    H, W = img.shape[:2]
    # far boxes first so near ones are drawn on top
    depth = [(T_ego_to_cam @ np.append(b[:3], 1.0))[2] for b in boxes]
    overlay = img.copy()
    lines = []
    for i in np.argsort(depth)[::-1]:
        corners = box_corners_3d(boxes[i])
        cam = (T_ego_to_cam @ np.concatenate([corners, np.ones((8, 1))], 1).T)[:3]  # [3,8]
        if np.any(cam[2] < 0.5):
            continue
        uv = (K @ cam)[:2] / cam[2]
        if uv[0].max() < 0 or uv[0].min() > W or uv[1].max() < 0 or uv[1].min() > H:
            continue
        uv = np.round(uv.T * 16).astype(np.int32)
        color = CLASS_COLORS[int(labels[i])]
        cv2.fillPoly(overlay, [uv[[0, 1, 5, 4]]], color, cv2.LINE_AA, shift=4)  # front face
        lines.append((uv, color))
    img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
    for uv, color in lines:
        for a, b in edges:
            cv2.line(img, tuple(uv[a]), tuple(uv[b]), color, 2, cv2.LINE_AA, shift=4)
    return img


def gt_boxes_ego(nusc: NuScenes, lidar_sd: Dict) -> np.ndarray:
    """Ground-truth boxes of the detected classes at this sweep's timestamp (devkit-interpolated between keyframes), ego frame."""
    ep = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    out = []
    for box in nusc.get_boxes(lidar_sd["token"]):
        if map_nuscenes_category_to_label(box.name) is None:
            continue
        b = transform_box_global_to_ego(box, ep)
        out.append([*b.center, *b.wlh, b.orientation.yaw_pitch_roll[0]])
    return np.array(out, dtype=np.float32).reshape(-1, 7)


def compose_frame(bev: np.ndarray, heat: np.ndarray, cam: np.ndarray, labels: np.ndarray,
                  scene_name: str, t_sec: float, split_tag: str, title: str, show_gt: bool = False) -> np.ndarray:
    """Layout: title bar, camera across the top, BEV boxes | heatmap side by side below."""
    W, gap, top = cam.shape[1], 12, 70
    side = (W - gap) // 2
    bev = cv2.resize(bev, (side, side), interpolation=cv2.INTER_AREA)
    heat = cv2.resize(heat, (side, side), interpolation=cv2.INTER_AREA)
    y_bev = top + cam.shape[0] + gap
    canvas = np.full((y_bev + side, W, 3), BG, dtype=np.uint8)
    canvas[top:top + cam.shape[0]] = cam
    canvas[y_bev:, :side] = bev
    canvas[y_bev:, side + gap:side + gap + side] = heat

    pil = Image.fromarray(canvas)
    d = ImageDraw.Draw(pil)
    f_title, f_small = load_font(30), load_font(20)
    d.text((20, 18), title, font=f_title, fill=(240, 240, 245))
    info = f"nuScenes {scene_name} ({split_tag})   t = {t_sec:4.1f} s"
    d.text((canvas.shape[1] - 20 - d.textlength(info, font=f_small), 26), info, font=f_small, fill=(150, 155, 170))

    # labels on the panels
    d.text((16, top + 12), "CAM_FRONT  ·  predicted 3D boxes projected", font=f_small, fill=(245, 245, 248))
    d.text((16, y_bev + 12), "LiDAR bird's-eye view  ·  predicted boxes (ego frame)" + ("  ·  white = ground truth" if show_gt else ""), font=f_small, fill=(200, 205, 215))
    d.text((side + gap + 16, y_bev + 12), "CenterHead heatmap  ·  sigmoid score > 0.05 (√ scaled), strongest class",
           font=f_small, fill=(200, 205, 215))

    # legend with per-class counts
    x, y = 16, canvas.shape[0] - 36
    for c, name in enumerate(CLASS_NAMES):
        n = int((labels == c).sum())
        d.rounded_rectangle((x, y + 3, x + 16, y + 19), radius=3, fill=CLASS_COLORS[c])
        txt = f"{name}  {n}"
        d.text((x + 24, y), txt, font=f_small, fill=(220, 222, 230))
        x += 24 + int(d.textlength(txt, font=f_small)) + 28
    return np.asarray(pil)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scene", default="scene-0103")
    ap.add_argument("--data_root", default="nuscenes-mini")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--out", default="outputs/demo")
    ap.add_argument("--stride", type=int, default=2, help="use every Nth LiDAR sweep (LiDAR is 20 Hz)")
    ap.add_argument("--score", type=str, default="0.2,0.3,0.3",
                    help="score threshold, one value or one per class (vehicle,pedestrian,cyclist)")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--camera_cache", default="cache/dinov2_vits14", help="cached DINOv2 features (camera-fusion models)")
    ap.add_argument("--gt", action="store_true", help="also draw ground-truth boxes (white) in the BEV panel")
    ap.add_argument("--gif_width", type=int, default=720)
    ap.add_argument("--gif_seconds", type=float, default=8.0, help="GIF length (the MP4 always has the full scene)")
    ap.add_argument("--crf", type=int, default=26, help="H.264 quality for the MP4 (lower = bigger/better)")
    args = ap.parse_args()
    thresholds = [float(v) for v in args.score.split(",")]
    thresholds = thresholds * len(CLASS_NAMES) if len(thresholds) == 1 else thresholds

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = LitPerceptionModel.load_from_checkpoint(args.ckpt, map_location=device, weights_only=False).eval()  # hparams are an OmegaConf DictConfig
    grid = model.grid

    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)
    scene = next(s for s in nusc.scene if s["name"] == args.scene)
    splits = create_splits_scenes()
    split_tag = "training scene" if args.scene in splits["mini_train"] or args.scene in splits["train"] else "held-out"
    lidar_tokens = scene_channel_tokens(nusc, scene, "LIDAR_TOP")[:: args.stride]
    if args.max_frames:
        lidar_tokens = lidar_tokens[: args.max_frames]
    cam_tokens = scene_channel_tokens(nusc, scene, "CAM_FRONT")
    cam_ts = np.array([nusc.get("sample_data", t)["timestamp"] for t in cam_tokens])

    # camera-fusion models need all 6 cameras per sweep; uncached (non-keyframe) images are encoded on the fly
    if model.use_camera:
        all_cams = {c: scene_channel_tokens(nusc, scene, c) for c in CAMERAS}
        all_cam_ts = {c: np.array([nusc.get("sample_data", t)["timestamp"] for t in all_cams[c]]) for c in CAMERAS}
        store = CameraFeatureStore(args.camera_cache, allow_compute=True)

    title = ("3D agent detection  ·  LiDAR + camera fusion (BEVFusion-style)" if model.use_camera
             else "3D agent detection  ·  PointPillars + CenterPoint-style head")
    bev_renderer = BEVRenderer((grid.x_min, grid.x_max), (grid.y_min, grid.y_max), res=0.1)
    frames_dir = os.path.join(args.out, f"{args.scene}_frames")
    shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir)

    t0 = nusc.get("sample_data", lidar_tokens[0])["timestamp"]
    for i, tok in enumerate(lidar_tokens):
        lidar_sd = nusc.get("sample_data", tok)
        points = load_lidar_points_ego(nusc, tok)

        batch = {"lidar_points": [points]}
        if model.use_camera:
            nearest = [all_cams[c][int(np.argmin(np.abs(all_cam_ts[c] - lidar_sd["timestamp"])))] for c in CAMERAS]
            cam_inputs = build_camera_inputs(nusc, tok, nearest, points.numpy(), store)
            batch.update({k: v.unsqueeze(0).to(device) for k, v in cam_inputs.items()})

        with torch.no_grad():
            bev_feat = model(batch)["bev"]
            hm_logits, reg = model.head(bev_feat)
            det = model.decode(hm_logits, reg, score_thresh=min(thresholds))[0]
        keep = det["scores"].numpy() >= np.asarray(thresholds)[det["labels"].numpy()]
        boxes = det["boxes"].numpy()[keep]
        labels = det["labels"].numpy()[keep]
        hm = torch.sigmoid(hm_logits)[0].cpu().numpy()  # [C, H_grid, W_grid]

        cam_sd = nusc.get("sample_data", cam_tokens[int(np.argmin(np.abs(cam_ts - lidar_sd["timestamp"])))])
        cam_img = cv2.cvtColor(cv2.imread(os.path.join(nusc.dataroot, cam_sd["filename"])), cv2.COLOR_BGR2RGB)
        K = np.array(nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])["camera_intrinsic"])
        cam_img = draw_boxes_on_camera(cam_img, boxes, labels, ego_lidar_to_camera(nusc, lidar_sd, cam_sd), K)

        gt = gt_boxes_ego(nusc, lidar_sd) if args.gt else None
        bev_img = bev_renderer.render(points.numpy(), boxes, labels, gt)
        heat_img = bev_renderer.render_heatmap(hm, points.numpy(), boxes)
        frame = compose_frame(bev_img, heat_img, cam_img, labels, args.scene, (lidar_sd["timestamp"] - t0) / 1e6, split_tag, title, args.gt)
        cv2.imwrite(os.path.join(frames_dir, f"{i:04d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"\rframe {i + 1}/{len(lidar_tokens)}  ({len(boxes)} boxes)", end="", flush=True)
    print()

    fps = 20 / args.stride
    pattern = os.path.join(frames_dir, "%04d.png")
    mp4 = os.path.join(args.out, f"{args.scene}.mp4")
    gif = os.path.join(args.out, f"{args.scene}.gif")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pattern,
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", str(args.crf), "-movflags", "+faststart", mp4], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pattern,
                    "-frames:v", str(int(args.gif_seconds * fps)), "-vf",
                    f"scale={args.gif_width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];"
                    f"[b][p]paletteuse=dither=bayer:bayer_scale=4", "-loop", "0", gif], check=True)
    print(f"wrote {mp4}\nwrote {gif}")


if __name__ == "__main__":
    main()
