"""
Precompute frozen DINOv2 features for the 6 cameras of every nuScenes keyframe (needed for camera fusion).

Usage:
  python cache_camera_features.py --data_root nuscenes-mini --version v1.0-mini --out cache/dinov2_vits14
"""
from __future__ import annotations
import argparse
import os

import torch
from nuscenes.nuscenes import NuScenes

from src.data.camera import CAMERAS, DinoV2Extractor, load_rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="nuscenes-mini")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--out", default="cache/dinov2_vits14")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)
    extractor = DinoV2Extractor()

    for i, sample in enumerate(nusc.sample):
        tokens = [sample["data"][c] for c in CAMERAS]
        todo = [t for t in tokens if not os.path.exists(os.path.join(args.out, f"{t}.pt"))]
        if todo:
            feats = extractor([load_rgb(nusc, t) for t in todo])
            for t, f in zip(todo, feats):
                torch.save(f.clone(), os.path.join(args.out, f"{t}.pt"))
        print(f"\rkeyframe {i + 1}/{len(nusc.sample)}", end="", flush=True)
    print(f"\ncached features for {len(nusc.sample) * len(CAMERAS)} images in {args.out}")


if __name__ == "__main__":
    main()
