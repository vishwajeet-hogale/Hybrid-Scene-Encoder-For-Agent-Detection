from __future__ import annotations
from typing import Any, Dict, List, Optional

import lightning.pytorch as L
from torch.utils.data import DataLoader

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from src.data.nuscenes_dataset import NuScenesDataset
from src.data.collate import hybrid_collate


def build_samples_from_scenes(nusc: NuScenes, scene_names: List[str]) -> List[str]:
    """
    Returns a list of sample_tokens for all samples in the given scenes.
    """
    scene_name_set = set(scene_names)
    sample_tokens: List[str] = []

    for scene in nusc.scene:
        if scene["name"] not in scene_name_set:
            continue

        token = scene["first_sample_token"]
        while token:
            sample_tokens.append(token)
            sample = nusc.get("sample", token)
            token = sample["next"] if sample["next"] != "" else None

    return sample_tokens


class NuScenesDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        version: str,
        train_split: str,
        val_split: str,
        test_split: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        shuffle: bool,
        drop_last: bool,
        bev: Dict[str, Any],
        map: Dict[str, Any],
    ):
        super().__init__()
        self.data_root = data_root
        self.version = version

        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.bev_cfg = bev
        self.map_cfg = map

        self.nusc: Optional[NuScenes] = None
        self.ds_train: Optional[NuScenesDataset] = None
        self.ds_val: Optional[NuScenesDataset] = None
        self.ds_test: Optional[NuScenesDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Create NuScenes handle once
        if self.nusc is None:
            self.nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=False)

        splits = create_splits_scenes()

        if stage in (None, "fit"):
            train_scenes = splits[self.train_split]
            val_scenes = splits[self.val_split]

            train_tokens = build_samples_from_scenes(self.nusc, train_scenes)
            val_tokens = build_samples_from_scenes(self.nusc, val_scenes)

            self.ds_train = NuScenesDataset(self.nusc, train_tokens, split="train", bev_cfg=self.bev_cfg, map_cfg=self.map_cfg)
            self.ds_val = NuScenesDataset(self.nusc, val_tokens, split="val", bev_cfg=self.bev_cfg, map_cfg=self.map_cfg)

        if stage in (None, "test"):
            test_scenes = splits[self.test_split]
            test_tokens = build_samples_from_scenes(self.nusc, test_scenes)

            self.ds_test = NuScenesDataset(self.nusc, test_tokens, split="test", bev_cfg=self.bev_cfg, map_cfg=self.map_cfg)

    def train_dataloader(self) -> DataLoader:
        assert self.ds_train is not None
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=hybrid_collate,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.ds_val is not None
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=hybrid_collate,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.ds_test is not None
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=hybrid_collate,
        )
