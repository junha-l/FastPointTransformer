import os.path as osp
import logging
from typing import Optional

import gin
import numpy as np
from plyfile import PlyData
from pandas import DataFrame
import torch
import pytorch_lightning as pl

from src.data.collate import CollationFunctionFactory
import src.data.transforms as T

SCANNET_COLOR_MAP = {
    0: (0.0, 0.0, 0.0),
    1: (174.0, 199.0, 232.0),
    2: (152.0, 223.0, 138.0),
    3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0),
    5: (188.0, 189.0, 34.0),
    6: (140.0, 86.0, 75.0),
    7: (255.0, 152.0, 150.0),
    8: (214.0, 39.0, 40.0),
    9: (197.0, 176.0, 213.0),
    10: (148.0, 103.0, 189.0),
    11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0),  # No 13
    14: (247.0, 182.0, 210.0),
    15: (66.0, 188.0, 102.0),
    16: (219.0, 219.0, 141.0),
    17: (140.0, 57.0, 197.0),
    18: (202.0, 185.0, 52.0),
    19: (51.0, 176.0, 203.0),
    20: (200.0, 54.0, 131.0),
    21: (92.0, 193.0, 61.0),
    22: (78.0, 71.0, 183.0),
    23: (172.0, 114.0, 82.0),
    24: (255.0, 127.0, 14.0),
    25: (91.0, 163.0, 138.0),
    26: (153.0, 98.0, 156.0),
    27: (140.0, 153.0, 101.0),
    28: (158.0, 218.0, 229.0),
    29: (100.0, 125.0, 154.0),
    30: (178.0, 127.0, 135.0),  # No 31
    32: (146.0, 111.0, 194.0),
    33: (44.0, 160.0, 44.0),
    34: (112.0, 128.0, 144.0),
    35: (96.0, 207.0, 209.0),
    36: (227.0, 119.0, 194.0),
    37: (213.0, 92.0, 176.0),
    38: (94.0, 106.0, 211.0),
    39: (82.0, 84.0, 163.0),
    40: (100.0, 85.0, 144.0),
}
VALID_CLASS_LABELS = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    14,
    16,
    24,
    28,
    33,
    34,
    36,
    39,
)
VALID_CLASS_NAMES = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
)


def read_ply(filename):
    with open(osp.join(filename), "rb") as f:
        plydata = PlyData.read(f)
    assert plydata.elements
    data = DataFrame(plydata.elements[0].data).values
    return data


@gin.configurable
class ScanNetDatasetBase(torch.utils.data.Dataset):
    IN_CHANNELS = None
    CLASS_LABELS = None
    SPLIT_FILES = {
        "train": "scannetv2_train.txt",
        "val": "scannetv2_val.txt",
        "trainval": "scannetv2_trainval.txt",
        "test": "scannetv2_test.txt",
        "overfit": "scannetv2_overfit.txt",
    }

    def __init__(self, phase, data_root, transform=None, ignore_label=255):
        assert self.IN_CHANNELS is not None
        assert self.CLASS_LABELS is not None
        assert phase in self.SPLIT_FILES.keys()
        super(ScanNetDatasetBase, self).__init__()

        self.phase = phase
        self.data_root = data_root
        self.transform = transform
        self.ignore_label = ignore_label
        self.split_file = self.SPLIT_FILES[phase]
        self.ignore_class_labels = tuple(set(range(41)) - set(self.CLASS_LABELS))
        self.labelmap = self.get_labelmap()
        self.labelmap_inverse = self.get_labelmap_inverse()

        with open(osp.join(self.data_root, "meta_data", self.split_file), "r") as f:
            filenames = f.read().splitlines()

        sub_dir = "test" if phase == "test" else "train"
        self.filenames = [
            osp.join(self.data_root, "scannet_processed", sub_dir, f"{filename}.ply")
            for filename in filenames
        ]

    def __len__(self):
        return len(self.filenames)

    def get_classnames(self):
        classnames = {}
        for class_id in self.CLASS_LABELS:
            classnames[self.labelmap[class_id]] = VALID_CLASS_NAMES[
                VALID_CLASS_LABELS.index(class_id)
            ]
        return classnames

    def get_colormaps(self):
        colormaps = {}
        for class_id in self.CLASS_LABELS:
            colormaps[self.labelmap[class_id]] = SCANNET_COLOR_MAP[class_id]
        return colormaps

    def get_labelmap(self):
        labelmap = {}
        for k in range(41):
            if k in self.ignore_class_labels:
                labelmap[k] = self.ignore_label
            else:
                labelmap[k] = self.CLASS_LABELS.index(k)
        return labelmap

    def get_labelmap_inverse(self):
        labelmap_inverse = {}
        for k, v in self.labelmap.items():
            labelmap_inverse[v] = self.ignore_label if v == self.ignore_label else k
        return labelmap_inverse


@gin.configurable
class ScanNetRGBDataset(ScanNetDatasetBase):
    IN_CHANNELS = 3
    CLASS_LABELS = VALID_CLASS_LABELS
    NUM_CLASSES = len(VALID_CLASS_LABELS)  # 20

    def __getitem__(self, idx):
        data = self._load_data(idx)
        coords, feats, labels = self.get_cfl_from_data(data)
        if self.transform is not None:
            coords, feats, labels = self.transform(coords, feats, labels)
        coords = torch.from_numpy(coords)
        feats = torch.from_numpy(feats)
        labels = torch.from_numpy(labels)
        return coords.float(), feats.float(), labels.long(), None

    def _load_data(self, idx):
        filename = self.filenames[idx]
        data = read_ply(filename)
        return data

    def get_cfl_from_data(self, data):
        xyz, rgb, labels = data[:, :3], data[:, 3:6], data[:, -2]
        labels = np.array([self.labelmap[x] for x in labels])
        return (xyz.astype(np.float32), rgb.astype(np.float32), labels.astype(np.int64))


@gin.configurable
class PeRFceptionScannetDataset(ScanNetDatasetBase):
    IN_CHANNELS = 3
    CLASS_LABELS = VALID_CLASS_LABELS
    NUM_CLASSES = len(VALID_CLASS_LABELS)  # 20

    SPLIT_FILES = {
        "train": "scannet_256_train.txt",
        "val": "scannet_256_val.txt",
        "test": "scannet_256_val.txt",
    }

    def __init__(
        self,
        phase,
        data_root,
        transform=None,
        ignore_label=255,
        # PeRFception specific arguments
        features=["sh"],
        downsample_voxel_size=None,
        voxel_size: float = 0.02,
        ignore_void: Optional[bool] = False,
    ) -> None:
        assert self.IN_CHANNELS is not None
        assert self.CLASS_LABELS is not None
        assert phase in self.SPLIT_FILES.keys()
        # super(PeRFceptionScannetDataset, self).__init__()

        self.phase = phase
        self.data_root = data_root
        self.transform = transform
        self.ignore_label = ignore_label
        self.split_file = self.SPLIT_FILES[phase]
        self.ignore_class_labels = tuple(set(range(41)) - set(self.CLASS_LABELS))
        self.labelmap = self.get_labelmap()
        self.labelmap_inverse = self.get_labelmap_inverse()

        self.features = features
        self.voxel_size = voxel_size
        if downsample_voxel_size is None:
            self.downsample_voxel_size = voxel_size / 2
        else:
            self.downsample_voxel_size = downsample_voxel_size
        self.ignore_label = ignore_label
        self.ignore_void = ignore_void

        self.labelmap[888] = ignore_label
        self.labelmap[999] = ignore_label

        with open(osp.join(self.data_root, self.split_file), "r") as f:
            filenames = f.read().splitlines()
        self.filenames = filenames

    def _load_data(self, inst_id):
        ckpt_path = osp.join(self.data_root, f"plenoxel_scannet_{inst_id}.npz")
        ckpt = np.load(ckpt_path)

        pcd = ckpt["pcd"].astype(np.float32)
        density = ckpt["density"].astype(np.float32)
        sh = ckpt["sh"].astype(np.float32) * ckpt["sh_scale"] + ckpt["sh_min"]

        labels = ckpt["labels"]
        labels = np.array([self.labelmap[x] for x in labels])

        if self.ignore_void:
            not_void = labels <= 40
            pcd = pcd[not_void]
            density = density[not_void]
            sh = sh[not_void]
            labels = labels[not_void]
        return dict(pcd=pcd, density=density, sh=sh, labels=labels)

    def __getitem__(self, index) -> dict:
        inst_id = self.filenames[index]
        data = self._load_data(inst_id)
        coords, density, sh, labels = (
            data["pcd"],
            data["density"],
            data["sh"],
            data["labels"],
        )

        # normalize density
        if len(self.features) > 1:
            density /= np.abs(density).max() + 1e-5

        # concate feature
        feats = np.concatenate([coords, density, sh], axis=1)

        # apply transform
        if self.transform is not None:
            coords, feats, labels = self.transform(coords, feats, labels)

        density = feats[:, 3:4]
        sh = feats[:, 4:]
        ones = np.ones(density.shape)

        features = []
        for f in self.features:
            features.append(eval(f))
        features = np.concatenate(features, axis=1).astype(np.float32)

        coords = torch.from_numpy(coords)
        feats = torch.from_numpy(features)
        labels = torch.from_numpy(labels)
        return coords.float(), feats.float(), labels.long(), None


@gin.configurable
class ScanNetRGBDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root,
        train_batch_size,
        val_batch_size,
        train_num_workers,
        val_num_workers,
        collation_type,
        train_transforms,
        eval_transforms,
        perfception: bool = False,
    ):
        super(ScanNetRGBDataModule, self).__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.collate_fn = CollationFunctionFactory(collation_type)
        self.train_transforms_ = train_transforms
        self.eval_transforms_ = eval_transforms
        self.data_class = (
            PeRFceptionScannetDataset if perfception else ScanNetRGBDataset
        )

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_transforms = []
            if self.train_transforms_ is not None:
                for name in self.train_transforms_:
                    train_transforms.append(getattr(T, name)())
            train_transforms = T.Compose(train_transforms)
            self.dset_train = self.data_class("train", self.data_root, train_transforms)
        eval_transforms = []
        if self.eval_transforms_ is not None:
            for name in self.eval_transforms_:
                eval_transforms.append(getattr(T, name)())
        eval_transforms = T.Compose(eval_transforms)
        self.dset_val = self.data_class("val", self.data_root, eval_transforms)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dset_train,
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.train_num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dset_val,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.val_num_workers,
            drop_last=False,
            collate_fn=self.collate_fn,
        )


@gin.configurable
class ScanNetRGBDataset_(ScanNetRGBDataset):
    def __getitem__(self, idx):
        data, filename = self._load_data(idx)
        coords, feats, labels = self.get_cfl_from_data(data)
        if self.transform is not None:
            coords, feats, labels = self.transform(coords, feats, labels)
        coords = torch.from_numpy(coords)
        feats = torch.from_numpy(feats)
        labels = torch.from_numpy(labels)
        return coords.float(), feats.float(), labels.long(), filename

    def _load_data(self, idx):
        filename = self.filenames[idx]
        data = read_ply(filename)
        return data, filename
