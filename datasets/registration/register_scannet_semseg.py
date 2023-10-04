# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
import numpy as np
import os
import glob
from typing import List, Tuple, Union

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager

from utils.constants import SCAN_37, SCAN_40, SCAN_20

__all__ = ["load_scannet_instances", "register_scannet_context"]

name2folder = {"scannet_41_val_seg": "label41",
               "scannet_38_val_seg": "label38",
               "scannet_21_val_seg": "label21",}

name2class = {"scannet_41_val_seg": SCAN_40,
              "scannet_38_val_seg": SCAN_37,
              "scannet_21_val_seg": SCAN_20}


def load_scannet_instances(name: str, dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]]):
    """
    Load ScanNet annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    with PathManager.open(os.path.join(dirname, "meta", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)
        
    dicts = []
    for field in fileids:
        image_dir = os.path.join(dirname, 'images', field[0])
        semseg_dir = image_dir.replace('color', name2folder[name]).replace('jpg', 'png')
        r = {
            "file_name": image_dir,
            "sem_seg_file_name": semseg_dir,
            "image_id": semseg_dir.split('/')[-3] + semseg_dir.split('/')[-1].split('.')[0],
        }
        dicts.append(r)
    return dicts


def register_scannet_context(name, dirname, split, class_names=name2class):
    DatasetCatalog.register(name, lambda: load_scannet_instances(name, dirname, split, class_names))
    MetadataCatalog.get(name).set(
        stuff_classes=class_names[name],
        dirname=dirname,
        split=split,
        ignore_label=[0],
        thing_dataset_id_to_contiguous_id={},
        class_offset=1,
        keep_sem_bgd=False
    )


def register_all_sunrgbd_seg(root):
    SPLITS = [
            ("scannet_41_val_seg", "scannet_frames_25k", "val"),
            ("scannet_38_val_seg", "scannet_frames_25k", "val"),
            ("scannet_21_val_seg", "scannet_frames_25k", "val"),
        ]
        
    for name, dirname, split in SPLITS:
        register_scannet_context(name, os.path.join(root, dirname), split)
        MetadataCatalog.get(name).evaluator_type = "sem_seg"


_root = os.getenv("DATASET", "datasets")
register_all_sunrgbd_seg(_root)