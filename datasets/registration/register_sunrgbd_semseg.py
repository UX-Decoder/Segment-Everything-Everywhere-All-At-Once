
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

from utils.constants import SUN_RGBD_37

__all__ = ["load_sunrgbd_instances", "register_sunrgbd_context"]

def load_sunrgbd_instances(name: str, dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]]):
    """
    Load SUN-RGBD detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    if split == 'val':
        split = 'test'
        
    # Needs to read many small annotation files. Makes sense at local
    image_pths = sorted(glob.glob(os.path.join(dirname, 'image', split, '*.jpg')))
    semseg_pths = sorted(glob.glob(os.path.join(dirname, 'label37', split, '*.png')))
    
    assert len(image_pths) == len(semseg_pths)
    
    dicts = []
    for image_dir, semseg_dir in zip(image_pths, semseg_pths):
        r = {
            "file_name": image_dir,
            "sem_seg_file_name": semseg_dir,
            "image_id": semseg_dir.split('/')[-1].split('.')[0],
        }
        dicts.append(r)
    return dicts


def register_sun_context(name, dirname, split, class_names=SUN_RGBD_37):
    DatasetCatalog.register(name, lambda: load_sunrgbd_instances(name, dirname, split, class_names))
    MetadataCatalog.get(name).set(
        stuff_classes=class_names,
        dirname=dirname,
        split=split,
        ignore_label=[0],
        thing_dataset_id_to_contiguous_id={},
        class_offset=1,
        keep_sem_bgd=False
    )


def register_all_sunrgbd_seg(root):
    SPLITS = [
            ("sunrgbd_37_val_seg", "sun_rgbd", "val"),
        ]
        
    for name, dirname, split in SPLITS:
        register_sun_context(name, os.path.join(root, dirname), split)
        MetadataCatalog.get(name).evaluator_type = "sem_seg"


_root = os.getenv("DATASET", "datasets")
register_all_sunrgbd_seg(_root)