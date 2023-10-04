# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import os
import glob
from typing import List, Tuple, Union
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from scipy.io import loadmat

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager


__all__ = ["load_pascalvoc_instances", "register_pascalvoc_context"]

def get_labels_with_sizes(x):
    obj_sizes = np.bincount(x.flatten())
    labels = np.nonzero(obj_sizes)[0].tolist()
    labels = [x for x in labels if x != 0]
    return labels, obj_sizes[labels].tolist()

def load_pascalvoc_instances(name: str, dirname: str, mode: str, split: str):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    with PathManager.open(os.path.join(dirname, 'ImageSets', 'Segmentation', split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    dicts = []
    for field in fileids:
        anno_path = os.path.join(dirname, "Annotations", "{}.xml".format(field))
        image_path = os.path.join(dirname, "JPEGImages", "{}.jpg".format(field))
        inst_path = os.path.join(dirname, "SegmentationObject", "{}.png".format(field))
        semseg_path = os.path.join(dirname, "SegmentationClass", "{}.png".format(field))

        instances_mask = cv2.imread(inst_path)
        instances_mask = cv2.cvtColor(instances_mask, cv2.COLOR_BGR2GRAY).astype(np.int32)

        objects_ids = np.unique(instances_mask)
        objects_ids = [x for x in objects_ids if x != 0 and x != 220]

        slice_size = 5
        for i in range(0, len(objects_ids), slice_size):
            r = {
                "file_name": image_path,
                "inst_name": inst_path,
                "semseg_name": semseg_path,
                "objects_ids": objects_ids[i:i+slice_size],
            }
            dicts.append(r)
    return dicts

def register_pascalvoc_context(name, dirname, mode, split):
    DatasetCatalog.register("{}_{}".format(name, mode), lambda: load_pascalvoc_instances(name, dirname, mode, split))
    MetadataCatalog.get("{}_{}".format(name, mode)).set(
        dirname=dirname,
        thing_dataset_id_to_contiguous_id={},
    )

def register_all_sbd(root):
    SPLITS = [
            ("pascalvoc_val", "PascalVOC", "Point", "val"),
            ("pascalvoc_val", "PascalVOC", "Scribble", "val"),
            ("pascalvoc_val", "PascalVOC", "Polygon", "val"),
            ("pascalvoc_val", "PascalVOC", "Circle", "val"),
            ("pascalvoc_val", "PascalVOC", "Box", "val"),
        ]
        
    for name, dirname, mode, split in SPLITS:
        register_pascalvoc_context(name, os.path.join(root, dirname), mode, split)
        MetadataCatalog.get("{}_{}".format(name, mode)).evaluator_type = "interactive"

_root = os.getenv("DATASET", "datasets")
register_all_sbd(_root)