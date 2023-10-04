# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
import copy
from PIL import Image
# import logging

import cv2
import numpy as np

import torch
from torchvision import transforms

from modeling.utils import configurable

__all__ = ["ImageNetDatasetMapper"]


# This is specifically designed for the COCO dataset.
class ImageNetDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        size_train=None,
        size_test=None,
        size_crop=None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.is_train = is_train
        self.size_train = size_train
        self.size_test = size_test
        self.size_crop = size_crop

        t = []
        t.append(transforms.Resize(size_crop, interpolation=Image.BICUBIC))
        t.append(transforms.CenterCrop(size_test))
        self.transform = transforms.Compose(t)
        
    @classmethod
    def from_config(cls, cfg, is_train=True):
        ret = {
            "is_train": is_train,
            "size_train": cfg['INPUT']['SIZE_TRAIN'],
            "size_test": cfg['INPUT']['SIZE_TEST'],
            "size_crop": cfg['INPUT']['SIZE_CROP']
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        file_name = dataset_dict['file_name']
        image = Image.open(file_name).convert('RGB')

        if self.is_train == False:
            image = self.transform(image)
            image = torch.from_numpy(np.asarray(image).copy())            
            image = image.permute(2,0,1)

        dataset_dict['image'] = image
        dataset_dict['height'] = image.shape[1]
        dataset_dict['width'] = image.shape[2]
        return dataset_dict