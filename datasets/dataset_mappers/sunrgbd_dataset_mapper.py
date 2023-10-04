# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
import copy

import scipy.io
import numpy as np
import torch
from PIL import Image

from torchvision import transforms
from modeling.utils import configurable

__all__ = ["SunRGBDSegDatasetMapper"]


# This is specifically designed for the COCO dataset.
class SunRGBDSegDatasetMapper:
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
        min_size_test=None,
        max_size_test=None,
        mean=None,
        std=None,
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
        self.min_size_test = min_size_test
        self.max_size_test = max_size_test
        self.pixel_mean = torch.tensor(mean)[:,None,None]
        self.pixel_std = torch.tensor(std)[:,None,None]

        t = []
        t.append(transforms.Resize(self.min_size_test, interpolation=Image.BICUBIC))
        self.transform = transforms.Compose(t)
    
    @classmethod
    def from_config(cls, cfg, is_train=True):
        ret = {
            "is_train": is_train,
            "min_size_test": cfg['INPUT']['MIN_SIZE_TEST'],
            "max_size_test": cfg['INPUT']['MAX_SIZE_TEST'],
            "mean": cfg['INPUT']['PIXEL_MEAN'],
            "std": cfg['INPUT']['PIXEL_STD'],
        }
        return ret
    
    def read_semseg(self, file_name):
        if '.png' in file_name:
            semseg = np.asarray(Image.open(file_name))
        elif '.mat' in file_name:
            semseg = scipy.io.loadmat(file_name)['LabelMap']
        return semseg

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        file_name = dataset_dict['file_name']
        semseg_name = dataset_dict['sem_seg_file_name']
        image = Image.open(file_name).convert('RGB')

        dataset_dict['width'] = image.size[0]
        dataset_dict['height'] = image.size[1]

        if self.is_train == False:
            image = self.transform(image)
            image = torch.from_numpy(np.asarray(image).copy())
            image = image.permute(2,0,1)
            
        semseg = self.read_semseg(semseg_name)
        semseg = torch.from_numpy(semseg.astype(np.int32))
        dataset_dict['image'] = image
        dataset_dict['semseg'] = semseg
        return dataset_dict