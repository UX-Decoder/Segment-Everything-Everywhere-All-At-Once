# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import io
from PIL import Image
import numpy as np

import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog

from modeling.language.LangEncoder import build_tokenizer
from modeling.utils import configurable


__all__ = ["VLPreDatasetMapper"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    # The scope of vlp dataset may not need any augmentation.
    cfg_input = cfg['INPUT']
    image_size = cfg_input['IMAGE_SIZE']
    augmentation = []

    augmentation.extend([
        T.Resize((image_size, image_size)),
    ])
    
    return augmentation


# This is specifically designed for the COCO dataset.
class VLPreDatasetMapper:
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
        dataset_name=None,
        *,
        tfm_gens,
        image_format,
        tokenizer=None,
        max_token_num=None,
        device=None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            crop_gen: crop augmentation
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[PretrainDatasetMapper] Full TransformGens used in training: {}".format(
                str(self.tfm_gens)
            )
        )

        self.img_format = image_format
        self.is_train = is_train

        self.all_arrows = MetadataCatalog.get(dataset_name).arrows

        self.tokenizer = tokenizer
        self.max_token_num = max_token_num
        self.device = device

    @classmethod
    def from_config(cls, cfg, is_train=True, dataset_name=None):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        tokenizer = build_tokenizer(cfg['MODEL']['TEXT'])
        max_token_num = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']
        device = cfg['device']

        ret = {
            "is_train": is_train,
            "dataset_name": dataset_name,
            "tfm_gens": tfm_gens,
            "image_format": cfg['INPUT']['FORMAT'],
            "tokenizer": tokenizer,
            "max_token_num": max_token_num,
            "device": device,
        }
        return ret

    def get_image(self, inp):
        image_bytes = io.BytesIO(inp)
        image_bytes.seek(0)
        return Image.open(image_bytes)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        arr = self.all_arrows[dataset_dict['arr_id']]
        cur_id = dataset_dict['cur_id']
        image = self.get_image(arr['image'][cur_id].as_py())

        image = utils._apply_exif_orientation(image)
        image = utils.convert_PIL_to_numpy(image, self.img_format)
        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        captions = dataset_dict['captions']
        tokens = self.tokenizer(
            captions, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
        )
        dataset_dict['tokens'] = {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]}
        return dataset_dict