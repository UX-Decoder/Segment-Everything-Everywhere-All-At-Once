# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging
import random

import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Boxes, Instances
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.data import MetadataCatalog
from pycocotools import mask

from utils.prompt_engineering import prompt_engineering
from modeling.language.misc import text_noun_with_prompt_all
from modeling.utils import configurable

__all__ = ["COCOPanopticNewBaselineDatasetMapper"]

def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    cfg_input = cfg['INPUT']
    image_size = cfg_input['IMAGE_SIZE']
    min_scale = cfg_input['MIN_SCALE']
    max_scale = cfg_input['MAX_SCALE']

    augmentation = []


    if cfg_input['RANDOM_FLIP'] != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg_input['RANDOM_FLIP'] == "horizontal",
                vertical=cfg_input['RANDOM_FLIP'] == "vertical",
            )
        )

    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])
    
    return augmentation


# This is specifically designed for the COCO dataset.
class COCOPanopticNewBaselineDatasetMapper:
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
        *,
        tfm_gens,
        image_format,
        caption_thres,
        grounding,
        max_grounding_num,
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
            "[COCOPanopticNewBaselineDatasetMapper] Full TransformGens used in training: {}".format(
                str(self.tfm_gens)
            )
        )

        self.img_format = image_format
        self.is_train = is_train
        self.caption_thres = caption_thres
        self.grounding = grounding
        self.max_grounding_num = max_grounding_num
        self.caption_similarity = torch.load(MetadataCatalog.get('logistic').get('caption_similarity_pth'))

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg['INPUT']['FORMAT'],
            "caption_thres": cfg['MODEL']['DECODER']['CAPTION']['SIM_THRES'],
            "grounding": cfg['MODEL']['DECODER']['GROUNDING']['ENABLED'],
            "max_grounding_num": cfg['MODEL']['DECODER']['GROUNDING']['MAX_LEN'],
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
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # Add caption noun that is not in coco set to target
        captions = dataset_dict["captions"]
        captions_noun = []
        for caption in captions:
            nouns = np.array(text_noun_with_prompt_all(caption, phrase_prob=0.0, append_text=False)[1])
            cap_similarity = np.array([self.caption_similarity[noun][0] for noun in nouns])
            captions_noun.append(nouns[cap_similarity < self.caption_thres].tolist())
        dataset_dict["captions_noun"] = captions_noun
        
        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            segments_info = dataset_dict["segments_info"]

            # apply the same transformation to panoptic segmentation
            pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

            from panopticapi.utils import rgb2id

            pan_seg_gt = rgb2id(pan_seg_gt)

            instances = Instances(image_shape)
            classes = []
            masks = []
            for segment_info in segments_info:
                class_id = segment_info["category_id"]
                if not segment_info["iscrowd"]:
                    classes.append(class_id)
                    masks.append(pan_seg_gt == segment_info["id"])

            is_things = [COCO_CATEGORIES[idx]['isthing'] for idx in classes]
            classes = np.array(classes)
            is_things = np.array(is_things)
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            instances.is_things = torch.tensor(is_things, dtype=torch.int64)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
                instances.gt_boxes = Boxes(torch.zeros((0, 4)))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor
                instances.gt_boxes = masks.get_bounding_boxes()

            dataset_dict["instances"] = instances

        if self.grounding:
            grounding_anno = dataset_dict['grounding_info']
            grounding_len = random.randint(1, self.max_grounding_num-1)
            if len(grounding_anno) > 0:
                masks_grd = []
                texts_grd = []
                mode = 'text'
                random.shuffle(grounding_anno)
                for ann in grounding_anno:
                    rle = mask.frPyObjects(
                        ann['segmentation'], dataset_dict['height'], dataset_dict['width'])
                    m = mask.decode(rle)
                    # sometimes there are multiple binary map (corresponding to multiple segs)
                    m = np.sum(m, axis=2)
                    m = m.astype(np.uint8)  # convert to np.uint8
                    m = transforms.apply_segmentation(m[:,:,None])[:,:,0]
                    masks_grd += [m]
                    # random select a sentence of a single annotation.
                    rand_index = random.randint(0, len(ann['sentences'])-1)
                    texts_grd += [ann['sentences'][rand_index]['raw'].lower()]
                max_len = min(grounding_len, len(texts_grd))
                indices = np.random.permutation(max_len)
                texts_grd = list(np.array(texts_grd)[indices])
                masks_grd = torch.tensor(np.stack(masks_grd)[indices])
                hash_grd = np.array([hash(txt) for txt in texts_grd])
            else:
                masks_grd = instances.gt_masks
                mode = 'class'
                if len(masks_grd) == 0:
                    masks_grd = torch.tensor([])
                    texts_grd = ['none']
                    hash_grd = np.array([hash(txt) for txt in texts_grd])
                else:
                    texts_grd = np.array([COCO_CATEGORIES[idx]['name'] for idx in classes])
                    hash_grd = np.array([hash(txt) for txt in texts_grd])
                    unique_hash_grd = np.unique(hash_grd)
                    np.random.shuffle(unique_hash_grd)
                    max_len = min(grounding_len, len(unique_hash_grd))
                    indices = np.random.permutation(max_len)                    
                    selected_unique_hash_grd = unique_hash_grd[indices]
                    selected_mask = np.in1d(hash_grd, selected_unique_hash_grd)
                    texts_grd = texts_grd[selected_mask]
                    hash_grd = hash_grd[selected_mask]
                    masks_grd = masks_grd[selected_mask]
                    texts_grd = [prompt_engineering(text.replace('-other','').replace('-merged','').replace('-stuff',''), topk=10000, suffix='.') \
                                        for text in texts_grd]
            groundings = {'masks': masks_grd, 'texts': texts_grd, 'mode': mode, 'hash': hash_grd}
            dataset_dict["groundings"] = groundings        

        return dataset_dict
