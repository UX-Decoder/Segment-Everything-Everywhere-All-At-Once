import sys
import random

import torch
import torch.nn as nn

from .point import Point
from .polygon import Polygon
from .scribble import Scribble
from .circle import Circle

from modeling.utils import configurable


class ShapeSampler(nn.Module):
    @configurable
    def __init__(self, max_candidate=1, shape_prob=[], shape_candidate=[], is_train=True):
        super().__init__()
        self.max_candidate = max_candidate
        self.shape_prob = shape_prob
        self.shape_candidate = shape_candidate
        self.is_train = is_train

    @classmethod
    def from_config(cls, cfg, is_train=True, mode=None):
        max_candidate = cfg['STROKE_SAMPLER']['MAX_CANDIDATE']
        candidate_probs = cfg['STROKE_SAMPLER']['CANDIDATE_PROBS']
        candidate_names = cfg['STROKE_SAMPLER']['CANDIDATE_NAMES']

        if mode == 'hack_train':
            candidate_classes = [getattr(sys.modules[__name__], class_name)(cfg, True) for class_name in candidate_names]        
        else:
            # overwrite condidate_prob
            if not is_train:
                candidate_probs = [0.0 for x in range(len(candidate_names))]
                candidate_probs[candidate_names.index(mode)] = 1.0
            candidate_classes = [getattr(sys.modules[__name__], class_name)(cfg, is_train) for class_name in candidate_names]

        # Build augmentation
        return {
            "max_candidate": max_candidate,
            "shape_prob": candidate_probs,
            "shape_candidate": candidate_classes,
            "is_train": is_train,
        }

    def forward(self, instances):
        masks = instances.gt_masks.tensor
        boxes = instances.gt_boxes.tensor

        if len(masks) == 0:
            gt_masks = torch.zeros(masks.shape[-2:]).bool()
            rand_masks = torch.zeros(masks.shape[-2:]).bool()
            return {'gt_masks': gt_masks[None,:], 'rand_shape': torch.stack([rand_masks]), 'types': ['none']}
        indices = [x for x in range(len(masks))]
 
        if self.is_train:
            random.shuffle(indices)
            candidate_mask = masks[indices[:self.max_candidate]]
            candidate_box = boxes[indices[:self.max_candidate]]
        else:
            candidate_mask = masks
            candidate_box = boxes
        
        draw_funcs = random.choices(self.shape_candidate, weights=self.shape_prob, k=len(candidate_mask))
        rand_shapes = [d.draw(x,y) for d,x,y in zip(draw_funcs, candidate_mask, candidate_box)]
        types = [repr(x) for x in draw_funcs]
        for i in range(0, len(rand_shapes)):
            if rand_shapes[i].sum() == 0:
                candidate_mask[i] = candidate_mask[i] * 0
                types[i] = 'none'

        # candidate_mask: (c,h,w), bool. rand_shape: (c, iter, h, w), bool. types: list(c)
        return {'gt_masks': candidate_mask, 'rand_shape': torch.stack(rand_shapes).bool(), 'types': types, 'sampler': self}

def build_shape_sampler(cfg, **kwargs):
    return ShapeSampler(cfg, **kwargs)