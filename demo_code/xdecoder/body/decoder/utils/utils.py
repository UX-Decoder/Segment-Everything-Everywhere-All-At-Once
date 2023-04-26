import torch
import copy
from torch import nn, Tensor
import os

import math
import torch.nn.functional as F
from torch import nn


def rand_sample(x, max_len):
    if x.shape[1] <= max_len:
        return x
    else:
        rand_idx = torch.randperm(x.shape[1])[:max_len]
        return x[:,rand_idx]

def prepare_features(x, num_feature_levels, pe_layer, input_proj, level_embed):
    src = []
    pos = []
    size_list = []

    # disable mask, it does not affect performance
    for i in range(num_feature_levels):
        size_list.append(x[i].shape[-2:])
        pos.append(pe_layer(x[i], None).flatten(2))
        src.append(input_proj[i](x[i]).flatten(2) + level_embed.weight[i][None, :, None])

        # flatten NxCxHxW to HWxNxC
        pos[-1] = pos[-1].permute(2, 0, 1)
        src[-1] = src[-1].permute(2, 0, 1)
    return src, pos, size_list