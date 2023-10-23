import os
import copy
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F


def rand_sample(x, divisor, max_len):
    # non_zero_pos_point = [rand_sample((m.nonzero()/divisor).t(), self.max_spatial_len[-1]).t() for m in extra['spatial_query_pos_mask']]
    if len(x.nonzero()) == 0:
        return x.nonzero().t()

    non_zero_point_index = (x.nonzero()/divisor).t()
    mask_ids = non_zero_point_index[0].unique().long()

    # compute probability for each samle
    probs = torch.zeros_like(non_zero_point_index[0])
    for idx in mask_ids:
        prob = 1./(len(mask_ids)*((non_zero_point_index[0:1]==idx).sum()))
        probs[non_zero_point_index[0]==idx] = prob
    
    indices = torch.multinomial(probs, num_samples=min(max_len, len(probs)), replacement=False).sort()[0]
    non_zero_point_index = non_zero_point_index[:,indices]
    return non_zero_point_index # [n, 512]

def rand_sample_plain(x, max_len):
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