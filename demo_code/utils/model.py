import logging
import os
import time
import pickle

import torch
import torch.distributed as dist

from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from fvcore.nn import flop_count_str

logger = logging.getLogger(__name__)


NORM_MODULES = [
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
    # NaiveSyncBatchNorm inherits from BatchNorm2d
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.LayerNorm,
    torch.nn.LocalResponseNorm,
]

def register_norm_module(cls):
    NORM_MODULES.append(cls)
    return cls