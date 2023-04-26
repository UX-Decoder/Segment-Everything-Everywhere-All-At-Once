# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import logging
from utils.distributed import is_main_process
logger = logging.getLogger(__name__)


def align_and_update_state_dicts(model_state_dict, ckpt_state_dict):
    model_keys = sorted(model_state_dict.keys())
    ckpt_keys = sorted(ckpt_state_dict.keys())
    result_dicts = {}
    matched_log = []
    unmatched_log = []
    unloaded_log = []
    for model_key in model_keys:
        model_weight = model_state_dict[model_key]
        if model_key in ckpt_keys:
            ckpt_weight = ckpt_state_dict[model_key]
            if model_weight.shape == ckpt_weight.shape:
                result_dicts[model_key] = ckpt_weight
                ckpt_keys.pop(ckpt_keys.index(model_key))
                matched_log.append("Loaded {}, Model Shape: {} <-> Ckpt Shape: {}".format(model_key, model_weight.shape, ckpt_weight.shape))
            else:
                unmatched_log.append("*UNMATCHED* {}, Model Shape: {} <-> Ckpt Shape: {}".format(model_key, model_weight.shape, ckpt_weight.shape))
        else:
            unloaded_log.append("*UNLOADED* {}, Model Shape: {}".format(model_key, model_weight.shape))
            
    if is_main_process():
        for info in matched_log:
            logger.info(info)
        for info in unloaded_log:
            logger.warning(info)
        for key in ckpt_keys:
            logger.warning("$UNUSED$ {}, Ckpt Shape: {}".format(key, ckpt_state_dict[key].shape))
        for info in unmatched_log:
            logger.warning(info)
    return result_dicts