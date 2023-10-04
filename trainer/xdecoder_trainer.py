# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import logging
import os
import json
import random
import copy
import itertools
from typing import Any, Dict, List, Set, Union
from datetime import datetime
from mpi4py import MPI

import numpy as np
import torch
from torch.utils.data import DataLoader

from detectron2.projects.deeplab import build_lr_scheduler
from fvcore.common.config import CfgNode
from infinibatch import iterators

from utils.distributed import is_main_process, get_world_size
from .default_trainer import DefaultTrainer
from .utils.serialization import JSONEncoder, filter_jsonable

logger = logging.getLogger(__name__)


class XDecoder_Trainer(DefaultTrainer):
    """
    Construct Mask2Former_Trainer for optimizer and lr_scheduler
    """
    def create_optimizer_and_scheduler(self):
        """
        Set up self.optimizers and self.lr_schedulers

        This method initializes self.optimizers and self.lr_schedulers as dictionaries of
        instances of the classes that OPTIMIZER and LR_SCHEDULER in the config file points to.
        One optimizer and lr scheduler for each model in self.raw_models. They have the same keys
        as self.raw_models.
        """
        self.opt['init_optimizer_in_deepspeed'] = False
        self.opt['init_lr_scheduler_in_deepspeed'] = False
        
        self.optimizers = {module_name: None for module_name in self.model_names}
        self.lr_schedulers = {module_name: None for module_name in self.model_names}

        cfg_solver = self.opt['SOLVER']
        weight_decay_norm = cfg_solver['WEIGHT_DECAY_NORM']
        weight_decay_embed = cfg_solver['WEIGHT_DECAY_EMBED']
        weight_decay_bias = cfg_solver.get('WEIGHT_DECAY_BIAS', 0.0)

        defaults = {}
        defaults["lr"] = cfg_solver['BASE_LR']
        defaults["weight_decay"] = cfg_solver['WEIGHT_DECAY']

        norm_module_types = (
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
        )
        
        fix_param = self.opt['SOLVER'].get('FIX_PARAM',{})
        ignore_fix = self.opt['SOLVER'].get('IGNORE_FIX',[])
        for _module_name in self.model_names:

            flag_continue = False
            for name, param in self.raw_models[_module_name].named_parameters():
                for ig in ignore_fix:
                    if ig in name:
                        flag_continue = True
                        break

                if flag_continue:
                    flag_continue = False
                    continue

                for key, value in fix_param.items():
                    if key in name and value == True:
                        param.requires_grad = False

        lr_multiplier = self.opt['SOLVER']['LR_MULTIPLIER']

        for _module_name in self.model_names:
            # parameters = self.raw_models[module_name].get_training_parameters()
            # self.optimizers[module_name] = optimizer_class(parameters, **optimizer_parameters)
            # params = []
            # for module_param_name, value in self.raw_models[module_name].named_parameters(recurse=True):
            params: List[Dict[str, Any]] = []
            memo: Set[torch.nn.parameter.Parameter] = set()
            for module_name, module in self.raw_models[_module_name].named_modules():
                for module_param_name, value in module.named_parameters(recurse=False):
                    if not value.requires_grad:
                        continue
                    # Avoid duplicating parameters
                    if value in memo:
                        continue
                    memo.add(value)

                    hyperparams = copy.copy(defaults)

                    for key, lr_mul in lr_multiplier.items():
                        if key in "{}.{}".format(module_name, module_param_name):
                            hyperparams["lr"] = hyperparams["lr"] * lr_mul
                            if is_main_process():
                                logger.info("Modify Learning rate of {}: {}".format("{}.{}".format(module_name, module_param_name), lr_mul))

                    if (
                        "relative_position_bias_table" in module_param_name
                        or "absolute_pos_embed" in module_param_name
                    ):
                        hyperparams["weight_decay"] = 0.0
                    if isinstance(module, norm_module_types):
                        hyperparams["weight_decay"] = weight_decay_norm
                    if isinstance(module, torch.nn.Embedding):
                        hyperparams["weight_decay"] = weight_decay_embed
                    if "bias" in module_name:
                        hyperparams["weight_decay"] = weight_decay_bias
                    params.append({"params": [value], **hyperparams})

            def maybe_add_full_model_gradient_clipping(optim):
                # detectron2 doesn't have full model gradient clipping now
                clip_norm_val = cfg_solver['CLIP_GRADIENTS']['CLIP_VALUE']
                enable = (
                    cfg_solver['CLIP_GRADIENTS']['ENABLED']
                    and cfg_solver['CLIP_GRADIENTS']['CLIP_TYPE'] == "full_model"
                    and clip_norm_val > 0.0
                )

                class FullModelGradientClippingOptimizer(optim):
                    def step(self, closure=None):
                        all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                        torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                        super().step(closure=closure)

                return FullModelGradientClippingOptimizer if enable else optim

            optimizer_type = cfg_solver['OPTIMIZER']
            if optimizer_type == "SGD":
                optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                    params, cfg_solver['BASE_LR'], momentum=cfg_solver['MOMENTUM']
                )
            elif optimizer_type == "ADAMW":
                optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                    params, cfg_solver['BASE_LR']
                )
            else:
                raise NotImplementedError(f"no optimizer type {optimizer_type}")
            
            self.optimizers[_module_name] = optimizer
            self.optimizers[_module_name].zero_grad()

        num_epoch = self.opt['SOLVER']['MAX_NUM_EPOCHS']
        cfg_solver['MAX_ITER'] = num_epoch * self.train_params['updates_per_epoch']
        cfg_solver['STEPS'] = [int(x*cfg_solver['MAX_ITER']) for x in cfg_solver['STEPS']]
        logger.info(f"Calculate MAX_ITER @ {cfg_solver['MAX_ITER']} and STEPS @ {cfg_solver['STEPS']}")

        for module_name in self.model_names:
            scheduler_cfg = CfgNode({'SOLVER': cfg_solver})
            self.lr_schedulers[module_name] = build_lr_scheduler(scheduler_cfg, self.optimizers[module_name])

        for module_name in self.model_names:
            num_params = 0
            num_trainable_params = 0
            for name, param in self.raw_models[module_name].named_parameters():
                num_params += param.numel()
                if param.requires_grad:
                    num_trainable_params += param.numel()
            logger.info(f"Total number of parameters in {module_name} module (on each GPU): {num_params}")
            logger.info(f"Number of trainable parameters in {module_name} module (on each GPU): {num_trainable_params}")