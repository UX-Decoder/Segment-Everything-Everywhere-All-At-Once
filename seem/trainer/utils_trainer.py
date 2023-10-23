# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

from datetime import datetime
import time
import os
import sys
import importlib
import json
import random
import logging
import numpy as np
import copy
import contextlib
import shutil
from typing import Any, Callable, Union
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from mpi4py import MPI
from infinibatch import iterators

from .distributed_trainer import DistributedTrainer
from .utils.misc import *
from .utils.serialization import JSONEncoder, filter_jsonable
from utils.distributed import get_world_size

logger = logging.getLogger(__name__)


class UtilsTrainer(DistributedTrainer):

    def __init__(self, opt):
        super().__init__(opt)

    def is_gradient_accumulation_boundary(self):
        return (self.train_params['num_updates'] + 1) % self.grad_acc_steps == 0

    def get_batch_size(self, batch, module_name='default'):
        if hasattr(self.raw_models[module_name], 'get_batch_size'):
            if callable(self.raw_models[module_name].get_batch_size):
                return self.raw_models[module_name].get_batch_size(batch)
        return {}

    def _initialize_ddp(self):
        if self.opt['FP16']:
            from torch.cuda.amp import GradScaler
            self.grad_scaler = GradScaler()
            logger.warning("PyTorch AMP GradScaler initialized.")

        for module_name in self.model_names:
            if self.opt['world_size'] > 1:
                # ddp: wrap modules for distributed data parallel training
                self.models[module_name] = nn.parallel.DistributedDataParallel(self.models[module_name],
                                                        device_ids=[self.opt['local_rank']],
                                                        output_device=self.opt['local_rank'],
                                                        find_unused_parameters=self.opt.get('FIND_UNUSED_PARAMETERS', True))

    def _get_and_validate_current_optim_steps(self):
        current_optim_steps = set([self.train_params['optim_steps'][module_name] for module_name in self.model_names])
        assert len(current_optim_steps) == 1, f"All modules should be at the same optim step: {self.train_params['optim_steps']}"
        return next(iter(current_optim_steps))

    def load_model(self, load_path):
        for module_name in self.model_names:
            self.raw_models[module_name] = self.raw_models[module_name].from_pretrained(load_path)
            self.raw_models[module_name].to(self.opt['device'])

    def save_checkpoint(self, tag):
        tag = str(tag).zfill(8)
        logger.warning('Saving checkpoint...')

        resume_epoch_idx = self.train_params['current_epoch_idx']
        resume_batch_idx = self.train_params['current_batch_idx'] + 1

        if resume_batch_idx == self.train_params['updates_per_epoch']:
            self.train_params['start_batch_idx'] = 0
            self.train_params['start_epoch_idx'] = resume_epoch_idx + 1
        else:
            self.train_params['start_batch_idx'] = resume_batch_idx
            self.train_params['start_epoch_idx'] = resume_epoch_idx
        
        save_dir = os.path.join(self.save_folder, tag)

        if self.opt['world_size'] > 1:
            torch.distributed.barrier()

        if self.opt['rank'] == 0:
            os.makedirs(self.save_folder, exist_ok=True)

        if self.opt['world_size'] > 1:
            torch.distributed.barrier()

        if self.opt['rank'] == 0:
            os.makedirs(save_dir, exist_ok=True)

        if self.opt['rank'] == 0:
            if self.opt['FP16']:
                amp_state = self.grad_scaler.state_dict()
            else:
                amp_state = None
            for module_name in self.model_names:
                module_save_dir = os.path.join(save_dir, module_name)
                os.makedirs(module_save_dir, exist_ok=True)
                save_path = os.path.join(module_save_dir, 'module_training_states.pt')
                state = {'module': self.models[module_name].state_dict(),
                            'optimizer': self.optimizers[module_name].state_dict(),
                            'lr_scheduler': self.lr_schedulers[module_name].state_dict(),
                            'amp_state': amp_state,}
                torch.save(state, save_path)

        if self.opt['rank'] == 0:
            save_path = os.path.join(save_dir, 'trainer_states.pt')
            trainer_state = {'train_loss': self.train_loss,
                                'train_params': self.train_params,}
            torch.save(trainer_state, save_path)

        num_retries = 0
        while num_retries < 3:
            try:
                random_state_path = os.path.join(save_dir, f"random_state_rank_{self.opt['rank']:04d}")
                random_state = {'random': random.getstate(),
                                'numpy_random': np.random.get_state(),
                                'torch_random': torch.get_rng_state(),
                                'torch_cuda_random': torch.cuda.get_rng_state(device=self.opt['device']) if self.opt['CUDA'] else None
                                }
                torch.save(random_state, random_state_path)
                num_retries = 3
            except Exception as err:
                num_retries += 1
                logger.warning(err)
                logger.warning("Failed to save checkpoint at retry {}, waiting for 30s to retry.".format(num_retries))
                time.sleep(30)

        if self.opt['rank'] == 0:
            for module_name in self.model_names:
                module_save_dir = os.path.join(save_dir, module_name)
                self.raw_models[module_name].save_pretrained(module_save_dir)

        if self.opt['rank'] == 0:
            # save the latest checkpoint location to json file
            checkpoint_location = {'checkpoint_tag': tag,
                                    'checkpoint_path': os.path.relpath(self.save_folder, start=self.opt['SAVE_DIR'])}
            with open(os.path.join(self.opt['SAVE_DIR'], f"resume_checkpoint.json"), 'w', encoding='utf-8') as f:
                json.dump(checkpoint_location, f, cls=JSONEncoder)

        logger.warning(f'Finished saving checkpoint and model to {save_dir}.')

    def load_weight(self, checkpoint_path=None, must_exist=False):
        self.load_model(checkpoint_path)
        logger.warning(f'Load weights from {checkpoint_path}...')

    def load_checkpoint(self, checkpoint_path=None, must_exist=False):
        logger.warning(f'Resuming checkpoint from {checkpoint_path}...')

        for model_name in self.model_names:
            model_load_path = os.path.join(checkpoint_path, model_name, 'module_training_states.pt')
            state = torch.load(model_load_path, map_location=self.opt['device'])
            
            logger.warning(f'HACK to strip module from model state dict on single gpu debugging!')
            ckpt = state['module']
            if get_world_size() <= 1:
                ckpt = {key.replace('module.',''):ckpt[key] for key in ckpt.keys()}
                
            self.models[model_name].load_state_dict(ckpt)
            self.optimizers[model_name].load_state_dict(state['optimizer'])
            self.lr_schedulers[model_name].load_state_dict(state['lr_scheduler'])
            if self.opt['FP16']:
                self.grad_scaler.load_state_dict(state['amp_state'])

        load_path = os.path.join(checkpoint_path, 'trainer_states.pt')
        trainer_state = torch.load(load_path, map_location='cpu')
        self.train_loss = trainer_state['train_loss']
        self.train_params = trainer_state['train_params']

        random_state_path = os.path.join(checkpoint_path, f"random_state_rank_{self.opt['rank']:04d}")
        if os.path.exists(random_state_path):
            random_state = torch.load(random_state_path, map_location='cpu')
            random.setstate(random_state['random'])
            np.random.set_state(random_state['numpy_random'])
            torch.set_rng_state(random_state['torch_random'])
            if self.opt['CUDA']:
                torch.cuda.set_rng_state(random_state['torch_cuda_random'], device=self.opt['device'])
        else:
            logging.warning("Could not find random state for rank {}".format(self.opt['rank']))

        logger.warning(f'Finished loading checkpoint from {checkpoint_path}.')