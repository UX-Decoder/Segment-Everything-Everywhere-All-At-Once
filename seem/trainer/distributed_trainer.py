# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import os
import logging
from mpi4py import MPI

import torch

from .utils.hook import add_hook
from .utils.mpi_adapter import MPIAdapter
from .utils.misc import save_opt_to_yaml

logger = logging.getLogger(__name__)


class DistributedTrainer:
    def __init__(self, opt):
        self.opt = opt

        # parse environment information for distributed training
        adapter = MPIAdapter(self.opt['PORT'])
        self.opt['world_size'] = adapter.world_size
        self.opt['local_size'] = adapter.local_size
        self.opt['rank'] = adapter.rank
        self.opt['local_rank'] = adapter.local_rank

        self.set_opt_hook()

        # set up device
        if not self.opt['CUDA']:
            self.opt['device'] = torch.device("cpu")
            logger.info("Using CPU")
        else:
            torch.cuda.set_device(self.opt['local_rank'])
            self.opt['device'] = torch.device("cuda", self.opt['local_rank'])
            logger.info("Using CUDA")

        # init distributed training
        adapter.log_info()
        if torch.distributed.is_available() and self.opt['world_size'] > 1:
            adapter.init_process_group(backend='nccl')

        # save config file
        self.save_folder = self.opt['SAVE_DIR']

        if self.opt['world_size'] > 1:
            torch.distributed.barrier()

        if self.opt['rank'] == 0:
            os.makedirs(self.save_folder, exist_ok=True)

            logger.info(f"Save config file to {os.path.join(self.save_folder, 'conf_copy.yaml')}")
            save_opt_to_yaml(self.opt, os.path.join(self.save_folder, 'conf_copy.yaml'))

        # ddp: log stats and update learning rate
        self.grad_acc_steps = self.opt['GRADIENT_ACCUMULATE_STEP']
        logger.info(f"Base learning rate: {self.opt['SOLVER']['BASE_LR']}")
        logger.info(f"Number of GPUs: {self.opt['world_size']}")
        logger.info(f"Gradient accumulation steps: {self.grad_acc_steps}")

        if self.opt['world_size'] > 1:
            add_hook()

        # prepare metadata for save folder
        conf_file = self.opt['conf_files'][0]
        if 'BASENAME' not in self.opt:
            self.opt['BASENAME'] = os.path.basename(conf_file)
        
        self.init_save_folder()

    def set_opt_hook(self):
        # Fill in the default values for required keywords
        self.opt['CUDA'] = self.opt.get('CUDA', True) and torch.cuda.is_available()
        self.opt['FP16'] = self.opt.get('FP16', False) and self.opt['CUDA']
        self.opt['GRADIENT_ACCUMULATE_STEP'] = int(self.opt.get('GRADIENT_ACCUMULATE_STEP', 1))
        self.opt['EVAL_PER_UPDATE_NUM'] = int(self.opt.get('EVAL_PER_UPDATE_NUM', 0))
        self.opt['LR_SCHEDULER_PARAMS'] = self.opt.get('LR_SCHEDULER_PARAMS', {})

        if 'SAVE_DIR' not in self.opt:
            assert False, "Please initialize SAVE_DIR in your config file."
        self.opt['SAVE_DIR'] = os.path.normpath(self.opt['SAVE_DIR'])
        logger.info(f"Setting SAVE_DIR as {self.opt['SAVE_DIR']}")

    def init_save_folder(self):
        """
        Initialize the save folder for logs, model, checkpoint, and evaluation.
        """
        runid = 1

        if self.opt['world_size'] > 1:
            torch.distributed.barrier()

        if self.opt['rank'] == 0:
            while True:
                save_folder = os.path.join(self.opt['SAVE_DIR'], f"{self.opt['BASENAME']}_conf~", f"run_{runid}")
                try:
                    os.makedirs(save_folder, exist_ok=False)
                    break
                except FileExistsError:
                    runid = runid + 1

        if self.opt['world_size'] > 1:
            torch.distributed.barrier()

        if self.opt['world_size'] > 1:
            runid = 1
            while True:
                save_folder = os.path.join(self.opt['SAVE_DIR'], f"{self.opt['BASENAME']}_conf~", f"run_{runid}")
                if not os.path.exists(save_folder):
                    break
                else:
                    runid += 1

            runid -= 1
            save_folder = os.path.join(self.opt['SAVE_DIR'], f"{self.opt['BASENAME']}_conf~", f"run_{runid}")
            # this second os.makedirs() call on all ranks is to force sync the save_folder creation between blobFuse and local fs
            os.makedirs(save_folder, exist_ok=True)

        self.save_folder = save_folder