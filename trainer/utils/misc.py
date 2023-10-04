import math
import yaml
import logging
from typing import Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, decay=0):
        self.val = val
        if decay:
            alpha = math.exp(-n / decay)  # exponential decay over 100 updates
            self.sum = alpha * self.sum + (1 - alpha) * val * n
            self.count = alpha * self.count + (1 - alpha) * n
        else:
            self.sum += val * n
            self.count += n
        self.avg = self.sum / self.count


def move_batch_to_device(batch, device):
    """
    Move the batch to the device.
    It should be called before feeding the batch to the model.

    Args:
        batch (torch.tensor or container of torch.tensor): input batch
        device (torch.device): device to move the batch to
    Returns:
        return_batch: same type as the input batch with internal tensors moved to device
    """
    if torch.is_tensor(batch):
        return_batch = batch.to(device)
    elif isinstance(batch, list):
        return_batch = [move_batch_to_device(t, device) for t in batch]
    elif isinstance(batch, tuple):
        return_batch = tuple(move_batch_to_device(t, device) for t in batch)
    elif isinstance(batch, dict):
        return_batch = {}
        for k in batch:
            return_batch[k] = move_batch_to_device(batch[k], device)
    else:
        logger.debug(f"Can not move type {type(batch)} to device. Skipping it in the batch.")
        return_batch = batch

    return return_batch


def cast_batch_to_half(batch):
    """
    Cast the float32 tensors in a batch to float16.
    It should be called before feeding the batch to the FP16 DeepSpeed model.

    Args:
        batch (torch.tensor or container of torch.tensor): input batch
    Returns:
        return_batch: same type as the input batch with internal float32 tensors casted to float16
    """
    if torch.is_tensor(batch):
        if torch.is_floating_point(batch):
            return_batch = batch.to(torch.float16)
        else:
            return_batch = batch
    elif isinstance(batch, list):
        return_batch = [cast_batch_to_half(t) for t in batch]
    elif isinstance(batch, tuple):
        return_batch = tuple(cast_batch_to_half(t) for t in batch)
    elif isinstance(batch, dict):
        return_batch = {}
        for k in batch:
            return_batch[k] = cast_batch_to_half(batch[k])
    else:
        logger.debug(f"Can not cast type {type(batch)} to float16. Skipping it in the batch.")
        return_batch = batch

    return return_batch

# Adapted from https://github.com/marian-nmt/marian-dev/blob/master/src/training/exponential_smoothing.h
def apply_exponential_smoothing(avg_params: Tensor,
                                updated_params: Tensor,
                                steps: int,
                                beta: float=0.9999,  # noqa: E252
                                ref_target_words: Optional[int]=None,  # noqa: E252
                                actual_target_words: Optional[int]=None):  # noqa: E252
    r'''
        Applies exponential smoothing on a model's parameters, updating them in place.
        Can provide improved performance compared to inference using a single checkpoint.

        .. math::
            s_{t+1} = \beta \cdot s_t + (1-\beta) \cdot p_{t+1}
        where :math:`s_t` are the smoothed params (`avg_params`) at time :math:`t` and :math:`p_{t+1}` are the incoming
        updated_parameters from the most recent step (time :math:`t+1`).

        Args:
            avg_params List[Tensor]:
                Model parameters derived using the repeated average for all t < steps. Updated in-place.
            updated_params List[Tensor]:
                Model parameters from the latest update.
            steps int:
                Number of optimizer steps taken.
            beta float:
                Parameter that controls the decay speed. Default = 0.9999
            ref_target_words Optional[int]:
                Reference number of target labels expected in a batch.
            actual_target_words Optional[int]:
                The actual number of target labels in this batch.
        '''

    if ref_target_words is not None and actual_target_words is not None:
        beta = beta ** (actual_target_words / ref_target_words)
        steps = max(steps, steps * (actual_target_words / ref_target_words))  # BUG: does not account for changing batch size

    # Decay parameters more quickly at the beginning to avoid retaining the random initialization
    decay_by = min(beta, (steps + 1.) / (steps + 10))

    # Equivalent to: decay_by * avg_params + (1.0 - decay_by) * updated_params
    updated_params = updated_params.to(avg_params.dtype)
    avg_params.copy_(decay_by * (avg_params - updated_params) + updated_params)

def save_opt_to_yaml(opt, conf_file):
    with open(conf_file, 'w', encoding='utf-8') as f:
        yaml.dump(opt, f)

class LossMeter(object):
    def __init__(self):
        self.reset()

    def reset(self,):
        self.losses = {}
    
    def update_iter(self, losses):
        for key, value in losses.items():
            self.add(key, value)
    
    def add(self, name, loss):
        if name not in self.losses:
            self.losses[name] = AverageMeter()
        self.losses[name].update(loss)
    
    def get(self, name):
        if name not in self.losses:
            return 0
        return self.losses[name]