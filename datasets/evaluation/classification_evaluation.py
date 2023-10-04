# Copyright (c) Facebook, Inc. and its affiliates.
# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import torch
import logging

from detectron2.evaluation.evaluator import DatasetEvaluator

from utils.misc import AverageMeter
from utils.distributed import get_world_size


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if isinstance(output, list):
        output = output[-1]

    n_classes = output.size()[1]
    maxk = min(max(topk), n_classes)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res

class ClassificationEvaluator(DatasetEvaluator):
    def __init__(self, *args):
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()
        self._logger = logging.getLogger(__name__)

    def reset(self):
        self.top1.reset()
        self.top5.reset()

    def process(self, inputs, outputs):
        logits = torch.stack([o['pred_class'] for o in outputs])
        y = torch.tensor([t['class_id'] for t in inputs], device=logits.device)
        prec1, prec5 = accuracy(logits, y, (1, 5))
        self.top1.update(prec1, y.size(0))
        self.top5.update(prec5, y.size(0))

    def evaluate(self):
        if get_world_size() > 1:
            tmp_tensor = torch.tensor(
                [self.top1.sum, self.top5.sum, self.top1.count],
                device=torch.cuda.current_device()
            )
            torch.distributed.all_reduce(
                tmp_tensor, torch.distributed.ReduceOp.SUM
            )
            top1_sum, top5_sum, count = tmp_tensor.tolist()
        else:
            top1_sum = self.top1.sum
            top5_sum = self.top5.sum
            count = self.top1.count

        results = {}
        scores = {
            'top1': top1_sum / count,
            "top5": top5_sum / count
        }
        results['class'] = scores
        self._logger.info(results)
        return results
