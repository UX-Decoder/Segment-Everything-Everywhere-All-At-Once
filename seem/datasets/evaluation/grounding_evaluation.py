# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
import logging
import torch
from torchvision.ops import box_iou

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.evaluation.evaluator import DatasetEvaluator


class GroundingEvaluator(DatasetEvaluator):
    """
    Evaluate grounding segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        compute_box=False,
        distributed=True,
    ):
        self._logger = logging.getLogger(__name__)
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._cpu_device = torch.device("cpu")
        self._compute_box = compute_box
        meta = MetadataCatalog.get(dataset_name)

    def reset(self):
        self.cum_I = 0
        self.cum_U = 0
        self.mIoU = 0
        self.eval_seg_iou_list = [.5, .6, .7, .8, .9]
        self.seg_correct = torch.zeros(len(self.eval_seg_iou_list), device=self._cpu_device)
        self.seg_total = 0
        if self._compute_box:
            self.mIoU_box = 0
            self.seg_correct_box = torch.zeros(len(self.eval_seg_iou_list), device=self._cpu_device)

    @staticmethod
    def computeIoU(pred_seg, gd_seg):
        I = (pred_seg & gd_seg)
        U = (pred_seg | gd_seg)
        return I, U

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            pred = output['grounding_mask'].sigmoid() > 0.5
            gt = input['groundings']['masks'].bool()
            bsi = len(pred)
            I, U = self.computeIoU(pred, gt)
            self.cum_I += I.sum().cpu()
            self.cum_U += U.sum().cpu()
            IoU = I.reshape(bsi,-1).sum(-1)*1.0 / (U.reshape(bsi,-1).sum(-1) + 1e-6)
            self.mIoU += IoU.sum().cpu()

            if self._compute_box:
                pred_box = BoxMode.convert(output['grounding_box'], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
                gt_box = BoxMode.convert(input['groundings']['boxes'], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS).cpu()
                IoU_box = box_iou(pred_box, gt_box).diagonal()
                self.mIoU_box += IoU_box.sum()

            for idx in range(len(self.eval_seg_iou_list)):
                eval_seg_iou = self.eval_seg_iou_list[idx]
                self.seg_correct[idx] += (IoU >= eval_seg_iou).sum().cpu()
                if self._compute_box:
                    self.seg_correct_box[idx] += (IoU_box >= eval_seg_iou).sum().cpu()
            self.seg_total += bsi

    def evaluate(self):
        if self._distributed:
            synchronize()
            self.cum_I = torch.stack(all_gather(self.cum_I)).sum()
            self.cum_U = torch.stack(all_gather(self.cum_U)).sum()
            self.mIoU = torch.stack(all_gather(self.mIoU)).sum()
            self.seg_correct = torch.stack(all_gather(self.seg_correct)).sum(0)
            self.seg_total = sum(all_gather(self.seg_total))

            if self._compute_box:
                self.mIoU_box = torch.stack(all_gather(self.mIoU_box)).sum()
                self.seg_correct_box = torch.stack(all_gather(self.seg_correct_box)).sum(0)
            if not is_main_process():
                return

        results = {}
        for idx in range(len(self.eval_seg_iou_list)):
            result_str = 'precision@{}'.format(self.eval_seg_iou_list[idx])
            results[result_str] = (self.seg_correct[idx]*100 / self.seg_total).item()
        results['cIoU'] = (self.cum_I*100./self.cum_U).item()
        results['mIoU'] = (self.mIoU*100./self.seg_total).item()

        if self._compute_box:
            for idx in range(len(self.eval_seg_iou_list)):
                result_str = 'precisionB@{}'.format(self.eval_seg_iou_list[idx])
                results[result_str] = (self.seg_correct_box[idx]*100 / self.seg_total).item()
            results['mBIoU'] = (self.mIoU_box*100./self.seg_total).item()

        self._logger.info(results)
        return {'grounding': results}