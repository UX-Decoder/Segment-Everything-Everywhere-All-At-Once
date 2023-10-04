# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu), Ziyi Dou (zdou@cs.ucla.edu)
# --------------------------------------------------------
import copy
import itertools
import logging
from collections import OrderedDict
import torch
from pycocotools.cocoeval import COCOeval

import detectron2.utils.comm as comm
from detectron2.evaluation.evaluator import DatasetEvaluator

try:
    from detectron2.evaluation.fast_eval_api import COCOeval_opt
except ImportError:
    COCOeval_opt = COCOeval


class RetrievalEvaluator(DatasetEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.
    The metrics range from 0 to 100 (instead of 0 to 1), where a -1 or NaN means
    the metric cannot be computed (e.g. due to no predictions made).
    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def __init__(
        self,
        dataset_name=None,
        output_dir=None,
        ensemble=False,
        distributed=True,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:
                    "json_file": the path to the COCO format annotation
                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "bbox", "segm", "keypoints".
                By default, will infer this automatically from predictions.
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:
                1. "instances_predictions.pth" a file that can be loaded with `torch.load` and
                   contains all the results in the format they are produced by the model.
                2. "coco_instances_results.json" a json file in COCO's result format.
            max_dets_per_image (int): limit on the maximum number of detections per image.
                By default in COCO, this limit is to 100, but this can be customized
                to be greater, as is needed in evaluation metrics AP fixed and AP pool
                (see https://arxiv.org/pdf/2102.01066.pdf)
                This doesn't affect keypoint evaluation.
            use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
                Although the results should be very close to the official implementation in COCO
                API, it is still recommended to compute results with the official API for use in
                papers. The faster implementation also uses more RAM.
            kpt_oks_sigmas (list[float]): The sigmas used to calculate keypoint OKS.
                See http://cocodataset.org/#keypoints-eval
                When empty, it will use the defaults in COCO.
                Otherwise it should be the same length as ROI_KEYPOINT_HEAD.NUM_KEYPOINTS.
            allow_cached_coco (bool): Whether to use cached coco json from previous validation
                runs. You should set this to False if you need to use different validation data.
                Defaults to True.
        """
        self._logger = logging.getLogger(__name__)
        self._dataset_name = dataset_name
        self._output_dir = output_dir
        self._ensemble = ensemble
        self._distributed = distributed

        if 'p2i' in dataset_name:
            self.mode = 'patch2image'
        elif 'interactive2i' in dataset_name:
            self.mode = 'interactive2image'
        else:
            self.mode = 'default'

    def reset(self):
        self._text_embeds = []
        self._image_embeds = []
        self._image_embeds2 = []
        self._text_ids = []
        self._image_ids = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for output in outputs:
            self._text_ids.extend(output['caption']['caption_ids'])
            self._image_ids.append(output['caption']['image_ids'])
            self._text_embeds.append(output['caption']['text_embeds'])
            self._image_embeds.append(output['caption']['image_embeds'][0])
            if self._ensemble:
                self._image_embeds2.append(output['caption']['image_embeds'][1])

    def evaluate(self, img_ids=None):
        if self.mode == 'default':
            return self.evaluate_default(img_ids)
        elif self.mode in ['patch2image', 'interactive2image']:
            return self.evaluate_p2i(img_ids)
        else:
            assert False, "Unknown mode for retrieval evaluation"

    def evaluate_default(self, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """

        if self._distributed:
            comm.synchronize()
            def gather(x, move=False):
                x = comm.gather(x)
                x = list(itertools.chain(*x))
                if move:
                    x = [xx.to(self._text_embeds[0].device) for xx in x]
                return x
            text_embeds = gather(self._text_embeds, move=True)
            image_embeds = gather(self._image_embeds, move=True)
            if self._ensemble:
                image_embeds2 = gather(self._image_embeds2, move=True)
            text_ids = gather(self._text_ids)
            image_ids = gather(self._image_ids)
            if not comm.is_main_process():
                return {}
        else:
            text_embeds = self._text_embeds
            image_embeds = self._image_embeds
            if self._ensemble:
                image_embeds2 = self._image_embeds2
            text_ids = self._text_ids
            image_ids = self._image_ids
        if len(text_embeds) == 0:
            self._logger.warning("[COCOCaptionEvaluator] Did not receive valid predictions.")
            return {}
        iids = torch.tensor(image_ids).view(-1).cuda()
        tiids = torch.tensor(text_ids).view(-1).cuda()
        image_embeds = torch.cat(image_embeds)
        text_embeds = torch.cat(text_embeds)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True) 
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True) 
        scores = image_embeds @ text_embeds.t()

        if self._ensemble:
            image_embeds2 = torch.cat(image_embeds2)
            image_embeds2 = image_embeds2 / image_embeds2.norm(dim=-1, keepdim=True) 
            scores2 = image_embeds2 @ text_embeds.t()
            scores = scores2 * 0.5 + scores * 0.5

        topk10 = scores.topk(10, dim=1)
        topk5 = scores.topk(5, dim=1)
        topk1 = scores.topk(1, dim=1)
        topk10_iids = tiids[topk10.indices]
        topk5_iids = tiids[topk5.indices]
        topk1_iids = tiids[topk1.indices]
        tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
        tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
        tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()
        topk10 = scores.topk(10, dim=0)
        topk5 = scores.topk(5, dim=0)
        topk1 = scores.topk(1, dim=0)
        topk10_iids = iids[topk10.indices]
        topk5_iids = iids[topk5.indices]
        topk1_iids = iids[topk1.indices]
        ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
        ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
        ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()
        self._results = OrderedDict()
        # Copy so the caller can do whatever with results
        self._results['recall'] = {}
        self._results['recall']['irtr'] = float("{:.3f}".format((ir_r1 + tr_r1).item() * 100))
        self._results['recall']['ir1'] = float("{:.3f}".format(ir_r1.item() * 100))
        self._results['recall']['ir5'] = float("{:.3f}".format(ir_r5.item() * 100))
        self._results['recall']['ir10'] = float("{:.3f}".format(ir_r10.item() * 100))
        self._results['recall']['tr1'] = float("{:.3f}".format(tr_r1.item() * 100))
        self._results['recall']['tr5'] = float("{:.3f}".format(tr_r5.item() * 100))
        self._results['recall']['tr10'] = float("{:.3f}".format(tr_r10.item() * 100))
        self._logger.info(self._results)
        return copy.deepcopy(self._results)

    def evaluate_p2i(self, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """

        if self._distributed:
            comm.synchronize()
            def gather(x, move=False):
                x = comm.gather(x)
                x = list(itertools.chain(*x))
                if move:
                    x = [xx.to(self._text_embeds[0].device) for xx in x]
                return x
            text_embeds = gather(self._text_embeds, move=True)
            image_embeds = gather(self._image_embeds, move=True)
            image_embeds2 = gather(self._image_embeds2, move=True)
            text_ids = gather(self._text_ids)
            image_ids = gather(self._image_ids)
            if not comm.is_main_process():
                return {}
        else:
            text_embeds = self._text_embeds
            image_embeds = self._image_embeds
            image_embeds2 = self._image_embeds2
            text_ids = self._text_ids
            image_ids = self._image_ids

        if len(text_embeds) == 0:
            self._logger.warning("[COCOCaptionEvaluator] Did not receive valid predictions.")
            return {}

        iids = torch.tensor(image_ids).view(-1).cuda()
        tiids = torch.tensor(text_ids).view(-1).cuda()
        image_embeds = torch.cat(image_embeds)
        text_embeds = torch.cat(text_embeds)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True) 
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True) 

        image_embeds2 = torch.cat(image_embeds2)
        image_embeds2 = image_embeds2 / image_embeds2.norm(dim=-1, keepdim=True)

        # compute image to image retrieval
        self._results = OrderedDict()
        self._results['recall'] = {}
        ii_scores = image_embeds2 @ image_embeds.t()

        topk10 = ii_scores.topk(10, dim=1)
        topk5 = ii_scores.topk(5, dim=1)
        topk1 = ii_scores.topk(1, dim=1)
        topk10_iids = iids[topk10.indices]
        topk5_iids = iids[topk5.indices]
        topk1_iids = iids[topk1.indices]
        iir_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
        iir_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
        iir_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()
        # Copy so the caller can do whatever with results
        self._results['recall']['p2ir1'] = float("{:.3f}".format(iir_r1.item() * 100))
        self._results['recall']['p2ir5'] = float("{:.3f}".format(iir_r5.item() * 100))
        self._results['recall']['p2ir10'] = float("{:.3f}".format(iir_r10.item() * 100))
        self._logger.info(self._results)
        return copy.deepcopy(self._results)