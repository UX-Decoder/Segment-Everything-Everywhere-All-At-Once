# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from timm.loss import SoftTargetCrossEntropy
from .point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..language.loss import ql_multi_contrastive_loss, image_text_contrastive_loss_queue, vl_similarity, all_gather_grad
from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list, _max_by_axis
from ..utils import box_ops

# from image2html.visualizer import VL


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, top_x_layers, losses,
                 num_points, oversample_ratio, importance_sample_ratio, grounding_weight):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.top_x_layers = top_x_layers
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        # grounding
        self.grounding_weight = grounding_weight

    def loss_labels(self, outputs, targets, indices, num_masks, layer_id, extra):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        if layer_id > self.top_x_layers['mask']:
            return {"loss_mask_ce_0": 0}

        if indices is None or len(targets) == 0:
            loss_ce = outputs['pred_logits'].sum() * 0.0
            losses = {"loss_mask_ce_0": loss_ce}
            return losses

        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].type(self.empty_weight.dtype)

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        if src_logits.shape[2] == self.num_classes+1:
            empty_weight = torch.ones(self.num_classes + 1).to(src_logits.device).type(self.empty_weight.dtype)
            empty_weight[-1] = self.eos_coef
        else:
            empty_weight = torch.ones(self.num_classes + 1000 + 1).to(src_logits.device).type(self.empty_weight.dtype)
            empty_weight[self.num_classes] = self.eos_coef

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight)
        losses = {"loss_mask_ce_0": loss_ce}
        return losses

    def loss_labels_openimage(self, outputs, targets, indices, num_masks, layer_id, extra):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        if layer_id > self.top_x_layers['mask']:
            return {"loss_openimage_ce_0": 0}

        assert "pred_captions" in outputs

        if indices is None or len(targets) == 0 or (len(targets) > 0 and len(targets[0]['labels']) == 0):
            loss_ce = outputs['pred_captions'].sum() * 0.0
            losses = {"loss_openimage_ce_0": loss_ce}
            return losses

        # compute i2t loss
        loss_openimage_ce = 0
        losses = {}
        for b in range(len(indices)):
            pred_logit = outputs["pred_logits"][b][indices[b][0]]
            gt_logit = torch.zeros_like(pred_logit)
            select_idx = torch.stack((torch.arange(len(indices[b][1])), indices[b][1])).tolist()
            gt_logit[select_idx] = 1
            loss_openimage_ce += torch.sum(-gt_logit * F.log_softmax(pred_logit, dim=-1), dim=-1).mean()
        loss_openimage_ce = loss_openimage_ce / len(indices)
        losses.update({"loss_openimage_ce_0": loss_openimage_ce})
        return losses

    def loss_itc(self, outputs, targets, indices, num_masks, layer_id, extra):
        if layer_id >= self.top_x_layers['retrieval']:
            return {"loss_retrieval_decoder_0": 0}
        t_emb = torch.cat([x['caption_proj'] for x in targets], dim=0)
        v_emb = outputs['pred_captions'][:,-1]
        loss_contrast = image_text_contrastive_loss_queue(v_emb, t_emb, extra['lang_encoder'], extra['training'])

        # compute query-token contrastive loss
        ttk_emb = torch.cat([x['caption_tokens'] for x in targets], dim=0)
        ttk_mask = torch.cat([x['caption_mask'] for x in targets], dim=0).float()
        ttk_mask = ttk_mask * torch.cumsum(ttk_mask, dim=1)
        vtk_emb = outputs['pred_captions'][:,:-1]
        keep = torch.cat([x['caption_mask'] for x in targets], dim=0).bool()

        ttk_emb = ttk_emb / (ttk_emb.norm(dim=-1, keepdim=True) + 1e-7)
        vtk_emb = vtk_emb / (vtk_emb.norm(dim=-1, keepdim=True) + 1e-7)
        logit_scale = extra['lang_encoder'].logit_scale.exp().clamp(max=100)

        # prepare gt
        gt = (torch.eye(vtk_emb.shape[0]).type_as(ttk_mask).unsqueeze(-1) * ttk_mask.unsqueeze(0).repeat(vtk_emb.shape[0], 1, 1))[:,keep].flatten(1)
        gt = gt / (gt.sum(1, keepdim=True) + 1e-7)
        # compute i2t loss
        logits = logit_scale * (vtk_emb @ ttk_emb[keep].transpose(0, 1)).mean(1)
        loss_contrast_fine_vt = SoftTargetCrossEntropy()(logits, gt)
        # loss_contrast_fine = loss_contrast_fine_vt # i2t only

        # compute t2i loss
        bs, nq, _ = vtk_emb.shape
        logits = logit_scale * (ttk_emb @ vtk_emb.flatten(0,1).transpose(0, 1)).reshape(bs,-1,bs,nq).mean(dim=-1)[keep,:]
        loss_contrast_fine_tv = SoftTargetCrossEntropy()(logits, gt.t())
        # compute loss
        loss_contrast_fine = (loss_contrast_fine_vt * 0.7 + loss_contrast_fine_tv * 0.3)

        losses = {"loss_retrieval_decoder_0": loss_contrast + loss_contrast_fine * 0.5}
        return losses

    def loss_captionings(self, outputs, targets, indices, num_masks, layer_id, extra):
        if layer_id >= self.top_x_layers['captioning']:
            return {"loss_captioning_0": 0}

        pred_captions_gen = outputs['pred_captionings'][:, :-1]
        token_embs = extra['token_embedding'].weight
        # token_embs = (token_embs / token_embs.norm(dim=-1, keepdim=True) + 1e-7)
        # pred_captions_gen = (pred_captions_gen / pred_captions_gen.norm(dim=-1, keepdim=True) + 1e-7)
        pred_captions_gen = pred_captions_gen @ token_embs.t()

        # temperature = extra['lang_encoder'].logit_scale
        # logit_scale = temperature.exp().clamp(max=100)

        target_captions_gen = torch.cat([target['caption_tokenids'] for target in targets], 0)[:, 1:]
        target_captions_gen_mask = torch.cat([target['caption_mask'] for target in targets], 0)[:, 1:]

        # loss_caption = F.cross_entropy(pred_captions_gen.transpose(1,2) * logit_scale, target_captions_gen, reduction='none')
        loss_caption = F.cross_entropy(pred_captions_gen.transpose(1,2), target_captions_gen, reduction='none')
        loss_caption = (loss_caption * target_captions_gen_mask).sum() / (target_captions_gen_mask.sum() + 1)
        losses = {"loss_captioning_0": loss_caption}
        return losses

    def loss_captions(self, outputs, targets, indices, num_masks, layer_id, extra):
        if layer_id >= self.top_x_layers['caption']:
            return {"loss_caption_0": 0}
        matched_tokens = [m[0] for m in indices]
        t_emb_class = torch.cat([extra['class_embeddings'][targets[bs]['labels'][m[1]]] for bs, m in enumerate(indices)])    
        t_hash_class = torch.cat([torch.tensor(targets[bs]['labels_hash'])[m[1]] for bs, m in enumerate(indices)])
        
        # pred_captions denotes all unmatched object queries.
        unmatched_pred_captions = []
        matched_pred_captions = []
        for idx, m in enumerate(matched_tokens):
            unmatched_masks = torch.ones(outputs['pred_captions'].shape[1:-1]).bool()
            matched_masks = torch.zeros(outputs['pred_captions'].shape[1:-1]).bool()

            unmatched_masks[m] = False
            matched_masks[m] = True

            unmatched_pred_captions.append(outputs['pred_captions'][idx][unmatched_masks])
            matched_pred_captions.append(outputs['pred_captions'][idx][matched_masks])

        outputs['unmatched_pred_captions'] = unmatched_pred_captions
        v_emb_class = torch.cat(matched_pred_captions)
        v_emb_class = v_emb_class / (v_emb_class.norm(dim=-1, keepdim=True) + 1e-7)

        indices = self.matcher(outputs, targets, mode="caption_womask", extra={'temperature':extra['lang_logit']})
        src_idx = self._get_src_permutation_idx(indices)

        t_emb = torch.cat([t['captions'][indices[bs][1]] for bs,t in enumerate(targets)])
        t_hash = torch.cat([torch.tensor(t['captions_hash'])[indices[bs][1]] for bs,t in enumerate(targets)])

        unmatched_pred_captions, _ = nested_tensor_from_tensor_list(unmatched_pred_captions).decompose()
        v_emb = unmatched_pred_captions[src_idx]
        v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)
        
        loss_contrast = ql_multi_contrastive_loss(torch.cat((v_emb, v_emb_class)), torch.cat((t_emb, t_emb_class)), torch.cat((t_hash, t_hash_class)), temperature=extra['lang_logit'])
        losses = {"loss_caption_0": loss_contrast}

        return losses

    def loss_masks(self, outputs, targets, indices, num_masks, layer_id, extra):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        if layer_id >= self.top_x_layers['mask']:
            return {"loss_mask_bce_0": 0, "loss_mask_dice_0": 0}

        assert "pred_masks" in outputs
        if indices is None or len(targets) == 0:
            loss = outputs['pred_masks'].sum() * 0.0
            losses = {"loss_mask_bce_0": loss, "loss_mask_dice_0": loss}
            return losses

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]
        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]
        
        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            ).type(src_masks.dtype)
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask_bce_0": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_mask_dice_0": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def loss_groundings(self, outputs, targets, indices, num_masks, layer_id, extra):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_gmasks" in outputs
        assert "pred_gtexts" in outputs
        
        if layer_id >= self.top_x_layers['grounding']:
            return {"loss_grounding_bce_0": 0, "loss_grounding_dice_0": 0, "loss_grounding_ce_0": 0}

        masks = [t["grounding_masks"] for t in targets]
        if indices is None or None in masks:
            loss = outputs['pred_gmasks'].sum() * 0.0
            return {"loss_grounding_bce_0": loss, "loss_grounding_dice_0": loss, "loss_grounding_ce_0": loss}

        pred_logits = []
        for b in range(len(indices)):
            t_emb = targets[b]['grounding_class_embs']
            v_emb = outputs["pred_gtexts"][b]
            
            t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
            v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)            

            out_prob = vl_similarity(v_emb, t_emb, temperature=extra['lang_logit'])
            pred_logits += [out_prob]            
        outputs['pred_logits'] = pred_logits

        indices = self.matcher(outputs, targets, mode='grounding', extra={'temperature':extra['lang_logit']})
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_gmasks"]
        src_masks = src_masks[src_idx]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]
        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]
        
        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            ).type(src_masks.dtype)
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_grounding_bce_0": sigmoid_ce_loss_jit(point_logits, point_labels, len(src_masks)),
            "loss_grounding_dice_0": dice_loss_jit(point_logits, point_labels, len(src_masks)),
        }

        # compute query-token contrastive loss
        # ttk_emb = torch.cat([x['caption_tokens'] for x in targets], dim=0)
        # ttk_mask = torch.cat([x['caption_mask'] for x in targets], dim=0).float()
        # ttk_mask = ttk_mask * torch.cumsum(ttk_mask, dim=1)
        # vtk_emb = outputs['pred_captions'][:,:-1]
        # keep = torch.cat([x['caption_mask'] for x in targets], dim=0).bool()

        # ttk_emb = ttk_emb / (ttk_emb.norm(dim=-1, keepdim=True) + 1e-7)
        # vtk_emb = vtk_emb / (vtk_emb.norm(dim=-1, keepdim=True) + 1e-7)
        # logit_scale = extra['lang_encoder'].logit_scale.exp().clamp(max=100)

        # # prepare gt
        # gt = (torch.eye(vtk_emb.shape[0]).type_as(ttk_mask).unsqueeze(-1) * ttk_mask.unsqueeze(0).repeat(vtk_emb.shape[0], 1, 1))[:,keep].flatten(1)
        # gt = gt / (gt.sum(1, keepdim=True) + 1e-7)
        # # compute i2t loss
        # logits = logit_scale * (vtk_emb @ ttk_emb[keep].transpose(0, 1)).mean(1)
        # loss_contrast_fine_vt = SoftTargetCrossEntropy()(logits, gt)
        # # loss_contrast_fine = loss_contrast_fine_vt # i2t only

        # # compute t2i loss
        # bs, nq, _ = vtk_emb.shape
        # logits = logit_scale * (ttk_emb @ vtk_emb.flatten(0,1).transpose(0, 1)).reshape(bs,-1,bs,nq).mean(dim=-1)[keep,:]
        # loss_contrast_fine_tv = SoftTargetCrossEntropy()(logits, gt.t())
        # # compute loss
        # loss_contrast_fine = (loss_contrast_fine_vt * 0.7 + loss_contrast_fine_tv * 0.3)

        # compute t2i loss
        loss_grd_ce = 0
        for b in range(len(indices)):
            task = targets[b]['grounding_task']
            pred_logit = outputs["pred_logits"][b]
            gt_logit = torch.zeros_like(pred_logit)
            select_idx = torch.stack((indices[b][0], indices[b][1])).tolist()
            gt_logit[select_idx] = 1
            t_hash = torch.tensor(targets[b]['grounding_hash'], device=gt_logit.device)
            hash_table = torch.zeros((len(t_hash), len(t_hash)), device=gt_logit.device)
            for idx in range(0, len(hash_table)):
                hash_table[idx][t_hash==t_hash[idx]] = 1
            hash_table = hash_table / hash_table.sum(-1, keepdim=True)
            gt_logit = gt_logit @ hash_table
            loss_grd_ce += self.grounding_weight[task]*torch.sum(-gt_logit.t() * F.log_softmax(pred_logit.t(), dim=-1), dim=-1).mean()
        loss_grd_ce = loss_grd_ce / len(indices)
        losses.update({"loss_grounding_ce_0": loss_grd_ce})
        del src_masks
        del target_masks
        return losses

    def loss_spatials(self, outputs, targets, indices, num_masks, layer_id, extra):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_smasks" in outputs
        assert "pred_smaskembs" in outputs

        if layer_id >= self.top_x_layers['spatial']:
            loss = outputs['pred_smasks'].sum() * 0.0
            loss_grd_ce = outputs["pred_smasks"].sum() * 0.0
            return {"loss_spatial_bce_0": loss, "loss_spatial_dice_0": loss, "loss_spatial_ce_0": loss_grd_ce}

        gt_masks = [x['gt_spatial_masks'] for x in targets]
        # compute a keep index with batch size to avoid empty gt_masks
        stack_gt_mask = torch.cat(gt_masks)
        bs,_,_ = stack_gt_mask.shape
        stack_gt_mask = stack_gt_mask.view(bs,-1).sum(dim=-1)
        keep = stack_gt_mask > 0 # only keep sample contain positive mask

        if keep.sum() == 0:
            loss = outputs['pred_smasks'].sum() * 0.0
            loss_grd_ce = outputs["pred_smasks"].sum() * 0.0
            return {"loss_spatial_bce_0": loss, "loss_spatial_dice_0": loss, "loss_spatial_ce_0": loss_grd_ce}

        # mask embedding logits
        v_emb = outputs["pred_smaskembs"] # [bs, nq, 512]

        # pos mask
        s_emb = outputs["pred_pspatials"] # [bs, ns, 512]
        pred_logits = v_emb @ s_emb.transpose(1,2)
        outputs['pred_pos_logits'] = pred_logits # [bs, nq, 1]
        indices = self.matcher(outputs, targets, mode='spatial', extra={})
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        # pos class loss
        pred_logit = torch.cat([o[:len(t['gt_spatial_masks'])] for o,t in zip(outputs["pred_pos_logits"].transpose(1,2), targets)])
        gt_logit = torch.zeros_like(pred_logit)
        gt_logit = gt_logit[keep]
        _src_idx = [torch.arange(keep.sum(), device=src_idx[0].device), src_idx[1][keep.cpu()]]
        gt_logit[_src_idx] = 1
        pred_logit = pred_logit[keep]
        loss_spa_ce_pos = torch.sum(-gt_logit * F.log_softmax(pred_logit, dim=-1), dim=-1).mean()

        # neg mask
        # s_emb = outputs["pred_nspatials"] # [bs, ns, 512]
        # neg_mask = (s_emb.sum(dim=list(range(1, len(s_emb.shape)))) != 0).float()[keep]
        # pred_logits = v_emb @ s_emb.transpose(1,2)
        # outputs['pred_neg_logits'] = pred_logits # [bs, nq, 1]
        # indices = self.matcher(outputs, targets, mode='spatial_pn', extra=extra)
        # src_idx = self._get_src_permutation_idx(indices)
        # tgt_idx = self._get_tgt_permutation_idx(indices)
        # src_masks_neg = outputs["pred_smasks"][src_idx][keep]
        # src_masks_neg = src_masks_neg*(neg_mask[:,None,None])
        # src_masks_neg = src_masks_neg.clip(0) * (-1)

        # neg class loss
        # pred_logit = outputs["pred_neg_logits"]
        # gt_logit = torch.zeros_like(pred_logit)
        # gt_logit[src_idx] = 1
        # bs,_,ns = pred_logit[keep].shape
        # pred_logit = pred_logit[keep].transpose(1,2).view(bs*ns,-1)
        # gt_logit = gt_logit[keep].transpose(1,2).view(bs*ns,-1)
        # loss_spa_ce_neg = (torch.sum(-gt_logit * F.log_softmax(pred_logit, dim=-1), dim=-1)*neg_mask).sum() / (neg_mask.sum()+1e-6)

        # recompute a keep index with matched tgt
        stack_gt_mask = nn.utils.rnn.pad_sequence(gt_masks, padding_value=-1).transpose(0,1)[tgt_idx]        
        bs,_,_ = stack_gt_mask.shape
        target_masks = stack_gt_mask
        stack_gt_mask = stack_gt_mask.view(bs,-1).sum(dim=-1)
        keep = stack_gt_mask > 0 # only keep sample contain positive mask
        src_masks_pos = outputs["pred_smasks"][src_idx][keep]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks = target_masks.to(src_masks_pos)
        target_masks = target_masks[keep]

        # mul = extra['spatial_query_mode'][keep]
        # src_masks_cur = src_masks_cur.clip(0) * mul[:,None,None]
        # src_masks_cur = src_masks_cur

        # if neg_mask[0] == 1:
        #     import cv2
        #     print(src_masks_pos.shape)
        #     print(src_masks_neg.shape)
        #     print(target_masks.shape)
        #     # import pdb; pdb.set_trace()
        #     v_pos_mask = (src_masks_pos[0].sigmoid() > 0.5).float().cpu().detach().numpy() * 255
        #     v_neg_mask = (_src_masks_neg[0].sigmoid() > 0.5).float().cpu().detach().numpy() * 255
        #     v_sum = ((src_masks_pos[0]-_src_masks_neg[0].clip(0)).sigmoid() > 0.5).float().cpu().detach().numpy() * 255
        #     v_gt = target_masks[0].float().cpu().detach().numpy() * 255

        #     cv2.imwrite('v_pos_mask.png', v_pos_mask)
        #     cv2.imwrite('v_neg_mask.png', v_neg_mask)
        #     cv2.imwrite('v_sum.png', v_sum)
        #     cv2.imwrite('v_gt.png', v_gt)
        #     import pdb; pdb.set_trace()

        # src_masks = (src_masks_pos + src_masks_neg)[:, None]
        src_masks = src_masks_pos[:, None]
        target_masks = target_masks[:, None]

        # debug visualization
        # with torch.no_grad():
        #     import cv2
        #     import numpy as np

        #     v_src_masks = (F.interpolate(src_masks, size=target_masks.shape[-2:], mode='bilinear', align_corners=False).sigmoid() > 0.5).float().cpu().numpy()[:,0] * 255
        #     v_target_masks = target_masks.float().cpu().numpy()[:,0] * 255
        #     v_masks = np.concatenate([v_src_masks, v_target_masks], axis=2)

        #     for i in range(len(src_masks)):
        #         v1 = v_src_masks[i]
        #         v2 = v_target_masks[i]
        #         v = np.concatenate([v1,v2], axis=1)
        #         cv2.imwrite('v{}.png'.format(i), v)
        #     import pdb; pdb.set_trace()

        # visualization
        # VL.step()
        # v_img = batched_inputs[0]['image'].permute(1,2,0).cpu().numpy()
        # VL.add_image(v_img[:,:,::-1])
        # candidate_masks = batched_inputs[0]['spatial_query']['rand_shape'].float().cpu().numpy()
        # gt_masks = batched_inputs[0]['spatial_query']['gt_masks'].float().cpu().numpy()
        # texts = ['cmask' for i in range(len(candidate_masks))]
        # VL.overlay_obj_mask_to_image(v_img[:,:,::-1], candidate_masks, texts)
        # texts = ['gmask' for i in range(len(candidate_masks))]
        # VL.overlay_obj_mask_to_image(v_img[:,:,::-1], gt_masks, texts)

        # import cv2
        # for i in range(len(src_masks)):
        #     visual_src_mask_cur = (src_masks_cur[i].sigmoid()>0.5).detach().float().cpu().numpy() * 255
        #     visual_src_mask_mem = (src_masks_mem[i].sigmoid()>0.5).detach().float().cpu().numpy() * 255
        #     visual_src_mask = (src_masks[i,0].sigmoid()>0.5).detach().float().cpu().numpy() * 255
        #     visual_target_mask = (target_masks[i,0].sigmoid()>0.5).detach().float().cpu().numpy() * 255

        #     cv2.imwrite('visual_src_mask_cur_{}_{}.png'.format(i, mul[i].item()), visual_src_mask_cur)
        #     cv2.imwrite('visual_src_mask_mem_{}_{}.png'.format(i, mul[i].item()), visual_src_mask_mem)
        #     cv2.imwrite('visual_src_mask_{}_{}.png'.format(i, mul[i].item()), visual_src_mask)
        #     cv2.imwrite('visual_target_mask_{}_{}.png'.format(i, mul[i].item()), visual_target_mask)
        # import pdb; pdb.set_trace()

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            ).type(src_masks.dtype)
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        num_masks = len(src_masks)
        losses = {
            "loss_spatial_bce_0": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_spatial_dice_0": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        # losses.update({"loss_spatial_ce_0": loss_spa_ce_pos + loss_spa_ce_neg})
        losses.update({"loss_spatial_ce_0": loss_spa_ce_pos})

        del src_masks
        del target_masks
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, layer_id, extra):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        if layer_id >= self.top_x_layers['box']:
            return {"loss_bbox_0": 0, "loss_giou_0": 0}

        assert 'pred_boxes' in outputs

        if indices is None or len(targets) == 0:
            loss = outputs['pred_boxes'].sum() * 0.0
            losses = {"loss_bbox_0": loss, "loss_giou_0": loss}
            return losses

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"]
        src_boxes = src_boxes[src_idx].sigmoid()
        
        target_boxes = [t['boxes'] for t in targets]
        max_size = _max_by_axis([list(box.shape) for box in target_boxes])
        max_size = [len(target_boxes)] + max_size
        empty_boxes = torch.zeros(max_size).to(src_boxes.device)
        for idx, tar_box in enumerate(target_boxes):
            empty_boxes[idx,:tar_box.shape[0],:] = tar_box
        target_boxes = empty_boxes[tgt_idx]

        # target_isthings = [t['is_things'] for t in targets]
        # max_size = _max_by_axis([list(lab.shape) for lab in target_isthings])
        # max_size = [len(target_isthings)] + max_size
        # empty_lab = torch.zeros(max_size).to(src_boxes.device)

        # for idx, tar_thing in enumerate(target_isthings):
        #     empty_lab[idx,:tar_thing.shape[0]] = tar_thing
        # target_isthings = empty_lab[tgt_idx]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox_0'] = loss_bbox.sum() / num_boxes
        
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou_0'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks, layer_id, extra):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'boxes': self.loss_boxes,
            'captions': self.loss_captions,
            'retrievals': self.loss_itc,
            'captionings': self.loss_captionings,
            'groundings': self.loss_groundings,
            'labels_openimage': self.loss_labels_openimage,
            'spatials': self.loss_spatials,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, layer_id, extra)

    def forward(self, outputs, targets, extra=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs_without_aux.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, 0, extra))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            # NOTE: we reverse the aux_outputs so that the first is the second last layer
            for i, aux_outputs in enumerate(outputs["aux_outputs"][::-1]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks, (i+1), extra)
                    l_dict = {k.replace('_0', f"_{i+1}"): v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def forward_vlp(self, outputs, targets, extra=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute all the requested losses
        losses = {}
        num_masks = indices = None
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, 0, extra))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            # NOTE: we reverse the aux_outputs so that the first is the second last layer
            for i, aux_outputs in enumerate(outputs["aux_outputs"][::-1]):
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks, (i+1), extra)
                    l_dict = {k.replace('_0', f"_{i+1}"): v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def forward_grounding(self, outputs, targets, extra=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute all the requested losses
        losses = {}
        indices = [[] for i in range(len(targets))]

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["grounding_masks"]) for t in targets) + 1e-7
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, 0, extra))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            # NOTE: we reverse the aux_outputs so that the first is the second last layer
            for i, aux_outputs in enumerate(outputs["aux_outputs"][::-1]):
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks, (i+1), extra)
                    l_dict = {k.replace('_0', f"_{i+1}"): v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def forward_openimage(self, outputs, targets, extra=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        neg_class_emb =  all_gather_grad(torch.cat([x['neg_class_emb'] for x in targets]))
        neg_hash = all_gather_grad(torch.cat([x['neg_hash'] for x in targets]))

        extra['neg_class_emb'] = neg_class_emb
        extra['neg_hash'] = neg_hash
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices, pred_logits = self.matcher.openimage_forward(outputs_without_aux, targets, extra=extra)
        outputs['pred_logits'] = pred_logits

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=neg_class_emb.device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, 0, extra))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            # NOTE: we reverse the aux_outputs so that the first is the second last layer
            for i, aux_outputs in enumerate(outputs["aux_outputs"][::-1]):
                indices, pred_logits = self.matcher.openimage_forward(aux_outputs, targets, extra=extra)
                aux_outputs['pred_logits'] = pred_logits
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks, (i+1), extra)
                    l_dict = {k.replace('_0', f"_{i+1}"): v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
