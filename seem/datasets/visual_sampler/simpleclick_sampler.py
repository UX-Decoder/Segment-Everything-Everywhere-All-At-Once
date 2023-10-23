import sys
import random

import cv2
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.contrib import distance_transform

from .point import Point
from .polygon import Polygon, get_bezier_curve
from .scribble import Scribble
from .circle import Circle

from modeling.utils import configurable


class SimpleClickSampler(nn.Module):
    @configurable
    def __init__(self, mask_mode='point', sample_negtive=False, is_train=True, dilation=None, dilation_kernel=None, max_points=None):
        super().__init__()
        self.mask_mode = mask_mode
        self.sample_negtive = sample_negtive
        self.is_train = is_train
        self.dilation = dilation
        self.register_buffer("dilation_kernel", dilation_kernel)
        self.max_points = max_points

    @classmethod
    def from_config(cls, cfg, is_train=True, mode=None):
        mask_mode = mode
        sample_negtive = cfg['STROKE_SAMPLER']['EVAL']['NEGATIVE']

        dilation = cfg['STROKE_SAMPLER']['DILATION']
        dilation_kernel = torch.ones((1, 1, dilation, dilation), device=torch.cuda.current_device())

        max_points = cfg['STROKE_SAMPLER']['POLYGON']['MAX_POINTS']

        # Build augmentation
        return {
            "mask_mode": mask_mode,
            "sample_negtive": sample_negtive,
            "is_train": is_train,
            "dilation": dilation,
            "dilation_kernel": dilation_kernel,
            "max_points": max_points,
        }

    def forward_point(self, instances, pred_masks=None, prev_masks=None):
        gt_masks = instances.gt_masks.tensor
        n,h,w = gt_masks.shape

        # We only consider positive points
        pred_masks = torch.zeros(gt_masks.shape, device=torch.cuda.current_device()).bool() if pred_masks is None else pred_masks[:,:h,:w]
        prev_masks = torch.zeros(gt_masks.shape, device=torch.cuda.current_device()).bool() if prev_masks is None else prev_masks

        if not gt_masks.is_cuda:
            gt_masks = gt_masks.to(pred_masks.device)

        fp = gt_masks & (~(gt_masks & pred_masks)) & (~prev_masks)

        # conv implementation
        mask_dt = (distance_transform((~F.pad(fp[None,], pad=(1, 1, 1, 1), mode='constant', value=0)).float())[0,:,1:-1,1:-1]).reshape(n,-1)
        max_xy_idx = torch.stack([torch.arange(n), mask_dt.max(dim=-1)[1].cpu()]).tolist()
        next_mask = torch.zeros(gt_masks.shape, device=torch.cuda.current_device()).bool()
        next_mask = next_mask.view(n,-1)

        next_mask[max_xy_idx] = True
        next_mask = next_mask.reshape((n,h,w)).float()
        next_mask = F.conv2d(next_mask[None,], self.dilation_kernel.repeat(len(next_mask),1,1,1), padding=self.dilation//2, groups=len(next_mask))[0] > 0
        # end conv implementation

        # disk implementation
        # mask_dt = distance_transform((~fp)[None,].float())[0].view(n,-1)
        # max_xy = mask_dt.max(dim=-1)[1]
        # max_y, max_x = max_xy//w, max_xy%w
        # max_xy_idx = torch.stack([max_y, max_x]).transpose(0,1)[:,:,None,None]
        # y_idx = torch.arange(start=0, end=h, step=1, dtype=torch.float32, device=torch.cuda.current_device())
        # x_idx = torch.arange(start=0, end=w, step=1, dtype=torch.float32, device=torch.cuda.current_device())
        # coord_y, coord_x = torch.meshgrid(y_idx, x_idx)
        # coords = torch.stack((coord_y, coord_x), dim=0).unsqueeze(0).repeat(len(max_xy_idx),1,1,1) # [bsx2,2,h,w], corresponding to 2d coordinate
        # coords.add_(-max_xy_idx)
        # coords.mul_(coords)
        # next_mask = coords[:, 0] + coords[:, 1]
        # next_mask = (next_mask <= 5**2)
        # end disk implementation

        rand_shapes = prev_masks | next_mask

        types = ['point' for i in range(len(gt_masks))]
        return {'gt_masks': instances.gt_masks.tensor, 'rand_shape': rand_shapes[:,None], 'types': types}

    def forward_circle(self, instances, pred_masks=None, prev_masks=None):
        gt_masks = instances.gt_masks.tensor
        n,h,w = gt_masks.shape

        # We only consider positive points
        pred_masks = torch.zeros(gt_masks.shape, device=torch.cuda.current_device()).bool() if pred_masks is None else pred_masks[:,:h,:w]
        prev_masks = torch.zeros(gt_masks.shape, device=torch.cuda.current_device()).bool() if prev_masks is None else prev_masks

        if not gt_masks.is_cuda:
            gt_masks = gt_masks.to(pred_masks.device)

        fp = gt_masks & (~(gt_masks & pred_masks)) & (~prev_masks)

        # conv implementation
        mask_dt = (distance_transform((~F.pad(fp[None,], pad=(1, 1, 1, 1), mode='constant', value=0)).float())[0,:,1:-1,1:-1]).reshape(n,-1)
        max_xy_idx = torch.stack([torch.arange(n), mask_dt.max(dim=-1)[1].cpu()]).tolist()
        next_mask = torch.zeros(gt_masks.shape, device=torch.cuda.current_device()).bool()
        next_mask = next_mask.view(n,-1)

        next_mask[max_xy_idx] = True
        next_mask = next_mask.reshape((n,h,w)).float()

        _next_mask = []
        for idx in range(len(next_mask)):
            points = next_mask[idx].nonzero().flip(dims=[-1]).cpu().numpy()
            _next_mask += [Circle.draw_by_points(points, gt_masks[idx:idx+1].cpu(), h, w)]
        next_mask = torch.cat(_next_mask, dim=0).bool().cuda()
        rand_shapes = prev_masks | next_mask

        types = ['circle' for i in range(len(gt_masks))]
        return {'gt_masks': instances.gt_masks.tensor, 'rand_shape': rand_shapes[:,None], 'types': types}

    def forward_scribble(self, instances, pred_masks=None, prev_masks=None):
        gt_masks = instances.gt_masks.tensor
        n,h,w = gt_masks.shape

        # We only consider positive points
        pred_masks = torch.zeros(gt_masks.shape, device=torch.cuda.current_device()).bool() if pred_masks is None else pred_masks[:,:h,:w]
        prev_masks = torch.zeros(gt_masks.shape, device=torch.cuda.current_device()).bool() if prev_masks is None else prev_masks

        if not gt_masks.is_cuda:
            gt_masks = gt_masks.to(pred_masks.device)

        fp = gt_masks & (~(gt_masks & pred_masks)) & (~prev_masks)

        # conv implementation
        mask_dt = (distance_transform((~F.pad(fp[None,], pad=(1, 1, 1, 1), mode='constant', value=0)).float())[0,:,1:-1,1:-1]).reshape(n,-1)
        max_xy_idx = torch.stack([torch.arange(n), mask_dt.max(dim=-1)[1].cpu()]).tolist()
        next_mask = torch.zeros(gt_masks.shape, device=torch.cuda.current_device()).bool()
        next_mask = next_mask.view(n,-1)

        next_mask[max_xy_idx] = True
        next_mask = next_mask.reshape((n,h,w)).float()

        _next_mask = []
        for idx in range(len(next_mask)):
            points = next_mask[idx].nonzero().flip(dims=[-1]).cpu().numpy()
            _next_mask += [Scribble.draw_by_points(points, gt_masks[idx:idx+1].cpu(), h, w)]
        next_mask = torch.cat(_next_mask, dim=0).bool().cuda()
        rand_shapes = prev_masks | next_mask

        types = ['scribble' for i in range(len(gt_masks))]
        return {'gt_masks': instances.gt_masks.tensor, 'rand_shape': rand_shapes[:,None], 'types': types}

    def forward_polygon(self, instances, pred_masks=None, prev_masks=None):
        gt_masks = instances.gt_masks.tensor
        gt_boxes = instances.gt_boxes.tensor
        n,h,w = gt_masks.shape

        # We only consider positive points
        pred_masks = torch.zeros(gt_masks.shape, device=torch.cuda.current_device()).bool() if pred_masks is None else pred_masks[:,:h,:w]
        prev_masks = torch.zeros(gt_masks.shape, device=torch.cuda.current_device()).bool() if prev_masks is None else prev_masks

        if not gt_masks.is_cuda:
            gt_masks = gt_masks.to(pred_masks.device)

        fp = gt_masks & (~(gt_masks & pred_masks)) & (~prev_masks)

        next_mask = []
        for i in range(len(fp)):
            rad = 0.2
            edgy = 0.05
            num_points = random.randint(1, min(self.max_points, fp[i].sum()))

            h,w = fp[i].shape
            view_mask = fp[i].reshape(h*w)
            non_zero_idx = view_mask.nonzero()[:,0]
            selected_idx = torch.randperm(len(non_zero_idx))[:num_points]
            non_zero_idx = non_zero_idx[selected_idx]
            y = (non_zero_idx // w)*1.0/(h+1)
            x = (non_zero_idx % w)*1.0/(w+1)
            coords = torch.cat((x[:,None],y[:,None]), dim=1).cpu().numpy()

            x1,y1,x2,y2 = gt_boxes[i].int().unbind()
            x,y, _ = get_bezier_curve(coords, rad=rad, edgy=edgy)
            x = x.clip(0.0, 1.0)
            y = y.clip(0.0, 1.0)
            points = torch.from_numpy(np.concatenate((y[None,]*(y2-y1-1).item(),x[None,]*(x2-x1-1).item()))).int()
            canvas = torch.zeros((y2-y1, x2-x1))
            canvas[points.long().tolist()] = 1
            rand_mask = torch.zeros(fp[i].shape)
            rand_mask[y1:y2,x1:x2] = canvas
            next_mask += [rand_mask]

        next_mask = torch.stack(next_mask).to(pred_masks.device).bool()
        rand_shapes = prev_masks | next_mask

        types = ['polygon' for i in range(len(gt_masks))]
        return {'gt_masks': instances.gt_masks.tensor, 'rand_shape': rand_shapes[:,None], 'types': types}

    def forward_box(self, instances, pred_masks=None, prev_masks=None):
        gt_masks = instances.gt_masks.tensor
        gt_boxes = instances.gt_boxes.tensor
        n,h,w = gt_masks.shape

        for i in range(len(gt_masks)):
            x1,y1,x2,y2 = gt_boxes[i].int().unbind()
            gt_masks[i,y1:y2,x1:x2] = 1

        # We only consider positive points
        pred_masks = torch.zeros(gt_masks.shape, device=torch.cuda.current_device()).bool() if pred_masks is None else pred_masks[:,:h,:w]
        prev_masks = torch.zeros(gt_masks.shape, device=torch.cuda.current_device()).bool() if prev_masks is None else prev_masks

        if not gt_masks.is_cuda:
            gt_masks = gt_masks.to(pred_masks.device)

        fp = gt_masks & (~(gt_masks & pred_masks)) & (~prev_masks)

        # conv implementation
        mask_dt = (distance_transform((~F.pad(fp[None,], pad=(1, 1, 1, 1), mode='constant', value=0)).float())[0,:,1:-1,1:-1]).reshape(n,-1)
        max_xy_idx = torch.stack([torch.arange(n), mask_dt.max(dim=-1)[1].cpu()]).tolist()
        next_mask = torch.zeros(gt_masks.shape, device=torch.cuda.current_device()).bool()
        next_mask = next_mask.view(n,-1)

        next_mask[max_xy_idx] = True
        next_mask = next_mask.reshape((n,h,w)).float()
        next_mask = F.conv2d(next_mask[None,], self.dilation_kernel.repeat(len(next_mask),1,1,1), padding=self.dilation//2, groups=len(next_mask))[0] > 0
        # end conv implementation

        rand_shapes = prev_masks | next_mask

        types = ['box' for i in range(len(gt_masks))]
        return {'gt_masks': instances.gt_masks.tensor, 'rand_shape': rand_shapes[:,None], 'types': types}

    def forward(self, instances, *args, **kwargs):
        if self.mask_mode == 'Point':
            return self.forward_point(instances, *args, **kwargs)
        elif self.mask_mode == 'Circle':
            return self.forward_circle(instances, *args, **kwargs)
        elif self.mask_mode == 'Scribble':
            return self.forward_scribble(instances, *args, **kwargs)
        elif self.mask_mode == 'Polygon':
            return self.forward_polygon(instances, *args, **kwargs)
        elif self.mask_mode == 'Box':
            return self.forward_box(instances, *args, **kwargs)

def build_shape_sampler(cfg, **kwargs):
    return ShapeSampler(cfg, **kwargs)