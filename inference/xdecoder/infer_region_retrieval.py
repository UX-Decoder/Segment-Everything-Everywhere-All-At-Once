# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import os
import sys
import logging

pth = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, pth)

from PIL import Image
import numpy as np
np.random.seed(0)
import cv2

import torch
from torchvision import transforms

from utils.arguments import load_opt_command

from detectron2.data import MetadataCatalog
from modeling.language.loss import vl_similarity
from modeling.BaseModel import BaseModel
from modeling import build_model
from utils.visualizer import Visualizer
from utils.distributed import init_distributed

logger = logging.getLogger(__name__)


def main(args=None):
    '''
    Main execution point for PyLearn.
    '''
    opt, cmdline_args = load_opt_command(args)
    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['base_path'] = absolute_user_dir
    opt = init_distributed(opt)

    # META DATA
    pretrained_pth = os.path.join(opt['RESUME_FROM'])
    output_root = './output'

    image_list = ['inference/images/coco/000.jpg', 'inference/images/coco/001.jpg', 'inference/images/coco/002.jpg', 'inference/images/coco/003.jpg']
    text = ['pizza on the plate']

    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(["background", "background"], is_eval=False)

    t = []
    t.append(transforms.Resize(224, interpolation=Image.BICUBIC))
    transform_ret = transforms.Compose(t)
    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform_grd = transforms.Compose(t)

    metadata = MetadataCatalog.get('ade20k_panoptic_train')
    color = [0/255, 255/255, 0/255]

    with torch.no_grad():
        batch_inputs = []
        candidate_list = []
        for j in range(len(image_list)):
            image_ori = Image.open(image_list[j])
            width = image_ori.size[0]
            height = image_ori.size[1]

            image = transform_ret(image_ori)
            image = np.asarray(image)
            candidate_list += [image]
            image_list += [image]
            image_ori = np.asarray(image_ori)
            images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()
            batch_inputs += [{'image': images, 'height': height, 'width': width}]

        outputs = model.model.evaluate(batch_inputs)
        v_emb = torch.cat([x['captions'][-1:] for x in outputs])
        v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)

        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(text, is_eval=False, name='caption', prompt=False)
        t_emb = getattr(model.model.sem_seg_head.predictor.lang_encoder, '{}_text_embeddings'.format('caption'))
        temperature = model.model.sem_seg_head.predictor.lang_encoder.logit_scale
        logits = vl_similarity(v_emb, t_emb, temperature)
        max_prob, max_id = logits.softmax(0).max(dim=0)
        
        frame_pth = image_list[max_id.item()]
        image_ori = Image.open(frame_pth)
        width = image_ori.size[0]
        height = image_ori.size[1]

        image = transform_grd(image_ori)
        image = np.asarray(image)
        image_ori = np.asarray(image_ori)
        images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()
        batch_inputs = [{'image': images, 'height': height, 'width': width, 'groundings': {'texts': [text]}}]
        outputs = model.model.evaluate_grounding(batch_inputs, None)

        visual = Visualizer(image_ori, metadata=metadata)
        grd_masks = (outputs[0]['grounding_mask'] > 0).float().cpu().numpy()
        for text_, mask in zip(text, grd_masks):
            demo = visual.draw_binary_mask(mask, color=color, text='', alpha=0.5)
        region_img = demo.get_image()
        candidate_list[max_id.item()] = region_img
        out_image = np.zeros((224*4+60, 448*4, 3))
        for ii, img in enumerate(candidate_list):
            img = cv2.resize(img, (448, 224))
            if ii != max_id.item():
                img = img * 0.4
            hs, ws = 60+(ii//4)*224, (ii%4)*448
            out_image[hs:hs+224,ws:ws+448,:] = img[:,:,::-1]

        font                   = cv2.FONT_HERSHEY_DUPLEX
        fontScale              = 1.2
        thickness              = 3
        lineType               = 2
        bottomLeftCornerOfText = (10, 40)
        fontColor              = [255,255,255]
        cv2.putText(out_image, text[0],
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

        x1 = (max_id.item()%4) * 448
        y1 = (max_id.item()//4) * 224 + 60
        cv2.rectangle(out_image, (x1, y1), (x1+448, y1+224), (0,0,255), 3)

        x1 = x1
        y1 = y1 + 224 - 30
        cv2.rectangle(out_image, (x1+2, y1), (x1+60, y1+28), (0,0,0), -1)

        fontScale              = 1.0
        thickness              = 2

        bottomLeftCornerOfText = (x1, y1+21)
        cv2.putText(out_image, str(max_prob.item())[0:4],
            bottomLeftCornerOfText,
            font,
            0.8,
            [0,0,255],
            thickness,
            lineType)

        if not os.path.exists(output_root):
            os.makedirs(output_root)
        cv2.imwrite(os.path.join(output_root, 'region_retrieval.png'), out_image)


if __name__ == "__main__":
    main()
    sys.exit(0)