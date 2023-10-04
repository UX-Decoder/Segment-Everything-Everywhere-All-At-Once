# --------------------------------------------------------
# SEEM -- Segment Everything Everywhere All At Once
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu), Jianwei Yang (jianwyan@microsoft.com)
# --------------------------------------------------------

import os
import warnings
import PIL
from PIL import Image
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import gradio as gr
import torch
import argparse
import whisper
import numpy as np

from gradio import processing_utils
from modeling.BaseModel import BaseModel
from modeling import build_model
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
from utils.constants import COCO_PANOPTIC_CLASSES

from demo.seem.tasks import *

def parse_option():
    parser = argparse.ArgumentParser('SEEM Demo', add_help=False)
    parser.add_argument('--conf_files', default="configs/seem/focall_unicl_lang_demo.yaml", metavar="FILE", help='path to config file', )
    cfg = parser.parse_args()
    return cfg

'''
build args
'''
cfg = parse_option()
opt = load_opt_from_config_files([cfg.conf_files])
opt = init_distributed(opt)

# META DATA
cur_model = 'None'
if 'focalt' in cfg.conf_files:
    pretrained_pth = os.path.join("seem_focalt_v2.pt")
    if not os.path.exists(pretrained_pth):
        os.system("wget {}".format("https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focalt_v2.pt"))
    cur_model = 'Focal-T'
elif 'focal' in cfg.conf_files:
    pretrained_pth = os.path.join("seem_focall_v1.pt")
    if not os.path.exists(pretrained_pth):
        os.system("wget {}".format("https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v1.pt"))
    cur_model = 'Focal-L'

'''
build model
'''
model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)

'''
audio
'''
audio = whisper.load_model("base")

@torch.no_grad()
def inference(image, task, *args, **kwargs):
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        if 'Video' in task:
            return interactive_infer_video(model, audio, image, task, *args, **kwargs)
        else:
            return interactive_infer_image(model, audio, image, task, *args, **kwargs)

class ImageMask(gr.components.Image):
    """
    Sets: source="canvas", tool="sketch"
    """

    is_template = True

    def __init__(self, **kwargs):
        super().__init__(source="upload", tool="sketch", interactive=True, **kwargs)

    def preprocess(self, x):
        return super().preprocess(x)

class Video(gr.components.Video):
    """
    Sets: source="canvas", tool="sketch"
    """

    is_template = True

    def __init__(self, **kwargs):
        super().__init__(source="upload", **kwargs)

    def preprocess(self, x):
        return super().preprocess(x)


'''
launch app
'''
title = "SEEM: Segment Everything Everywhere All At Once"
description = """
<div style="text-align: center; font-weight: bold;">
    <span style="font-size: 18px" id="paper-info">
        [<a href="https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once" target="_blank">GitHub</a>]
        [<a href="https://arxiv.org/pdf/2304.06718.pdf" target="_blank">arXiv</a>]
    </span>
</div>
<div style="text-align: left; font-weight: bold;">
    <br>
    &#x1F32A Note: The current model is run on <span style="color:blue;">SEEM {}</span>, for <span style="color:blue;">best performance</span> refer to <a href="https://huggingface.co/spaces/xdecoder/SEEM" target="_blank"><span style="color:red;">our demo</span></a>.
    </p>
</div>
""".format(cur_model)

'''Usage
Instructions:
&#x1F388 Try our default examples first (Sketch is not automatically drawed on input and example image);
&#x1F388 For video demo, it takes about 30-60s to process, please refresh if you meet an error on uploading;
&#x1F388 Upload an image/video (If you want to use referred region of another image please check "Example" and upload another image in referring image panel);
&#x1F388 Select at least one type of prompt of your choice (If you want to use referred region of another image please check "Example");
&#x1F388 Remember to provide the actual prompt for each promt type you select, otherwise you will meet an error (e.g., rember to draw on the referring image);
&#x1F388 Our model by default support the vocabulary of COCO 133 categories, others will be classified to 'others' or misclassifed.
'''

article = "The Demo is Run on SEEM-Tiny."
inputs = [ImageMask(label="[Stroke] Draw on Image",type="pil"), gr.inputs.CheckboxGroup(choices=["Stroke", "Example", "Text", "Audio", "Video", "Panoptic"], type="value", label="Interative Mode"), ImageMask(label="[Example] Draw on Referring Image",type="pil"), gr.Textbox(label="[Text] Referring Text"), gr.Audio(label="[Audio] Referring Audio", source="microphone", type="filepath"), gr.Video(label="[Video] Referring Video Segmentation",format="mp4",interactive=True)]
gr.Interface(
    fn=inference,
    inputs=inputs,
    outputs=[
        gr.outputs.Image(
        type="pil",
        label="Segmentation Results (COCO classes as label)"),
        gr.Video(
        label="Video Segmentation Results (COCO classes as label)", format="mp4"
        ),
    ],
    examples=[
    ["demo/seem/examples/corgi1.webp", ["Text"], "demo/seem/examples/corgi2.jpg", "The corgi.", None, None],
    ["demo/seem/examples/river1.png", ["Text", "Audio"], "demo/seem/examples/river2.png", "The green trees.", "demo/seem/examples/river1.wav", None],
    ["demo/seem/examples/zebras1.jpg", ["Example"], "demo/seem/examples/zebras2.jpg", "", None, None],
    ["demo/seem/examples/fries1.png", ["Example"], "demo/seem/examples/fries2.png", "", None, None],
    ["demo/seem/examples/placeholder.png", ["Video"], "demo/seem/examples/ref_vase.JPG", "", None, "demo/seem/examples/vasedeck.mp4"],
    ],
    title=title,
    description=description,
    article=article,
    allow_flagging='never',
    cache_examples=False,
).launch(share=True)
