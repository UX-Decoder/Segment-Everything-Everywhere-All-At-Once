# Installation Guide

**General Environment**
* Linux System
* CUDA enabled GPU with Memory > 8GB (Evaluation)
* CUDA enabled GPU with Memory > 12GB (Training)

**Installation**

```sh
# Python Package Installation
pip install -r assets/requirements/requirements.txt
pip install -r assets/requirements/requirements_custom.txt

# Customer Operator [only need training deformable vision encoder]
cd modeling/vision/encoder/ops && sh make.sh && cd ../../../../

# System Package [only need for demo in SEEM]
sudo apt update
sudo apt install ffmpeg
```

**Dataset Preparation**

Please refer to [DATASET.md](assets/readmes/DATASET.md).

**Evaluation Tool**
```sh
# save coco_caption.zip to .xdecoder_data
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/coco_caption.zip
unzip coco_caption.zip
```

**Environment Variables**
```sh
export DETECTRON2_DATASETS=/pth/to/xdecoder_data
export DATASET=/pth/to/xdecoder_data
export DATASET2=/pth/to/xdecoder_data
export VLDATASET=/pth/to/xdecoder_data
export PATH=$PATH:/pth/to/xdecoder_data/coco_caption/jre1.8.0_321/bin
export PYTHONPATH=$PYTHONPATH:/pth/to/xdecoder_data/coco_caption
```

**Pretrained Checkpoint**

X-Decoder:
```sh
# Focal-T UniCL
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/focalt_in21k_yfcc_gcc_xdecoder_unicl.pt

# Focal-L UniCL
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/focall_vision_focalb_lang_unicl.pt
```

SEEM:
```
# Focal-T X-Decoder
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/xdecoder_focalt_last.pt

# Focal-L X-Decoder
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/xdecoder_focall_last_oq101.pt

# Focal-B UniCL Language
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/focalb_lang_unicl.pt

# ViT-B SAM
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# ViT-L SAM
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

```

