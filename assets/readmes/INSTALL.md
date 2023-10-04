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
```
```

**Environment Variables**

**Pretrained Checkpoint**
