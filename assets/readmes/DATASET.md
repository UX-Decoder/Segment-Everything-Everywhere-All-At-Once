# Preparing Dataset

:bangbang: The dataset preparation contains many details, welcome community contribution to fix any bug, Thanks!

Our dataloader follows [Detectron2](https://github.com/facebookresearch/detectron2) that contains: <br/>
(1) [A dataset registrator](datasets/registration) <br/>
(2) [A dataset mapper](datasets/dataset_mappers) <br/>
We modify the dataset registration and mapper for custom datasets.

## Training Dataset
We assume all the datasets are stored under:
```
.xdecoder_data
```

### COCO (SEEM & X-Decoder)

```sh
# Prepare panoptic_train2017, panoptic_semseg_train2017 exactly the same as [Mask2Fomer](https://github.com/facebookresearch/Mask2Former/tree/main/datasets)

# (SEEM & X-Decoder) Download additional logistic and custom annotation files to .xdecoder_data/coco/annotations
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/caption_class_similarity.pth
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/captions_train2017_filtrefgumdval_filtvlp.json
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/grounding_train2017_filtrefgumdval_filtvlp.json
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/panoptic_train2017_filtrefgumdval_filtvlp.json
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/refcocog_umd_val.json
wget https://github.com/peteanderson80/coco-caption/blob/master/annotations/captions_val2014.json

# (SEEM) Download LVIS annotations for mask preparation
wget https://huggingface.co/xdecoder/SEEM/resolve/main/coco_train2017_filtrefgumdval_lvis.json
```

After dataset preparation, the dataset structure would be:
```
.xdecoder_data
└── coco/
    ├── train2017/
    ├── val2017/
    ├── panoptic_train2017/
    ├── panoptic_semseg_train2017/
    ├── panoptic_val2017/
    ├── panoptic_semseg_val2017/
    └── annotations/
        ├── refcocog_umd_val.json
        ├── captions_val2014.json
        ├── panoptic_val2017.json
        ├── caption_class_similarity.pth
        ├── panoptic_train2017_filtrefgumdval_filtvlp.json
        └── grounding_train2017_filtrefgumdval_filtvlp.json
└── lvis/
    └── coco_train2017_filtrefgumdval_lvis.json
```

#### 4M Image Text Pairs (X-Decoder)
We follow the exact data preparation for the image text pairs data with [ViLT](https://github.com/dandelin/ViLT/blob/master/DATA.md).
```
# The pretrained arrow file are put under .xdecoder_data/pretrain_arrows_code224 with the following list of files.
["filtcoco2017val_caption_karpathy_train.arrow", "filtcoco2017val_caption_karpathy_val.arrow", "filtcoco2017val_caption_karpathy_restval.arrow"] + ["code224_vg.arrow"] + [f"code224_sbu_{i}.arrow" for i in range(9)] + [f"code224_conceptual_caption_train_{i}.arrow" for i in range(31)]
# ["filtcoco2017val_caption_karpathy_train.arrow", "filtcoco2017val_caption_karpathy_val.arrow", "filtcoco2017val_caption_karpathy_restval.arrow"] are originated from ["filtcoco2017val_caption_karpathy_train.arrow", "filtcoco2017val_caption_karpathy_val.arrow", "filtcoco2017val_caption_karpathy_restval.arrow"] with deletion of coco val2017 overlapped images to avoid information leakage.
```

To get quick started:
```sh 
# Download coco karparthy test set (we hack the training data to be coco_caption_karpathy_test.arrow only for quick start in the codebase)
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/coco_caption_karpathy_test.arrow
```

After dataset preparation, the dataset structure would be:
```
.xdecoder_data
└── pretrain_arrows_code224/
    ├── coco_caption_karpathy_test.arrow
    ├── *filtcoco2017val_caption_karpathy_train.arrow
    ├── ...
    ├── *code224_vg.arrow
    ├── *code224_sbu_0.arrow
    ├── ...
    ├── *code224_conceptual_caption_train_0.arrow
    └── ...
* Those datasets are optional for debugging the pipeline. ! NEED to add back when you are training the model.
```

***NOTE:***

<img src="https://user-images.githubusercontent.com/11957155/226159078-7f817452-76f8-44f4-af7a-9f13f3e02554.png" width="500">
There are overlap between COCO2017, COCO-Karpathy and REF-COCO dataset, and ref-coco is all overlapped with the COCO2017 training data, we have exclude the refcocog-umd validation, coco-karpathy test split during training.

## Evaluation Dataset

### RefCOCO (SEEM & X-Decoder)
Please refer to COCO Preparation on [line](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once/blob/v1.0/assets/readmes/DATASET.md#coco-seem--x-decoder).

### ADE20K, Cityscapes (X-Decoder)
Please Refer to [Mask2Former](https://github.com/facebookresearch/Mask2Former/tree/main/datasets).

### BDD100K (X-Decoder)
Please download the 10k split of BDD100k at https://doc.bdd100k.com/download.html#id1

### PascalVOC and all other interactive evaluation datasets (SEEM)
Please follow the instruction on [RITM](https://github.com/SamsungLabs/ritm_interactive_segmentation)

After dataset preparation, the dataset structure would be:
```
.xdecoder_data
└── PascalVOC/
    ├── Annotations/
    ├── ImageSets
    ├── JPEGImages/
    ├── SegmentationClass/
    └── SegmentationObject/
```

