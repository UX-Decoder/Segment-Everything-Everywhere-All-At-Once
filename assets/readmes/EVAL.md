## X-Decoder

**Focal-T:**
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun -n 8 python entry.py evaluate \
            --conf_files configs/xdecoder/focalt_unicl_lang.yaml \
            --overrides \
            COCO.INPUT.IMAGE_SIZE 1024 \
            MODEL.DECODER.CAPTIONING.ENABLED True \
            MODEL.DECODER.RETRIEVAL.ENABLED True \
            MODEL.DECODER.GROUNDING.ENABLED True \
            COCO.TEST.BATCH_SIZE_TOTAL 8 \
            COCO.TRAIN.BATCH_SIZE_TOTAL 8 \
            COCO.TRAIN.BATCH_SIZE_PER_GPU 1 \
            VLP.TEST.BATCH_SIZE_TOTAL 8 \
            VLP.TRAIN.BATCH_SIZE_TOTAL 8 \
            VLP.TRAIN.BATCH_SIZE_PER_GPU 1 \
            MODEL.DECODER.HIDDEN_DIM 512 \
            MODEL.ENCODER.CONVS_DIM 512 \
            MODEL.ENCODER.MASK_DIM 512 \
            ADE20K.TEST.BATCH_SIZE_TOTAL 8 \
            FP16 True \
            WEIGHT True \
            RESUME_FROM /pth/to/xdecoder_data/xdecoder/xdecoder_focalt_last.pt
```

**Focal-L:**
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun -n 8 python entry.py evaluate \
            --conf_files configs/xdecoder/focall_unicl_lang.yaml \
            --overrides \
            COCO.INPUT.IMAGE_SIZE 1024 \
            MODEL.DECODER.CAPTIONING.ENABLED True \
            MODEL.DECODER.RETRIEVAL.ENABLED True \
            MODEL.DECODER.GROUNDING.ENABLED True \
            COCO.TEST.BATCH_SIZE_TOTAL 8 \
            COCO.TRAIN.BATCH_SIZE_TOTAL 8 \
            COCO.TRAIN.BATCH_SIZE_PER_GPU 1 \
            VLP.TEST.BATCH_SIZE_TOTAL 8 \
            VLP.TRAIN.BATCH_SIZE_TOTAL 8 \
            VLP.TRAIN.BATCH_SIZE_PER_GPU 1 \
            MODEL.DECODER.HIDDEN_DIM 512 \
            MODEL.ENCODER.CONVS_DIM 512 \
            MODEL.ENCODER.MASK_DIM 512 \
            ADE20K.TEST.BATCH_SIZE_TOTAL 8 \
            FP16 True \
            WEIGHT True \
            RESUME_FROM /pth/to/xdecoder_data/xdecoder/xdecoder_focall_last.pt
```

## SEEM

**Focal-T v0:**
```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun -n 8 python entry.py evaluate \
            --conf_files configs/seem/focalt_unicl_lang_v0.yaml \
            --overrides \
            COCO.INPUT.IMAGE_SIZE 1024 \
            MODEL.DECODER.HIDDEN_DIM 512 \
            MODEL.ENCODER.CONVS_DIM 512 \
            MODEL.ENCODER.MASK_DIM 512 \
            VOC.TEST.BATCH_SIZE_TOTAL 8 \
            TEST.BATCH_SIZE_TOTAL 8 \
            REF.TEST.BATCH_SIZE_TOTAL 8 \
            FP16 True \
            WEIGHT True \
            RESUME_FROM /pth/to/xdecoder_data/seem/seem_focalt_v0.pt
```

**Focal-T v1:**
```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun -n 8 python entry.py evaluate \
            --conf_files configs/seem/focalt_unicl_lang_v1.yaml \
            --overrides \
            COCO.INPUT.IMAGE_SIZE 1024 \
            MODEL.DECODER.HIDDEN_DIM 512 \
            MODEL.ENCODER.CONVS_DIM 512 \
            MODEL.ENCODER.MASK_DIM 512 \
            VOC.TEST.BATCH_SIZE_TOTAL 8 \
            TEST.BATCH_SIZE_TOTAL 8 \
            REF.TEST.BATCH_SIZE_TOTAL 8 \
            FP16 True \
            WEIGHT True \
            RESUME_FROM /pth/to/xdecoder_data/seem/seem_focalt_v1.pt
```

**ViT-B SAM v1:**
```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun -n 8 python entry.py evaluate \
            --conf_files configs/seem/samvitb_unicl_lang_v1.yaml \
            --overrides \
            COCO.INPUT.IMAGE_SIZE 1024 \
            MODEL.DECODER.HIDDEN_DIM 512 \
            MODEL.ENCODER.CONVS_DIM 512 \
            MODEL.ENCODER.MASK_DIM 512 \
            VOC.TEST.BATCH_SIZE_TOTAL 8 \
            TEST.BATCH_SIZE_TOTAL 8 \
            REF.TEST.BATCH_SIZE_TOTAL 8 \
            FP16 True \
            WEIGHT True \
            RESUME_FROM /pth/to/xdecoder_data/seem/seem_samvitb_v1.pt
```

**ViT-L SAM v1:**
```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun -n 8 python entry.py evaluate \
            --conf_files configs/seem/samvitl_unicl_lang_v1.yaml \
            --overrides \
            COCO.INPUT.IMAGE_SIZE 1024 \
            MODEL.DECODER.HIDDEN_DIM 512 \
            MODEL.ENCODER.CONVS_DIM 512 \
            MODEL.ENCODER.MASK_DIM 512 \
            VOC.TEST.BATCH_SIZE_TOTAL 8 \
            TEST.BATCH_SIZE_TOTAL 8 \
            REF.TEST.BATCH_SIZE_TOTAL 8 \
            FP16 True \
            WEIGHT True \
            RESUME_FROM /pth/to/xdecoder_data/seem/seem_samvitl_v1.pt
```

**Focal-L v0:**
```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun -n 8 python entry.py evaluate \
            --conf_files configs/seem/focall_unicl_lang_v0.yaml \
            --overrides \
            COCO.INPUT.IMAGE_SIZE 1024 \
            MODEL.DECODER.HIDDEN_DIM 512 \
            MODEL.ENCODER.CONVS_DIM 512 \
            MODEL.ENCODER.MASK_DIM 512 \
            VOC.TEST.BATCH_SIZE_TOTAL 8 \
            TEST.BATCH_SIZE_TOTAL 8 \
            REF.TEST.BATCH_SIZE_TOTAL 8 \
            FP16 True \
            WEIGHT True \
            RESUME_FROM /pth/to/xdecoder_data/seem/seem_focall_v0.pt
```
