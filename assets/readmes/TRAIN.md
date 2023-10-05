## X-Decoder

**Focal-T**
```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun -n 8 python entry.py train \
            --conf_files configs/xdecoder/focalt_unicl_lang.yaml \
            --overrides \
            FP16 True \
            COCO.INPUT.IMAGE_SIZE 1024 \
            MODEL.DECODER.HIDDEN_DIM 512 \
            MODEL.ENCODER.CONVS_DIM 512 \
            MODEL.ENCODER.MASK_DIM 512 \
            MODEL.DECODER.CAPTIONING.ENABLED True \
            MODEL.DECODER.RETRIEVAL.ENABLED True \
            MODEL.DECODER.GROUNDING.ENABLED True \
            MODEL.DECODER.CAPTIONING_WEIGHT 8 \
            MODEL.DECODER.RETRIEVAL_WEIGHT 8 \
            MODEL.DECODER.TOP_CAPTIONING_LAYERS 3 \
            MODEL.DECODER.TOP_RETRIEVAL_LAYERS 3 \
            MODEL.DECODER.TOP_GROUNDING_LAYERS 6 \
            MODEL.DECODER.GROUNDING.TEXT_WEIGHT 2.0 \
            MODEL.DECODER.GROUNDING.CLASS_WEIGHT 0.5 \
            COCO.TEST.BATCH_SIZE_TOTAL 8 \
            COCO.TRAIN.BATCH_SIZE_TOTAL 8 \
            COCO.TRAIN.BATCH_SIZE_PER_GPU 1 \
            VLP.TEST.BATCH_SIZE_TOTAL 8 \
            VLP.TRAIN.BATCH_SIZE_TOTAL 256 \
            VLP.TRAIN.BATCH_SIZE_PER_GPU 32 \
            VLP.DATALOADER.NUM_WORKERS 32
            ADE20K.TEST.BATCH_SIZE_TOTAL 8 \
            REF.TEST.BATCH_SIZE_TOTAL 8 \
            SOLVER.LR_MULTIPLIER.lang_encoder 0.1 \
            WEIGHT True \
            RESUME_FROM /pth/to/xdecoder_data/pretrained/focalt_in21k_yfcc_gcc_xdecoder_unicl.pt
```

**Focal-L**
```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun -n 8 python entry.py train \
            --conf_files configs/xdecoder/focall_unicl_lang.yaml \
            --overrides \
            FP16 True \
            COCO.INPUT.IMAGE_SIZE 1024 \
            MODEL.DECODER.HIDDEN_DIM 512 \
            MODEL.ENCODER.CONVS_DIM 512 \
            MODEL.ENCODER.MASK_DIM 512 \
            MODEL.DECODER.CAPTIONING.ENABLED True \
            MODEL.DECODER.RETRIEVAL.ENABLED True \
            MODEL.DECODER.GROUNDING.ENABLED True \
            MODEL.DECODER.CAPTIONING_WEIGHT 8 \
            MODEL.DECODER.RETRIEVAL_WEIGHT 8 \
            MODEL.DECODER.TOP_CAPTIONING_LAYERS 3 \
            MODEL.DECODER.TOP_RETRIEVAL_LAYERS 3 \
            MODEL.DECODER.TOP_GROUNDING_LAYERS 6 \
            MODEL.DECODER.GROUNDING.TEXT_WEIGHT 2.0 \
            MODEL.DECODER.GROUNDING.CLASS_WEIGHT 0.5 \
            COCO.TEST.BATCH_SIZE_TOTAL 8 \
            COCO.TRAIN.BATCH_SIZE_TOTAL 8 \
            COCO.TRAIN.BATCH_SIZE_PER_GPU 1 \
            VLP.TEST.BATCH_SIZE_TOTAL 8 \
            VLP.TRAIN.BATCH_SIZE_TOTAL 256 \
            VLP.TRAIN.BATCH_SIZE_PER_GPU 32 \
            VLP.DATALOADER.NUM_WORKERS 32
            ADE20K.TEST.BATCH_SIZE_TOTAL 8 \
            REF.TEST.BATCH_SIZE_TOTAL 8 \
            SOLVER.LR_MULTIPLIER.lang_encoder 0.1 \
            WEIGHT True \
            RESUME_FROM /pth/to/xdecoder_data/pretrained/focall_vision_focalb_lang_unicl.pt
```

## SEEM
