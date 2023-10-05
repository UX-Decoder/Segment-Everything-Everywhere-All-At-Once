## X-Decoder

**Focal-T:**
```bash
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

**Focal-L:**
```bash
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

**Focal-T:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun -n 8 python entry.py train \
            --conf_files configs/seem/focalt_unicl_lang_v1.yaml \
            --overrides \
            FP16 True \
            COCO.INPUT.IMAGE_SIZE 1024 \
            MODEL.DECODER.HIDDEN_DIM 512 \
            MODEL.ENCODER.CONVS_DIM 512 \
            MODEL.ENCODER.MASK_DIM 512 \
            TEST.BATCH_SIZE_TOTAL 8 \
            TRAIN.BATCH_SIZE_TOTAL 16 \
            TRAIN.BATCH_SIZE_PER_GPU 2 \
            SOLVER.MAX_NUM_EPOCHS 50 \
            SOLVER.BASE_LR 0.0001 \
            SOLVER.FIX_PARAM.backbone True \
            SOLVER.FIX_PARAM.lang_encoder True \
            SOLVER.FIX_PARAM.pixel_decoder True \
            MODEL.DECODER.COST_SPATIAL.CLASS_WEIGHT 5.0 \
            MODEL.DECODER.COST_SPATIAL.MASK_WEIGHT 2.0 \
            MODEL.DECODER.COST_SPATIAL.DICE_WEIGHT 2.0 \
            MODEL.DECODER.TOP_SPATIAL_LAYERS 10 \
            MODEL.DECODER.SPATIAL.ENABLED True \
            MODEL.DECODER.GROUNDING.ENABLED True \
            FIND_UNUSED_PARAMETERS True \
            ATTENTION_ARCH.SPATIAL_MEMORIES 32 \
            MODEL.DECODER.SPATIAL.MAX_ITER 5 \
            ATTENTION_ARCH.QUERY_NUMBER 3 \
            STROKE_SAMPLER.MAX_CANDIDATE 10 \
            WEIGHT True \
            RESUME_FROM /pth/to/xdecoder_data/pretrained/xdecoder_focalt_last.pt
```

**Focal-L:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun -n 8 python entry.py train \
            --conf_files configs/seem/focall_unicl_lang_v1.yaml \
            --overrides \
            FP16 True \
            COCO.INPUT.IMAGE_SIZE 1024 \
            MODEL.DECODER.HIDDEN_DIM 512 \
            MODEL.ENCODER.CONVS_DIM 512 \
            MODEL.ENCODER.MASK_DIM 512 \
            TEST.BATCH_SIZE_TOTAL 8 \
            TRAIN.BATCH_SIZE_TOTAL 16 \
            TRAIN.BATCH_SIZE_PER_GPU 2 \
            SOLVER.MAX_NUM_EPOCHS 50 \
            SOLVER.BASE_LR 0.0001 \
            SOLVER.FIX_PARAM.backbone True \
            SOLVER.FIX_PARAM.lang_encoder True \
            SOLVER.FIX_PARAM.pixel_decoder True \
            MODEL.DECODER.COST_SPATIAL.CLASS_WEIGHT 5.0 \
            MODEL.DECODER.COST_SPATIAL.MASK_WEIGHT 2.0 \
            MODEL.DECODER.COST_SPATIAL.DICE_WEIGHT 2.0 \
            MODEL.DECODER.TOP_SPATIAL_LAYERS 10 \
            MODEL.DECODER.SPATIAL.ENABLED True \
            MODEL.DECODER.GROUNDING.ENABLED True \
            FIND_UNUSED_PARAMETERS True \
            ATTENTION_ARCH.SPATIAL_MEMORIES 32 \
            MODEL.DECODER.SPATIAL.MAX_ITER 5 \
            ATTENTION_ARCH.QUERY_NUMBER 3 \
            STROKE_SAMPLER.MAX_CANDIDATE 10 \
            WEIGHT True \
            RESUME_FROM /pth/to/xdecoder_data/pretrained/xdecoder_focall_last_oq101.pt
```

**SAM ViT-B:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun -n 8 python entry.py train \
            --conf_files configs/seem/samvitb_unicl_lang_v1.yaml \
            --overrides \
            FP16 True \
            COCO.INPUT.IMAGE_SIZE 1024 \
            MODEL.DECODER.HIDDEN_DIM 512 \
            MODEL.ENCODER.CONVS_DIM 512 \
            MODEL.ENCODER.MASK_DIM 512 \
            TEST.BATCH_SIZE_TOTAL 8 \
            TRAIN.BATCH_SIZE_TOTAL 16 \
            TRAIN.BATCH_SIZE_PER_GPU 2 \
            SOLVER.MAX_NUM_EPOCHS 50 \
            SOLVER.BASE_LR 0.0001 \
            SOLVER.FIX_PARAM.backbone True \
            SOLVER.FIX_PARAM.lang_encoder True \
            SOLVER.FIX_PARAM.pixel_decoder True \
            MODEL.DECODER.COST_SPATIAL.CLASS_WEIGHT 5.0 \
            MODEL.DECODER.COST_SPATIAL.MASK_WEIGHT 2.0 \
            MODEL.DECODER.COST_SPATIAL.DICE_WEIGHT 2.0 \
            MODEL.DECODER.TOP_SPATIAL_LAYERS 10 \
            MODEL.DECODER.SPATIAL.ENABLED True \
            MODEL.DECODER.GROUNDING.ENABLED True \
            FIND_UNUSED_PARAMETERS True \
            ATTENTION_ARCH.SPATIAL_MEMORIES 32 \
            MODEL.DECODER.SPATIAL.MAX_ITER 5 \
            ATTENTION_ARCH.QUERY_NUMBER 3 \
            STROKE_SAMPLER.MAX_CANDIDATE 10 \
            MODEL.BACKBONE.PRETRAINED /pth/to/xdecoder_data/pretrained/sam_vit_b_01ec64.pth \
            WEIGHT True \
            RESUME_FROM /pth/to/xdecoder_data/pretrained/focalb_lang_unicl.pt
```

**SAM ViT-L:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun -n 8 python entry.py train \
            --conf_files configs/seem/samvitl_unicl_lang_v1.yaml \
            --overrides \
            FP16 True \
            COCO.INPUT.IMAGE_SIZE 1024 \
            MODEL.DECODER.HIDDEN_DIM 512 \
            MODEL.ENCODER.CONVS_DIM 512 \
            MODEL.ENCODER.MASK_DIM 512 \
            TEST.BATCH_SIZE_TOTAL 8 \
            TRAIN.BATCH_SIZE_TOTAL 16 \
            TRAIN.BATCH_SIZE_PER_GPU 2 \
            SOLVER.MAX_NUM_EPOCHS 50 \
            SOLVER.BASE_LR 0.0001 \
            SOLVER.FIX_PARAM.backbone True \
            SOLVER.FIX_PARAM.lang_encoder True \
            SOLVER.FIX_PARAM.pixel_decoder True \
            MODEL.DECODER.COST_SPATIAL.CLASS_WEIGHT 5.0 \
            MODEL.DECODER.COST_SPATIAL.MASK_WEIGHT 2.0 \
            MODEL.DECODER.COST_SPATIAL.DICE_WEIGHT 2.0 \
            MODEL.DECODER.TOP_SPATIAL_LAYERS 10 \
            MODEL.DECODER.SPATIAL.ENABLED True \
            MODEL.DECODER.GROUNDING.ENABLED True \
            FIND_UNUSED_PARAMETERS True \
            ATTENTION_ARCH.SPATIAL_MEMORIES 32 \
            MODEL.DECODER.SPATIAL.MAX_ITER 5 \
            ATTENTION_ARCH.QUERY_NUMBER 3 \
            STROKE_SAMPLER.MAX_CANDIDATE 10 \
            MODEL.BACKBONE.PRETRAINED /pth/to/xdecoder_data/pretrained/sam_vit_l_0b3195.pth \
            WEIGHT True \
            RESUME_FROM /pth/to/xdecoder_data/pretrained/focalb_lang_unicl.pt
```
