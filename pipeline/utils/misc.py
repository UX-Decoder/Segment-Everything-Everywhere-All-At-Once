import logging
import torch

logger = logging.getLogger(__name__)

def hook_opt(opt):

    try:
        grounding_flag = opt['REF']['INPUT']['SPATIAL']
    except:
        grounding_flag = False

    if grounding_flag:
        opt['ATTENTION_ARCH']['SELF_ATTENTION']['queries']['grounding'] = ['queries_grounding', 'tokens_grounding', 'tokens_spatial']

    try:
        spatial_flag = opt['STROKE_SAMPLER']['EVAL']['GROUNDING']
    except:
        spatial_flag = False

    if spatial_flag:
        opt['ATTENTION_ARCH']['SELF_ATTENTION']['queries']['spatial'] = ['queries_spatial', 'tokens_spatial', 'memories_spatial', 'tokens_grounding']

    return opt

# HACK for evalution 
def hook_metadata(metadata, name):
    return metadata

# HACK for evalution 
def hook_switcher(model, name):
    mappings = {}
    if name in ['cityscapes_fine_sem_seg_val', 'scannet_21_val_seg', 'scannet_38_val_seg', 'scannet_41_val_seg', 'sunrgbd_37_val_seg', 'context_59_val_seg', 'context_459_val_seg', 'voc_2012_val_seg', 'bdd10k_val_sem_seg', 'ade20k_full_sem_seg_val']:
        mappings = {'SEMANTIC_ON': True, 'INSTANCE_ON': False, 'PANOPTIC_ON': False}
    elif name in ['cityscapes_fine_instance_seg_val'] or 'seginw' in name:
        mappings = {'SEMANTIC_ON': False, 'INSTANCE_ON': True, 'PANOPTIC_ON': False}
    elif name in ['cityscapes_fine_panoptic_val', 'scannet_21_panoptic_val', 'bdd10k_40_panoptic_val']:
        mappings = {'SEMANTIC_ON': True, 'INSTANCE_ON': False, 'PANOPTIC_ON': True}
    elif name in ['coco_2017_val_panoptic_with_sem_seg', 'ade20k_panoptic_val', 'coco_2017_test-dev']:
        mappings = {'SEMANTIC_ON': True, 'INSTANCE_ON': True, 'PANOPTIC_ON': True}
    else:
        if name not in ["vlp_val", "vlp_captioning_val", "vlp_val2017", "vlp_captioning_val2017", "imagenet_val", "refcocog_val_google", "phrasecut_val", "phrasecut_test", "refcocop_val_unc", "refcoco_val_unc", "refcocog_val_umd", "pascalvoc_val_Point", "grounding_coco_entity_val", "vlp_coco_entity_val"]:
            assert False, "dataset switcher is not defined"

    for key, value in mappings.items():
        if key == 'SEMANTIC_ON':
            model.model.semantic_on = value
        if key == 'INSTANCE_ON':
            model.model.instance_on = value
        if key == 'PANOPTIC_ON':
            model.model.panoptic_on = value