# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os
import collections

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.utils.file_io import PathManager


_PREDEFINED_SPLITS_COCO_PANOPTIC_CAPTION = {
    "coco_2017_train_panoptic": (
        # This is the original panoptic annotation directory
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017.json",
        "coco/panoptic_semseg_train2017",
        "coco/annotations/captions_train2017.json",
        "coco/annotations/grounding_train2017.json",
        "coco/annotations/caption_class_similarity.pth"
    ),
    "coco_2017_train_panoptic_filtkar": (
        # This is the original panoptic annotation directory
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017_filtkar.json",
        "coco/panoptic_semseg_train2017",
        "coco/annotations/captions_train2017_filtkar.json",
        "coco/annotations/grounding_train2017.json",
        "coco/annotations/caption_class_similarity.pth"
    ),
    "coco_2017_train_panoptic_filtrefgumdval": (
        # This is the original panoptic annotation directory
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017_filtrefgumdval.json",
        "coco/panoptic_semseg_train2017",
        "coco/annotations/captions_train2017_filtrefgumdval.json",
        "coco/annotations/grounding_train2017_filtrefgumd.json",
        "coco/annotations/caption_class_similarity.pth"
    ),
    "coco_2017_train_panoptic_filtall": (
        # This is the original panoptic annotation directory
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017_filtrefgumdval_filtvlp.json",
        "coco/panoptic_semseg_train2017",
        "coco/annotations/captions_train2017_filtrefgumdval_filtvlp.json",
        "coco/annotations/grounding_train2017_filtrefgumdval_filtvlp.json",
        "coco/annotations/caption_class_similarity.pth"
    ),
}


def get_metadata():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in COCO_CATEGORIES]
    stuff_colors = [k["color"] for k in COCO_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(COCO_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta


def load_coco_panoptic_json(json_file, image_dir, gt_dir, semseg_dir, caption_file, grounding_file, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    with PathManager.open(caption_file) as f:
        caption_info = json.load(f)
    
    with PathManager.open(grounding_file) as f:
        grounding_info = json.load(f)

    # build dict {image_id: Listof[captions]}
    cap_dict = collections.defaultdict(list)
    for cap_ann in caption_info['annotations']:
        image_id = int(cap_ann["image_id"])
        cap_dict[image_id].append(cap_ann["caption"])
        
    # build dictionary for grounding
    grd_dict = collections.defaultdict(list)
    for grd_ann in grounding_info['annotations']:
        image_id = int(grd_ann["image_id"])
        grd_dict[image_id].append(grd_ann)
    
    ret = []
    for ann in json_info["annotations"]:
        image_id = int(ann["image_id"])
        # TODO: currently we assume image and label has the same filename but
        # different extension, and images have extension ".jpg" for COCO. Need
        # to make image extension a user-provided argument if we extend this
        # function to support other COCO-like datasets.
        image_file = os.path.join(image_dir, os.path.splitext(ann["file_name"])[0] + ".jpg")
        label_file = os.path.join(gt_dir, ann["file_name"])
        sem_label_file = os.path.join(semseg_dir, ann["file_name"])
        segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]

        grounding_anno = grd_dict[image_id] if image_id in grd_dict else []
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "captions": cap_dict[image_id],
                "grounding_info": grounding_anno,
                "pan_seg_file_name": label_file,
                "sem_seg_file_name": sem_label_file,
                "segments_info": segments_info,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    assert PathManager.isfile(ret[0]["sem_seg_file_name"]), ret[0]["sem_seg_file_name"]
    return ret


def register_coco_panoptic_annos_caption_grounding_sem_seg(
    name, metadata, image_root, panoptic_root, panoptic_json, sem_seg_root, caption_root, grounding_root, similarity_pth, instances_json
):
    panoptic_name = '_'.join(name.split('_')[0:4])
    delattr(MetadataCatalog.get(panoptic_name), "thing_classes")
    delattr(MetadataCatalog.get(panoptic_name), "thing_colors")
    MetadataCatalog.get(panoptic_name).set(
        thing_classes=metadata["thing_classes"],
        thing_colors=metadata["thing_colors"],
        # thing_dataset_id_to_contiguous_id=metadata["thing_dataset_id_to_contiguous_id"],
    )
    
    # the name is "coco_2017_train_panoptic_with_sem_seg" and "coco_2017_val_panoptic_with_sem_seg"
    semantic_name = name + "_with_sem_seg_caption_grounding"
    DatasetCatalog.register(
        semantic_name,
        lambda: load_coco_panoptic_json(panoptic_json, image_root, panoptic_root, sem_seg_root, caption_root, grounding_root, metadata),
    )
    MetadataCatalog.get('logistic').set(caption_similarity_pth=similarity_pth)
    MetadataCatalog.get(semantic_name).set(
        sem_seg_root=sem_seg_root,
        panoptic_root=panoptic_root,
        caption_root=caption_root,         
        image_root=image_root,
        panoptic_json=panoptic_json,
        json_file=instances_json,
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )


def register_all_coco_panoptic_annos_caption_grounding_sem_seg(root):
    for (
        prefix,
        (panoptic_root, panoptic_json, semantic_root, caption_root, grounding_root, similarity_pth),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC_CAPTION.items():
        prefix_instances = '_'.join(prefix.split('_')[0:3])
        instances_meta = MetadataCatalog.get(prefix_instances)
        image_root, instances_json = instances_meta.image_root, instances_meta.json_file
        image_root = image_root.replace('datasets', root)

        register_coco_panoptic_annos_caption_grounding_sem_seg(
            prefix,
            get_metadata(),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
            os.path.join(root, caption_root),
            os.path.join(root, grounding_root),
            os.path.join(root, similarity_pth), 
            instances_json,
        )


_root = os.getenv("DATASET", "datasets")
register_all_coco_panoptic_annos_caption_grounding_sem_seg(_root)
