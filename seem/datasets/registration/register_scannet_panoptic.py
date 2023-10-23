# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

SCANNET_20_CATEGORIES = [
    {"color": [174, 199, 232], "id": 1, "isthing": 0, "name": "wall"},
    {"color": [152, 223, 138], "id": 2, "isthing": 0, "name": "floor"},
    {"color": [31, 119, 180], "id": 3, "isthing": 1, "name": "cabinet"},
    {"color": [255, 187, 120], "id": 4, "isthing": 1, "name": "bed"},
    {"color": [188, 189, 34], "id": 5, "isthing": 1, "name": "chair"},
    {"color": [140, 86, 75], "id": 6, "isthing": 1, "name": "sofa"},
    {"color": [255, 152, 150], "id": 7, "isthing": 1, "name": "table"},
    {"color": [214, 39, 40], "id": 8, "isthing": 1, "name": "door"},
    {"color": [197, 176, 213], "id": 9, "isthing": 1, "name": "window "},
    {"color": [148, 103, 189], "id": 10, "isthing": 1, "name": "bookshelf"},
    {"color": [196, 156, 148], "id": 11, "isthing": 1, "name": "picture"},
    {"color": [23, 190, 207], "id": 12, "isthing": 1, "name": "counter"},
    {"color": [247, 182, 210], "id": 14, "isthing": 1, "name": "desk"},
    {"color": [219, 219, 141], "id": 16, "isthing": 1, "name": "curtain"},
    {"color": [255, 127, 14], "id": 24, "isthing": 1, "name": "refrigerator"},
    {"color": [158, 218, 229], "id": 28, "isthing": 1, "name": "shower curtain"},
    {"color": [44, 160, 44], "id": 33, "isthing": 1, "name": "toilet"},
    {"color": [112, 128, 144], "id": 34, "isthing": 1, "name": "sink"},
    {"color": [227, 119, 194], "id": 36, "isthing": 1, "name": "bathtub"},
    {"color": [82, 84, 163], "id": 39, "isthing": 1, "name": "otherfurniture"},
]

SCANNET_COLORS = [k["color"] for k in SCANNET_20_CATEGORIES]

MetadataCatalog.get("scannet20_pano_val").set(
    stuff_colors=SCANNET_COLORS[:],
)

def load_scannet_panoptic_json(json_file, image_dir, gt_dir, meta):
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

    ret = []
    for ann in json_info["annotations"]:
        image_id = ann["image_id"]
        # TODO: currently we assume image and label has the same filename but
        # different extension, and images have extension ".jpg" for COCO. Need
        # to make image extension a user-provided argument if we extend this
        # function to support other COCO-like datasets.
        folder_name = ann['file_name'].split('__')[0]
        file_name = ann['file_name'].split('__')[1].replace('png', 'jpg')

        image_file = os.path.join(image_dir, folder_name, 'color', file_name)
        label_file = os.path.join(gt_dir, ann["file_name"])

        segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "pan_seg_file_name": label_file,
                "segments_info": segments_info,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    return ret


def register_scannet_panoptic(
    name, metadata, image_root, panoptic_root, panoptic_json,
):
    """
    Register a "standard" version of ADE20k panoptic segmentation dataset named `name`.
    The dictionaries in this registered dataset follows detectron2's standard format.
    Hence it's called "standard".
    Args:
        name (str): the name that identifies a dataset,
            e.g. "ade20k_panoptic_train"
        metadata (dict): extra metadata associated with this dataset.
        image_root (str): directory which contains all the images
        panoptic_root (str): directory which contains panoptic annotation images in COCO format
        panoptic_json (str): path to the json panoptic annotation file in COCO format
        sem_seg_root (none): not used, to be consistent with
            `register_coco_panoptic_separated`.
        instances_json (str): path to the json instance annotation file
    """
    panoptic_name = name
    DatasetCatalog.register(
        panoptic_name,
        lambda: load_scannet_panoptic_json(
            panoptic_json, image_root, panoptic_root, metadata
        ),
    )
    MetadataCatalog.get(panoptic_name).set(
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        evaluator_type="scannet_panoptic_seg",
        ignore_label=0,
        label_divisor=1000,
        object_mask_threshold=0.4,
        **metadata,
    )


_PREDEFINED_SPLITS_SCANNET_PANOPTIC = {
    "scannet_21_panoptic_val": (
        "scannet_frames_25k/images",
        "scannet_frames_25k/scannet_panoptic",
        "scannet_frames_25k/scannet_panoptic.json",
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
    thing_classes = [k["name"] for k in SCANNET_20_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in SCANNET_20_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in SCANNET_20_CATEGORIES]
    stuff_colors = [k["color"] for k in SCANNET_20_CATEGORIES]

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

    for i, cat in enumerate(SCANNET_20_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta


def register_all_scannet_panoptic(root):
    metadata = get_metadata()
    for (
        prefix,
        (image_root, panoptic_root, panoptic_json),
    ) in _PREDEFINED_SPLITS_SCANNET_PANOPTIC.items():
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_scannet_panoptic(
            prefix,
            metadata,
            os.path.join(root, image_root),
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_scannet_panoptic(_root)
