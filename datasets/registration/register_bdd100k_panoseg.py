# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import json
import os

from collections import namedtuple
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

Label = namedtuple(
    "Label",
    [
        "name",  # The identifier of this label, e.g. 'car', 'person', ... .
        # We use them to uniquely name a class
        "id",  # An integer ID that is associated with this label.
        # The IDs are used to represent the label in ground truth images An ID
        # of -1 means that this label does not have an ID and thus is ignored
        # when creating ground truth images (e.g. license plate). Do not modify
        # these IDs, since exactly these IDs are expected by the evaluation
        # server.
        "trainId",
        # Feel free to modify these IDs as suitable for your method. Then
        # create ground truth images with train IDs, using the tools provided
        # in the 'preparation' folder. However, make sure to validate or submit
        # results to our evaluation server using the regular IDs above! For
        # trainIds, multiple labels might have the same ID. Then, these labels
        # are mapped to the same class in the ground truth images. For the
        # inverse mapping, we use the label that is defined first in the list
        # below. For example, mapping all void-type classes to the same ID in
        # training, might make sense for some approaches. Max value is 255!
        "category",  # The name of the category that this label belongs to
        "categoryId",
        # The ID of this category. Used to create ground truth images
        # on category level.
        "hasInstances",
        # Whether this label distinguishes between single instances or not
        "ignoreInEval",
        # Whether pixels having this class as ground truth label are ignored
        # during evaluations or not
        "color",  # The color of this label
    ],
)


# Our extended list of label types. Our train id is compatible with Cityscapes
BDD_CATEGORIES = [
    #       name                     id    trainId   category catId
    #       hasInstances   ignoreInEval   color
    # Label("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
    Label("dynamic", 1, 255, "void", 0, False, True, (111, 74, 0)),
    Label("ego vehicle", 2, 255, "void", 0, False, True, (0, 0, 0)),
    Label("ground", 3, 255, "void", 0, False, True, (81, 0, 81)),
    Label("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
    Label("parking", 5, 255, "flat", 1, False, True, (250, 170, 160)),
    Label("rail track", 6, 255, "flat", 1, False, True, (230, 150, 140)),
    Label("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
    Label("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
    Label("bridge", 9, 255, "construction", 2, False, True, (150, 100, 100)),
    Label("building", 10, 2, "construction", 2, False, False, (70, 70, 70)),
    Label("fence", 11, 4, "construction", 2, False, False, (190, 153, 153)),
    Label("garage", 12, 255, "construction", 2, False, True, (180, 100, 180)),
    Label(
        "guard rail", 13, 255, "construction", 2, False, True, (180, 165, 180)
    ),
    Label("tunnel", 14, 255, "construction", 2, False, True, (150, 120, 90)),
    Label("wall", 15, 3, "construction", 2, False, False, (102, 102, 156)),
    Label("banner", 16, 255, "object", 3, False, True, (250, 170, 100)),
    Label("billboard", 17, 255, "object", 3, False, True, (220, 220, 250)),
    Label("lane divider", 18, 255, "object", 3, False, True, (255, 165, 0)),
    Label("parking sign", 19, 255, "object", 3, False, False, (220, 20, 60)),
    Label("pole", 20, 5, "object", 3, False, False, (153, 153, 153)),
    Label("polegroup", 21, 255, "object", 3, False, True, (153, 153, 153)),
    Label("street light", 22, 255, "object", 3, False, True, (220, 220, 100)),
    Label("traffic cone", 23, 255, "object", 3, False, True, (255, 70, 0)),
    Label(
        "traffic device", 24, 255, "object", 3, False, True, (220, 220, 220)
    ),
    Label("traffic light", 25, 6, "object", 3, False, False, (250, 170, 30)),
    Label("traffic sign", 26, 7, "object", 3, False, False, (220, 220, 0)),
    Label(
        "traffic sign frame",
        27,
        255,
        "object",
        3,
        False,
        True,
        (250, 170, 250),
    ),
    Label("terrain", 28, 9, "nature", 4, False, False, (152, 251, 152)),
    Label("vegetation", 29, 8, "nature", 4, False, False, (107, 142, 35)),
    Label("sky", 30, 10, "sky", 5, False, False, (70, 130, 180)),
    Label("person", 31, 11, "human", 6, True, False, (220, 20, 60)),
    Label("rider", 32, 12, "human", 6, True, False, (255, 0, 0)),
    Label("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
    Label("bus", 34, 15, "vehicle", 7, True, False, (0, 60, 100)),
    Label("car", 35, 13, "vehicle", 7, True, False, (0, 0, 142)),
    Label("caravan", 36, 255, "vehicle", 7, True, True, (0, 0, 90)),
    Label("motorcycle", 37, 17, "vehicle", 7, True, False, (0, 0, 230)),
    Label("trailer", 38, 255, "vehicle", 7, True, True, (0, 0, 110)),
    Label("train", 39, 16, "vehicle", 7, True, False, (0, 80, 100)),
    Label("truck", 40, 14, "vehicle", 7, True, False, (0, 0, 70)),
]

BDD_COLORS = [k.color for k in BDD_CATEGORIES]

MetadataCatalog.get("bdd100k_pano_val").set(
    stuff_colors=BDD_COLORS[:],
)

def load_bdd_panoptic_json(json_file, image_dir, gt_dir, meta):
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
        file_name = ann['file_name'].replace('png', 'jpg')

        image_file = os.path.join(image_dir, file_name)
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


def register_bdd_panoptic(
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
        lambda: load_bdd_panoptic_json(
            panoptic_json, image_root, panoptic_root, metadata
        ),
    )
    MetadataCatalog.get(panoptic_name).set(
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        evaluator_type="bdd_panoptic_pano",
        ignore_label=0,
        label_divisor=1000,
        **metadata,
    )


_PREDEFINED_SPLITS_SCANNET_PANOPTIC = {
    "bdd10k_40_panoptic_val": (
        "bdd100k/images/10k/val",
        "bdd100k/labels/pan_seg/coco_pano/val",
        "bdd100k/labels/pan_seg/meta/coco_val.json",
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
    thing_classes = [k.name for k in BDD_CATEGORIES if k.hasInstances == True]
    thing_colors = [k.color for k in BDD_CATEGORIES if k.hasInstances == True]
    stuff_classes = [k.name for k in BDD_CATEGORIES]
    stuff_colors = [k.color for k in BDD_CATEGORIES]

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

    for i, cat in enumerate(BDD_CATEGORIES):
        if cat.hasInstances:
            thing_dataset_id_to_contiguous_id[cat.id] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat.id] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id
    meta["offset"] = 0
    meta["object_mask_threshold"] = 0.4
    return meta


def register_all_scannet_panoptic(root):
    metadata = get_metadata()
    for (
        prefix,
        (image_root, panoptic_root, panoptic_json),
    ) in _PREDEFINED_SPLITS_SCANNET_PANOPTIC.items():
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_bdd_panoptic(
            prefix,
            metadata,
            os.path.join(root, image_root),
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_scannet_panoptic(_root)