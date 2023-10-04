import os
import json
from refer import REFER

coco_root = '/pth/to/coco'
ref_root = '/pth/to/refcocoseg'

coco_train_annot = json.load(open(os.path.join(coco_root, 'annotations/instances_train2017.json')))
coco_train_id = []
image_annot = {}
for i in range(len(coco_train_annot['images'])):
    coco_train_id.append(coco_train_annot['images'][i]['id'])
    image_annot[coco_train_annot['images'][i]['id']] = coco_train_annot['images'][i]

refg = REFER(data_root=ref_root,
                dataset='refcocog', splitBy='umd')
refg_val_ids = refg.getRefIds(split='val')

full_anno = []
for ref_id in refg_val_ids:
    ref = refg.loadRefs(ref_id)[0]
    anno = refg.refToAnn[ref_id]
    anno.update(ref)
    full_anno.append(anno)

imageid_list = []
final_anno = {}
for anno in full_anno:
    imageid_list += [anno['image_id']]
    final_anno[anno['ann_id']] = anno
    
annotations = [value for key, value in final_anno.items()]

iamges = []
for image_id in list(set(imageid_list)):
    iamges += [image_annot[image_id]]

outputs = {'images': iamges, 'annotations': annotations}
print(len(iamges))
print(len(annotations))
json.dump(outputs, open(os.path.join(coco_root, 'annotations/refcocog_umd_train.json'), 'w'))
