from pycocotools.coco import COCO
import json
import copy


class CustomCOCO(COCO):
    def __init__(self, ann_file):
        super().__init__(ann_file)

    def __add__(self, other):
        anns1 = [self.anns[key] for key in self.anns.keys()]
        anns2 = [other.anns[key] for key in other.anns.keys()]

        img_names = []
        new_imgs = []
        new_anns = []
        id = 0

        for ann in anns1:
            img = self.imgs[ann['image_id']]
            img_name = img['file_name']
            img['id'] = id
            ann['id'] = id
            ann['image_id'] = id
            new_anns.append(ann)
            new_imgs.append(img)
            img_names.append(img_name)

            id += 1

        for ann in anns2:
            img = other.imgs[ann['image_id']]
            img_name = img['file_name']
            img['id'] = id
            ann['id'] = id
            ann['image_id'] = id
            if img_name not in img_names:
                new_anns.append(ann)
                new_imgs.append(img)

            id += 1

        new_dataset = {
            'images': new_imgs,
            'annotations': new_anns,
            'categories': self.dataset['categories']
        }

        return new_dataset


for i in range(10):
    coco1 = CustomCOCO(f'../datasets/annotations/20kp-real-simillar-val-{i}.json')
    coco2 = CustomCOCO(f'../datasets/annotations/20kp-synthetic-varied-val-{i}.json')

    with open(f'../datasets/annotations/20kp-combined-val-{i}.json', 'w') as f:
        json.dump(coco1 + coco2, f)