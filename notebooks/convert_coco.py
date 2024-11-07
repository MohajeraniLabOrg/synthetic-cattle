import os.path as osp
import sys
from glob import glob
from typing import Dict, Optional
import datumaro as dm
from datumaro.plugins.cvat_format.converter import _SubsetWriter, XmlAnnotationWriter
from unittest import mock

class RemapBboxLabels(dm.ItemTransform):
    def __init__(self, extractor: dm.IExtractor, new_label_id: int):
        super().__init__(extractor)
        self._new_label_id = new_label_id

    def transform_item(self, item: dm.DatasetItem) -> Optional[dm.DatasetItem]:
        updated_annotations = []
        for ann in item.annotations:
            if isinstance(ann, dm.Bbox):
                ann = ann.wrap(label=self._new_label_id)
            updated_annotations.append(ann)

        return self.wrap_item(item, annotations=updated_annotations)

class PatchedCvatSubsetWriter(_SubsetWriter):
    # CVAT will require 'outside' property on the skeleton points,
    # but it is missing in the datumaro export in CVAT format
    # Here we fix this by monkey-patching the export method.

    def _write_shape(self, shape, item):
        xml_writer = self._writer

        def patched_open_points(points: Dict):
            if isinstance(shape, dm.Points):
                points['outside'] = str(int(shape.visibility[0] == dm.Points.Visibility.absent))
                points['occluded'] = str(int(shape.visibility[0] == dm.Points.Visibility.hidden))

            XmlAnnotationWriter.open_points(xml_writer, points)

        with mock.patch.object(self._writer, 'open_points', patched_open_points):
            return super()._write_shape(shape, item)

def prepare_import_dataset(kp_dataset_dir: str, dst_dir: str):
    kp_dataset_annotation_filename = next(
        fn for fn in glob(osp.join(kp_dataset_dir, 'annotations', '*.json'))
        if 'person_keypoints' in osp.basename(fn)
    )
    bbox_dataset = dm.Dataset.import_from(kp_dataset_annotation_filename, 'coco_instances')
    kp_dataset = dm.Dataset.import_from(kp_dataset_annotation_filename, 'coco_person_keypoints')

    # Boxes need to have a separate label in CVAT,
    # but they will be parsed with the same label as skeletons,
    # since they are read from the same annotation. So, we just remap the labels.
    resulting_labels = kp_dataset.categories()[dm.AnnotationType.label]
    bbox_dataset.transform('project_labels', dst_labels=resulting_labels)
    bbox_label_id = resulting_labels.find('person_bbox')[0] # <<<< use your bbox label name here
    assert bbox_label_id is not None

    output_dataset = dm.Dataset.from_extractors(bbox_dataset, kp_dataset)
    output_dataset.transform(RemapBboxLabels, new_label_id=bbox_label_id)

    with mock.patch('datumaro.plugins.cvat_format.converter._SubsetWriter', PatchedCvatSubsetWriter):
        output_dataset.export(dst_dir, 'cvat', save_images=True)


if __name__ == '__main__':
    prepare_import_dataset(sys.argv[1], sys.argv[2])
