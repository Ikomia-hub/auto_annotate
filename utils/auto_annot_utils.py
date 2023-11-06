import os
import json
from typing import List
import numpy as np
from segment_anything import SamPredictor


def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]

def make_folders_and_files(output_folder, task, ratio, export_coco):
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    annot_directory_voc = os.path.join(output_folder, "voc_annotations")
    if not os.path.isdir(annot_directory_voc):
        os.mkdir(annot_directory_voc)

    annot_dir_coco = os.path.join(output_folder, "coco_annotations")
    json_coco_train_bb = os.path.join(output_folder, "coco_annotations", "train_bbox.json")
    json_coco_test_bb = os.path.join(output_folder, "coco_annotations", "test_bbox.json")
    json_coco_train_s = os.path.join(output_folder, "coco_annotations", "train_segmentation.json")
    json_coco_test_s = os.path.join(output_folder, "coco_annotations", "test_segmentation.json")

    if export_coco:
        if not os.path.isdir(annot_dir_coco):
            os.mkdir(annot_dir_coco)
        if not os.path.isfile(json_coco_train_bb):
            filename = json_coco_train_bb
            data = {}
            # Open the file in write mode
            with open(filename, 'w') as file:
                json.dump(data, file)

        if ratio < 1:
            if not os.path.isfile(json_coco_test_bb):
                filename = json_coco_train_bb
                data = {}
                # Open the file in write mode
                with open(filename, 'w') as file:
                    json.dump(data, file)

        if task == "segmentation":
            if not os.path.isfile(json_coco_train_s):
                filename = json_coco_train_s
                data = {}
                # Open the file in write mode
                with open(filename, 'w') as file:
                    json.dump(data, file)

            if ratio < 1:
                if not os.path.isfile(json_coco_test_s):
                    filename = json_coco_train_s
                    data = {}
                    # Open the file in write mode
                    with open(filename, 'w') as file:
                        json.dump(data, file)

    return annot_directory_voc, json_coco_train_bb, json_coco_test_bb, json_coco_train_s, json_coco_test_s
