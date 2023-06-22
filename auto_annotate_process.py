# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import copy
from ikomia import core, dataprocess, utils
from segment_anything import SamPredictor
from auto_annotate.utils.voc2coco import convert, get_xml_files
from auto_annotate.utils.load_models import load_grounding_dino, load_sam_predictor
from auto_annotate.utils.auto_annot_utils import make_folders_and_files, enhance_class_name
from ikomia.dnn.dataset import read_class_names
import os
import torch
from typing import List
import numpy as np
import cv2
import supervision as sv
from tqdm import tqdm


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class AutoAnnotateParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        self.classes = 'car, person, dog, chair'
        self.task = 'object detection'
        self.dataset_split_ratio = 0.8
        self.model_name_grounding_dino = "Swin-B"
        self.model_name_sam = "vit_l"
        self.conf_thres = 0.35
        self.conf_thres_text = 0.25
        self.cuda = torch.cuda.is_available()
        self.update = False
        self.min_image_area_percent = 0.002
        self.max_image_area_percent = 0.80
        self.approximation_percent = 0.2
        self.input_image_folder = ""
        self.output_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "annotations")

    def set_values(self, params):
        self.classes = params["classes"]
        self.task = params["task"]
        self.dataset_split_ratio = float(params["dataset_split_ratio"])
        self.model_name_grounding_dino = params["model_name_grounding_dino"]
        self.model_name_sam = params["model_name_sam"]
        self.conf_thres = float(params["conf_thres"])
        self.conf_thres_text = float(params["conf_thres_text"])
        self.cuda = utils.strtobool(params["cuda"])
        self.update = True
        self.min_image_area_percent = float(params["min_image_area_percent"])
        self.max_image_area_percent = float(params["max_image_area_percent"])
        self.approximation_percent = float(params["approximation_percent"])
        self.input_image_folder = params["input_image_folder"]
        self.output_folder = params["output_folder"]

    def get_values(self):
        params = {}
        params["classes"] = str(self.classes)
        params["task"] = str(self.task)
        params["dataset_split_ratio"] = str(self.dataset_split_ratio)
        params["model_name_grounding_dino"] = str(self.model_name_grounding_dino)
        params["model_name_sam"] = str(self.model_name_sam)
        params["conf_thres"] = str(self.conf_thres)
        params["conf_thres_text"] = str(self.conf_thres_text)
        params["cuda"] = str(self.cuda)
        params["min_image_area_percent"] = str(self.min_image_area_percent)
        params["max_image_area_percent"] = str(self.max_image_area_percent)
        params["approximation_percent"] = str(self.approximation_percent)
        params["input_image_folder"] = str(self.input_image_folder)
        params["output_folder"] = str(self.output_folder)
        return params

# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class AutoAnnotate(core.CWorkflowTask):

    def __init__(self, name, param):
        core.CWorkflowTask.__init__(self, name)
        # Create parameters class
        if param is None:
            self.set_param_object(AutoAnnotateParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.device = torch.device("cpu")
        self.grounding_dino_model = None
        self.sam_predictor = None
        self.img_extension = ["jpeg", "jpg", "png", "bmp", "tiff",
                              "tif", "dib", "jpe", "jp2", "webp", 
                              "pbm", "pgm","ppm", "pxm", "pnm", "sr", 
                              "ras", "exr", "hdr", "pic"
        ]
        self.class_list_enhanced = None

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def segment(self, sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    def run(self):
        # Core function of your process
        # Call begin_task_run() for initialization
        self.begin_task_run()
        # Get parameters :
        param = self.get_param_object()

        if param.update or self.grounding_dino_model or self.sam_predictor is None:
            self.device = torch.device("cuda") if param.cuda else torch.device("cpu")

            self.grounding_dino_model = load_grounding_dino(
                                            param.model_name_grounding_dino,
                                            self.device
                                            )
            if param.task == "segmentation":
                self.sam_predictor = load_sam_predictor(param.model_name_sam, self.device)

            param.update = False

        # Get image list:
        image_paths = sv.list_files_with_extensions(
            directory=param.input_image_folder,
            extensions=self.img_extension
        )

        # Edit classes list from prompt or file:
        if os.path.isfile(param.classes):
            class_list = read_class_names(param.classes)
            print(class_list)
        else:
            class_list = param.classes.split(', ')
        self.class_list_enhanced = enhance_class_name(class_list)

        images = {}
        annotations = {}

        # Run models over images:
        for image_path in tqdm(image_paths):
            image_name = image_path.name
            image_path = str(image_path)
            image = cv2.imread(image_path)

            # Grounding DINO object detection:
            detections = self.grounding_dino_model.predict_with_classes(
                image=image,
                classes= self.class_list_enhanced,
                box_threshold=param.conf_thres,
                text_threshold=param.conf_thres_text,
            )

            detections = detections[detections.class_id != None]

            if param.task == "segmentation":
                # SAM segmentation:
                detections.mask = self.segment(
                    sam_predictor=self.sam_predictor,
                    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    xyxy=detections.xyxy
                )

            images[image_name] = image
            annotations[image_name] = detections

        # Create output folder & file
        annot_dir_voc, json_coco_train_bbox, json_coco_test_bbox, \
            json_coco_train_seg, json_coco_test_seg = make_folders_and_files(
            param.output_folder,
            param.task,
            param.dataset_split_ratio
        )

        # Export annotations as Pascal VOC
        sv.Dataset(
            classes=class_list,
            images=images,
            annotations=annotations
        ).as_pascal_voc(
            annotations_directory_path=annot_dir_voc,
            min_image_area_percentage=param.min_image_area_percent,
            max_image_area_percentage=param.max_image_area_percent,
            approximation_percentage=param.approximation_percent
        )

        # List of xml annotation files
        xml_files_train ,xml_files_test  = get_xml_files(annot_dir_voc, param.dataset_split_ratio)

        # Convert voc to coco annotations
        convert(
            xml_files=xml_files_train,
            json_file=json_coco_train_bbox,
            task="object detection"
        )
        if param.dataset_split_ratio < 1:
            convert(
                xml_files=xml_files_test,
                json_file=json_coco_test_bbox,
                task="object detection"
            )

        if param.task == "segmentation":
            convert(
                xml_files=xml_files_train,
                json_file=json_coco_train_seg,
                task=param.task
            )
            if param.dataset_split_ratio < 1:
                convert(
                    xml_files=xml_files_test,
                    json_file=json_coco_test_seg,
                    task=param.task
                )

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class AutoAnnotateFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "auto_annotate"
        self.info.short_description = "Auto-annotate images with GroundingDINO and SAM models"
        self.info.description = "Auto-annotate images using text prompt." \
                                "GroundingDINO is used for object detection (bbox) " \
                                "followed by SAM for segmentation. Annotations are saved in " \
                                "Pascal VOC format and COCO format. The COCO annotation (.json) is " \
                                "compatible with the Ikomia 'dataset_coco' dataloader."
        # relative path -> as displayed in Ikomia application process tree
        self.info.icon_path = "icons/icon.png"
        self.info.path = "Plugins/Python/Dataset"
        self.info.version = "1.0.0"
        self.info.authors = "Liu et al. (GroundingDINO), Kirillov et al. (SAM)"
        self.info.article = ""
        self.info.journal = "ArXiv"
        self.info.year = 2023
        self.info.license = "Apache 2.0"
        # URL of documentation
        self.info.documentation_link = ""
        # Code source repository
        self.info.repository = ""
        # Keywords used for search
        self.info.keywords = "auto-annotation, labelling, groundingdino, SAM, coco, voc, segmentation"

    def create(self, param=None):
        # Create process object
        return AutoAnnotate(self.info.name, param)
