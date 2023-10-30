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

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from auto_annotate.auto_annotate_process import AutoAnnotateParam

# PyQt GUI framework
from PyQt5.QtWidgets import *
from torch.cuda import is_available


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class AutoAnnotateWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = AutoAnnotateParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        # Cuda
        self.check_cuda = pyqtutils.append_check(
                                    self.grid_layout,
                                    "Cuda",
                                    self.parameters.cuda and is_available()
        )
        self.check_cuda.setEnabled(is_available())

        # Task name
        self.combo_task = pyqtutils.append_combo(self.grid_layout, "Task")
        self.combo_task.addItems(["object detection", "segmentation"])
        self.combo_task.setCurrentText(self.parameters.task)

        # Prompt
        self.edit_prompt = pyqtutils.append_browse_file(
            self.grid_layout, label="Classes list or file (.txt)",
            path=self.parameters.classes,
            filter="*.txt",
        )

        # Train test split
        self.spin_train_test_split = pyqtutils.append_double_spin(
            self.grid_layout,
            "Train/test split",
            self.parameters.dataset_split_ratio,
            min=0.01, max=1.0,
            step=0.05, decimals=2
        )

        # Model name GroundingDino
        self.combo_model_dino = pyqtutils.append_combo(self.grid_layout, "Model name GroundingDino")
        self.combo_model_dino.addItem("Swin-T")
        self.combo_model_dino.addItem("Swin-B")
        self.combo_model_dino.setCurrentText(self.parameters.model_name_grounding_dino)

        row_dino = self.grid_layout.rowCount()
        self.qlabel_dino_param = QLabel('Grounding Dino parameters:')
        self.grid_layout.addWidget(self.qlabel_dino_param, row_dino, 0)

        # Confidence thresholds
        self.spin_conf_thres_box = pyqtutils.append_double_spin(
                                                self.grid_layout,
                                                "Confidence threshold boxes",
                                                self.parameters.conf_thres,
                                                min=0., max=1., step=0.01, decimals=2
        )

        self.spin_conf_thres_text = pyqtutils.append_double_spin(
                                                    self.grid_layout,
                                                    "Confidence threshold text",
                                                    self.parameters.conf_thres_text,
                                                    min=0., max=1., step=0.01, decimals=2
        )

        # Model name SAM
        self.combo_model_name_sam = pyqtutils.append_combo(self.grid_layout, "Model name SAM")
        self.combo_model_name_sam.addItem("mobile_sam")
        self.combo_model_name_sam.addItem("vit_b")
        self.combo_model_name_sam.addItem("vit_l")
        self.combo_model_name_sam.addItem("vit_h")
        self.combo_model_name_sam.setCurrentText(self.parameters.model_name_sam)

        row_annot = self.grid_layout.rowCount()
        self.qlabel_annot_param = QLabel('Annotation parameters:')
        self.grid_layout.addWidget(self.qlabel_annot_param, row_annot, 0)

        self.spin_min_relative_object_size = pyqtutils.append_double_spin(
                                                    self.grid_layout,
                                                    "min_relative_object_size",
                                                    self.parameters.min_relative_object_size,
                                                    min=0., max=1., step=0.01, decimals=3)

        self.spin_max_relative_object_size = pyqtutils.append_double_spin(self.grid_layout,
                                                    "max_relative_object_size",
                                                    self.parameters.max_relative_object_size,
                                                    min=0., max=1., step=0.01, decimals=2)

        self.spin_approximation_percent = pyqtutils.append_double_spin(self.grid_layout,
                                                "polygon_simplification_factor",
                                                self.parameters.approximation_percent,
                                                min=0., max=0.99, step=0.01, decimals=2)

        # Input image folder
        self.browse_in_folder = pyqtutils.append_browse_file(
            self.grid_layout, label="Image folder",
            path=self.parameters.image_folder,
            tooltip="Select folder",
            mode=QFileDialog.Directory
        )


        # Output folder
        self.browse_out_folder = pyqtutils.append_browse_file(
            self.grid_layout, label="Output folder",
            path=self.parameters.output_folder,
            tooltip="Select folder",
            mode=QFileDialog.Directory
        )

        # Dataset name 
        self.qlabel_output_dataset_name = QLabel('Output dataset name:')
        self.grid_layout.addWidget(self.qlabel_output_dataset_name, self.grid_layout.rowCount(), 0)
        self.edit_output_dataset_name = QLineEdit()
        self.grid_layout.addWidget(self.edit_output_dataset_name, self.grid_layout.rowCount()-1, 1)

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.set_layout(layout_ptr)


    def on_apply(self):
        # Apply button clicked slot

        # Get parameters from widget
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.task = self.combo_task.currentText()
        self.parameters.dataset_split_ratio = self.spin_train_test_split.value(
        )
        self.parameters.model_name_grounding_dino = self.combo_model_dino.currentText()
        self.parameters.model_name_sam = self.combo_model_name_sam.currentText()
        self.parameters.classes = self.edit_prompt.path
        self.parameters.conf_thres = self.spin_conf_thres_box.value()
        self.parameters.conf_thres_text = self.spin_conf_thres_text.value()
        self.parameters.min_relative_object_size = self.spin_min_relative_object_size.value()
        self.parameters.max_relative_object_size = self.spin_max_relative_object_size.value()
        self.parameters.approximation_percent = self.spin_approximation_percent.value()
        self.parameters.image_folder = self.browse_in_folder.path
        self.parameters.output_folder = self.browse_out_folder.path
        self.parameters.output_dataset_name = self.edit_output_dataset_name.text()

        self.parameters.update = True

        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class AutoAnnotateWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "auto_annotate"

    def create(self, param):
        # Create widget object
        return AutoAnnotateWidget(param, None)
