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
from PyQt5.QtCore import Qt
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

        # Input image folder
        self.browse_in_folder = pyqtutils.append_browse_file(
            self.grid_layout, label="Image folder",
            path=self.parameters.image_folder,
            tooltip="Select folder",
            mode=QFileDialog.Directory
        )

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

        self.checkbox_edit_model = QCheckBox("Show model settings:")
        self.checkbox_edit_model.setStyleSheet("text-decoration: underline; font-weight: bold; color: #ed8302")
        self.grid_layout.addWidget(self.checkbox_edit_model, self.grid_layout.rowCount(), 0)
        self.checkbox_edit_model.stateChanged.connect(self.toggleModelSettingsVisibility)

        # GroupBox for model's settings
        self.model_settings_group = QWidget()
        self.model_settings_layout = QGridLayout(self.model_settings_group)
        self.grid_layout.addWidget(self.model_settings_group, self.grid_layout.rowCount(), 0, 1, 2)

        # Model name GroundingDino
        self.combo_model_dino = pyqtutils.append_combo(self.model_settings_layout, "Model name GroundingDino")
        self.combo_model_dino.addItem("Swin-T")
        self.combo_model_dino.addItem("Swin-B")
        self.combo_model_dino.setCurrentText(self.parameters.model_name_grounding_dino)

        # Confidence thresholds
        self.spin_conf_thres_box = pyqtutils.append_double_spin(
                                                self.model_settings_layout,
                                                "Confidence threshold boxes",
                                                self.parameters.conf_thres,
                                                min=0., max=1., step=0.01, decimals=2
        )

        self.spin_conf_thres_text = pyqtutils.append_double_spin(
                                                    self.model_settings_layout,
                                                    "Confidence threshold text",
                                                    self.parameters.conf_thres_text,
                                                    min=0., max=1., step=0.01, decimals=2
        )

        # Model name SAM
        self.combo_model_name_sam = pyqtutils.append_combo(self.model_settings_layout, "Model name SAM")
        self.combo_model_name_sam.addItem("mobile_sam")
        self.combo_model_name_sam.addItem("vit_b")
        self.combo_model_name_sam.addItem("vit_l")
        self.combo_model_name_sam.addItem("vit_h")
        self.combo_model_name_sam.setCurrentText(self.parameters.model_name_sam)

        self.model_settings_group.setVisible(self.checkbox_edit_model.isChecked())

        # Edit annotation's parameters
        self.checkbox_edit_annotation = QCheckBox("Show annotation settings:")
        self.checkbox_edit_annotation.setStyleSheet("font-weight: bold; text-decoration: underline; color: #ed8302")

        self.grid_layout.addWidget(self.checkbox_edit_annotation, self.grid_layout.rowCount(), 0)
        self.checkbox_edit_annotation.stateChanged.connect(self.toggleAnnotationSettingsVisibility)

        # GroupBox for model's settings
        self.annotation_settings_group = QWidget()
        self.annotation_settings_layout = QGridLayout(self.annotation_settings_group)
        self.grid_layout.addWidget(self.annotation_settings_group, self.grid_layout.rowCount(), 0, 1, 2)

        # Train test split
        self.spin_train_test_split = pyqtutils.append_double_spin(
            self.annotation_settings_layout,
            "Train/test split (COCO)",
            self.parameters.dataset_split_ratio,
            min=0.01, max=1.0,
            step=0.05, decimals=2
        )

        self.spin_min_relative_object_size = pyqtutils.append_double_spin(
                                                    self.annotation_settings_layout,
                                                    "min_relative_object_size",
                                                    self.parameters.min_relative_object_size,
                                                    min=0., max=1., step=0.01, decimals=3
        )

        self.spin_max_relative_object_size = pyqtutils.append_double_spin(
                                                    self.annotation_settings_layout,
                                                    "max_relative_object_size",
                                                    self.parameters.max_relative_object_size,
                                                    min=0., max=1., step=0.01, decimals=2
        )

        self.spin_approximation_percent = pyqtutils.append_double_spin(
                                                self.annotation_settings_layout,
                                                "polygon_simplification_factor",
                                                self.parameters.approximation_percent,
                                                min=0., max=0.99, step=0.01, decimals=2
        )

        # Export COCO
        self.check_export_coco = pyqtutils.append_check(
                                    self.annotation_settings_layout,
                                    "Export in COCO format",
                                    self.parameters.export_coco)

        # Export Pascal VOC
        self.check_export_pascal_voc = pyqtutils.append_check(
                                    self.annotation_settings_layout,
                                    "Export in Pascal VOC format",
                                    self.parameters.export_pascal_voc)

        self.annotation_settings_group.setVisible(self.checkbox_edit_annotation.isChecked())


        # Output folder
        self.browse_out_folder = pyqtutils.append_browse_file(
            self.grid_layout, label="Output folder",
            path=self.parameters.output_folder,
            tooltip="Select folder",
            mode=QFileDialog.Directory
        )

        # Dataset name 
        self.qlabel_output_dataset_name = QLabel('Output dataset name (optional)')
        self.grid_layout.addWidget(self.qlabel_output_dataset_name, self.grid_layout.rowCount(), 0)
        self.edit_output_dataset_name = QLineEdit()
        self.grid_layout.addWidget(self.edit_output_dataset_name, self.grid_layout.rowCount()-1, 1)


        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def toggleModelSettingsVisibility(self, state):
        if state == Qt.Checked:
            self.model_settings_group.show()
        else:
            self.model_settings_group.hide()

    def toggleAnnotationSettingsVisibility(self, state):
        if state == Qt.Checked:
            self.annotation_settings_group.show()
        else:
            self.annotation_settings_group.hide()

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
        self.parameters.export_coco = self.check_export_coco.isChecked()
        self.parameters.export_pascal_voc = self.check_export_pascal_voc.isChecked()

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
