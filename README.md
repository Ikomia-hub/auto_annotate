# auto_annotate

Auto-annotate images using a text prompt. GroundingDINO is employed for object detection (bounding boxes), followed by SAM for segmentation. The annotations are then saved in both Pascal VOC format and COCO format. The COCO annotation file (.json) is compatible with the Ikomia 'dataset_coco' dataloader.


## :rocket: Ikomia API

### :wrench: Parameters
- **classes**: String or path to text file
- **task**: 'object detection' or 'segmentation'
- **model_name_grounding_dino**: 'Swin-T- or 'Swin-B'
- **model_name_sam**: 'vit_b', 'vit_l' or 'vit_h'
- **conf_thres**: Box confidence threshold of the GroundingDINO model
- **conf_thres_text**: Text confidence threshold of the GroundingDINO model
- **cuda**: (Bool) Workflow run of the specified device. (CPU if cuda = False)
- **min_image_area_percent**: The minimum percentage of detection area relative to the image area for a detection to be included
- **max_image_area_percent**: The maximum percentage of detection area relative to the image area for a detection to be included
- **approximation_percentage**: The percentage of polygon points to be removed from the input polygon, in the range [0, 1).
- **input_image_folder**: Path of your image folder
- **output_folder**: Path of the output annotation file


### :milky_way: Code snippet
```Python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display
from ikomia.utils import ik
import os


# Init your workflow
wf = Workflow()

# Add the auto_annotate process to the workflow and set parameters
annotate = wf.add_task(ik.auto_annotate(
        input_image_folder = "C:/Path/To/Your/Image/Folder"),
        classes = 'car, person, dog, chair'
        task = 'segmentation',
        model_name_grounding_dino = "Swin-T",
        model_name_sam = "vit_l",
        conf_thres = 0.50,
        conf_thres_text = 0.25,
        cuda = True,
        min_image_area_percent = 0.002,
        max_image_area_percent = 0.80,
        approximation_percent = 0.2,
        input_image_folder = "",
        output_folder = os.path.join(os.getcwd(), "annotations")
        )

# Run auto_annotate
wf.run()
```


## :black_nib: Citation

- GroundingDino
    - [Documentation](https://github.com/IDEA-Research/GroundingDINO)
    - [Code source](https://github.com/IDEA-Research/GroundingDINO) 

```bibtex
@article{liu2023grounding,
  title={Grounding dino: Marrying dino with grounded pre-training for open-set object detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}
```


- Segment Anything
    - [Documentation](https://segment-anything.com/)
    - [Code source](https://github.com/facebookresearch/segment-anything)   
```bibtex
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
