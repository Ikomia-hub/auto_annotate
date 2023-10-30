<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/auto_annotate/main/icons/icon.png" alt="Algorithm icon">
  <h1 align="center">auto_annotate</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/auto_annotate">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/auto_annotate">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/auto_annotate/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/auto_annotate.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>


Auto-annotate images using a text prompt. GroundingDINO is employed for object detection (bounding boxes), followed by MobileSAM or SAM for segmentation. The annotations are then saved in both Pascal VOC format and COCO format. 

The COCO annotation file (.json) is compatible with the Ikomia [dataset_coco](https://app.ikomia.ai/hub/algorithms/dataset_coco/) dataloader.


![label illustration](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*SsCYoXlIRaWQcWcfbPeCew.png)


## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```



#### 2. Create your workflow


- **classes** (str) - default 'car, person, dog, chair' : list of classes (string) or path to file.txt (see template utils/classes_list_template.txt).
- **task** (str) - default 'object detection': 'object detection' or 'segmentation'.
- **dataset_split_ratio** (float) - default '0.8': Image split between train and test coco annotations.
- **model_name_grounding_dino** (str) - default 'Swin-B': 'Swin-T' or 'Swin-B'.
- **model_name_sam** (str) - default 'mobile_sam': 'mobile_sam', 'vit_b', 'vit_l' or 'vit_h'.
- **conf_thres** (float) - default '0.35': Box confidence threshold of the GroundingDINO model.
- **conf_thres_text** (float) - default '0.25': Text confidence threshold of the GroundingDINO model.
- **min_relative_object_size** (float) - default '0.002': The minimum percentage of detection area relative to the image area for a detection to be included.
- **max_relative_object_size** (float) - default '0.8': The maximum percentage of detection area relative to the image area for a detection to be included.
- **polygon_simplification_factor** (float) - default '0.8': The percentage of polygon points to be removed from the input polygon, in the range [0, 1[.
- **image_folder** (str): Path of your image folder.
- **output_folder** (str): Path of the output annotation file.
- **export_coco** (bool) - default 'True': Save annotation in COCO format.
- **export_pascal_voc** (bool) - default 'False': Save annotation in Pascal VOC format.

**Parameters** should be in **strings format**  when added to the dictionary.

*The code snippet below requires 6Gb of GPU memory*

```Python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add the auto_annotate process to the workflow and set parameters
annotate = wf.add_task(name = "auto_annotate")

annotate.set_parameters({
    "image_folder": "Path/To/Your/Image/Folder",
    "classes": "car, person, dog, chair",
    "task": "segmentation",
    "dataset_split_ratio": "0.8",
    "model_name_grounding_dino": "Swin-T",
    "model_name_sam" = "mobile_sam",
    "conf_thres": "0.35",
    "conf_thres_text": "0.25",
    "min_relative_object_size": "0.80",
    "output_folder": "Path/To/Annotations/Output/Folder"
})

# Run auto_annotate
wf.run()
```


## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).



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


- Segment Anything (SAM)
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

- Mobile SAM
    - [Documentation](https://segment-anything.com/)
    - [Code source](https://github.com/ChaoningZhang/MobileSAM) 

```bibtex
@article{mobile_sam,
  title={Faster Segment Anything: Towards Lightweight SAM for Mobile Applications},
  author={Zhang, Chaoning and Han, Dongshen and Qiao, Yu and Kim, Jung Uk and Bae, Sung Ho and Lee, Seungkyu and Hong, Choong Seon},
  journal={arXiv preprint arXiv:2306.14289},
  year={2023}
}
```





