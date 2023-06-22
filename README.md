# auto_annotate

Auto-annotate images using a text prompt. GroundingDINO is employed for object detection (bounding boxes), followed by SAM for segmentation. The annotations are then saved in both Pascal VOC format and COCO format. The COCO annotation file (.json) is compatible with the Ikomia 'dataset_coco' dataloader.





## :black_nib: Citation

- GroundingDino   

```bibtex
@article{liu2023grounding,
  title={Grounding dino: Marrying dino with grounded pre-training for open-set object detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}
```





- Segment Anything
```bibtex
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```


    - [Documentation](https://segment-anything.com/)
    - [Code source](https://github.com/facebookresearch/segment-anything)