# Open Vocabulary Compositional Zero-shot Learning (OV-CZSL) 
This repository provides dataset splits and code for Paper:
### Beyond Seen Primitive Concepts and Attribute-Object Compositional Learning, CVPR 2024
[Nirat Saini](https://scholar.google.com/citations?hl=en&view_op=list_works&gmla=AJsN-F4kgg1kbcLx0j2dkvo5bGoQb9BU8bNEaEkiOirw72JFqU1cdNGVo3r8KTG7pq0yHTgIZ1M6jqtUUbXRAz_6YPTAeJjMwA&user=VsTvk-8AAAAJ),
[Khoi Pham](https://scholar.google.com/citations?user=o7hS8EcAAAAJ&hl=en),
[Abhinav Shrivastava](http://www.cs.umd.edu/~abhinav/)

 ## Code Instructions:
Pre-requisites:
Create the conda environment by:
```
conda env create -f environment_ov_czsl.yml
```

For each of the dataset, the config file needs to be used. For example, dataset MIT-states needs:
```
python train.py --cfg config/mit.yml
```
### For more details, refer to the [Project Page](https://ov-czsl.github.io/)
For questions and queries, feel free to reach out to [Nirat](niratsaini@gmail.com).

## Citation
Please cite our CVPR 2024 paper if you use the this repo for OV-CZSL.
```
@InProceedings{Saini_2024_CVPR,
    author    = {Saini, Nirat and Pham, Khoi and Shrivastava, Abhinav},
    title     = {Beyond Seen Primitive Concepts and Attribute-Object Compositional Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {14466-14476}
}
```
