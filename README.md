# LineReg: Semantic edge-guided single-view 2D/3D registration for vertebrae in X-rays

![Registration Animation](notebooks/combined_animation.gif)

## 📖 Introduction
**LineReg** is a 2D-3D spine (vertebra) registration tool specifically designed for intraoperative X-ray images and preoperative CT scans. This project leverages Digitally Reconstructed Radiograph (DRR) generation technology, combining pixel-level Normalized Cross-Correlation (NCC) with morphological edge alignment (Dice Loss). It utilizes the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) to search for the optimal camera pose in a 6 Degrees of Freedom (6DoF) space, achieving high-precision single-view registration.

## 📰 News
* Our project has been accepted by *Medical Physics*! ✔
* Single-view registration on simulated X-rays. ✔
* Dual-view registration on simulated X-rays. ☐
* Dual-view registration on Real X-rays. ☐

## ✨ Key Features
* **High-Speed DRR Generation**: GPU-accelerated forward projection via the `diffdrr` framework.
* **Edge-Aware Optimization**: Combines GNCC and edge Dice loss to robustly handle soft tissue interference and complex anatomy.
* **Derivative-free Optimization**: Uses CMA-ES for 6DoF optimization, avoiding local optima without requiring differentiable renderers.
* **2D-3D Joint Visualization**: One-click generation of dynamic registration animations (2D overlays + 3D camera tracking) using `OpenCV` and `PyVista`.
* **Automated Full-Spine Segmentation Parsing**: Directly parses multi-label full-spine segmentations to automatically crop specific target vertebrae (e.g., L1-L5).

## 🛠️ Requirements
This project requires the following core libraries:
* `torch` >= 1.10.0
* `numpy`
* `pandas`
* `opencv-python`
* `SimpleITK`
* `pyvista`
* `cmaes`
* `tqdm`
* [`diffdrr`](https://github.com/eigenvivek/DiffDRR)

## 🚀 Quick Start

### 1. Data Preparation
Please place your CT data and segmentation files in the `Data` directory. The recommended directory structure is as follows:
```text
Data/
└── case1/
    ├── ct.nii.gz          # Preoperative full-body/full-spine CT
    └── ct_seg.nii.gz      # CT multi-label segmentation file (e.g., 21-25 corresponds to L1-L5)
```
### 2. Run Registration
You can modify the parameters in `lineReg_main.py` to set the registration target (e.g., `caseName = 'case1', vertName = 'L2'`), and then run the main program:
```bash
python lineReg_main.py
```
```text
Once the registration is complete, the system will automatically save the following in the results/ directory:

Pose records for each generation during the registration process (.csv)

2D-3D joint evolution visualization animation (.gif)

Static reference projection images and edge reference images (.png)
```

### 3. visualization
You can modify the parameters in `reg_process_vis.py` to set the visualization registration results (e.g., `caseName = 'case1', vertName = 'L2'`), and then run the main program:
```bash
python reg_process_vis.py
```


## Citing `LineReg`

If you find `LineReg` useful in your work, please cite our
[paper](https://arxiv.org/abs/2208.12737):

[//]: # (    @inproceedings{gopalakrishnan2022fast,)

[//]: # (      title={Fast auto-differentiable digitally reconstructed radiographs for solving inverse problems in intraoperative imaging},)

[//]: # (      author={Gopalakrishnan, Vivek and Golland, Polina},)

[//]: # (      booktitle={Workshop on Clinical Image-Based Procedures},)

[//]: # (      pages={1--11},)

[//]: # (      year={2022},)

[//]: # (      organization={Springer})

[//]: # (    })
