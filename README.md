# LineReg

![Registration Animation](notebooks/combined_animation.gif)

## 📖 Introduction
**LineReg** enables single-view 2D-3D spine registration by optimizing 6DoF
camera poses with DRR generation and CMA-ES search. The current configuration
uses normalized cross-correlation (NCC); the optional edge Dice loss is disabled
in `lineReg_main.py`.

## 📰 News
* Our paper has been accepted by *Medical Physics* ! ✔
* Single-view registration on simulated X-rays. ✔
* Dual-view registration on simulated X-rays. ☐
* Dual-view registration on Real X-rays. ☐

## ✨ Key Features
* **High-Speed DRR Generation**: GPU-accelerated forward projection via [`nanodrr`](https://github.com/eigenvivek/nanodrr).
* **Edge Reference Generation**: Extracts reference edges and projects 3D side fiducials. Edge Dice loss code is available but currently disabled; optimization uses NCC.
* **Derivative-free Optimization**: Uses CMA-ES for 6DoF optimization, avoiding local optima without requiring differentiable renderers.
* **2D-3D Joint Visualization**: One-click generation of dynamic registration animations (2D overlays + 3D camera tracking) using `OpenCV` and `PyVista`.
* **Automated Full-Spine Segmentation Parsing**: Dynamically parses multi-label full-spine segmentations, allowing users to specify the target vertebra (e.g., L1-L5) for registration based on segmentation labels.

## 🛠️ Requirements
This project requires the following core libraries:
* `torch` >= 2.5.0
* `numpy`
* `pandas`
* `opencv-python`
* `SimpleITK`
* `pyvista`
* `cmaes`
* `tqdm`
* [`nanodrr`](https://github.com/eigenvivek/nanodrr) (`0.1.6`, pinned to the GitHub commit in `requirements.txt`)

On Windows, activate the Conda environment and install the requirements with:

```powershell
conda activate D:\anaconda\envs\linereg
pip install -r requirements.txt
```

Use a CUDA-enabled PyTorch build for GPU rendering. The installed `linereg`
environment contains PyTorch `2.12.1+cu126` and was verified on an RTX 4060 Ti.

The adapter in `nanodrr_adapter.py` preserves the original DiffDRR geometry:
centered RAS CT coordinates, AP view with reversed detector x axis, `ZXY` Euler
angles in radians, and the original vertebra-offset multiplication order.
It also preserves the old HU-to-density contrast. `Data/` and `results/` are
ignored by Git so patient data and generated images stay local.

Run the geometry and real CT projection check before registration:

```powershell
python verify_nanodrr.py
```

This saves `results/case1_L2/nanodrr_gt.png` and checks that the detector
center and x/z directions agree with the former DiffDRR setup.

## Why nanodrr can render faster

[`nanodrr`](https://github.com/eigenvivek/nanodrr) combines the camera-to-world,
world-to-voxel, and voxel-to-sampling-grid transforms into one matrix. It caches
volume transforms and detector rays, and applies ray-length scaling after the
sample reduction. These changes reduce repeated tensor operations and memory
traffic in its PyTorch renderer.

Its optional fused Triton backend performs ray sampling and accumulation inside
a GPU kernel, avoiding the large intermediate sample grids of the PyTorch path.
`torch.compile` and mixed precision can provide additional gains. See the
[upstream benchmarks](https://github.com/eigenvivek/nanodrr#benchmarks) for the
hardware, image dimensions, and backend settings behind the reported speedups;
those numbers are not measurements of LineReg on this machine.

The verified Windows environment currently uses nanodrr's **PyTorch CUDA
backend**, because Triton is unavailable. The adapter uses 500 samples per ray
and float32, without `torch.compile`. The old LineReg renderer used DiffDRR's
default Siddon integration, while nanodrr uses sampled trilinear ray marching.
Camera geometry is preserved, but image intensities can differ due to the
integration method and sample count. Compare image quality and runtime together
when tuning `LineRegDRR.n_samples`.

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
The script saves `results/case1_L2/nanodrr_gt.png` and the other reference
projections, displays the reference image, and runs CMA-ES after the plot window
is closed. The current optimization uses NCC only. After it completes, pose
results are saved in `results/`:

Pose records for each generation during the registration process (.csv).

### 3. visualization
You can modify the parameters in `reg_process_vis.py` to set the visualization registration results (e.g., `caseName = 'case1', vertName = 'L2'`), and then run the main program:
```bash
python reg_process_vis.py
```
```text
2D-3D joint evolution visualization animation (.gif)

Static reference projection images and edge reference images (.png)
```

## Citing `LineReg`

The DRR rendering backend is provided by
[`nanodrr` by Vivek Gopalakrishnan](https://github.com/eigenvivek/nanodrr).
Please also follow the upstream project's citation guidance when using it in
research.

If you find `LineReg` useful in your work, please cite our
[paper](https://doi.org/10.1002/mp.70385):

[//]: # (    @inproceedings{gopalakrishnan2022fast,)

[//]: # (      title={Fast auto-differentiable digitally reconstructed radiographs for solving inverse problems in intraoperative imaging},)

[//]: # (      author={Gopalakrishnan, Vivek and Golland, Polina},)

[//]: # (      booktitle={Workshop on Clinical Image-Based Procedures},)

[//]: # (      pages={1--11},)

[//]: # (      year={2022},)

[//]: # (      organization={Springer})

[//]: # (    })
