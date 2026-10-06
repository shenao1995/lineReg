# LineReg

![Registration Animation](notebooks/combined_animation.gif)

## 📖 Introduction
**LineReg** provides simulated single-view registration and a separate real
X-ray single/dual-view 2D-3D registration entry point. It combines nanodrr
trilinear DRR rendering, gradient normalized cross-correlation (GNCC), and
6DoF CMA-ES optimization. Real dual-view registration jointly optimizes one
shared CT-to-world pose; optional edge Dice loss remains disabled in
`lineReg_main.py`.

## 📰 News
* Our paper has been accepted by *Medical Physics* ! ✔
* Single-view registration on simulated X-rays. ✔
* Dual-view registration on simulated X-rays. ☐
* Real X-ray single/dual-view GNCC registration entry point implemented. ✔
* Real-data single/dual-view nonblank DRR smoke checks completed. ✔
* Quantitative real X-ray registration accuracy validation. ☐

## ✨ Key Features
* **High-Speed DRR Generation**: GPU-accelerated forward projection via [`nanodrr`](https://github.com/eigenvivek/nanodrr).
* **Real X-ray Single/Dual-View Registration**: Select views using command-line parameters, with ROI-aware Sobel GNCC and one shared CT pose for dual-view optimization.
* **Explicit Coordinate Handling**: Reference-compatible `legacy_temp` and affine-aware `ras` modes, per-view XML calibration, and declared image/annotation transformations.
* **Edge Reference Generation**: Extracts reference edges and projects 3D side fiducials. Edge Dice loss code is available but currently disabled; optimization uses GNCC.
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

The adapter explicitly uses nanodrr's **Triton backend**. It was verified with
`triton-windows` 3.7.1 and PyTorch 2.12.1 on the Windows RTX 4060 Ti environment.
For this PyTorch version, install the matching Windows package with
`python -m pip install "triton-windows>=3.7,<3.8"`. Other PyTorch versions need
the corresponding Triton version; see the
[Windows compatibility table](https://github.com/triton-lang/triton-windows#3-pytorch).
The adapter uses 500 samples per ray and float32, without `torch.compile`.
To use the PyTorch fallback, change `backend="triton"` to `backend="torch"`
in `nanodrr_adapter.py`. The old LineReg renderer used DiffDRR's
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
is closed. The current optimization uses GNCC. After it completes, pose
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

## Real X-ray registration (single or dual view)

The independent entry point `real_xray_reg.py` reads real radiographs and XML
calibration from the reference project's data format. Both views optimize ONE
rigid CT-to-acquisition-world pose. Rendering uses
[`nanodrr`](https://github.com/eigenvivek/nanodrr) trilinear ray integration.

Copy `examples/real_xray_config.json` and edit the data paths. Relative paths are
resolved against the CONFIG directory, not the working directory. Activate the
`linereg` conda environment, then run from the project directory:

```powershell
python real_xray_reg.py --config examples/real_xray_config.json --mode dual
python real_xray_reg.py --config examples/real_xray_config.json --mode single --view ap
# Read/project only, without CMA-ES optimization:
python real_xray_reg.py --config examples/real_xray_config.json --mode dual --iterations 0
```

`gncc.py` implements signed Sobel gradient NCC: the average NCC of the X and Y
gradients. Loss = `1 - GNCC`; dual-view loss is the weighted mean of per-view
losses. With an ROI, means, variances, and covariance are computed only inside
the mask for BOTH images, after differentiation. Constant gradients score zero.
Target gradients are cached. The simulated entry also now uses this true GNCC.

Configuration details:

- `image_size`: `[width, height]`; each view retains its own XML spacing and SDD.
- `segmentation` and `label`: optional target-vertebra extraction, supporting
  binary or multi-label segmentation. Omit segmentation to render the whole CT.
- `segmentation_space="physical"` (default) requires a matching CT/seg affine.
  Some reference masks have their origin reset to zero: explicitly choose
  `"voxel"` ONLY when you know mask indices correspond directly to CT indices.
  This mode checks matching dimensions and deliberately ignores mask metadata.
- `bbox_json`: reference-format list of objects with `category_name` and
  `bbox: [x,y,width,height]`; `vertebra` selects the object. Alternatively supply
  `bbox` directly in a view. Omit both to use the full image.
- `bbox_size`: annotation image `[width,height]`, default original X-ray size.
  `bbox_frame="processed"` means coordinates AFTER the optional 180-degree
  rotation; `"raw"` rotates the box with the raw image. `bbox_margin` is in
  resized pixels. Review the saved ROI masks before a long optimization.
- `rotate_180`, `invert`, and `clahe` default true, following reference
  preprocessing but removing its random noise injection for reproducibility.
  Adjust these explicitly for different acquisition/intensity conventions.
- `initial_ct_to_world`: optional rigid 4x4 matrix in the declared CT frame;
  if present, it overrides automatic initialization. Dual-view initialization
  triangulates the two bbox-center rays (image-center rays if no boxes).
  This assumes both boxes depict the SAME vertebra. Single-view initialization
  requires this matrix OR `source_to_target_mm` (distance along selected ray).
  The example's 750 mm is a placeholder, NOT a depth estimate for your case.
- Optional `landmarks_json` reads `prediction[0][vertebra]`, as in the reference;
  or supply `landmark: [x,y,z]`. Explicitly set `landmark_space`:
  `"legacy_lps_mm"` for the reference's volume-local LPS mm in legacy mode
  (converted by `[-x+size_x*spacing_x,-y+size_y*spacing_y,z]`), or
  `"physical_ras"` / `"voxel"` (continuous CT voxel index) in RAS mode.
  Initialization then aligns this point, rather than the crop center, to the
  bbox ray(s). The rendering volume center remains tracked independently.
- `parameter_scales`: CMA-ES search scales for XYZ degrees and independent XYZ
  world translations in mm. Rotations are about the initially placed target.
  They are NOT the original `lineReg_main.py` ZXY camera-pose parameters.
- `backend`: `triton` (CUDA), `torch` (CPU or CUDA), or `auto`; `n_samples`
  controls integration resolution. `iterations=0` performs an initial projection
  check. Increase iterations/sample count after checking geometry.

### Coordinate conventions: choose explicitly

`coordinate_convention="legacy_temp"` (example/default) matches the inspected
`lineReg_temp/tools2.py` `get_ext_pose`, `crop_ct_vert`, and `update_pose_v2`
matrix chain, including row-stacked camera axes, old AP detector orientation,
180-degree target-only rotation, and historical `size*spacing/2` offsets. This
compatibility mode requires identity CT direction in LPS and fails on oblique
CTs rather than silently applying wrong offsets. CT origin is intentionally
ignored in this legacy LOCAL frame, as in the reference. The output transform
maps this local frame, NOT physical patient RAS, into acquisition world.

For column vectors, legacy camera-to-centered-volume is:

```text
V_offset @ inverse(T_half_extent) @ inverse(T_ct_to_world) @ C_reference @ B_AP
```

The reference's Euler implementation built `T_ct_to_world = R_XYZ @ T`, so its
translation column is `R @ t`. To reuse old numerical parameters, construct
that matrix and supply `initial_ct_to_world`; do not copy its six-vector as the
new optimizer's delta parameters. Without an explicitly declared landmark,
auto-init aligns the rendered vertebra crop center with the bbox ray(s).

`coordinate_convention="ras"` instead uses the complete CT affine (origin,
spacing, direction, and the exact voxel-center coordinate), named XML detector
axes as COLUMN vectors, and the XML `ImgCenter` principal point. Intrinsics are
updated together with image rotation/resizing. No legacy AP reorientation is
applied in this mode. It assumes the XML describes the RAW detector image and
the initialization matrix maps physical patient RAS to acquisition world. Do
not switch conventions while reusing an unchanged initial matrix. XML pixel
coordinates are interpreted with pixel centers at `(column+.5,row+.5)`.

Outputs: `result.json` (explicit CT-to-world and centered-volume-to-CT matrices,
per-view GNCC and units), `history.csv`, target/mask/initial/final DRRs, and
color overlays. These files remain under ignored `results/`; medical data is
not committed. A nonblank projection and improved GNCC do not establish
registration accuracy: check anatomical correspondence and, when available,
landmark/TRE validation. Single-view depth remains weakly constrained.

For a local geometry/rendering check with the example CT data, run
`python verify_nanodrr.py`. For your real acquisition data, use the real X-ray
entry point with `--iterations 0` and inspect the generated DRRs and ROI masks
before optimization. These checks do not establish anatomical registration
accuracy.

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
