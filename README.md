LineReg
================

> Semantic edges-guided 2D/3D registration in PyTorch

[![Paper shield](notebooks/mp_title.png)](http://www.cjig.cn/jig/ch/reader/view_abstract.aspx?flag=2&file_no=202401020000001&journal_id=jig)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

`LineReg` is a PyTorch-based semantic edges-guided 2D/3D registration methods for vertebrae in radiographs

1. Single-View/Dual-View 2D/3D registration methods for vertebrae
2. Simulation images study or real x-ray study 
3. Differentiable X-ray rendering
4. GPU-accelerated synthesis and optimization

Most importantly, `DiffDRR` implements DRR rendering as a PyTorch module, making it interoperable in deep learning pipelines.

## Install

To install the latest stable release (**recommended**):

```zsh
pip install diffdrr
```

To install the development version:

```zsh
git clone https://github.com/eigenvivek/DiffDRR.git --depth 1
pip install -e 'DiffDRR/[dev]'
```

## Hello, World!

The following minimal example specifies the geometry of the projectional radiograph imaging system and traces rays through a CT volume:


![](notebooks/index_files/figure-commonmark/cell-2-output-1.png)

On a single NVIDIA RTX 2080 Ti GPU, producing such an image takes

    25.2 ms ± 10.5 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

The full example is available at
[`introduction.ipynb`](https://vivekg.dev/DiffDRR/tutorials/introduction.html).

## Usage

### Rendering

The physics-based pipeline in `DiffDRR` renders photorealistic X-rays. For example, compare 
a real X-ray to a synthetic X-ray rendered from a CT of the same patient using `DiffDRR`
(X-rays and CTs from the [DeepFluoro dataset](https://github.com/rg2/DeepFluoroLabeling-IPCAI2020)):

![`DiffDRR` rendering from the same camera pose as a real X-ray.](notebooks/index_files/deepfluoro.png)

### 2D/3D Registration

The impotus for developing `DiffDRR` was to solve 2D/3D registration
problems with gradient-based optimization. Here, we demonstrate `DiffDRR`'s
capabilities by generating two DRRs:

1.  A fixed DRR from a set of ground truth parameters
2.  A moving DRR from randomly initialized parameters

To align the two images, we use gradient descent to maximize
an image similarity metric between the two DRRs. This produces
optimization runs like this:

![](experiments/registration.gif)

The full example is available at
[`optimizers.ipynb`](https://vivekg.dev/DiffDRR/tutorials/optimizers.html).

#### *🆕 Examples on Real-World Data 🆕*



## How does `DiffDRR` work?

`DiffDRR` reformulates Siddon’s method,[^1] an exact
algorithm for calculating the radiologic path of an X-ray
through a volume, as a series of vectorized tensor operations. This
version of the algorithm is easily implemented in tensor algebra
libraries like PyTorch to achieve a fast auto-differentiable DRR
generator.

[^1]: [Siddon RL. Fast calculation of
the exact radiological path for a three-dimensional CT array. Medical
Physics, 2(12):252–5, 1985.](https://doi.org/10.1118/1.595715)

## Citing `DiffDRR`

If you find `DiffDRR` useful in your work, please cite our
[paper](https://arxiv.org/abs/2208.12737):

    @inproceedings{gopalakrishnan2022fast,
      title={Fast auto-differentiable digitally reconstructed radiographs for solving inverse problems in intraoperative imaging},
      author={Gopalakrishnan, Vivek and Golland, Polina},
      booktitle={Workshop on Clinical Image-Based Procedures},
      pages={1--11},
      year={2022},
      organization={Springer}
    }

If the 2D/3D registration capabilities are helpful, please cite our followup, [`DiffPose`](https://arxiv.org/abs/2312.06358):

    @article{gopalakrishnan2023intraoperative,
      title={Intraoperative 2D/3D Image Registration via Differentiable X-ray Rendering},
      author={Gopalakrishnan, Vivek and Dey, Neel and Golland, Polina},
      journal={arXiv preprint arXiv:2312.06358},
      year={2023}
    }
