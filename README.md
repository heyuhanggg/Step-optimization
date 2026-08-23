# Step Optimization for Fast Diffusion Sampling

> **Research preview.** This repository accompanies a manuscript that is currently under review. The present release intentionally contains only the core timestep-optimization module and a small set of precomputed CIFAR-10 schedules. Additional experiment runners, model-specific adapters, configurations, and evaluation scripts are planned for release after the review process.

## Overview

This repository provides a compact, DPM-Solver-compatible implementation of an offline timestep optimization method for fast diffusion sampling. It requires no parameter training, but it does use calibration data and pretrained-denoiser evaluations during offline schedule construction. The method constructs a non-uniform sampling schedule for a fixed pretrained diffusion model and noise schedule under a prescribed number of neural function evaluations (NFEs).

The optimized schedule is computed before sampling and can be reused for subsequent runs performed under the same model and sampling configuration. 
## Release Scope

Included in this preview:

- the core schedule-optimization implementation;
- selected precomputed CIFAR-10 timestep schedules;
- basic instructions for loading and validating the released schedules.

Not included at this stage:

- pretrained model checkpoints;
- dataset and model adapters;
- full generation and evaluation pipelines;
- complete benchmark configurations;
- scripts used to produce every result reported in the manuscript.

These components are planned for the full release after completion of the review process. 

## Repository Structure

```text
.
|-- code_opt.py
|   Core utilities for timestep calibration, schedule initialization,
|   constrained optimization, and schedule serialization.
|
`-- cifar10_demo_time/
    |-- t10_opt_cifar10.pt
    |-- t12_opt_cifar10.pt
    `-- t15_opt_cifar10.pt
```

## Requirements

The core module requires:

- Python 3.9--3.11;
- PyTorch 2.1.x;
- NumPy;
- SciPy.

A minimal environment can be prepared with:

```bash
pip install torch==2.1.1 numpy==1.26.4 scipy==1.11.4
```

For CUDA-enabled PyTorch installations, follow the installation command appropriate for your CUDA version on the official PyTorch website.

## Precomputed CIFAR-10 Schedules

The preview contains the following time-domain schedules:

| NFE budget | File | Number of stored nodes |
|---:|---|---:|
| 10 | `cifar10_demo_time/t10_opt_cifar10.pt` | 11 |
| 12 | `cifar10_demo_time/t12_opt_cifar10.pt` | 13 |
| 15 | `cifar10_demo_time/t15_opt_cifar10.pt` | 16 |

Each file stores a one-dimensional PyTorch tensor containing `NFE + 1` time nodes, including both sampling endpoints. The nodes are ordered from the noisy endpoint to the clean endpoint.

Load and validate a released schedule as follows:

```python
from pathlib import Path

import torch

nfe = 10
schedule_path = Path("cifar10_demo_time") / f"t{nfe}_opt_cifar10.pt"

timesteps = torch.load(schedule_path, map_location="cpu", weights_only=True)
timesteps = timesteps.detach().flatten()

assert timesteps.numel() == nfe + 1
assert torch.all(timesteps[:-1] > timesteps[1:])
assert torch.isclose(timesteps[0], torch.tensor(1.0, dtype=timesteps.dtype))
assert torch.isclose(timesteps[-1], torch.tensor(1e-3, dtype=timesteps.dtype))

print(timesteps)
```

The exact integration hook depends on the sampler implementation. Convert the loaded tensor to the device and dtype expected by the downstream sampler before use.

## Core Optimizer

`code_opt.py` contains the reusable schedule-optimization components. Constructing a new schedule requires a compatible pretrained denoiser, its noise schedule, and calibration data. Because these interfaces differ across diffusion codebases, model-specific wrappers are intentionally not included in this review-stage release.

The released CIFAR-10 schedules provide the simplest entry point for checking schedule loading and sampler integration without rerunning the offline optimization procedure.

The released demo schedules are stored as bare `.pt` tensors for direct sampler integration. By contrast, the optional `save_schedule()` helper in `code_opt.py` writes a metadata-bearing JSON record. These are two intentionally different serialization formats.


## Citation

Citation information for the accompanying manuscript will be added after the review process.

## Contact

For questions about this preview, please open an issue in this repository.
