# AI-Based Restoration of Degraded Images

**Team:** We4Restore
**Event:** KLA Hackathon 2026

## Overview

We4Restore is an AI-based image restoration pipeline developed for the KLA Hackathon 2026 problem statement on restoring degraded images.

Our solution uses a highly optimized **Native NAFNet architecture (Width = 32) with 2× Super-Resolution** to reconstruct high-quality images from degraded inputs. The system is designed to balance structural fidelity and pixel-level accuracy while remaining suitable for efficient offline inference.

To address the **Perception-Distortion trade-off**, the model was optimized using a composite **Charbonnier + MS-SSIM loss** formulation and evaluated across the complete **3,200-image dataset**.

During inference, the system employs an **8× Geometric Test-Time Augmentation (TTA)** strategy. The input is transformed through eight spatial orientations consisting of rotations and reflections. Each transformed image is restored independently, inverse transformations are applied to the predictions, and the results are geometrically ensembled. This reduces directional bias and helps suppress restoration artifacts without increasing the number of trainable network parameters.

## Key Features

* Native NAFNet-based image restoration
* 2× Super-Resolution capability
* Width-32 lightweight architecture
* Composite Charbonnier + MS-SSIM training loss
* 8× geometric Test-Time Augmentation
* Offline inference with no API dependencies
* `.npy` input/output support
* Automatic NaN/Inf sanitization
* Output values strictly clamped to `[0.0, 1.0]`
* Grayscale output preserving the original filenames
* Reproducible training pipeline

## Repository Structure

```text
We4Restore-KLA/
├── models/
│   └── nafnet_sr_best.pth       # Best trained model weights (Git LFS)
├── results/                     # Training logs, metrics & visual samples
├── run.py                       # Offline inference engine
├── train.py                     # Reproducible training pipeline
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## Results & Metrics

The model was evaluated against the complete validation dataset.

| Metric               |       Result |
| -------------------- | -----------: |
| Peak Validation PSNR | **28.64 dB** |
| SSIM                 |   **0.7773** |
| LPIPS                |   **0.3923** |

These results demonstrate the model's ability to recover degraded structural information while maintaining perceptual and pixel-level fidelity.

## Restoration Samples

The qualitative restoration results follow the format:

```text
Top    → Degraded Input
Middle → Model Output
Bottom → Ground Truth
```

Representative samples are available in the `results/` directory.

## Methodology

The restoration pipeline consists of four major stages:

```text
Degraded Image
      │
      ▼
8× Geometric TTA
      │
      ▼
Native NAFNet
(Width = 32)
      │
      ▼
2× Super-Resolution
      │
      ▼
Inverse TTA + Ensemble
      │
      ▼
Restored Image
```

During training, the network minimizes a composite loss consisting of **Charbonnier loss** for robust pixel-level reconstruction and **MS-SSIM loss** for structural similarity.

During inference, the eight transformed predictions are aligned back to the original orientation and averaged to obtain the final restoration.

## Input / Output Contract

The inference engine expects directories containing NumPy arrays.

### Input

```text
<input-dir>/
├── image_001.npy
├── image_002.npy
├── image_003.npy
└── ...
```

Each input is read as a degraded grayscale image stored as a `.npy` array.

### Output

```text
<output-dir>/
├── image_001.npy
├── image_002.npy
├── image_003.npy
└── ...
```

The output arrays:

* Preserve the exact original filenames.
* Are stored as grayscale `(H, W)` NumPy arrays.
* Contain floating-point values.
* Are strictly clamped to the `[0.0, 1.0]` range.
* Have all `NaN` and `Inf` values removed before saving.

This provides a deterministic and clean interface for offline evaluation.

## Failure Analysis & Limitations

The model performs well on degraded images containing recoverable high-frequency structures, including fine semiconductor traces and other detailed patterns.

However, extremely low-contrast regions remain challenging. When the intensity of a degraded trace becomes very close to that of the surrounding wafer substrate, the available information becomes ambiguous. In these cases, the network can produce slightly over-smoothed edges or lose extremely fine structural details.

The current system also prioritizes restoration quality over minimum possible inference latency because of the 8× TTA ensemble.

### Future Improvements

Potential improvements include:

* FP8/low-precision quantization for edge deployment
* Knowledge distillation into a smaller restoration network
* Hardware-aware model optimization
* Faster TTA strategies
* Improved low-contrast feature recovery
* Additional perceptual and frequency-domain loss functions
* Deployment using optimized inference runtimes

## Environment & Setup

The project is designed for **standalone offline execution**.

No external repositories, API keys, cloud services, or internet connectivity are required after the required Python dependencies and model weights are available locally.

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Execution

### 1. Offline Inference

Run the restoration engine using:

```bash
python run.py <input-dir> <output-dir>
```

Example:

```bash
python run.py ./input ./output
```

The script automatically discovers all `.npy` files in the input directory, performs restoration using the trained NAFNet model and 8× TTA pipeline, sanitizes the predictions, and writes the restored arrays to the output directory.

### 2. Reproducible Training

Training can be launched using:

```bash
python train.py --data_dir "/path/to/dataset" --output_dir "./results" --epochs 25
```

Example:

```bash
python train.py \
    --data_dir "/path/to/dataset" \
    --output_dir "./results" \
    --epochs 25
```

The training pipeline stores model checkpoints, metrics, and training outputs in the specified results directory.

## Model

The primary trained checkpoint is:

```text
models/nafnet_sr_best.pth
```

The checkpoint contains the best-performing model obtained during training based on validation performance.

## Reproducibility

The repository contains the complete inference and training entry points required to reproduce the system:

```text
train.py → Model Training
run.py   → Offline Evaluation / Inference
```

All preprocessing, model execution, post-processing, and output sanitization required by the inference pipeline are handled programmatically.

## Team

**We4Restore**

KLA Hackathon 2026

---

*Built for efficient, high-fidelity restoration of degraded images using deep learning.*
