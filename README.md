We4Restore: AI-Based Restoration of Degraded Images

Team: We4Restore
Event: KLA Hackathon 2026

Overview

We4Restore is an AI-based image restoration pipeline developed for the KLA Hackathon 2026 problem statement on restoring degraded images. Our solution uses a highly optimized Native NAFNet architecture (Width = 32) with 2× Super-Resolution to reconstruct high-quality images from degraded inputs. The system is designed to balance structural fidelity and pixel-level accuracy while remaining suitable for efficient offline inference.

To address the Perception-Distortion trade-off, the model was optimized using a composite Charbonnier + MS-SSIM loss formulation and evaluated across the complete 3,200-image dataset.

During inference, the system employs an 8× Geometric Test-Time Augmentation (TTA) strategy. The input is transformed through eight spatial orientations consisting of rotations and reflections. Each transformed image is restored independently, inverse transformations are applied to the predictions, and the results are geometrically ensembled. This reduces directional bias and helps suppress restoration artifacts without increasing the number of trainable network parameters.

Key Features
Native NAFNet-based image restoration with 2× Super-Resolution and a Width-32 lightweight architecture.
Composite Charbonnier + MS-SSIM training loss alongside MixUp Augmentation.
8× Geometric Test-Time Augmentation utilizing $D_4$ dihedral symmetry.
Offline inference with no API dependencies.
.npy input/output support with automatic NaN/Inf sanitization.
Output values strictly clamped to [0.0, 1.0].
Grayscale output preserving the original filenames.
Reproducible training pipeline featuring cosine annealing learning-rate scheduling.
Repository Structure
We4Restore-KLA/
├── models/
│   └── nafnet_sr_best.pth       # Best trained model weights
├── results/                     # Training logs, metrics & visual samples
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── run.py                       # Offline inference engine
├── train.py                     # Reproducible training pipeline
└── visualize.py                 # Side-by-side array visualizer
Results & Metrics

The model was evaluated against the complete validation dataset.

Metric	Result
Peak Validation PSNR	28.64 dB
SSIM	0.7773
LPIPS	0.3923

These results demonstrate the model's ability to recover degraded structural information while maintaining perceptual and pixel-level fidelity. Representative sample grids are also saved automatically during validation.

Methodology

The restoration pipeline consists of four major stages:

Input: Ingestion of raw degraded .npy arrays.
Pre-Processing: 8× Geometric Test-Time Augmentation.
Core Processing: Native NAFNet (Width = 32).
Synthesis: 2× Super-Resolution → Inverse TTA → Ensemble → Restored Image.

During training, the network minimizes a composite loss consisting of Charbonnier loss for robust pixel-level reconstruction and MS-SSIM loss for structural similarity.

During inference, the eight transformed predictions are aligned back to the original orientation and averaged to obtain the final restoration.

Input / Output Contract

The inference engine expects directories containing NumPy arrays.

Input:
Each input is read as a degraded grayscale image stored as a .npy array.

Output Arrays:

Preserve the exact original filenames.
Are stored as grayscale NumPy arrays.
Contain floating-point values.
Are strictly clamped to the [0.0, 1.0] range.
Have all NaN and Inf values mathematically removed before saving.

This provides a deterministic and clean interface for offline evaluation.

Failure Analysis & Limitations

The model performs well on degraded images containing recoverable high-frequency structures, including fine semiconductor traces and other detailed patterns.

However, extremely low-contrast regions remain challenging. When the intensity of a degraded trace becomes very close to that of the surrounding wafer substrate, the available information becomes ambiguous. In these cases, the network can produce slightly over-smoothed edges or lose extremely fine structural details.

The current system also prioritizes restoration metrology quality over minimum possible inference latency because of the 8× TTA ensemble.

Future Improvements
FP8 / low-precision quantization for edge deployment.
Knowledge distillation into a smaller restoration network.
Hardware-aware model optimization.
Faster TTA strategies.
Improved low-contrast feature recovery using frequency-domain loss functions.
Environment & Setup

The project is designed for standalone offline execution. No external repositories, API keys, cloud services, or internet connectivity are required after the required Python dependencies and model weights are available locally.

Install Dependencies
pip install -r requirements.txt
Execution
1. Offline Inference

Run the restoration engine using:

python run.py ./input ./output

The script automatically discovers all .npy files in the input directory, performs restoration using the trained NAFNet model and 8× TTA pipeline, sanitizes the predictions, and writes the restored arrays to the output directory.

2. Visual Metrology Verification

Because .npy files are raw arrays, a visualization tool is included to render a side-by-side structural comparison of the tensor data:

python visualize.py ./input/image_001.npy ./output/image_001.npy
3. Reproducible Training

Training can be launched using:

python train.py --data_dir "/path/to/dataset" --output_dir "./results" --epochs 25

The training pipeline stores model checkpoints, metrics, and visual validation grid images in the specified results directory.

Model & Reproducibility

The primary trained checkpoint is:

models/nafnet_sr_best.pth

The checkpoint contains the best-performing model obtained during training based on validation performance.

The repository contains the complete inference and training entry points required to reproduce the system. All preprocessing, model execution, post-processing, and output sanitization required by the inference pipeline are handled programmatically.

We4Restore — AI-Based Restoration of Degraded Images
KLA Hackathon 2026
