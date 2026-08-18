# AI-Based Restoration of Degraded Images
**Team:** We4Restore  
**Event:** KLA Hackathon 2026

## Overview
This repository contains our Phase 1 submission for the AI-Based Restoration of Degraded Images problem statement. Our solution provides an end-to-end, high-throughput deep learning pipeline using a highly optimized **Native NAFNet (Width=32, 2x Super-Resolution)** architecture.

To navigate the Perception-Distortion tradeoff and maximize both structural fidelity (SSIM) and pixel accuracy (PSNR), our model weights were optimized using a composite Charbonnier + MS-SSIM loss formulation evaluated against the complete 3,200-image dataset.

During inference, our standalone engine utilizes a geometric **8x Test-Time Augmentation (TTA)** pipeline. By mathematically ensembling predictions across all eight spatial orientations (rotations and reflections) and applying inverse geometry transforms, the model suppresses directional bias and hallucinated noise without adding any network parameters.

## Repository Structure
```text
We4Restore-KLA/
├── models/
│   └── nafnet_sr_best.pth # Pre-trained network weights (LFS)
├── results/               # Training progression & visual samples
├── run.py                 # Main offline inference engine
├── train.py               # Reproducible training pipeline
├── requirements.txt       # Minimal dependency list
└── README.md              # Documentation and execution instructions
