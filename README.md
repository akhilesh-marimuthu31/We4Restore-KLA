# AI-Based Restoration of Degraded Images
**Team:** We4Restore  
**Event:** KLA Hackathon 2026

## Overview
This repository contains our submission for the AI-Based Restoration of Degraded Images problem statement. Our solution leverages a custom, highly optimized Native NAFNet Super-Resolution (SR) architecture tailored specifically for high-frequency semiconductor imagery.

To navigate the Perception-Distortion tradeoff and maximize both structural fidelity (SSIM) and pixel accuracy (PSNR), our model weights were optimized using a composite Charbonnier + MS-SSIM loss formulation evaluated against the complete 3,200-image dataset.

During inference, this execution script utilizes a geometric **8x Test-Time Augmentation (TTA)** pipeline. By mathematically ensembling predictions across all eight spatial orientations (rotations and reflections) and applying inverse geometry transforms, the model suppresses directional bias and hallucinated artifacts without relying on additional network parameters.

## Repository Structure
```text
We4Restore/
├── run.py                 # Main offline inference engine
├── requirements.txt       # Minimal dependency list
├── README.md              # Documentation and execution instructions
└── models/                
    └── nafnet_sr_best.pth # Pre-trained network weights