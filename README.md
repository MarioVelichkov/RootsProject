# Roots Project

Computer vision and robotics simulation pipeline for detecting plant root tips and guiding an OT-2 pipette to inoculation targets.

This repository is a cleaned, portfolio-ready reconstruction of an older university project. The focus is not coursework submission evidence; it is a professional demo that shows an end-to-end applied AI system:

```text
Plate image -> U-Net root segmentation -> skeleton/root tip extraction
            -> robot coordinate conversion -> PID-controlled OT-2 simulation
```

## Demo Status

The current main path is PID-based because it is deterministic, explainable, and suitable for a portfolio video. Reinforcement learning and presentation coursework files are intentionally excluded from the active pipeline.

Recommended portfolio recording flow:

1. Run the CV pipeline and inspect `outputs/cv/*_diagnostic.png`.
2. Run the full GUI simulation demo.
3. Record the PyBullet window while the pipette visits detected root tips.

## Technical Highlights

- U-Net inference for root pixel segmentation.
- Classical image preprocessing for Petri dish cropping.
- Morphological cleanup and skeletonization for root structure extraction.
- Root tip detection from skeleton branches.
- Explicit pixel-to-robot coordinate transformation with calibration offsets.
- PID control loop for moving an OT-2 pipette inside the measured work envelope.
- Clean CLI scripts and Conda setup for reproducible local execution.

## Repository Layout

```text
roots-project/
├── data/samples/              # Small sample images used for smoke tests and demos
├── docs/                      # Architecture, setup, provenance, and limitations
├── models/                    # Local model placement; use Git LFS or external download for publishing
├── notebooks/legacy/          # Archived original/reference notebooks, not active code
├── outputs/                   # Generated masks, skeletons, coordinates, plots, and demo artifacts
├── scripts/                   # Runnable CLI entrypoints
├── src/roots_project/         # Production Python package
└── tests/                     # Lightweight tests for core logic
```

The original raw material is preserved under `notebooks/legacy/` for traceability. The active code path lives under `src/`, `scripts/`, and `docs/`.

## Setup

Install Conda, then create the environment:

```bash
conda env create -f environment.yml
conda activate roots-project
```

If you already created the environment and only need to update it:

```bash
conda env update -f environment.yml --prune
conda activate roots-project
```

The default model path is:

```text
models/mario_230623_unet_model_128px.h5
```

Large model files should be handled with Git LFS or documented as external assets before publishing the repository.

## Run the CV Pipeline

```bash
python scripts/run_cv_pipeline.py ^
  --image data/samples/task5_test_image.png ^
  --model models/mario_230623_unet_model_128px.h5 ^
  --output outputs/cv
```

Outputs include:

- `*_root_mask.png`
- `*_skeleton.png`
- `*_root_tips.csv`
- `*_metadata.json`
- `*_diagnostic.png`

## Run the PID Simulation Demo

Headless smoke test:

```bash
python scripts/run_pid_demo.py --headless --max-tips 1
```

Visible PyBullet demo for recording:

```bash
python scripts/run_pid_demo.py --max-tips 3
```

## Run the Full Demo

Use the current simulation plate texture as the image source:

```bash
python scripts/run_full_demo.py --max-tips 5
```

Run with a specific image:

```bash
python scripts/run_full_demo.py ^
  --image data/samples/task5_test_image.png ^
  --model models/mario_230623_unet_model_128px.h5 ^
  --max-tips 5
```


## What I Built

This project demonstrates a practical AI-to-robotics workflow: detect biologically meaningful plant root targets from imagery, transform those targets into robot-space coordinates, and drive a simulated OT-2 pipette to those targets with a feedback controller.

The cleaned version emphasizes production-quality structure: reusable modules, typed configuration, explicit data flow, runnable scripts, and documentation written for technical recruiters and engineering reviewers.

## Reconstruction Notes

This repo was reconstructed from older project files. The early computer vision and model work is based primarily on Mario Velichkov's original project material. Some later simulation and integration logic was adapted from a completed reference implementation because the original integration was incomplete, especially around coordinate conversion from image space to robot space.

The active repository intentionally avoids academic evidence language, presentation deliverables, and RL training workflows. Those files can remain archived as legacy/reference material, but they are not part of the professional demo path.
