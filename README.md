# Roots Project

Computer vision and robotics pipeline for detecting plant root tips and guiding a
simulated OT-2 pipette to inoculation targets.

```text
Plate image -> U-Net root segmentation -> skeleton/root tip extraction
            -> robot coordinate conversion -> PID-controlled OT-2 simulation
```

## Features

- U-Net inference for root pixel segmentation.
- Petri dish detection and image preprocessing.
- Mask cleanup, skeletonization, and root tip extraction.
- Calibrated conversion from image pixels to robot coordinates.
- PID-controlled movement in a PyBullet OT-2 simulation.
- Matched source images and simulation textures for bundled demonstrations.

## Repository Layout

```text
roots-project/
|-- data/samples/              # Sample input images
|-- docs/                      # Technical documentation
|-- models/                    # Local model placement
|-- scripts/                   # Command-line entry points
|-- src/roots_project/         # Python package
`-- tests/                     # Automated tests
```

## Setup

```bash
conda env create -f environment.yml
conda activate roots-project
```

Place the trained model at:

```text
models/mario_230623_unet_model_128px.h5
```

The model is not tracked because it exceeds GitHub's regular file limit. See
[`models/README.md`](models/README.md) for distribution options.

## Usage

Run the computer vision pipeline:

```bash
python scripts/run_cv_pipeline.py --image data/samples/task5_test_image.png --model models/mario_230623_unet_model_128px.h5 --output outputs/cv
```

Run a headless PID simulation:

```bash
python scripts/run_pid_demo.py --headless --max-tips 1
```

Run the full pipeline with a bundled plate:

```bash
python scripts/run_full_demo.py --max-tips 5 --seed 23
```

Run the full pipeline with a specific image:

```bash
python scripts/run_full_demo.py --image data/samples/test_image.png --model models/mario_230623_unet_model_128px.h5 --max-tips 5
```

Generated masks, skeletons, coordinates, diagnostics, and JSON results are written to
`outputs/`.

## Documentation

- [Architecture](docs/architecture.md)
- [Setup](docs/setup.md)
