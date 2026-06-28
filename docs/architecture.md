# Architecture

## Pipeline Overview

```mermaid
flowchart LR
  A["Raw plate image"] --> B["Petri dish crop"]
  B --> C["U-Net root segmentation"]
  C --> D["Morphological cleanup"]
  D --> E["Skeletonization"]
  E --> F["Root tip extraction"]
  F --> G["Pixel-to-robot transform"]
  G --> H["PID controller"]
  H --> I["OT-2 PyBullet simulation"]
```

## Active Modules

- `roots_project.cv.preprocessing`: image loading, Petri dish detection, cropping, patch padding.
- `roots_project.cv.segmentation`: TensorFlow model loading and patch-wise U-Net inference.
- `roots_project.cv.postprocessing`: binary mask cleanup, skeletonization, and root tip extraction.
- `roots_project.cv.coordinates`: image pixel coordinates to robot-space target coordinates.
- `roots_project.simulation`: PID control and OT-2 simulation orchestration.

## Coordinate System

The CV pipeline emits pixel coordinates as `(row, column)`. The active transform follows the working integration approach from the reference implementation:

- image row maps to plate X in millimeters,
- image column maps to plate Y in millimeters,
- plate millimeters are converted to meters,
- calibrated offsets align plate coordinates to the OT-2 work envelope,
- Z is held at a fixed hover height for stable inoculation.

Calibration lives in `PlateCalibration` in `src/roots_project/config.py`.
