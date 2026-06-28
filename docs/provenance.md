# Provenance

This repository separates the production demo from the old working material.

## Primary Original Sources

- `notebooks/legacy/mario/raw_project_files/task1` through `notebooks/legacy/mario/raw_project_files/task6-8`: original computer vision notebooks, masks, sample images, and model work.
- `notebooks/legacy/mario/raw_project_files/mario_230623_unet_model_128px.h5`: selected trained U-Net model for the active pipeline before it was copied into `models/`.
- `notebooks/legacy/mario/raw_project_files/Task_9_Evidence`: original robotics environment notes and work envelope values.
- `notebooks/legacy/mario/raw_project_files/Task_10_Evidence`: original wrapper/testing work related to simulation control.

## Reference Sources

- `notebooks/legacy/reference/raw_project_files/task9-10-11`: completed reference implementation for OT-2 simulation, PID integration, coordinate conversion experiments, URDF, meshes, and textures.
- `notebooks/legacy/reference/raw_project_files/Task8` and `notebooks/legacy/reference/raw_project_files/Pipeline`: reference implementations for root segmentation and root measurement experiments.

## Active Reconstruction Policy

- Use Mario's original CV/model work where it is complete and understandable.
- Use the reference implementation directly where it materially improves the final product, especially simulation integration and coordinate conversion.
- Keep legacy coursework artifacts out of the main README and runnable path.
- Document adapted logic honestly in this file and in the README.
