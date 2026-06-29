# Setup

## Conda Environment

```bash
conda env create -f environment.yml
conda activate roots-project
```

Python 3.10 is required for the pinned TensorFlow, PyBullet, OpenCV, scikit-image,
skan, and patchify dependencies.

## Model Placement

The default model path is:

```text
models/mario_230623_unet_model_128px.h5
```

The model exceeds GitHub's regular file limit. Distribute it using:

- Git LFS;
- a GitHub release asset; or
- an external download with placement instructions.

## Quick Checks

```bash
pytest
python scripts/run_pid_demo.py --headless --max-tips 1
python scripts/run_cv_pipeline.py --image data/samples/task5_test_image.png --model models/mario_230623_unet_model_128px.h5
python scripts/run_full_demo.py --headless --max-tips 1 --seed 23
python scripts/run_full_demo.py --image data/samples/test_image.png --headless --max-tips 1
```
