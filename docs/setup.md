# Setup

## Conda Environment

```bash
conda env create -f environment.yml
conda activate roots-project
```

The environment is pinned to Python 3.10 because the active stack relies on TensorFlow, PyBullet, OpenCV, scikit-image, skan, and patchify. The system Python 3.13 installation is intentionally not used.

## Model Placement

The active model path is:

```text
models/mario_230623_unet_model_128px.h5
```

The model is large. For GitHub publishing, use one of:

- Git LFS for `models/*.h5`
- a release asset
- documented manual placement in `models/`

## Quick Checks

```bash
pytest
python scripts/run_pid_demo.py --headless --max-tips 1
python scripts/run_cv_pipeline.py --image data/samples/task5_test_image.png --model models/mario_230623_unet_model_128px.h5
```


