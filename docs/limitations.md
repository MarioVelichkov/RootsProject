# Limitations

- The full pipeline requires a TensorFlow model that is distributed separately.
- Calibration is covered by geometry tests but may require adjustment for different
  images or simulation assets.
- The simulation uses PID control; reinforcement learning is not included.
- Custom images must match the plate and root imagery used to train the model.
- PyBullet GUI capture requires separate screen-recording software.
