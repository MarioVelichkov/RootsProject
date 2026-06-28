from __future__ import annotations

from pathlib import Path

import numpy as np
from patchify import patchify, unpatchify


def load_unet_model(model_path: str | Path):
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import load_model

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (positives + K.epsilon())

    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        return true_positives / (predicted_positives + K.epsilon())

    def f1(y_true, y_pred):
        precision = precision_m(y_true, y_pred)
        recall = recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    return load_model(path, custom_objects={"f1": f1, "recall_m": recall_m, "precision_m": precision_m})


def predict_mask(model, image: np.ndarray, patch_size: int, batch_size: int = 64) -> np.ndarray:
    input_channels = _model_input_channels(model)
    if input_channels not in {1, 3}:
        raise ValueError(f"Unsupported model channel count: {input_channels}")

    if image.ndim == 2:
        patches = patchify(image, (patch_size, patch_size), step=patch_size)
        patches_reshaped = patches.reshape(-1, patch_size, patch_size, 1).astype("float32") / 255.0
        if input_channels == 3:
            patches_reshaped = np.repeat(patches_reshaped, 3, axis=-1)
    elif image.ndim == 3:
        image_channels = image.shape[-1]
        if image_channels != input_channels:
            if image_channels == 3 and input_channels == 1:
                image = image[..., :1]
            else:
                raise ValueError(f"Image has {image_channels} channels, model expects {input_channels}")
        patches = patchify(image, (patch_size, patch_size, input_channels), step=patch_size)
        patches_reshaped = patches.reshape(-1, patch_size, patch_size, input_channels).astype("float32") / 255.0
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    predictions = model.predict(patches_reshaped, batch_size=batch_size, verbose=0)
    if predictions.ndim == 4 and predictions.shape[-1] == 1:
        predictions = predictions[..., 0]

    rows = image.shape[0] // patch_size
    cols = image.shape[1] // patch_size
    predictions = predictions.reshape(rows, cols, patch_size, patch_size)
    return unpatchify(predictions, image.shape[:2])


def _model_input_channels(model) -> int:
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    channels = input_shape[-1]
    if channels is None:
        return 1
    return int(channels)
