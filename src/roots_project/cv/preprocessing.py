from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class CropBox:
    left: int
    right: int
    top: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True)
class Padding:
    bottom: int
    right: int


@dataclass(frozen=True)
class ResizedRoi:
    image: np.ndarray
    x_start: int
    y_start: int
    side_length: int

    @property
    def crop_box(self) -> CropBox:
        return CropBox(
            left=self.x_start,
            right=self.x_start + self.side_length,
            top=self.y_start,
            bottom=self.y_start + self.side_length,
        )


def load_grayscale_image(image_path: str | Path) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    return image


def load_color_image(image_path: str | Path) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    return image


def extract_resized_petri_roi(image: np.ndarray, output_size: tuple[int, int] = (1024, 1024)) -> ResizedRoi:
    """Match the original Task 5 inference crop/resize behavior."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        side_length = min(image.shape[:2])
        x_start = 0
        y_start = 0
    else:
        x, y, width, height = cv2.boundingRect(max(contours, key=cv2.contourArea))
        side_length = max(width, height)
        x_center = x + width // 2
        y_center = y + height // 2
        x_start = max(0, x_center - side_length // 2)
        y_start = max(0, y_center - side_length // 2)
        x_end = min(image.shape[1], x_start + side_length)
        y_end = min(image.shape[0], y_start + side_length)
        x_start = max(0, x_end - side_length)
        y_start = max(0, y_end - side_length)

    cropped = image[y_start : y_start + side_length, x_start : x_start + side_length]
    resized = cv2.resize(cropped, output_size, interpolation=cv2.INTER_AREA)
    return ResizedRoi(image=resized, x_start=x_start, y_start=y_start, side_length=side_length)


def reverse_resized_roi(mask: np.ndarray, roi: ResizedRoi, original_shape: tuple[int, int]) -> np.ndarray:
    aligned = np.zeros(original_shape[:2], dtype=mask.dtype)
    resized_mask = cv2.resize(
        mask,
        (roi.side_length, roi.side_length),
        interpolation=cv2.INTER_NEAREST,
    )
    aligned[roi.y_start : roi.y_start + roi.side_length, roi.x_start : roi.x_start + roi.side_length] = resized_mask
    return aligned


def detect_petri_dish_crop(gray_image: np.ndarray, max_size: int = 2800) -> CropBox:
    blurred = cv2.GaussianBlur(gray_image, (51, 51), 0)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = cv2.magnitude(sobel_x, sobel_y)

    _, edges = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
    edges = edges.astype(np.uint8)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return CropBox(0, gray_image.shape[1], 0, gray_image.shape[0])

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    side_length = min(max(w, h), max_size)
    center_x, center_y = x + w // 2, y + h // 2
    half = side_length // 2

    left = max(center_x - half, 0)
    top = max(center_y - half, 0)
    side = min(side_length, gray_image.shape[1] - left, gray_image.shape[0] - top)
    return CropBox(left=left, right=left + side, top=top, bottom=top + side)


def crop_image(image: np.ndarray, crop_box: CropBox) -> np.ndarray:
    return image[crop_box.top : crop_box.bottom, crop_box.left : crop_box.right]


def pad_to_patch_size(image: np.ndarray, patch_size: int) -> tuple[np.ndarray, Padding]:
    height, width = image.shape[:2]
    pad_bottom = (patch_size - (height % patch_size)) % patch_size
    pad_right = (patch_size - (width % patch_size)) % patch_size
    padded = cv2.copyMakeBorder(
        image,
        0,
        pad_bottom,
        0,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    return padded, Padding(bottom=pad_bottom, right=pad_right)


def remove_padding(image: np.ndarray, original_shape: tuple[int, int]) -> np.ndarray:
    height, width = original_shape
    return image[:height, :width]
