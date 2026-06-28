from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from roots_project.config import PlateCalibration, WorkEnvelope


class CropLike(Protocol):
    left: int
    top: int


@dataclass(frozen=True)
class RootTip:
    index: int
    row: float
    col: float
    x_m: float
    y_m: float
    z_m: float
    in_work_envelope: bool


def pixel_to_robot_coordinates(
    pixel_tips: list[tuple[float, float]],
    crop_box: CropLike,
    crop_width_px: int,
    calibration: PlateCalibration,
    envelope: WorkEnvelope,
) -> list[RootTip]:
    if crop_width_px <= 0:
        raise ValueError("crop_width_px must be positive")

    mm_per_pixel = calibration.plate_size_mm / crop_width_px
    tips: list[RootTip] = []

    for index, (row, col) in enumerate(pixel_tips):
        if calibration.rotate_image_axes:
            mm_x = (crop_box.top + row) * mm_per_pixel
            mm_y = (crop_box.left + col) * mm_per_pixel
        else:
            mm_x = (crop_box.left + col) * mm_per_pixel
            mm_y = (crop_box.top + row) * mm_per_pixel

        x_m = calibration.origin_x_m + (mm_x / 1000.0) + calibration.x_offset_m
        y_m = calibration.origin_y_m + (mm_y / 1000.0) + calibration.y_offset_m
        z_m = calibration.hover_z_m
        tips.append(
            RootTip(
                index=index,
                row=float(row),
                col=float(col),
                x_m=float(x_m),
                y_m=float(y_m),
                z_m=float(z_m),
                in_work_envelope=envelope.contains_xyz(x_m, y_m, z_m),
            )
        )

    return tips
