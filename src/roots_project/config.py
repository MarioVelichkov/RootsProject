from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .paths import MODELS_DIR, OUTPUTS_DIR


@dataclass(frozen=True)
class WorkEnvelope:
    x_min: float = -0.187
    x_max: float = 0.253
    y_min: float = -0.171
    y_max: float = 0.220
    z_min: float = 0.169
    z_max: float = 0.290

    def contains_xy(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def contains_xyz(self, x: float, y: float, z: float) -> bool:
        return self.contains_xy(x, y) and self.z_min <= z <= self.z_max


@dataclass(frozen=True)
class PlateCalibration:
    plate_size_mm: float = 150.0
    origin_x_m: float = 0.10775
    origin_y_m: float = 0.024
    origin_z_m: float = 0.057
    x_offset_m: float = -0.00005
    y_offset_m: float = 0.038
    hover_z_m: float = 0.190
    rotate_image_axes: bool = True


@dataclass(frozen=True)
class PipelineSettings:
    model_path: Path = MODELS_DIR / "mario_230623_unet_model_128px.h5"
    output_dir: Path = OUTPUTS_DIR / "cv"
    patch_size: int = 128
    threshold: float = 0.5
    morph_min_size: int = 200
    morph_kernel_size: int = 5
    morph_dilate_iterations: int = 2
    morph_close_iterations: int = 1
    plant_tip_merge_distance_px: float = 128.0
    max_roots: int = 5
    calibration: PlateCalibration = PlateCalibration()
    envelope: WorkEnvelope = WorkEnvelope()
