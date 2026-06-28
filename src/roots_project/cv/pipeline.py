from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from roots_project.config import PipelineSettings
from .coordinates import RootTip, pixel_to_robot_coordinates
from .postprocessing import clean_root_mask, extract_root_tips, skeletonize_mask
from .preprocessing import CropBox, extract_resized_petri_roi, load_color_image, reverse_resized_roi
from .segmentation import load_unet_model, predict_mask


@dataclass(frozen=True)
class RootPipelineResult:
    image_path: Path
    output_dir: Path
    crop_shape: tuple[int, int]
    mask_path: Path
    skeleton_path: Path
    tips_csv_path: Path
    metadata_path: Path
    diagnostic_path: Path
    root_tips: list[RootTip]

    def to_dict(self) -> dict:
        payload = asdict(self)
        for key in ["image_path", "output_dir", "mask_path", "skeleton_path", "tips_csv_path", "metadata_path", "diagnostic_path"]:
            payload[key] = str(payload[key])
        return payload


class RootAnalysisPipeline:
    def __init__(self, settings: PipelineSettings):
        self.settings = settings
        self.model = load_unet_model(settings.model_path)

    def run(self, image_path: str | Path, output_dir: str | Path | None = None) -> RootPipelineResult:
        source_path = Path(image_path)
        destination = Path(output_dir) if output_dir else self.settings.output_dir
        destination.mkdir(parents=True, exist_ok=True)

        image = load_color_image(source_path)
        roi = extract_resized_petri_roi(image)
        prediction = predict_mask(self.model, roi.image, self.settings.patch_size)
        aligned_prediction = reverse_resized_roi(prediction, roi, image.shape[:2])

        mask = (aligned_prediction > self.settings.threshold).astype(np.uint8)
        skeleton_mask = clean_root_mask(
            prediction=aligned_prediction,
            threshold=self.settings.threshold,
            min_size=self.settings.morph_min_size,
            kernel_size=self.settings.morph_kernel_size,
            dilate_iterations=self.settings.morph_dilate_iterations,
            close_iterations=self.settings.morph_close_iterations,
        )
        skeleton = skeletonize_mask(skeleton_mask)
        pixel_tips = extract_root_tips(
            skeleton,
            max_roots=self.settings.max_roots,
            plant_merge_distance_px=self.settings.plant_tip_merge_distance_px,
        )
        coordinate_crop = CropBox(
            left=-roi.x_start,
            right=roi.side_length - roi.x_start,
            top=-roi.y_start,
            bottom=roi.side_length - roi.y_start,
        )
        root_tips = pixel_to_robot_coordinates(
            pixel_tips=pixel_tips,
            crop_box=coordinate_crop,
            crop_width_px=roi.side_length,
            calibration=self.settings.calibration,
            envelope=self.settings.envelope,
        )

        stem = source_path.stem
        mask_path = destination / f"{stem}_root_mask.png"
        skeleton_path = destination / f"{stem}_skeleton.png"
        tips_csv_path = destination / f"{stem}_root_tips.csv"
        metadata_path = destination / f"{stem}_metadata.json"
        diagnostic_path = destination / f"{stem}_diagnostic.png"

        cv2.imwrite(str(mask_path), mask * 255)
        cv2.imwrite(str(skeleton_path), skeleton.astype(np.uint8) * 255)
        self._write_tips_csv(tips_csv_path, root_tips)
        self._write_diagnostic(diagnostic_path, image, roi.image, mask, skeleton, root_tips)

        result = RootPipelineResult(
            image_path=source_path,
            output_dir=destination,
            crop_shape=roi.image.shape[:2],
            mask_path=mask_path,
            skeleton_path=skeleton_path,
            tips_csv_path=tips_csv_path,
            metadata_path=metadata_path,
            diagnostic_path=diagnostic_path,
            root_tips=root_tips,
        )
        metadata_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        return result

    @staticmethod
    def _write_tips_csv(path: Path, root_tips: list[RootTip]) -> None:
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["index", "row", "col", "x_m", "y_m", "z_m", "in_work_envelope"],
            )
            writer.writeheader()
            for tip in root_tips:
                writer.writerow(asdict(tip))

    @staticmethod
    def _write_diagnostic(
        path: Path,
        image: np.ndarray,
        roi_image: np.ndarray,
        mask: np.ndarray,
        skeleton: np.ndarray,
        root_tips: list[RootTip],
    ) -> None:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Raw image")
        axes[1].imshow(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Resized Petri ROI")
        axes[2].imshow(mask, cmap="gray")
        axes[2].set_title("Root mask")
        # Thicken the one-pixel skeleton only for the downscaled diagnostic.
        line_width = max(3, int(round(max(skeleton.shape) / 800)))
        if line_width % 2 == 0:
            line_width += 1
        display_skeleton = cv2.dilate(
            skeleton.astype(np.uint8),
            np.ones((line_width, line_width), dtype=np.uint8),
        )
        axes[3].imshow(display_skeleton, cmap="gray", vmin=0, vmax=1)
        axes[3].set_title("Skeleton + tips")
        for tip in root_tips:
            axes[3].plot(tip.col, tip.row, "ro", markersize=4)
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
