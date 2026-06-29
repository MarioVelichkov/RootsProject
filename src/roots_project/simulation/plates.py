from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from roots_project.cv.preprocessing import extract_resized_petri_roi, load_color_image


UV_ATLAS_SIZE = 1100
UV_FACE_SIZE = 278
UV_FACE_START = UV_ATLAS_SIZE - UV_FACE_SIZE


@dataclass(frozen=True)
class PlateSelection:
    pair_id: str
    source_image: Path
    texture_image: Path


def bundled_plate_pairs(assets_dir: str | Path) -> list[PlateSelection]:
    assets = Path(assets_dir)
    manifest_path = assets / "plate_pairs.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selections = [
        PlateSelection(
            pair_id=entry["id"],
            source_image=(assets / "textures" / "_plates" / entry["source"]).resolve(),
            texture_image=(assets / "textures" / entry["texture"]).resolve(),
        )
        for entry in manifest["pairs"]
    ]
    if not selections:
        raise ValueError(f"Plate pair manifest is empty: {manifest_path}")
    for selection in selections:
        if not selection.source_image.is_file() or not selection.texture_image.is_file():
            raise FileNotFoundError(f"Incomplete bundled plate pair: {selection.pair_id}")
    return selections


def select_bundled_plate(assets_dir: str | Path, seed: int | None = None) -> PlateSelection:
    return random.Random(seed).choice(bundled_plate_pairs(assets_dir))


def build_uv_texture(source_image: str | Path, output_dir: str | Path) -> PlateSelection:
    source = Path(source_image).resolve()
    roi = extract_resized_petri_roi(load_color_image(source)).image
    atlas = create_uv_atlas(roi)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    texture_path = destination / f"{source.stem}_simulation_texture.png"
    if not cv2.imwrite(str(texture_path), atlas):
        raise OSError(f"Could not write simulation texture: {texture_path}")
    return PlateSelection(source.stem, source, texture_path.resolve())


def create_uv_atlas(plate_roi: np.ndarray) -> np.ndarray:
    if plate_roi.ndim not in {2, 3}:
        raise ValueError(f"Unsupported plate ROI shape: {plate_roi.shape}")

    face = cv2.resize(
        cv2.flip(plate_roi, 1),
        (UV_FACE_SIZE, UV_FACE_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    atlas_shape = (UV_ATLAS_SIZE, UV_ATLAS_SIZE, *face.shape[2:])
    atlas = np.zeros(atlas_shape, dtype=face.dtype)
    atlas[UV_FACE_START:, UV_FACE_START:] = face
    return atlas
