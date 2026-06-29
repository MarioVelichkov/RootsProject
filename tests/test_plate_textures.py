from pathlib import Path

import numpy as np

from roots_project.simulation.demo import simulation_assets_dir
from roots_project.simulation.plates import (
    UV_FACE_START,
    bundled_plate_pairs,
    create_uv_atlas,
    select_bundled_plate,
)


def test_bundled_plate_pairs_are_stable_and_complete():
    pairs = bundled_plate_pairs(simulation_assets_dir())

    assert len(pairs) == 10
    assert pairs[0].pair_id == "01"
    assert pairs[0].source_image.name.startswith("030_43-2-")
    assert pairs[-1].pair_id == "10"
    assert pairs[-1].source_image.name.startswith("038_43-14-")


def test_bundled_plate_selection_is_reproducible_with_seed():
    first = select_bundled_plate(simulation_assets_dir(), seed=23)
    second = select_bundled_plate(simulation_assets_dir(), seed=23)

    assert first == second


def test_uv_atlas_places_horizontally_flipped_roi_on_box_face():
    roi = np.zeros((16, 16, 3), dtype=np.uint8)
    roi[:, :8] = (20, 40, 60)
    roi[:, 8:] = (180, 200, 220)

    atlas = create_uv_atlas(roi)
    face = atlas[UV_FACE_START:, UV_FACE_START:]

    assert atlas.shape == (1100, 1100, 3)
    assert not atlas[:UV_FACE_START, :UV_FACE_START].any()
    assert face[:, :100].mean() > face[:, -100:].mean()
