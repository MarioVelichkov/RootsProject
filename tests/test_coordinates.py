from types import SimpleNamespace

import pytest

from roots_project.config import PlateCalibration, WorkEnvelope
from roots_project.cv.coordinates import pixel_to_robot_coordinates


def test_pixel_to_robot_coordinates_returns_work_envelope_targets():
    tips = pixel_to_robot_coordinates(
        pixel_tips=[(100.0, 120.0), (500.0, 600.0)],
        crop_box=SimpleNamespace(left=0, top=0),
        crop_width_px=1024,
        calibration=PlateCalibration(),
        envelope=WorkEnvelope(),
    )

    assert len(tips) == 2
    assert all(tip.in_work_envelope for tip in tips)
    assert tips[0].z_m == 0.190

def test_plate_origin_matches_simulation_holder():
    tip = pixel_to_robot_coordinates(
        pixel_tips=[(0.0, 0.0)],
        crop_box=SimpleNamespace(left=0, top=0),
        crop_width_px=1024,
        calibration=PlateCalibration(),
        envelope=WorkEnvelope(),
    )[0]

    assert tip.x_m == pytest.approx(0.10770)
    assert tip.y_m == pytest.approx(0.06200)
