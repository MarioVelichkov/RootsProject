import numpy as np

from roots_project.cv.postprocessing import extract_root_tips


def test_extract_root_tips_keeps_lowest_tip_per_plant_column():
    skeleton = np.zeros((30, 35), dtype=bool)
    skeleton[2:11, 5] = True
    skeleton[2:22, 8] = True
    skeleton[2:17, 28] = True

    tips = extract_root_tips(skeleton, max_roots=5, plant_merge_distance_px=5)

    assert (21.0, 8.0) in tips
    assert (10.0, 5.0) not in tips
    assert len(tips) == 2
