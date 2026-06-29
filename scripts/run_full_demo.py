from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roots_project.config import PipelineSettings
from roots_project.cv import RootAnalysisPipeline
from roots_project.simulation.demo import load_simulation_class, run_pid_to_targets, simulation_assets_dir
from roots_project.simulation.plates import build_uv_texture, select_bundled_plate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full CV-to-PID OT-2 demo.")
    parser.add_argument("--image", type=Path, help="Plate image. If omitted, select a paired bundled plate.")
    parser.add_argument("--model", default=PipelineSettings.model_path, type=Path, help="Trained U-Net model path.")
    parser.add_argument("--output", default=PipelineSettings.output_dir, type=Path, help="CV output directory.")
    parser.add_argument("--max-tips", default=5, type=int, help="Maximum root tips to visit.")
    parser.add_argument("--seed", type=int, help="Optional seed for reproducible bundled plate selection.")
    parser.add_argument("--headless", action="store_true", help="Run PyBullet without the GUI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assets_dir = simulation_assets_dir()
    settings = PipelineSettings(model_path=args.model, output_dir=args.output, max_roots=args.max_tips)

    if args.image is None:
        plate = select_bundled_plate(assets_dir, seed=args.seed)
    else:
        plate = build_uv_texture(args.image, args.output / "simulation_textures")

    cv_result = RootAnalysisPipeline(settings).run(plate.source_image, args.output)
    calibration = settings.calibration
    plate_origin_xy = (
        calibration.origin_x_m + calibration.x_offset_m,
        calibration.origin_y_m + calibration.y_offset_m,
    )

    Simulation = load_simulation_class()
    sim = Simulation(
        num_agents=1,
        render=not args.headless,
        assets_dir=assets_dir,
        plate_image_path=plate.source_image,
        texture_path=plate.texture_image,
        plate_origin_xy=plate_origin_xy,
        plate_size_m=calibration.plate_size_mm / 1000.0,
    )
    try:
        pid_result = run_pid_to_targets(sim, cv_result.root_tips, max_tips=args.max_tips)
        payload = {
            "plate": {
                "pair_id": plate.pair_id,
                "source_image": str(plate.source_image),
                "texture_image": str(plate.texture_image),
                "bounds_xy_m": sim.get_plate_bounds_xy(),
            },
            "cv": cv_result.to_dict(),
            "pid": pid_result,
        }
        print(json.dumps(payload, default=lambda value: value.__dict__, indent=2))
    finally:
        sim.close()


if __name__ == "__main__":
    main()
