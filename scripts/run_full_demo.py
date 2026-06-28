from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roots_project.config import PipelineSettings
from roots_project.cv import RootAnalysisPipeline
from roots_project.simulation.demo import load_simulation_class, run_pid_to_targets, simulation_assets_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full CV-to-PID OT-2 demo.")
    parser.add_argument("--image", type=Path, help="Plate image. If omitted, use the current simulation plate texture.")
    parser.add_argument("--model", default=PipelineSettings.model_path, type=Path, help="Trained U-Net model path.")
    parser.add_argument("--output", default=PipelineSettings.output_dir, type=Path, help="CV output directory.")
    parser.add_argument("--max-tips", default=5, type=int, help="Maximum root tips to visit.")
    parser.add_argument("--headless", action="store_true", help="Run PyBullet without the GUI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Simulation = load_simulation_class()
    sim = Simulation(num_agents=1, render=not args.headless, assets_dir=simulation_assets_dir())
    try:
        image_path = args.image or Path(sim.get_plate_image())
        settings = PipelineSettings(model_path=args.model, output_dir=args.output, max_roots=args.max_tips)
        cv_result = RootAnalysisPipeline(settings).run(image_path, args.output)
        pid_result = run_pid_to_targets(sim, cv_result.root_tips, max_tips=args.max_tips)
        print(json.dumps({"cv": cv_result.to_dict(), "pid": pid_result}, default=lambda value: value.__dict__, indent=2))
    finally:
        sim.close()


if __name__ == "__main__":
    main()
