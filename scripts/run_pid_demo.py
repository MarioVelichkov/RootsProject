from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roots_project.simulation.demo import load_simulation_class, run_pid_to_targets, simulation_assets_dir


DEFAULT_TARGETS = [
    (0.140, 0.090, 0.190),
    (0.175, 0.120, 0.190),
    (0.205, 0.155, 0.190),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a PID-only OT-2 simulation demo against fixed targets.")
    parser.add_argument("--max-tips", default=3, type=int, help="Number of fixed demo targets to visit.")
    parser.add_argument("--headless", action="store_true", help="Run PyBullet without the GUI.")
    parser.add_argument("--threshold", default=0.001, type=float, help="XY success threshold in meters.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Simulation = load_simulation_class()
    sim = Simulation(num_agents=1, render=not args.headless, assets_dir=simulation_assets_dir())
    try:
        result = run_pid_to_targets(
            sim,
            DEFAULT_TARGETS,
            max_tips=args.max_tips,
            threshold_m=args.threshold,
        )
        print(json.dumps(result, default=lambda value: value.__dict__, indent=2))
    finally:
        sim.close()


if __name__ == "__main__":
    main()
