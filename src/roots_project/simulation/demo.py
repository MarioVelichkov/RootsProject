from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from roots_project.config import WorkEnvelope
from roots_project.cv.coordinates import RootTip
from .controller import PIDController


@dataclass(frozen=True)
class TargetResult:
    index: int
    target: tuple[float, float, float]
    final_position: tuple[float, float, float]
    final_xy_error_m: float
    final_z_error_m: float
    steps: int
    reached: bool
    deposited: bool
    droplet_position: tuple[float, float, float] | None


@dataclass(frozen=True)
class PidDemoResult:
    targets_attempted: int
    targets_reached: int
    targets_deposited: int
    results: list[TargetResult]


def run_pid_to_targets(
    simulation,
    targets: Iterable[RootTip | tuple[float, float, float]],
    max_tips: int = 5,
    threshold_m: float = 0.001,
    threshold_z_m: float = 0.002,
    max_steps_per_tip: int = 1200,
    hover_z_m: float = 0.190,
    envelope: WorkEnvelope = WorkEnvelope(),
) -> PidDemoResult:
    controllers = [
        PIDController(kp=5.0, ki=0.1, kd=0.01, dt=0.1, integral_limit=0.1),
        PIDController(kp=5.0, ki=0.1, kd=0.01, dt=0.1, integral_limit=0.1),
        PIDController(kp=5.0, ki=0.1, kd=0.01, dt=0.1, integral_limit=0.1),
    ]
    selected_targets = list(targets)[:max_tips]
    results: list[TargetResult] = []
    states = simulation.get_states()
    robot_key = next(iter(states))

    for raw_target in selected_targets:
        target = _normalise_target(raw_target, hover_z_m)
        target_is_valid = envelope.contains_xyz(*target) and simulation.contains_plate_xy(*target[:2])
        if not target_is_valid:
            results.append(
                TargetResult(
                    _target_index(raw_target),
                    target,
                    target,
                    float("inf"),
                    float("inf"),
                    0,
                    False,
                    False,
                    None,
                )
            )
            continue

        for controller in controllers:
            controller.reset()

        reached = False
        deposited = False
        droplet_position = None
        final_position = (0.0, 0.0, 0.0)
        final_xy_error = float("inf")
        final_z_error = float("inf")

        for step in range(max_steps_per_tip):
            current_states = simulation.get_states()
            current = np.array(current_states[robot_key]["pipette_position"], dtype=float)
            final_position = tuple(float(value) for value in current)
            final_xy_error = float(np.linalg.norm(np.array(target[:2]) - current[:2]))
            final_z_error = abs(float(target[2] - current[2]))
            if final_xy_error < threshold_m and final_z_error < threshold_z_m:
                reached = True
                simulation.run([[0.0, 0.0, 0.0, 1.0]], num_steps=1)
                simulation.run([[0.0, 0.0, 0.0, 0.0]], num_steps=80)
                droplet_position = getattr(simulation, "last_droplet_contact_position", None)
                deposited = droplet_position is not None
                break

            action = [
                controllers[0].compute(target[0], current[0]),
                controllers[1].compute(target[1], current[1]),
                controllers[2].compute(target[2], current[2]),
                0.0,
            ]
            simulation.run([action], num_steps=1)
        else:
            step = max_steps_per_tip

        results.append(
            TargetResult(
                index=_target_index(raw_target),
                target=target,
                final_position=final_position,
                final_xy_error_m=final_xy_error,
                final_z_error_m=final_z_error,
                steps=step + 1,
                reached=reached,
                deposited=deposited,
                droplet_position=_extract_droplet_position(droplet_position),
            )
        )

    return PidDemoResult(
        targets_attempted=len(results),
        targets_reached=sum(result.reached for result in results),
        targets_deposited=sum(result.deposited for result in results),
        results=results,
    )


def load_simulation_class():
    from .sim_class import Simulation

    return Simulation


def simulation_assets_dir() -> Path:
    return Path(__file__).resolve().parent / "assets"


def _normalise_target(target: RootTip | tuple[float, float, float], hover_z_m: float) -> tuple[float, float, float]:
    if isinstance(target, RootTip):
        return (target.x_m, target.y_m, hover_z_m)
    return (float(target[0]), float(target[1]), float(target[2]))


def _target_index(target: RootTip | tuple[float, float, float]) -> int:
    return target.index if isinstance(target, RootTip) else 0


def _extract_droplet_position(raw_value) -> tuple[float, float, float] | None:
    if raw_value is None or isinstance(raw_value, dict):
        return None
    try:
        return tuple(float(value) for value in raw_value[:3])
    except Exception:
        return None
