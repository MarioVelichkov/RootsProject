from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PIDController:
    kp: float
    ki: float
    kd: float
    dt: float = 0.1
    integral_limit: float | None = None

    def __post_init__(self) -> None:
        self.previous_error = 0.0
        self.integral = 0.0

    def reset(self) -> None:
        self.previous_error = 0.0
        self.integral = 0.0

    def compute(self, target: float, current: float) -> float:
        error = target - current
        self.integral += error * self.dt
        if self.integral_limit is not None:
            self.integral = max(-self.integral_limit, min(self.integral, self.integral_limit))
        derivative = (error - self.previous_error) / self.dt
        self.previous_error = error
        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
