from roots_project.simulation.controller import PIDController


def test_pid_output_moves_toward_positive_error():
    controller = PIDController(kp=2.0, ki=0.0, kd=0.0)

    assert controller.compute(target=1.0, current=0.25) > 0
    assert controller.compute(target=0.0, current=0.25) < 0
