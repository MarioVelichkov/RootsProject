import pytest

pybullet = pytest.importorskip("pybullet")

from roots_project.simulation.demo import load_simulation_class, simulation_assets_dir
from roots_project.simulation.plates import select_bundled_plate


def test_specimen_bounds_match_configured_plate_origin():
    selection = select_bundled_plate(simulation_assets_dir(), seed=1)
    Simulation = load_simulation_class()
    origin = (0.10770, 0.062)
    sim = Simulation(
        num_agents=1,
        render=False,
        assets_dir=simulation_assets_dir(),
        plate_image_path=selection.source_image,
        texture_path=selection.texture_image,
        plate_origin_xy=origin,
        plate_size_m=0.150,
    )
    try:
        bounds = sim.get_plate_bounds_xy()
        aabb_min, aabb_max = pybullet.getAABB(sim.specimenIds[0])

        assert bounds == pytest.approx((0.10770, 0.25770, 0.062, 0.212))
        assert (aabb_min[0], aabb_max[0]) == pytest.approx(bounds[:2])
        assert (aabb_min[1], aabb_max[1]) == pytest.approx(bounds[2:])
    finally:
        sim.close()


def test_droplet_reaches_plate_when_spawn_overlaps_source_robot():
    selection = select_bundled_plate(simulation_assets_dir(), seed=1)
    Simulation = load_simulation_class()
    sim = Simulation(
        num_agents=1,
        render=False,
        assets_dir=simulation_assets_dir(),
        plate_image_path=selection.source_image,
        texture_path=selection.texture_image,
        plate_origin_xy=(0.10770, 0.062),
        plate_size_m=0.150,
    )
    try:
        target = (0.1975, 0.0865, 0.1906)
        sim.set_start_position(*target)
        sim.run([[0.0, 0.0, 0.0, 1.0]], num_steps=1)
        sim.run([[0.0, 0.0, 0.0, 0.0]], num_steps=80)

        assert sim.last_droplet_contact_position is not None
        assert sim.last_droplet_contact_position[:2] == pytest.approx(target[:2], abs=1e-5)
        specimen_top = pybullet.getAABB(sim.specimenIds[0])[1][2]
        assert sim.last_droplet_contact_position[2] == pytest.approx(
            specimen_top + sim.droplet_radius
        )
    finally:
        sim.close()
