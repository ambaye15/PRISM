import numpy as np

from prism.validation.multiview.geometry import look_at_camera, project, triangulate_sequence
from prism.validation.multiview.measurements import estimate_frequency
from prism.validation.multiview.noninferiority import (
    camera_pool,
    dominant_motion_axis,
    projected_motion_fraction,
)
from prism.validation.multiview.construct_recovery import (
    irregular_trajectory,
    parse_design,
)


def test_triangulation_recovers_exact_points():
    rng = np.random.default_rng(4)
    points = rng.normal(size=(10, 6, 3)) * 0.2
    points[..., 1] += 1.0
    cameras = [look_at_camera(0, 10), look_at_camera(90, 10)]
    observations = []
    for camera in cameras:
        image, valid = project(points, camera["projection"])
        image[~valid] = np.nan
        observations.append(image)
    observations = np.stack(observations, axis=1)
    reconstructed = triangulate_sequence(
        observations, [camera["projection"] for camera in cameras]
    )
    assert np.nanmax(np.abs(reconstructed - points)) < 1e-6


def test_frequency_estimation():
    fps = 30.0
    frequency = 1.5
    time = np.arange(240) / fps
    points = np.zeros((len(time), 4, 3))
    points[:, :, 1] = np.array([0.0, 0.4, 0.8, 1.2])
    points[:, 0, 0] = 0.2 * np.sin(2 * np.pi * frequency * time)
    points[:, 3, 0] = -0.2 * np.sin(2 * np.pi * frequency * time)
    estimate = estimate_frequency(points, fps)
    assert abs(estimate - frequency) / frequency < 0.05


def test_clear_view_preserves_more_of_dominant_axis():
    points = np.zeros((60, 4, 3))
    time = np.linspace(0, 2 * np.pi, len(points))
    points[:, :, 1] = np.array([0.0, 0.4, 0.8, 1.2])
    points[:, 0, 0] = np.sin(time)
    points[:, 3, 0] = -np.sin(time)
    axis = dominant_motion_axis(points)
    quality = [projected_motion_fraction(axis, camera) for camera in camera_pool()]
    assert max(quality) > 0.99
    assert min(quality) < 0.10


def test_parse_design():
    values = parse_design("arm_flapping_f0.75_T10_A0.6_seed2")
    assert values["frequency_design_hz"] == 0.75
    assert values["duration_design_sec"] == 10.0
    assert values["amplitude_design"] == 0.6
    assert values["simulation_seed"] == 2
    assert np.isclose(values["vigor_design"], 0.45)


def test_zero_irregularity_preserves_trajectory():
    rng = np.random.default_rng(12)
    points = rng.normal(size=(60, 8, 3))
    changed = irregular_trajectory(points, 30.0, 0.0, 8)
    assert np.array_equal(points, changed)
