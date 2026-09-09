from __future__ import annotations

import numpy as np
import pytest

from bundle_adjustment import (
    BundleProblem,
    Camera,
    CameraIntrinsics,
    Observation,
    OptimizationOptions,
    Pose,
    optimize,
)
from bundle_adjustment.geometry import apply_pose_increment, project_point


def _look_at_pose(center: np.ndarray, target: np.ndarray) -> Pose:
    forward = target - center
    forward /= np.linalg.norm(forward)
    nominal_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, nominal_up)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    rotation = np.vstack((right, up, forward))
    return Pose(rotation=rotation, translation=-rotation @ center)


def _synthetic_problem() -> tuple[BundleProblem, BundleProblem]:
    generator = np.random.default_rng(7)
    intrinsics = CameraIntrinsics(fx=800.0, fy=810.0, cx=320.0, cy=240.0)
    true_poses = (
        _look_at_pose(np.array([-4.0, -3.0, 2.5]), np.zeros(3)),
        _look_at_pose(np.array([4.0, -2.5, 2.0]), np.zeros(3)),
        _look_at_pose(np.array([0.5, 4.5, 3.0]), np.zeros(3)),
    )
    true_points = generator.uniform([-1.3, -1.2, -0.8], [1.4, 1.1, 1.2], size=(12, 3))
    true_cameras = tuple(
        Camera(intrinsics, pose, str(index)) for index, pose in enumerate(true_poses)
    )
    observations = tuple(
        Observation(camera_index, point_index, project_point(intrinsics, pose, point))
        for camera_index, pose in enumerate(true_poses)
        for point_index, point in enumerate(true_points)
    )
    truth = BundleProblem(true_cameras, true_points, observations, name="synthetic")

    initial_poses = [true_poses[0]]
    for pose in true_poses[1:]:
        initial_poses.append(apply_pose_increment(pose, generator.normal(0.0, 0.015, size=6)))
    initial_points = true_points + generator.normal(0.0, 0.08, size=true_points.shape)
    initial_points[0] = true_points[0]
    initial = BundleProblem(
        tuple(Camera(intrinsics, pose, str(index)) for index, pose in enumerate(initial_poses)),
        initial_points,
        observations,
        name="synthetic",
    )
    return truth, initial


def test_problem_copies_points_and_optimizer_does_not_mutate_input() -> None:
    _, initial = _synthetic_problem()
    points_before = initial.points.copy()
    pose_before = initial.cameras[1].pose.translation.copy()

    result = optimize(
        initial,
        OptimizationOptions(
            loss="squared",
            max_iterations=80,
            auto_gauge_fix=False,
            fixed_camera_indices=(0,),
            fixed_point_indices=(0,),
        ),
    )

    np.testing.assert_allclose(initial.points, points_before)
    np.testing.assert_allclose(initial.cameras[1].pose.translation, pose_before)
    assert result.diagnostics.final_cost < result.diagnostics.initial_cost * 1e-5
    assert result.diagnostics.final_rmse < 1e-3
    assert result.converged
    assert result.diagnostics.accepted_steps > 0
    assert result.diagnostics.elapsed_seconds >= 0.0


def test_validation_rejects_observations_behind_camera() -> None:
    intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=0.0, cy=0.0)
    problem = BundleProblem(
        cameras=(Camera(intrinsics, Pose(np.eye(3), np.zeros(3))),),
        points=np.array([[0.0, 0.0, -1.0]]),
        observations=(Observation(0, 0, np.array([0.0, 0.0])),),
    )

    with pytest.raises(ValueError, match="depth"):
        optimize(problem)
