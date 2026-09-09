from __future__ import annotations

import numpy as np

from bundle_adjustment import CameraIntrinsics, Pose
from bundle_adjustment.geometry import (
    apply_pose_increment,
    camera_coordinates,
    project_point,
    project_point_jacobians,
)


def test_camera_center_and_projection_follow_world_to_camera_contract() -> None:
    pose = Pose(rotation=np.eye(3), translation=np.array([-2.0, 1.0, 3.0]))
    intrinsics = CameraIntrinsics(fx=100.0, fy=200.0, cx=10.0, cy=20.0)

    np.testing.assert_allclose(pose.camera_center, [2.0, -1.0, -3.0])
    np.testing.assert_allclose(camera_coordinates(pose, [3.0, 1.0, 2.0]), [1.0, 2.0, 5.0])
    np.testing.assert_allclose(project_point(intrinsics, pose, [3.0, 1.0, 2.0]), [30.0, 100.0])


def test_analytic_projection_jacobians_match_central_differences() -> None:
    intrinsics = CameraIntrinsics(fx=950.0, fy=980.0, cx=640.0, cy=480.0)
    pose = Pose(
        rotation=np.array(
            [
                [0.93629336, -0.31299183, -0.15934508],
                [0.28962948, 0.94470249, -0.15379200],
                [0.19866933, 0.09784340, 0.97517033],
            ]
        ),
        translation=np.array([0.2, -0.4, 4.0]),
    )
    point = np.array([0.8, -0.6, 2.1])
    pose_jacobian, point_jacobian = project_point_jacobians(intrinsics, pose, point)
    epsilon = 1e-7

    numeric_pose = np.empty((2, 6))
    for column in range(6):
        delta = np.zeros(6)
        delta[column] = epsilon
        numeric_pose[:, column] = (
            project_point(intrinsics, apply_pose_increment(pose, delta), point)
            - project_point(intrinsics, apply_pose_increment(pose, -delta), point)
        ) / (2.0 * epsilon)
    numeric_point = np.empty((2, 3))
    for column in range(3):
        delta = np.zeros(3)
        delta[column] = epsilon
        numeric_point[:, column] = (
            project_point(intrinsics, pose, point + delta)
            - project_point(intrinsics, pose, point - delta)
        ) / (2.0 * epsilon)

    np.testing.assert_allclose(pose_jacobian, numeric_pose, rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(point_jacobian, numeric_point, rtol=1e-6, atol=1e-5)
