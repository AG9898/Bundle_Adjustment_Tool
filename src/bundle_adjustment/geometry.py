"""Projection and pose-update operations using one explicit convention."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from .models import CameraIntrinsics, Pose

FloatArray: TypeAlias = npt.NDArray[np.float64]


def skew(vector: npt.ArrayLike) -> FloatArray:
    """Return the matrix ``[vector]_x`` such that ``[vector]_x @ b = vector × b``."""
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def camera_coordinates(pose: Pose, point_world: npt.ArrayLike) -> FloatArray:
    """Transform one world point with ``x_camera = R @ X_world + t``."""
    point = np.asarray(point_world, dtype=np.float64)
    if point.shape != (3,):
        raise ValueError(f"point_world must have shape (3,), got {point.shape}")
    return pose.rotation @ point + pose.translation


def project_camera_point(intrinsics: CameraIntrinsics, point_camera: npt.ArrayLike) -> FloatArray:
    """Project a positive-depth camera-space point using calibrated pinhole geometry."""
    x, y, z = np.asarray(point_camera, dtype=np.float64)
    if not np.isfinite((x, y, z)).all() or z <= 0.0:
        raise ValueError("A finite, positive-depth camera-space point is required")
    return np.array(
        [intrinsics.fx * x / z + intrinsics.cx, intrinsics.fy * y / z + intrinsics.cy],
        dtype=np.float64,
    )


def project_point(
    intrinsics: CameraIntrinsics, pose: Pose, point_world: npt.ArrayLike
) -> FloatArray:
    """Project one world point.

    Invalid depth raises ``ValueError`` rather than fabricating pixels.
    """
    return project_camera_point(intrinsics, camera_coordinates(pose, point_world))


def projection_jacobian_camera(
    intrinsics: CameraIntrinsics, point_camera: npt.ArrayLike
) -> FloatArray:
    """Return the 2x3 derivative of pixel coordinates with respect to camera coordinates."""
    x, y, z = np.asarray(point_camera, dtype=np.float64)
    if not np.isfinite((x, y, z)).all() or z <= 0.0:
        raise ValueError("A finite, positive-depth camera-space point is required")
    return np.array(
        [
            [intrinsics.fx / z, 0.0, -intrinsics.fx * x / (z * z)],
            [0.0, intrinsics.fy / z, -intrinsics.fy * y / (z * z)],
        ],
        dtype=np.float64,
    )


def project_point_jacobians(
    intrinsics: CameraIntrinsics, pose: Pose, point_world: npt.ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Return projection derivatives for a left-SE(3) pose update and world point.

    The pose increment is ``T_new = exp([omega, upsilon]) @ T``. Therefore
    ``d x_camera / d omega = -[x_camera]_x`` and
    ``d x_camera / d upsilon = I``.
    """
    point_camera = camera_coordinates(pose, point_world)
    jacobian_camera = projection_jacobian_camera(intrinsics, point_camera)
    pose_jacobian = jacobian_camera @ np.hstack((-skew(point_camera), np.eye(3)))
    point_jacobian = jacobian_camera @ pose.rotation
    return pose_jacobian, point_jacobian


def apply_pose_increment(pose: Pose, increment: npt.ArrayLike) -> Pose:
    """Apply a six-vector left-SE(3) increment to a world-to-camera pose."""
    delta = np.asarray(increment, dtype=np.float64)
    if delta.shape != (6,):
        raise ValueError(f"pose increment must have shape (6,), got {delta.shape}")
    if not np.all(np.isfinite(delta)):
        raise ValueError("pose increment must be finite")
    delta_rotation = Rotation.from_rotvec(delta[:3]).as_matrix()
    return Pose(
        rotation=delta_rotation @ pose.rotation,
        translation=delta_rotation @ pose.translation + delta[3:],
    )
