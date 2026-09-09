"""Typed, immutable-facing data models for bundle-adjustment problems."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
LossName = Literal["squared", "huber"]
TerminationReason = Literal[
    "converged_step",
    "converged_cost",
    "max_iterations",
    "linear_solver_failure",
    "invalid_initial_geometry",
    "damping_limit_reached",
]


def _readonly_copy(values: npt.ArrayLike, shape: tuple[int, ...], name: str) -> FloatArray:
    """Return a finite, immutable float64 array with an exact shape."""
    array = np.array(values, dtype=np.float64, copy=True)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Pinhole intrinsics in pixel units."""

    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        values = (self.fx, self.fy, self.cx, self.cy)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Camera intrinsics must be finite")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("fx and fy must be positive")


@dataclass(frozen=True, slots=True)
class Pose:
    """World-to-camera extrinsics: ``x_camera = rotation @ X_world + translation``."""

    rotation: FloatArray
    translation: FloatArray

    def __post_init__(self) -> None:
        rotation = _readonly_copy(self.rotation, (3, 3), "rotation")
        translation = _readonly_copy(self.translation, (3,), "translation")
        orthogonality_error = np.linalg.norm(rotation @ rotation.T - np.eye(3))
        determinant = np.linalg.det(rotation)
        if orthogonality_error > 1e-7 or not np.isclose(determinant, 1.0, atol=1e-7):
            raise ValueError("rotation must be a proper orthonormal 3x3 matrix")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    @property
    def camera_center(self) -> FloatArray:
        """Return the world-space camera centre derived from the extrinsics."""
        center = -self.rotation.T @ self.translation
        center.setflags(write=False)
        return center


@dataclass(frozen=True, slots=True)
class Camera:
    """A calibrated pinhole camera and its world-to-camera pose."""

    intrinsics: CameraIntrinsics
    pose: Pose
    identifier: str | None = None


@dataclass(frozen=True, slots=True)
class Observation:
    """A 2D measurement of a landmark in one camera."""

    camera_index: int
    point_index: int
    xy: FloatArray
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.camera_index < 0 or self.point_index < 0:
            raise ValueError("camera_index and point_index must be non-negative")
        if not np.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("observation weight must be finite and positive")
        object.__setattr__(self, "xy", _readonly_copy(self.xy, (2,), "observation xy"))


@dataclass(frozen=True, slots=True)
class BundleProblem:
    """An initialized sparse reconstruction to refine.

    The object copies and freezes its point array. It is therefore safe for the
    optimizer to promise that the caller's problem will not be mutated.
    """

    cameras: Sequence[Camera]
    points: FloatArray
    observations: Sequence[Observation]
    name: str = "unnamed"

    def __post_init__(self) -> None:
        cameras = tuple(self.cameras)
        observations = tuple(self.observations)
        points = np.array(self.points, dtype=np.float64, copy=True)
        if not cameras:
            raise ValueError("At least one camera is required")
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError(f"points must have shape (N, 3) with N > 0, got {points.shape}")
        if not np.all(np.isfinite(points)):
            raise ValueError("points must contain only finite values")
        if not observations:
            raise ValueError("At least one observation is required")
        for index, observation in enumerate(observations):
            if observation.camera_index >= len(cameras):
                raise ValueError(
                    f"Observation {index} references unknown camera {observation.camera_index}"
                )
            if observation.point_index >= len(points):
                raise ValueError(
                    f"Observation {index} references unknown point {observation.point_index}"
                )
        points.setflags(write=False)
        object.__setattr__(self, "cameras", cameras)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "observations", observations)

    @property
    def num_cameras(self) -> int:
        return len(self.cameras)

    @property
    def num_points(self) -> int:
        return len(self.points)


@dataclass(frozen=True, slots=True)
class OptimizationOptions:
    """Controls for the reference Levenberg--Marquardt optimizer.

    A gauge is fixed automatically when the relevant parameter class is
    optimized: camera zero anchors the global rigid transform and point zero
    anchors the remaining calibrated scale ambiguity. Applications with their
    own constraints can disable ``auto_gauge_fix`` and supply fixed indices.
    """

    max_iterations: int = 50
    initial_damping: float = 1e-3
    damping_multiplier: float = 10.0
    max_damping: float = 1e12
    step_tolerance: float = 1e-8
    cost_tolerance: float = 1e-12
    loss: LossName = "huber"
    huber_delta: float = 2.0
    optimize_poses: bool = True
    optimize_points: bool = True
    fixed_camera_indices: tuple[int, ...] = ()
    fixed_point_indices: tuple[int, ...] = ()
    auto_gauge_fix: bool = True
    minimum_depth: float = 1e-8

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.initial_damping <= 0.0 or not np.isfinite(self.initial_damping):
            raise ValueError("initial_damping must be finite and positive")
        if self.damping_multiplier <= 1.0 or not np.isfinite(self.damping_multiplier):
            raise ValueError("damping_multiplier must be finite and greater than one")
        if self.max_damping < self.initial_damping or not np.isfinite(self.max_damping):
            raise ValueError("max_damping must be finite and at least initial_damping")
        if self.step_tolerance < 0.0 or self.cost_tolerance < 0.0:
            raise ValueError("tolerances must be non-negative")
        if self.loss not in ("squared", "huber"):
            raise ValueError("loss must be 'squared' or 'huber'")
        if self.huber_delta <= 0.0 or not np.isfinite(self.huber_delta):
            raise ValueError("huber_delta must be finite and positive")
        if self.minimum_depth <= 0.0 or not np.isfinite(self.minimum_depth):
            raise ValueError("minimum_depth must be finite and positive")


@dataclass(frozen=True, slots=True)
class IterationSummary:
    """Metrics recorded after one LM proposal."""

    iteration: int
    cost: float
    damping: float
    step_norm: float
    accepted: bool
    invalid_projection_count: int


@dataclass(frozen=True, slots=True)
class OptimizationDiagnostics:
    """Metrics needed to assess and reproduce one optimization run."""

    initial_cost: float
    final_cost: float
    initial_rmse: float
    final_rmse: float
    initial_invalid_projection_count: int
    final_invalid_projection_count: int
    accepted_steps: int
    rejected_steps: int
    elapsed_seconds: float
    iterations: tuple[IterationSummary, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """The immutable result of one optimization attempt."""

    problem: BundleProblem
    diagnostics: OptimizationDiagnostics
    termination_reason: TerminationReason
    converged: bool
