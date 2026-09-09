"""Explicit input and initial-geometry validation."""

from __future__ import annotations

from collections import Counter

from .geometry import camera_coordinates
from .models import BundleProblem, OptimizationOptions


def validate_problem(problem: BundleProblem, options: OptimizationOptions) -> None:
    """Validate graph coverage, fixed indices, and initial positive depths."""
    for camera_index in options.fixed_camera_indices:
        if camera_index < 0 or camera_index >= problem.num_cameras:
            raise ValueError(f"fixed camera index {camera_index} is outside the problem")
    for point_index in options.fixed_point_indices:
        if point_index < 0 or point_index >= problem.num_points:
            raise ValueError(f"fixed point index {point_index} is outside the problem")
    camera_counts: Counter[int] = Counter()
    point_counts: Counter[int] = Counter()
    invalid_depths = 0
    for observation in problem.observations:
        camera_counts[observation.camera_index] += 1
        point_counts[observation.point_index] += 1
        depth = camera_coordinates(
            problem.cameras[observation.camera_index].pose,
            problem.points[observation.point_index],
        )[2]
        if depth <= options.minimum_depth:
            invalid_depths += 1
    empty_cameras = [index for index in range(problem.num_cameras) if not camera_counts[index]]
    empty_points = [index for index in range(problem.num_points) if not point_counts[index]]
    if empty_cameras:
        raise ValueError(f"Cameras without observations: {empty_cameras}")
    if empty_points:
        raise ValueError(f"Points without observations: {empty_points}")
    if invalid_depths:
        raise ValueError(
            f"{invalid_depths} observation(s) have depth at or below minimum_depth "
            f"({options.minimum_depth}) in the initial geometry"
        )
