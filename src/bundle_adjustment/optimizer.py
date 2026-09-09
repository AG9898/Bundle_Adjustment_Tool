"""Correctness-first sparse Levenberg--Marquardt bundle adjustment."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_array, diags, lil_array
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from .geometry import apply_pose_increment, camera_coordinates, project_point_jacobians
from .losses import robust_cost_and_weights
from .models import (
    BundleProblem,
    Camera,
    IterationSummary,
    OptimizationDiagnostics,
    OptimizationOptions,
    OptimizationResult,
    Pose,
    TerminationReason,
)
from .validation import validate_problem

FloatArray: TypeAlias = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _ParameterLayout:
    pose_offsets: dict[int, int]
    point_offsets: dict[int, int]
    size: int


@dataclass(frozen=True, slots=True)
class _Evaluation:
    residuals: FloatArray
    depths: FloatArray
    valid: npt.NDArray[np.bool_]
    cost: float
    rmse: float

    @property
    def invalid_count(self) -> int:
        return int(np.count_nonzero(~self.valid))


def _effective_fixed_indices(
    problem: BundleProblem, options: OptimizationOptions
) -> tuple[set[int], set[int]]:
    fixed_cameras = set(options.fixed_camera_indices)
    fixed_points = set(options.fixed_point_indices)
    if options.auto_gauge_fix and options.optimize_poses:
        fixed_cameras.add(0)
    if options.auto_gauge_fix and options.optimize_points:
        fixed_points.add(0)
    return fixed_cameras, fixed_points


def _parameter_layout(problem: BundleProblem, options: OptimizationOptions) -> _ParameterLayout:
    fixed_cameras, fixed_points = _effective_fixed_indices(problem, options)
    offset = 0
    pose_offsets: dict[int, int] = {}
    point_offsets: dict[int, int] = {}
    if options.optimize_poses:
        for index in range(problem.num_cameras):
            if index not in fixed_cameras:
                pose_offsets[index] = offset
                offset += 6
    if options.optimize_points:
        for index in range(problem.num_points):
            if index not in fixed_points:
                point_offsets[index] = offset
                offset += 3
    return _ParameterLayout(pose_offsets=pose_offsets, point_offsets=point_offsets, size=offset)


def _make_problem(
    problem: BundleProblem, poses: tuple[Pose, ...], points: FloatArray
) -> BundleProblem:
    cameras = tuple(
        Camera(intrinsics=camera.intrinsics, pose=pose, identifier=camera.identifier)
        for camera, pose in zip(problem.cameras, poses, strict=True)
    )
    return BundleProblem(
        cameras=cameras, points=points, observations=problem.observations, name=problem.name
    )


def _evaluate(problem: BundleProblem, options: OptimizationOptions) -> _Evaluation:
    residuals: FloatArray = np.full((len(problem.observations), 2), np.nan, dtype=np.float64)
    depths: FloatArray = np.empty(len(problem.observations), dtype=np.float64)
    valid: npt.NDArray[np.bool_] = np.zeros(len(problem.observations), dtype=bool)
    for row, observation in enumerate(problem.observations):
        camera = problem.cameras[observation.camera_index]
        point_camera = camera_coordinates(camera.pose, problem.points[observation.point_index])
        depths[row] = point_camera[2]
        if point_camera[2] <= options.minimum_depth:
            continue
        predicted = np.array(
            [
                camera.intrinsics.fx * point_camera[0] / point_camera[2] + camera.intrinsics.cx,
                camera.intrinsics.fy * point_camera[1] / point_camera[2] + camera.intrinsics.cy,
            ],
            dtype=np.float64,
        )
        residuals[row] = observation.xy - predicted
        valid[row] = True
    if not np.all(valid):
        return _Evaluation(
            residuals=residuals,
            depths=depths,
            valid=valid,
            cost=float("inf"),
            rmse=float("inf"),
        )
    unweighted_cost, _ = robust_cost_and_weights(residuals, options.loss, options.huber_delta)
    weights = np.array([observation.weight for observation in problem.observations])
    if options.loss == "squared":
        cost = (
            float(unweighted_cost)
            if np.all(weights == 1.0)
            else float(0.5 * np.sum(weights * np.sum(residuals * residuals, axis=1)))
        )
    else:
        norms = np.linalg.norm(residuals, axis=1)
        inlier = norms <= options.huber_delta
        robust_terms = np.where(
            inlier,
            0.5 * norms * norms,
            options.huber_delta * (norms - 0.5 * options.huber_delta),
        )
        cost = float(np.sum(weights * robust_terms))
    rmse = float(np.sqrt(np.mean(residuals * residuals)))
    return _Evaluation(residuals=residuals, depths=depths, valid=valid, cost=cost, rmse=rmse)


def _linearize(
    problem: BundleProblem, options: OptimizationOptions, layout: _ParameterLayout
) -> tuple[csr_array, FloatArray]:
    """Build a weighted projection Jacobian J and residual r for J delta ~= r."""
    if layout.size == 0:
        return csr_array((2 * len(problem.observations), 0)), np.empty(
            2 * len(problem.observations), dtype=np.float64
        )
    evaluation = _evaluate(problem, options)
    if evaluation.invalid_count:
        raise ValueError("Cannot linearize a problem containing invalid projections")
    _, robust_weights = robust_cost_and_weights(
        evaluation.residuals, options.loss, options.huber_delta
    )
    jacobian = lil_array((2 * len(problem.observations), layout.size), dtype=np.float64)
    weighted_residuals: FloatArray = np.empty(2 * len(problem.observations), dtype=np.float64)
    for row, observation in enumerate(problem.observations):
        camera = problem.cameras[observation.camera_index]
        pose_jacobian, point_jacobian = project_point_jacobians(
            camera.intrinsics,
            camera.pose,
            problem.points[observation.point_index],
        )
        scale = np.sqrt(observation.weight * robust_weights[row])
        row_slice = slice(2 * row, 2 * row + 2)
        weighted_residuals[row_slice] = scale * evaluation.residuals[row]
        pose_offset = layout.pose_offsets.get(observation.camera_index)
        if pose_offset is not None:
            jacobian[row_slice, pose_offset : pose_offset + 6] = scale * pose_jacobian
        point_offset = layout.point_offsets.get(observation.point_index)
        if point_offset is not None:
            jacobian[row_slice, point_offset : point_offset + 3] = scale * point_jacobian
    return jacobian.tocsr(), weighted_residuals


def _apply_step(
    problem: BundleProblem, layout: _ParameterLayout, step: FloatArray
) -> BundleProblem:
    poses = list(camera.pose for camera in problem.cameras)
    points = np.array(problem.points, copy=True)
    for camera_index, offset in layout.pose_offsets.items():
        poses[camera_index] = apply_pose_increment(poses[camera_index], step[offset : offset + 6])
    for point_index, offset in layout.point_offsets.items():
        points[point_index] += step[offset : offset + 3]
    return _make_problem(problem, tuple(poses), points)


def optimize(
    problem: BundleProblem, options: OptimizationOptions | None = None
) -> OptimizationResult:
    """Refine initialized poses and landmarks with robust sparse LM.

    The solver leaves ``problem`` unchanged. It uses a left-SE(3) update for
    world-to-camera poses and solves the projection linearization
    ``J delta ~= observed - projected``. Candidate steps that put an observed
    point behind a camera are rejected.
    """
    started_at = perf_counter()
    options = options or OptimizationOptions()
    validate_problem(problem, options)
    layout = _parameter_layout(problem, options)
    initial = _evaluate(problem, options)
    if initial.invalid_count:
        return _result(
            problem, initial, initial, (), "invalid_initial_geometry", perf_counter() - started_at
        )
    if layout.size == 0:
        return _result(problem, initial, initial, (), "converged_step", perf_counter() - started_at)

    current = problem
    current_evaluation = initial
    damping = options.initial_damping
    history: list[IterationSummary] = []
    termination: TerminationReason = "max_iterations"
    for iteration in range(1, options.max_iterations + 1):
        try:
            jacobian, residuals = _linearize(current, options, layout)
            normal = (jacobian.T @ jacobian).tocsr()
            diagonal = normal.diagonal()
            diagonal[diagonal <= 0.0] = 1.0
            right_hand_side = np.asarray(jacobian.T @ residuals).reshape(-1)
            with np.errstate(all="raise"):
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("error", MatrixRankWarning)
                    step = np.asarray(spsolve(normal + damping * diags(diagonal), right_hand_side))
        except (ArithmeticError, MatrixRankWarning, ValueError, np.linalg.LinAlgError):
            termination = "linear_solver_failure"
            break
        if step.shape != (layout.size,) or not np.all(np.isfinite(step)):
            termination = "linear_solver_failure"
            break
        step_norm = float(np.linalg.norm(step))
        candidate = _apply_step(current, layout, step)
        candidate_evaluation = _evaluate(candidate, options)
        accepted = candidate_evaluation.cost < current_evaluation.cost
        if accepted:
            previous_cost = current_evaluation.cost
            current = candidate
            current_evaluation = candidate_evaluation
            damping = max(damping / options.damping_multiplier, float(np.finfo(np.float64).eps))
        else:
            damping *= options.damping_multiplier
        history.append(
            IterationSummary(
                iteration=iteration,
                cost=current_evaluation.cost,
                damping=damping,
                step_norm=step_norm,
                accepted=accepted,
                invalid_projection_count=candidate_evaluation.invalid_count,
            )
        )
        if accepted and step_norm <= options.step_tolerance:
            termination = "converged_step"
            break
        if accepted and abs(previous_cost - current_evaluation.cost) <= options.cost_tolerance:
            termination = "converged_cost"
            break
        if damping > options.max_damping:
            termination = "damping_limit_reached"
            break
    final = _evaluate(current, options)
    return _result(
        current, initial, final, tuple(history), termination, perf_counter() - started_at
    )


def _result(
    problem: BundleProblem,
    initial: _Evaluation,
    final: _Evaluation,
    history: tuple[IterationSummary, ...],
    termination: TerminationReason,
    elapsed_seconds: float,
) -> OptimizationResult:
    """Construct a result from evaluation state without exposing mutable solver state."""
    diagnostics = OptimizationDiagnostics(
        initial_cost=initial.cost,
        final_cost=final.cost,
        initial_rmse=initial.rmse,
        final_rmse=final.rmse,
        initial_invalid_projection_count=initial.invalid_count,
        final_invalid_projection_count=final.invalid_count,
        accepted_steps=sum(summary.accepted for summary in history),
        rejected_steps=sum(not summary.accepted for summary in history),
        elapsed_seconds=elapsed_seconds,
        iterations=history,
    )
    return OptimizationResult(
        problem=problem,
        diagnostics=diagnostics,
        termination_reason=termination,
        converged=termination in {"converged_step", "converged_cost"},
    )
