# Geometry and Data Contract

## Purpose

This document is the authoritative contract for the public library API and the
COLMAP text adapter. It removes the ambiguity that commonly makes bundle
adjustment implementations appear to converge while optimizing inconsistent
geometry.

## Coordinate convention

Every `Pose` stores a world-to-camera transform:

```text
x_c = R_cw X_w + t_cw
```

`R_cw` is a proper rotation matrix and `t_cw` is a world-to-camera translation;
it is not the camera centre. Derive a world camera centre only as:

```text
C_w = -R_cw^T t_cw
```

For a calibrated pinhole camera, pixels are:

```text
u = fx * x_c / z_c + cx
v = fy * y_c / z_c + cy
```

An observed point must satisfy `z_c > minimum_depth` in the initial problem.
Proposals that move an observed point behind the camera are rejected rather
than assigned invented pixel coordinates.

## Pose parameterization and residual sign

The optimizer applies a left-SE(3) increment:

```text
T_cw,new = exp([omega, upsilon]) T_cw
```

Thus, to first order,

```text
d x_c / d omega   = -[x_c]_x
d x_c / d upsilon = I
```

For an observation `z`, residual is `r = z - project(T, X)`. The sparse system
uses the projection derivative `J` and solves `J delta ~= r`, which is
equivalent to the usual negative-residual Jacobian formulation. Analytic pose
and landmark derivatives are checked against central finite differences in the
test suite.

## Gauge handling

Calibrated bundle adjustment has a global similarity ambiguity when only image
measurements are optimized. With automatic gauge fixing enabled, the optimizer
keeps camera index 0 and point index 0 fixed. Applications that know better
anchors should provide `fixed_camera_indices` and `fixed_point_indices`; they
can disable `auto_gauge_fix` once sufficient independent constraints are fixed.

Index 0 is only a default. It is the caller's responsibility to select a better
anchor when the first camera or first point is unsuitable for their graph.

## Losses and weights

Each `Observation` has a positive scalar weight. The objective is the weighted
sum of either:

- squared 2D residual cost, `0.5 ||r||²`; or
- Huber cost with a per-observation Euclidean residual norm and configurable
  `huber_delta` in pixels.

Huber is assembled with iteratively reweighted least squares at each accepted
linearization. Metrics include the corresponding weighted robust cost and the
unweighted coordinate RMSE.

## COLMAP text import

`load_colmap_text(model_directory)` requires the following files as one coherent
model:

- `cameras.txt` supplies calibration. `PINHOLE(fx, fy, cx, cy)` and
  `SIMPLE_PINHOLE(f, cx, cy)` are supported.
- `images.txt` supplies COLMAP's world-to-camera quaternion and translation.
  The adapter converts COLMAP quaternion order `(qw, qx, qy, qz)` correctly.
- `points3D.txt` supplies world landmarks.

Untriangulated observations (`POINT3D_ID == -1`), missing point IDs, and images
without any remaining triangulated observations are omitted. Retained camera
and point IDs are remapped to dense zero-based indices in input order. The
camera identifier in the library result retains the source image ID and name
for traceability.

Unsupported models deliberately fail instead of silently dropping distortion or
inventing intrinsics. Exporting COLMAP text or handling binary models is not yet
implemented.

## Result schema

`save_result` emits schema version 1:

- `result.npz` stores arrays named `camera_rotations`, `camera_translations`,
  `camera_intrinsics`, `camera_identifiers`, `points`, and the observation
  arrays.
- `result.json` stores `schema_version`, package version, problem counts,
  termination/convergence state, and diagnostics. Diagnostics include initial
  and final cost/RMSE, invalid-projection counts, accepted/rejected steps,
  elapsed seconds, and one summary per proposal.

Use `read_result_metadata` or `bundle-adjust report` rather than assuming a
future schema can be interpreted as version 1.

## Initialized reconstruction import

`load_initialized_reconstruction(directory)` consumes the schema-v1 hand-off
artifact produced by the Spatial Intersection Tool. It requires
`reconstruction.json` to identify `initialized_reconstruction`, schema version
1, and the `world_to_camera` convention. `reconstruction.npz` must contain
per-camera rotations, translations, pinhole intrinsics and identifiers; world
points and identifiers; and dense camera/point/measurement/weight arrays.

This input artifact is distinct from the `result.npz` and `result.json` output
created by `save_result`. The latter records an optimization result and is not
silently treated as a new initialization.

## Termination states

`converged_step` means an accepted update was below `step_tolerance`.
`converged_cost` means an accepted update changed the robust cost by less than
`cost_tolerance`. `max_iterations`, `damping_limit_reached`, and
`linear_solver_failure` are non-converged outcomes. Any result is saved for
diagnosis; the CLI communicates non-convergence with process exit status 1.
