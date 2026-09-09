# Bundle Adjustment

Bundle Adjustment is a correctness-first Python library and command-line tool
for refining initialized sparse reconstructions. It is designed for
computer-vision, photogrammetry, robotics, graphics, and serious student
projects that already have calibrated cameras, 3D landmarks, and 2D feature
observations.

It is not an SfM pipeline: it does not match features, triangulate points, or
operate on raw images. Its job is to make an existing reconstruction more
consistent, explain what happened, and save a reproducible result.

## Status

This is pre-release software (`0.1.0a0`). The current implementation supports
calibrated pinhole bundle adjustment and COLMAP text sparse models with
`PINHOLE` and `SIMPLE_PINHOLE` cameras. It also accepts schema-v1 initialized
reconstructions produced by the companion Spatial Intersection Tool. The public
API and file schema are documented, but should be pinned when used in a
production system.

## Install

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install .
```

For contributors, install development tools instead:

```bash
.venv/bin/python -m pip install -e '.[dev]'
```

## Quick start: COLMAP text model

The input directory must contain COLMAP's `cameras.txt`, `images.txt`, and
`points3D.txt` files. Validate the whole model before running an optimization:

```bash
bundle-adjust validate path/to/sparse/0
```

Run a robust solve and save arrays plus a JSON report:

```bash
bundle-adjust optimize path/to/sparse/0 output/ba-result \
  --loss huber --huber-delta 2.0 --max-iterations 50

bundle-adjust report output/ba-result
```

### Quick start: pipeline hand-off

An initialized-reconstruction directory containing `reconstruction.npz` and
`reconstruction.json` may be passed to the same commands. This is the
versioned artifact emitted by `spatial-intersect triangulate`:

```bash
bundle-adjust validate initialized/
bundle-adjust optimize initialized/ output/ba-result --loss huber
```

The artifact is accepted only when it declares schema version 1 and the shared
`world_to_camera` pose convention; the loader does not guess fields from an
unrelated result directory.

The command refuses to overwrite a non-empty output directory unless
`--overwrite` is supplied. It exits with status 0 only when the optimizer
reaches a convergence criterion; it still writes the result and diagnostics if
it stops at the iteration limit.

## Library API

```python
from bundle_adjustment import OptimizationOptions, optimize
from bundle_adjustment.io import load_colmap_text, save_result

problem = load_colmap_text("path/to/sparse/0")
result = optimize(
    problem,
    OptimizationOptions(loss="huber", huber_delta=2.0, max_iterations=50),
)

print(result.termination_reason, result.diagnostics.final_rmse)
save_result(result, "output/ba-result")
```

`BundleProblem`, `Camera`, `Pose`, `CameraIntrinsics`, and `Observation` are
also public if another system needs to construct a problem directly.

## Geometry and solver contract

The sole pose convention is world to camera:

```text
x_camera = R_world_to_camera @ X_world + t_world_to_camera
C_world  = -R_world_to_camera.T @ t_world_to_camera
```

The solver uses a left-SE(3) pose increment and does not mutate the supplied
problem. By default it fixes camera 0 and landmark 0 to remove the global pose
and scale gauge; supply explicit fixed indices or disable automatic gauge fixing
when your application owns those constraints. All initially observed points must
have positive camera depth.

The reference backend is sparse Levenberg–Marquardt with step acceptance,
bounded damping, Huber or squared loss, per-observation weights, structured
termination state, and residual diagnostics. It is intentionally a correctness
baseline, not yet a claim of Ceres/g2o-scale performance.

See [the geometry and data contract](docs/geometry_and_data_contract.md) for
the full supported-format, parameterization, and artifact details.

## Output artifacts

Each optimization creates:

- `result.npz` — poses, intrinsics, points, and observations;
- `result.json` — schema version, package version, input counts, termination
  reason, timing, residual metrics, and per-iteration history.

These artifacts are intended for scripts and CI as well as human inspection.

## Current limitations

- Only calibrated `PINHOLE` and `SIMPLE_PINHOLE` COLMAP text cameras are
  accepted. Distortion, fisheye, binary models, and camera-intrinsic refinement
  are not implemented yet.
- Feature matching, triangulation, raw-image processing, UI tooling,
  incremental BA, GPU acceleration, covariance estimates, and a custom Schur
  backend are out of scope for this release.
- Large-scale performance should be benchmarked for the intended workload.

## Development

```bash
.venv/bin/python -m ruff check src/bundle_adjustment tests
.venv/bin/python -m mypy
.venv/bin/python -m pytest -q
```

The test suite includes finite-difference checks for the projection Jacobians,
synthetic optimizer convergence, and an end-to-end COLMAP/CLI fixture. See
[CONTRIBUTING.md](CONTRIBUTING.md) before opening a change.

## License

[MIT](LICENSE)
