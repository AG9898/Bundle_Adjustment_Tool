# Contributing

Contributions should strengthen the documented data contract or expand it with
tests and migration notes. Please open an issue before proposing a large new
backend or camera model so its performance and compatibility implications are
clear.

Every change to projection, pose updates, residual assembly, or a camera model
needs a finite-difference derivative test and at least one integration or
regression fixture. New format adapters must preserve the source coordinate
convention and calibration instead of filling in defaults.

Before submitting a pull request, run:

```bash
.venv/bin/python -m ruff check src/bundle_adjustment tests
.venv/bin/python -m mypy
.venv/bin/python -m pytest -q
```

Do not add large datasets, generated plots, virtual environments, or benchmark
downloads to the repository. Benchmarks must document source, environment,
hardware, and correctness equivalence to the reference backend.
