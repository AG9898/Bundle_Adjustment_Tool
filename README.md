# Bundle Adjustment Tool

A Python library for precise and efficient photogrammetric bundle adjustment using sparse Levenberg–Marquardt optimization and Schur complement solvers.  

---

## Features

- **Sparse Levenberg–Marquardt solver** with adaptive damping
- **Block-sparse Schur complement reduction** for efficient large-scale optimization
- **Pinhole camera model** with optional radial distortion support
- **Structured data classes** for cameras, observations, and 3D points
- **Synthetic dataset generation** for validation and testing
- **Comprehensive diagnostics** including reprojection error plots and statistics
- **3D visualization** of cameras and reconstructed points
- **Convergence analysis** with damping parameter sensitivity testing
- **Type annotations** throughout for code clarity and IDE support

---

## Project Structure

```
src/
├── data/           # Camera models and observation structures
│   ├── camera_models.py    # Pinhole camera with projection
│   └── observations.py     # CameraPose, Observation, BundleAdjustmentData
├── core/           # Core optimization components
│   ├── bundle_adjustment.py # Main BA interface (future)
│   └── residuals.py        # Residual computation and Jacobians
├── solvers/        # Optimization algorithms
│   ├── sparse_lm_solver.py # Levenberg-Marquardt implementation
│   ├── schur_complement.py # Block-sparse Schur complement solver
│   └── jacobians.py        # Analytical Jacobian computations
├── visualizations/ # Plotting and diagnostics
│   ├── plot_reprojection_error.py # Error histograms and scatter plots
│   └── plot_cameras.py     # 3D camera and point visualization
└── tests/          # Validation and testing
    ├── test_synthetic_dataset.py # Comprehensive synthetic testing
    └── test_convergence.py       # Damping behavior analysis

main.py             # Entry point with end-to-end demonstration
requirements.txt    # Python dependencies
docs/
└── design_overview.md # Technical architecture and design rationale
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Bundle_adjustment_tool.git
cd Bundle_adjustment_tool

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- **numpy** - Numerical computing and array operations
- **scipy** - Sparse linear algebra and optimization
- **matplotlib** - Visualization and plotting

---

##  Usage

### Quick Start

Run the complete bundle adjustment demonstration:

```bash
# Synthetic data (default)
python main.py --dataset synthetic

# COLMAP data
python main.py --dataset colmap --images_txt path/to/images.txt --points3D_txt path/to/points3D.txt
```

This will:
1. Load or generate a dataset (synthetic or COLMAP)
2. Add realistic perturbations to initial parameters (synthetic only)
3. Run sparse Levenberg-Marquardt optimization
4. Display comprehensive visualizations and statistics

### Visual Outputs

The demonstration produces:
- **Reprojection error histograms** (before/after optimization)
- **3D scene visualizations** showing cameras and points
- **Convergence curves** and damping parameter evolution
- **Detailed performance metrics** and statistics

### Code Example

```python
from src.data.camera_models import CameraModel
from src.data.observations import BundleAdjustmentData, CameraPose, Observation
from src.solvers.sparse_lm_solver import SparseLMSolver

# Create camera model
camera_model = CameraModel(focal_length=1000.0, principal_point=(640, 480))

# Set up bundle adjustment data
bundle_data = BundleAdjustmentData(
    camera_poses=camera_poses,
    points_3d=points_3d,
    observations=observations,
    camera_model=camera_model
)

# Run optimization
solver = SparseLMSolver(data=bundle_data, max_iterations=20)
optimized_poses, optimized_points, final_residual = solver.run()
```

---

## Testing

Run comprehensive tests to validate the implementation:

```bash
# Test synthetic dataset optimization
python -m src.tests.test_synthetic_dataset

# Test convergence behavior with different damping parameters
python -m src.tests.test_convergence
```

### Test Coverage

- **Synthetic Dataset Testing**: Validates optimization correctness with known ground truth
- **Convergence Analysis**: Tests damping parameter sensitivity and convergence behavior
- **Visualization Testing**: Ensures proper plotting and diagnostics
- **Robustness Testing**: Handles edge cases and numerical stability

---

## Documentation

### Technical Architecture

See `/docs/design_overview.md` for detailed technical architecture and design rationale.

### Key Components

- **Camera Models**: Pinhole projection with optional distortion
- **Data Structures**: Efficient representation of observations and poses
- **Optimization**: Sparse LM with Schur complement for scalability
- **Visualization**: Comprehensive diagnostics and 3D scene rendering

---

##  Assumptions & Scope

This tool is designed to perform bundle adjustment for refinement purposes only.
It assumes:

- **Initial Interior Orientation Parameters (IOP)** are already known or provided (e.g., from lab calibration, metadata, or SfM software outputs).

- **Initial Exterior Orientation Parameters (EOP)** are available from prior Structure-from-Motion (SfM) processes (e.g., COLMAP, OpenMVG).

This repository does not include routines for recovering initial orientations from tie points or raw photo measurements.
Such functionality is typically performed in upstream photogrammetric software prior to bundle adjustment.

---

## Technical Details

### Algorithm

The implementation uses:
- **Levenberg-Marquardt** optimization with adaptive damping
- **Schur complement** reduction for efficient large-scale problems
- **Analytical Jacobians** for precision and speed
- **Block-sparse matrices** for memory efficiency

### Performance

- **Scalable**: Handles hundreds of cameras and thousands of points
- **Efficient**: Sparse matrix operations and optimized linear algebra
- **Robust**: Adaptive damping and convergence monitoring
- **Precise**: Analytical derivatives and numerical stability

---

## Examples (Coming Soon)

Planned additions for:
- **Benchmark datasets** (BAL, 1DSfM, etc.)
- **Comparison to ground truth** with real photogrammetric data
- **Real-world dataset adapters** (COLMAP, OpenMVS, etc.)
- **Performance benchmarks** against other BA implementations

---

## Contributing

We welcome contributions! Areas of interest:

- **Robust loss functions** (Huber, Cauchy, etc.)
- **GPU backends** (CUDA, OpenCL acceleration)
- **Real-world dataset adapters** (COLMAP, Bundler, etc.)
- **Advanced camera models** (fisheye, omnidirectional)
- **Performance optimizations** and profiling tools
- **Documentation improvements** and tutorials

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest src/tests/

# Run linting
python -m flake8 src/
```

---

## License

MIT License - see LICENSE file for details.

---

## Acknowledgments

This project builds upon established photogrammetric and computer vision research, particularly:
- Levenberg-Marquardt optimization techniques
- Schur complement methods for bundle adjustment
- Sparse matrix algorithms for large-scale optimization

---

## Contact

For questions, issues, or contributions:
- Open an issue on GitHub
- Submit a pull request
- Contact the maintainers

---

*Built for the computer vision and photogrammetry community* 