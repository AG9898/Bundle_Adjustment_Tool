# Bundle Adjustment Tool — Design Overview

---

## Project Purpose

This project implements a photogrammetric bundle adjustment tool 

It serves as both an educational example and a foundation for future extensions (e.g., Structure-from-Motion pipelines).

---

## Architecture Overview

### Key Components:

```
data/
├── camera_models.py    # Pinhole camera projections
└── observations.py     # Cameras, 3D points, 2D observations

core/
├── residuals.py        # Collinearity-based reprojection residuals
└── bundle_adjustment.py # (Wrapper, optional)

solvers/
├── sparse_lm_solver.py # Levenberg–Marquardt with damping
└── schur_complement.py # Block-sparse normal equations

visualizations/
├── plot_reprojection_error.py
└── plot_cameras.py

tests/
└── Synthetic datasets for validation
```

### Data Flow Diagram

```plaintext
3D Points ───┐
Cameras ─────┴─> residuals.py ──> Jacobians ──> Schur Complement ──> Parameter Updates
Observations ──────────────────────────────────────────────────────────────────────┘
```

---

##  Optimization Strategy

### Residuals
- **Collinearity equations** via projection
- 2D reprojection errors: `observed - projected`
- Robust handling of points behind cameras

### Jacobian Structure
- **Block-sparse** (camera vs. points)
- Analytical derivatives for precision
- Sparse matrix format for efficiency

### Solver
- **Sparse Levenberg–Marquardt** with adaptive damping
- Damping adapted per iteration based on residual improvement
- Convergence monitoring via parameter update norms

### Schur Complement
- **Reduces dimensionality** by eliminating 3D points
- Solves smaller camera-only system
- Back-substitution for point updates

### Linear Solver
- **scipy.sparse.linalg.spsolve** for efficiency
- CSR matrix format for optimal performance
- Fallback to iterative solvers for large systems

---

##  Extensibility

### Future Directions

#### Robust Loss Functions
- **Huber loss** for outlier rejection
- **Cauchy loss** for heavy-tailed noise
- **Adaptive weighting** based on residual magnitude

#### Camera Models
- **Radial distortion** models (Brown, OpenCV)
- **Fisheye** and omnidirectional cameras
- **Multi-camera rigs** and calibration

#### Performance Optimization
- **GPU acceleration** (CuPy / torch.sparse)
- **Parallel processing** for large datasets
- **Memory optimization** for very large problems

#### Real-world Integration
- **COLMAP** dataset adapters
- **OpenMVG** format support
- **Bundler** compatibility
- **Structure-from-Motion** pipeline integration

#### Advanced Features
- **Incremental bundle adjustment** for real-time applications
- **Loop closure** detection and handling
- **Multi-scale** optimization strategies
- **Uncertainty quantification** and covariance estimation

---

##  Diagnostics & Outputs

### Visualization Tools

#### Reprojection Error Analysis
- **Histograms** of error distributions
- **Scatter plots** showing systematic errors
- **Statistics** (mean, std, max, percentiles)
- **Before/after** comparison plots

#### 3D Scene Visualization
- **Camera positions** and orientations
- **3D point clouds** with color coding
- **Trajectory plots** for sequential cameras
- **Frustum visualization** for camera orientations

#### Convergence Monitoring
- **Residual norm** vs. iteration
- **Damping parameter** evolution
- **Parameter update** magnitudes
- **Convergence rate** analysis

### Performance Metrics

#### Optimization Quality
- **Final residual norm** and improvement percentage
- **Average reprojection error** in pixels
- **Convergence speed** (iterations to convergence)
- **Parameter stability** (camera/point shifts)

#### Computational Performance
- **Memory usage** and matrix sparsity
- **Linear solve time** per iteration
- **Total optimization time**
- **Scalability** with problem size

---

##  Technical Implementation Details

### Mathematical Foundation

#### Bundle Adjustment Problem
The optimization minimizes the sum of squared reprojection errors:

```
min Σ ||x_ij - π(P_i, X_j)||²
```

Where:
- `x_ij` is the observed 2D point
- `π(P_i, X_j)` is the projection of 3D point `X_j` by camera `P_i`
- `P_i` represents camera pose (rotation + translation)
- `X_j` represents 3D point coordinates

#### Levenberg-Marquardt Algorithm
The LM algorithm solves the normal equations:

```
(J^T J + λI) δ = J^T r
```

Where:
- `J` is the Jacobian matrix
- `r` is the residual vector
- `λ` is the damping parameter
- `δ` is the parameter update

#### Schur Complement Reduction
For the block system:

```
[A  B] [δ_c]   [r_c]
[B^T C] [δ_p] = [r_p]
```

The Schur complement reduces to:

```
(A - B C^(-1) B^T) δ_c = r_c - B C^(-1) r_p
```

### Implementation Choices

#### Sparse Matrix Format
- **CSR (Compressed Sparse Row)** for efficient matrix-vector operations
- **LIL (List of Lists)** for construction, then convert to CSR
- **Block structure** exploited for memory efficiency

#### Numerical Stability
- **Analytical Jacobians** for precision
- **Adaptive damping** for robust convergence
- **Condition number** monitoring for ill-conditioned problems
- **Regularization** via damping parameter

#### Memory Management
- **In-place updates** where possible
- **Sparse matrix operations** to minimize memory usage
- **Efficient data structures** for large-scale problems

---

##  Design Principles

### Modularity
- **Separation of concerns** between data, computation, and visualization
- **Clean interfaces** between components
- **Minimal dependencies** between modules

### Clarity
- **Type annotations** throughout for code clarity
- **Comprehensive documentation** and docstrings
- **Clear variable names** and function signatures
- **Educational comments** explaining mathematical concepts

### Efficiency
- **Sparse algorithms** for large-scale problems
- **Vectorized operations** where possible
- **Optimized linear algebra** using scipy
- **Memory-conscious** data structures

### Robustness
- **Input validation** at all levels
- **Error handling** for numerical issues
- **Graceful degradation** for edge cases
- **Comprehensive testing** with synthetic data

---

##  Performance Characteristics

### Scalability
- **Linear complexity** in number of observations
- **Block-sparse structure** reduces memory requirements
- **Efficient linear solvers** for large systems
- **Parallel potential** for future GPU implementation

### Accuracy
- **Analytical derivatives** ensure mathematical precision
- **Robust convergence** with adaptive damping
- **Comprehensive validation** against synthetic ground truth
- **Numerical stability** considerations throughout

### Usability
- **Simple API** for basic usage
- **Comprehensive diagnostics** for debugging
- **Visualization tools** for result interpretation
- **Extensible design** for custom applications

---

##  Scope Clarification

This tool focuses exclusively on the refinement step in photogrammetric pipelines:

- It refines camera poses (EOPs) and 3D point positions given initial approximations.

- It assumes camera intrinsics (IOPs) are provided.

- It does not include:
  - Initial orientation recovery (resection)
  - Camera calibration routines
  - Block adjustments with ground control points (future extension potential)

This mirrors industry-standard workflows where initial IOPs and EOPs are derived from separate photogrammetric processes (e.g., COLMAP, Metashape, OpenMVG).

---

*This design overview provides the technical foundation for understanding and extending the bundle adjustment tool.* 