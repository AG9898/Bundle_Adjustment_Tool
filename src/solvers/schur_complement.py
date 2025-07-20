import numpy as np
import numpy.typing as npt
from typing import Tuple
from scipy.sparse import csr_matrix, csr_array, eye
from scipy.sparse.linalg import spsolve


def build_normal_equations(
    J_cam: csr_array,
    J_points: csr_array,
    residuals: npt.NDArray[np.float64],
    damping: float
) -> Tuple[csr_matrix, csr_matrix, csr_matrix, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Build block-normal equations for bundle adjustment using Schur complement.
    
    The normal equations are structured as:
    [A  B] [delta_cam]   [rhs_cam]
    [B^T C] [delta_points] = [rhs_points]
    
    where:
    A = J_cam^T * J_cam + lambda * I
    B = J_cam^T * J_points
    C = J_points^T * J_points + lambda * I
    
    Args:
        J_cam: Sparse Jacobian wrt camera parameters (2M, 6*num_cameras)
        J_points: Sparse Jacobian wrt 3D points (2M, 3*num_points)
        residuals: Residual vector (2M,)
        damping: Levenberg-Marquardt damping parameter (lambda)
        
    Returns:
        Tuple of (A, B, C, rhs_cam, rhs_points) where:
        - A: Block-diagonal matrix for cameras (6*num_cameras, 6*num_cameras)
        - B: Cross-term matrix (6*num_cameras, 3*num_points)
        - C: Block-diagonal matrix for points (3*num_points, 3*num_points)
        - rhs_cam: Right-hand side for cameras (6*num_cameras,)
        - rhs_points: Right-hand side for points (3*num_points,)
    """
    # Validate inputs
    assert J_cam.shape[0] == J_points.shape[0], \
        f"Jacobian row dimensions must match: {J_cam.shape[0]} vs {J_points.shape[0]}"
    assert J_cam.shape[0] == len(residuals), \
        f"Jacobian rows must match residual length: {J_cam.shape[0]} vs {len(residuals)}"
    assert damping >= 0, f"Damping must be non-negative, got {damping}"
    
    # Compute normal equation matrices
    # A = J_cam^T * J_cam + lambda * I
    A = J_cam.T @ J_cam + damping * eye(J_cam.shape[1], format='csr')
    
    # B = J_cam^T * J_points
    B = J_cam.T @ J_points
    
    # C = J_points^T * J_points + lambda * I
    C = J_points.T @ J_points + damping * eye(J_points.shape[1], format='csr')
    
    # Compute right-hand side vectors
    # rhs_cam = J_cam^T * residuals
    rhs_cam = J_cam.T @ residuals
    
    # rhs_points = J_points^T * residuals
    rhs_points = J_points.T @ residuals
    
    return A, B, C, rhs_cam, rhs_points


def solve_schur(
    A: csr_matrix,
    B: csr_matrix,
    C: csr_matrix,
    rhs_cam: npt.NDArray[np.float64],
    rhs_points: npt.NDArray[np.float64]
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Solve the reduced bundle adjustment problem using Schur complement.
    
    The Schur complement method solves the block system:
    [A  B] [delta_cam]   [rhs_cam]
    [B^T C] [delta_points] = [rhs_points]
    
    by computing:
    1. Schur complement: S = A - B * C^(-1) * B^T
    2. Solve for cameras: S * delta_cam = rhs_cam - B * C^(-1) * rhs_points
    3. Back-substitute for points: delta_points = C^(-1) * (rhs_points - B^T * delta_cam)
    
    Args:
        A: Block-diagonal matrix for cameras (6*num_cameras, 6*num_cameras)
        B: Cross-term matrix (6*num_cameras, 3*num_points)
        C: Block-diagonal matrix for points (3*num_points, 3*num_points)
        rhs_cam: Right-hand side for cameras (6*num_cameras,)
        rhs_points: Right-hand side for points (3*num_points,)
        
    Returns:
        Tuple of (delta_cam, delta_points) where:
        - delta_cam: Updates for camera parameters (6*num_cameras,)
        - delta_points: Updates for 3D points (3*num_points,)
    """
    # Validate input shapes
    num_cam_params = A.shape[0]
    num_point_params = C.shape[0]
    
    assert A.shape == (num_cam_params, num_cam_params), \
        f"A must be square: {A.shape}"
    assert C.shape == (num_point_params, num_point_params), \
        f"C must be square: {C.shape}"
    assert B.shape == (num_cam_params, num_point_params), \
        f"B shape {B.shape} must match A and C dimensions"
    assert len(rhs_cam) == num_cam_params, \
        f"rhs_cam length {len(rhs_cam)} must match A dimension {num_cam_params}"
    assert len(rhs_points) == num_point_params, \
        f"rhs_points length {len(rhs_points)} must match C dimension {num_point_params}"
    
    # Step 1: Solve C * temp = rhs_points for temporary variable
    # This gives us C^(-1) * rhs_points
    temp = spsolve(C, rhs_points)
    
    # Step 2: Compute modified right-hand side for cameras
    # rhs_cam_modified = rhs_cam - B * C^(-1) * rhs_points
    rhs_cam_modified = rhs_cam - B @ temp
    
    # Step 3: Compute Schur complement S = A - B * C^(-1) * B^T
    # First compute B * C^(-1) * B^T efficiently
    # Since C is block-diagonal, we can solve C * X = B^T for X
    # Then S = A - B * X
    X = spsolve(C, B.T.toarray())  # Solve C * X = B^T
    S = A - B @ X
    
    # Step 4: Solve Schur complement system for camera updates
    # S * delta_cam = rhs_cam_modified
    delta_cam = spsolve(S, rhs_cam_modified)
    
    # Step 5: Back-substitute to get point updates
    # delta_points = C^(-1) * (rhs_points - B^T * delta_cam)
    delta_points = spsolve(C, rhs_points - B.T @ delta_cam)
    
    return delta_cam, delta_points


def solve_schur_efficient(
    A: csr_matrix,
    B: csr_matrix,
    C: csr_matrix,
    rhs_cam: npt.NDArray[np.float64],
    rhs_points: npt.NDArray[np.float64]
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Efficient Schur complement solver optimized for bundle adjustment.
    
    This version avoids explicit computation of the full Schur complement
    matrix and uses more efficient sparse operations.
    
    Args:
        A: Block-diagonal matrix for cameras
        B: Cross-term matrix
        C: Block-diagonal matrix for points
        rhs_cam: Right-hand side for cameras
        rhs_points: Right-hand side for points
        
    Returns:
        Tuple of (delta_cam, delta_points)
    """
    # Validate inputs
    num_cam_params = A.shape[0]
    num_point_params = C.shape[0]
    
    assert A.shape == (num_cam_params, num_cam_params)
    assert C.shape == (num_point_params, num_point_params)
    assert B.shape == (num_cam_params, num_point_params)
    
    # Step 1: Solve C * temp1 = rhs_points
    temp1 = spsolve(C, rhs_points)
    
    # Step 2: Solve C * temp2 = B^T (for each column of B^T)
    # This is equivalent to solving C * temp2 = B^T
    temp2 = spsolve(C, B.T.toarray())
    
    # Step 3: Compute modified right-hand side
    rhs_cam_modified = rhs_cam - B @ temp1
    
    # Step 4: Compute Schur complement action without forming full matrix
    # S * delta_cam = (A - B * temp2) * delta_cam = rhs_cam_modified
    # Use iterative solver or direct solver depending on size
    if num_cam_params < 1000:  # Small enough for direct solve
        S = A - B @ temp2
        delta_cam = spsolve(S, rhs_cam_modified)
    else:
        # For larger systems, use iterative solver
        from scipy.sparse.linalg import cg
        S = A - B @ temp2
        delta_cam, _ = cg(S, rhs_cam_modified, tol=1e-6, maxiter=100)
    
    # Step 5: Back-substitute for points
    delta_points = spsolve(C, rhs_points - B.T @ delta_cam)
    
    return delta_cam, delta_points 