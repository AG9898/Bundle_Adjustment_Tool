import numpy as np
from typing import Tuple, Optional, Union
import numpy.typing as npt


class CameraModel:
    """
    Simple pinhole camera model for bundle adjustment.
    
    Implements perspective projection with focal length and principal point offset.
    Optional radial distortion support (placeholder).
    """
    
    def __init__(
        self,
        focal_length: float,
        principal_point: Tuple[float, float],
        distortion_coeffs: Optional[npt.NDArray[np.float64]] = None
    ) -> None:
        """
        Initialize camera model.
        
        Args:
            focal_length: Camera focal length in pixels
            principal_point: Principal point (cx, cy) in pixels
            distortion_coeffs: Optional radial distortion coefficients
        """
        self.focal_length = focal_length
        self.principal_point = principal_point
        self.distortion_coeffs = distortion_coeffs
    
    def project(
        self, 
        points_3d: npt.NDArray[np.float64], 
        pose: Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ) -> npt.NDArray[np.float64]:
        """
        Project 3D points to 2D image coordinates.
        
        Args:
            points_3d: 3D points in world coordinates (Nx3 array)
            pose: Camera pose as (R, t) where R is 3x3 rotation matrix, t is 3x1 translation
            
        Returns:
            2D points in image coordinates (Nx2 array)
        """
        R, t = pose
        
        # Validate input shapes
        assert points_3d.ndim == 2 and points_3d.shape[1] == 3, \
            f"points_3d must be Nx3 array, got shape {points_3d.shape}"
        assert R.shape == (3, 3), f"R must be 3x3 matrix, got shape {R.shape}"
        assert t.shape == (3,) or t.shape == (3, 1), f"t must be 3x1 or (3,), got shape {t.shape}"
        
        # Ensure t is a column vector
        if t.ndim == 1:
            t = t.reshape(3, 1)
        
        # Transform points from world to camera coordinates
        points_cam = (R @ points_3d.T) + t  # (3, N)
        points_cam = points_cam.T  # (N, 3)
        
        # Perspective projection: x' = X/Z, y' = Y/Z
        # Avoid division by zero
        z_coords = points_cam[:, 2]
        valid_mask = z_coords > 0
        
        points_2d = np.zeros((points_3d.shape[0], 2))
        points_2d[valid_mask] = points_cam[valid_mask, :2] / z_coords[valid_mask, np.newaxis]
        
        # Apply focal length and principal point offset
        points_2d[valid_mask, 0] = points_2d[valid_mask, 0] * self.focal_length + self.principal_point[0]
        points_2d[valid_mask, 1] = points_2d[valid_mask, 1] * self.focal_length + self.principal_point[1]
        
        # Apply distortion if coefficients are provided
        if self.distortion_coeffs is not None:
            points_2d = self._apply_distortion(points_2d)
        
        return points_2d
    
    def _apply_distortion(self, points_2d: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Apply radial distortion to 2D points.
        
        Args:
            points_2d: 2D points in image coordinates (Nx2 array)
            
        Returns:
            Distorted 2D points (Nx2 array)
        """
        # Placeholder for radial distortion implementation
        # TODO: Implement radial distortion using self.distortion_coeffs
        return points_2d 