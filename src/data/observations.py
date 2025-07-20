import numpy as np
import numpy.typing as npt
from typing import List, Tuple
from .camera_models import CameraModel


class CameraPose:
    """
    Represents a camera pose with rotation and translation.
    
    Attributes:
        rotation: 3x3 rotation matrix
        translation: 3D translation vector
    """
    
    def __init__(
        self, 
        rotation: npt.NDArray[np.float64], 
        translation: npt.NDArray[np.float64]
    ) -> None:
        """
        Initialize camera pose.
        
        Args:
            rotation: 3x3 rotation matrix
            translation: 3D translation vector
        """
        self.rotation = rotation
        self.translation = translation
    
    def __repr__(self) -> str:
        return f"CameraPose(rotation_shape={self.rotation.shape}, translation_shape={self.translation.shape})"


class Observation:
    """
    Represents a single observation of a 3D point in an image.
    
    Attributes:
        camera_index: Index of the camera that made this observation
        point_index: Index of the 3D point being observed
        image_point: 2D image coordinates of the observation
    """
    
    def __init__(
        self, 
        camera_index: int, 
        point_index: int, 
        image_point: npt.NDArray[np.float64]
    ) -> None:
        """
        Initialize observation.
        
        Args:
            camera_index: Index of the camera
            point_index: Index of the 3D point
            image_point: 2D image coordinates (x, y)
        """
        self.camera_index = camera_index
        self.point_index = point_index
        self.image_point = image_point
    
    def __repr__(self) -> str:
        return f"Observation(camera={self.camera_index}, point={self.point_index}, coords={self.image_point})"


class BundleAdjustmentData:
    """
    Container for all data needed for bundle adjustment optimization.
    
    Attributes:
        camera_poses: List of camera poses
        points_3d: 3D points in world coordinates (Nx3 array)
        observations: List of observations
        camera_model: Camera model for projection
    """
    
    def __init__(
        self,
        camera_poses: List[CameraPose],
        points_3d: npt.NDArray[np.float64],
        observations: List[Observation],
        camera_model: CameraModel
    ) -> None:
        """
        Initialize bundle adjustment data.
        
        Args:
            camera_poses: List of camera poses
            points_3d: 3D points in world coordinates (Nx3 array)
            observations: List of observations
            camera_model: Camera model for projection
        """
        self.camera_poses = camera_poses
        self.points_3d = points_3d
        self.observations = observations
        self.camera_model = camera_model
    
    def get_observation_matrices(self) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """
        Convert observations to matrix format for optimization.
        
        Returns:
            Tuple of (camera_indices, point_indices, image_points) where:
            - camera_indices: Mx1 array of camera indices
            - point_indices: Mx1 array of point indices  
            - image_points: Mx2 array of image coordinates
        """
        num_observations = len(self.observations)
        
        camera_indices = np.zeros((num_observations, 1), dtype=np.int64)
        point_indices = np.zeros((num_observations, 1), dtype=np.int64)
        image_points = np.zeros((num_observations, 2), dtype=np.float64)
        
        for i, obs in enumerate(self.observations):
            camera_indices[i, 0] = obs.camera_index
            point_indices[i, 0] = obs.point_index
            image_points[i, :] = obs.image_point
        
        return camera_indices, point_indices, image_points
    
    def validate(self) -> None:
        """
        Validate the bundle adjustment data for consistency.
        
        Raises:
            ValueError: If data is inconsistent or invalid
        """
        # Validate camera poses
        if len(self.camera_poses) == 0:
            raise ValueError("At least one camera pose must be provided")
        
        for i, pose in enumerate(self.camera_poses):
            if pose.rotation.shape != (3, 3):
                raise ValueError(f"Camera pose {i}: rotation must be 3x3, got {pose.rotation.shape}")
            if pose.translation.shape != (3,):
                raise ValueError(f"Camera pose {i}: translation must be (3,), got {pose.translation.shape}")
        
        # Validate 3D points
        if self.points_3d.ndim != 2 or self.points_3d.shape[1] != 3:
            raise ValueError(f"points_3d must be Nx3 array, got shape {self.points_3d.shape}")
        
        num_points = self.points_3d.shape[0]
        if num_points == 0:
            raise ValueError("At least one 3D point must be provided")
        
        # Validate observations
        if len(self.observations) == 0:
            raise ValueError("At least one observation must be provided")
        
        num_cameras = len(self.camera_poses)
        
        for i, obs in enumerate(self.observations):
            if obs.camera_index < 0 or obs.camera_index >= num_cameras:
                raise ValueError(f"Observation {i}: camera_index {obs.camera_index} out of range [0, {num_cameras-1}]")
            
            if obs.point_index < 0 or obs.point_index >= num_points:
                raise ValueError(f"Observation {i}: point_index {obs.point_index} out of range [0, {num_points-1}]")
            
            if obs.image_point.shape != (2,):
                raise ValueError(f"Observation {i}: image_point must be (2,), got {obs.image_point.shape}")
        
        # Validate camera model
        if self.camera_model is None:
            raise ValueError("Camera model must be provided")
    
    def __repr__(self) -> str:
        return (f"BundleAdjustmentData("
                f"cameras={len(self.camera_poses)}, "
                f"points={self.points_3d.shape[0]}, "
                f"observations={len(self.observations)})") 