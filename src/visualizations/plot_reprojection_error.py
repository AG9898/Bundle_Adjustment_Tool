import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from typing import Optional


def plot_reprojection_errors(
    residuals: npt.NDArray[np.float64],
    title: Optional[str] = None
) -> None:
    """
    Plot histogram of reprojection errors from residual vector.
    
    Args:
        residuals: Residual vector of shape (2M,) where M is number of observations.
                  Each pair of consecutive elements represents (dx, dy) for one observation.
        title: Optional title for the plot. If None, auto-generates title with statistics.
    """
    # Validate input
    assert residuals.ndim == 1, f"Residuals must be 1D array, got shape {residuals.shape}"
    assert len(residuals) % 2 == 0, f"Residuals length must be even, got {len(residuals)}"
    
    # Compute reprojection errors (Euclidean norm of 2D residual pairs)
    num_observations = len(residuals) // 2
    reprojection_errors = np.zeros(num_observations)
    
    for i in range(num_observations):
        dx = residuals[2 * i]
        dy = residuals[2 * i + 1]
        reprojection_errors[i] = np.sqrt(dx**2 + dy**2)
    
    # Compute statistics
    mean_error = np.mean(reprojection_errors)
    max_error = np.max(reprojection_errors)
    median_error = np.median(reprojection_errors)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot histogram
    plt.hist(reprojection_errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add vertical lines for statistics
    plt.axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.3f} px')
    plt.axvline(median_error, color='green', linestyle='--', linewidth=2, label=f'Median: {median_error:.3f} px')
    
    # Set title
    if title is None:
        title = f"Reprojection Error Distribution\n"
        title += f"Mean: {mean_error:.3f} px, Max: {max_error:.3f} px, "
        title += f"Observations: {num_observations}"
    
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Label axes
    plt.xlabel('Reprojection Error (pixels)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # Add legend
    plt.legend(fontsize=10)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()


def plot_residual_scatter(
    residuals: npt.NDArray[np.float64],
    title: Optional[str] = None
) -> None:
    """
    Plot scatter plot of 2D residuals showing systematic errors.
    
    Args:
        residuals: Residual vector of shape (2M,) where M is number of observations.
                  Each pair of consecutive elements represents (dx, dy) for one observation.
        title: Optional title for the plot.
    """
    # Validate input
    assert residuals.ndim == 1, f"Residuals must be 1D array, got shape {residuals.shape}"
    assert len(residuals) % 2 == 0, f"Residuals length must be even, got {len(residuals)}"
    
    # Extract x and y residuals
    dx = residuals[::2]  # Every even index
    dy = residuals[1::2]  # Every odd index
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot scatter
    plt.scatter(dx, dy, alpha=0.6, s=20, c='blue', edgecolors='black', linewidth=0.5)
    
    # Add reference lines
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.5, linewidth=1)
    plt.axvline(x=0, color='red', linestyle='-', alpha=0.5, linewidth=1)
    
    # Set title
    if title is None:
        title = f"2D Residual Scatter Plot\n"
        title += f"RMS: {np.sqrt(np.mean(dx**2 + dy**2)):.3f} px, "
        title += f"Observations: {len(dx)}"
    
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Label axes
    plt.xlabel('X Residual (pixels)', fontsize=12)
    plt.ylabel('Y Residual (pixels)', fontsize=12)
    
    # Make axes equal for proper aspect ratio
    plt.axis('equal')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()


if __name__ == "__main__":
    # Example usage and testing
    print("Testing reprojection error plotting...")
    
    # Generate synthetic residuals for testing
    np.random.seed(42)
    num_observations = 1000
    residuals = np.random.normal(0, 2.0, 2 * num_observations)  # 2D residuals
    
    # Plot histogram
    plot_reprojection_errors(residuals, "Synthetic Test Data")
    
    # Plot scatter
    plot_residual_scatter(residuals, "Synthetic Test Data") 