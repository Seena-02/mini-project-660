"""
Dataset Generators for EE660 Mini-Project
==========================================
This module provides 2D toy dataset generators for training generative models.
Datasets: Checkerboard, Gaussian Mixtures

Author: Seena
Course: EE660 - Machine Learning
"""

import numpy as np
from typing import Tuple, Optional


def generate_checkerboard(n_samples: int, 
                          n_squares: int = 4, 
                          noise: float = 0.05,
                          seed: Optional[int] = None) -> np.ndarray:
    """
    Generate samples from a checkerboard distribution.
    
    Args:
        n_samples: Number of samples to generate
        n_squares: Number of squares per side (default 4)
        noise: Standard deviation of Gaussian noise added to samples
        seed: Random seed for reproducibility
        
    Returns:
        np.ndarray: Array of shape (n_samples, 2) containing 2D samples
    """
    if seed is not None:
        np.random.seed(seed)
    
    samples = []
    # Determine which squares are "filled" (checkerboard pattern)
    # Squares where (i + j) % 2 == 0 are filled
    filled_squares = []
    for i in range(n_squares):
        for j in range(n_squares):
            if (i + j) % 2 == 0:
                filled_squares.append((i, j))
    
    # Sample uniformly from filled squares
    n_filled = len(filled_squares)
    samples_per_square = n_samples // n_filled
    extra_samples = n_samples % n_filled
    
    for idx, (i, j) in enumerate(filled_squares):
        # Determine number of samples for this square
        n_sq = samples_per_square + (1 if idx < extra_samples else 0)
        
        # Generate uniform samples within the square
        x_min, x_max = i / n_squares, (i + 1) / n_squares
        y_min, y_max = j / n_squares, (j + 1) / n_squares
        
        x = np.random.uniform(x_min, x_max, n_sq)
        y = np.random.uniform(y_min, y_max, n_sq)
        
        # Stack and add noise
        sq_samples = np.stack([x, y], axis=1)
        sq_samples += np.random.normal(0, noise, sq_samples.shape)
        samples.append(sq_samples)
    
    samples = np.vstack(samples)
    
    # Shuffle and center the data
    np.random.shuffle(samples)
    samples = samples * 2 - 1  # Scale to [-1, 1] range
    
    return samples.astype(np.float32)


def generate_gaussian_mixture(n_samples: int,
                              n_components: int = 8,
                              std: float = 0.05,
                              radius: float = 0.8,
                              seed: Optional[int] = None) -> np.ndarray:
    """
    Generate samples from a Gaussian mixture model arranged in a circle.
    
    Args:
        n_samples: Number of samples to generate
        n_components: Number of Gaussian components
        std: Standard deviation of each Gaussian component
        radius: Radius of the circle on which centers are placed
        seed: Random seed for reproducibility
        
    Returns:
        np.ndarray: Array of shape (n_samples, 2) containing 2D samples
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate centers arranged in a circle
    angles = np.linspace(0, 2 * np.pi, n_components, endpoint=False)
    centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    
    # Sample from mixture
    samples_per_component = n_samples // n_components
    extra_samples = n_samples % n_components
    
    samples = []
    for idx, center in enumerate(centers):
        n_comp = samples_per_component + (1 if idx < extra_samples else 0)
        comp_samples = np.random.normal(center, std, size=(n_comp, 2))
        samples.append(comp_samples)
    
    samples = np.vstack(samples)
    np.random.shuffle(samples)
    
    return samples.astype(np.float32)


def get_dataset(name: str, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Get dataset by name.
    
    Args:
        name: Dataset name ('checkerboard' or 'gaussian_mixture')
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        np.ndarray: Generated samples
    """
    name = name.lower().replace(' ', '_').replace('-', '_')
    
    if name == 'checkerboard':
        return generate_checkerboard(n_samples, seed=seed)
    elif name in ['gaussian_mixture', 'gaussian_mixtures', 'gmm']:
        return generate_gaussian_mixture(n_samples, seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {name}. Choose 'checkerboard' or 'gaussian_mixture'")


def visualize_dataset(data: np.ndarray, title: str = "Dataset", 
                      save_path: Optional[str] = None) -> None:
    """
    Visualize a 2D dataset.
    
    Args:
        data: Array of shape (n_samples, 2)
        title: Plot title
        save_path: If provided, save figure to this path
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(data[:, 0], data[:, 1], s=5, alpha=0.6, c='blue')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Test dataset generation
    print("Testing dataset generators...")
    
    # Generate and visualize checkerboard
    cb_data = generate_checkerboard(2000, seed=42)
    print(f"Checkerboard: shape={cb_data.shape}, range=[{cb_data.min():.3f}, {cb_data.max():.3f}]")
    
    # Generate and visualize Gaussian mixture
    gm_data = generate_gaussian_mixture(2000, seed=42)
    print(f"Gaussian Mixture: shape={gm_data.shape}, range=[{gm_data.min():.3f}, {gm_data.max():.3f}]")
    
    print("Dataset generation successful!")
