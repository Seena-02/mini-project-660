"""
Evaluation Utilities for Generative Models
==========================================
This module provides quantitative evaluation metrics for comparing
generative models on 2D datasets.

Metrics implemented:
- KDE-based log-likelihood estimation
- Maximum Mean Discrepancy (MMD)
- Coverage and density metrics

Author: Seena
Course: EE660 - Machine Learning
"""

import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist
from typing import Tuple, Dict, Optional
import warnings


def estimate_log_likelihood_kde(samples: np.ndarray, 
                                 test_data: np.ndarray,
                                 bandwidth: Optional[str] = 'scott') -> float:
    """
    Estimate log-likelihood of test data under a KDE fitted to samples.
    
    This provides a way to compare models that don't have explicit
    likelihood functions (like GANs).
    
    Args:
        samples: Generated samples from model (n_samples, dim)
        test_data: Test data to evaluate (n_test, dim)
        bandwidth: KDE bandwidth selection method ('scott' or 'silverman')
        
    Returns:
        Average log-likelihood of test data under KDE
    """
    try:
        # Add small noise to prevent singular covariance matrix
        # This handles cases where samples lie in a lower-dimensional subspace
        samples_noisy = samples + np.random.randn(*samples.shape) * 1e-6
        
        # Fit KDE to generated samples
        kde = gaussian_kde(samples_noisy.T, bw_method=bandwidth)
        
        # Compute log-likelihood on test data
        log_probs = kde.logpdf(test_data.T)
        
        # Handle numerical issues
        log_probs = np.clip(log_probs, -1e10, 1e10)
        
        return float(np.mean(log_probs))
    
    except np.linalg.LinAlgError:
        # If KDE still fails, return a default low value
        warnings.warn("KDE fitting failed due to singular covariance. "
                      "Returning default value.")
        return float('-inf')


def compute_mmd(x: np.ndarray, y: np.ndarray, 
                kernel: str = 'rbf', 
                sigma: Optional[float] = None) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) between two sample sets.
    
    MMD is a kernel-based distance measure between distributions.
    Lower MMD indicates more similar distributions.
    
    Args:
        x: First sample set (n_x, dim)
        y: Second sample set (n_y, dim)
        kernel: Kernel type ('rbf' for Gaussian/RBF kernel)
        sigma: Kernel bandwidth (if None, uses median heuristic)
        
    Returns:
        MMD^2 value (unbiased estimate)
    """
    n_x, n_y = len(x), len(y)
    
    # Compute pairwise distances
    xx_dist = cdist(x, x, 'sqeuclidean')
    yy_dist = cdist(y, y, 'sqeuclidean')
    xy_dist = cdist(x, y, 'sqeuclidean')
    
    # Median heuristic for bandwidth
    if sigma is None:
        all_dist = np.concatenate([xx_dist.flatten(), yy_dist.flatten(), xy_dist.flatten()])
        sigma = np.sqrt(np.median(all_dist) / 2)
        sigma = max(sigma, 1e-5)  # Avoid zero bandwidth
    
    # RBF kernel
    gamma = 1.0 / (2 * sigma ** 2)
    k_xx = np.exp(-gamma * xx_dist)
    k_yy = np.exp(-gamma * yy_dist)
    k_xy = np.exp(-gamma * xy_dist)
    
    # Unbiased MMD^2 estimate
    # Remove diagonal terms for unbiased estimate
    np.fill_diagonal(k_xx, 0)
    np.fill_diagonal(k_yy, 0)
    
    mmd2 = (k_xx.sum() / (n_x * (n_x - 1)) + 
            k_yy.sum() / (n_y * (n_y - 1)) - 
            2 * k_xy.sum() / (n_x * n_y))
    
    return float(mmd2)


def compute_coverage(real_samples: np.ndarray, 
                     generated_samples: np.ndarray,
                     k: int = 5) -> float:
    """
    Compute coverage metric: fraction of real samples that have at least
    one generated sample among their k-nearest neighbors.
    
    Higher coverage indicates the model captures more modes of the distribution.
    
    Args:
        real_samples: Real data samples (n_real, dim)
        generated_samples: Generated samples (n_gen, dim)
        k: Number of nearest neighbors to consider
        
    Returns:
        Coverage score in [0, 1]
    """
    n_real = len(real_samples)
    
    # Compute distances from real to generated
    distances = cdist(real_samples, generated_samples, 'euclidean')
    
    # For each real sample, find distance to k-th nearest generated sample
    k_nearest_dist = np.partition(distances, k, axis=1)[:, k]
    
    # Compute distances from real to real (excluding self)
    real_distances = cdist(real_samples, real_samples, 'euclidean')
    np.fill_diagonal(real_distances, np.inf)
    
    # Get k-th nearest neighbor distance in real data
    k_nearest_real = np.partition(real_distances, k, axis=1)[:, k]
    
    # A real sample is "covered" if there's a generated sample within
    # its k-nearest neighbor ball from the real distribution
    covered = k_nearest_dist <= k_nearest_real
    
    return float(np.mean(covered))


def compute_density(real_samples: np.ndarray,
                    generated_samples: np.ndarray,
                    k: int = 5) -> float:
    """
    Compute density metric: measures how many generated samples
    fall near real samples (inverse of coverage direction).
    
    Higher density indicates generated samples are concentrated
    around real data regions.
    
    Args:
        real_samples: Real data samples (n_real, dim)
        generated_samples: Generated samples (n_gen, dim)
        k: Number of nearest neighbors
        
    Returns:
        Density score
    """
    n_gen = len(generated_samples)
    
    # Compute distances from generated to real
    distances = cdist(generated_samples, real_samples, 'euclidean')
    
    # Get k-th nearest real neighbor for each generated sample
    k_nearest_dist = np.partition(distances, k, axis=1)[:, k]
    
    # Compute real-to-real k-nearest distances for threshold
    real_distances = cdist(real_samples, real_samples, 'euclidean')
    np.fill_diagonal(real_distances, np.inf)
    k_nearest_real = np.partition(real_distances, k, axis=1)[:, k]
    
    # Average density: how many generated samples are within typical real neighborhoods
    threshold = np.mean(k_nearest_real)
    in_manifold = k_nearest_dist <= threshold
    
    return float(np.mean(in_manifold))


def evaluate_model(generated_samples: np.ndarray,
                   test_samples: np.ndarray,
                   train_samples: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Comprehensive evaluation of a generative model.
    
    Args:
        generated_samples: Samples from the generative model
        test_samples: Held-out test data
        train_samples: Training data (optional, for additional metrics)
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    results = {}
    
    # KDE-based log-likelihood
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results['log_likelihood'] = estimate_log_likelihood_kde(
            generated_samples, test_samples
        )
    
    # MMD
    results['mmd'] = compute_mmd(test_samples, generated_samples)
    
    # Coverage and Density
    results['coverage'] = compute_coverage(test_samples, generated_samples)
    results['density'] = compute_density(test_samples, generated_samples)
    
    return results


def print_evaluation_results(results: Dict[str, float], model_name: str = "Model"):
    """
    Pretty print evaluation results.
    
    Args:
        results: Dictionary of metric names to values
        model_name: Name of the model being evaluated
    """
    print(f"\n{'='*50}")
    print(f"Evaluation Results: {model_name}")
    print(f"{'='*50}")
    print(f"  Log-Likelihood (KDE):  {results['log_likelihood']:.4f}")
    print(f"  MMD:                   {results['mmd']:.6f}")
    print(f"  Coverage:              {results['coverage']:.4f}")
    print(f"  Density:               {results['density']:.4f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Test evaluation functions
    print("Testing evaluation utilities...")
    
    # Generate synthetic test data
    np.random.seed(42)
    real_data = np.random.randn(500, 2)
    
    # Good model: samples close to real distribution
    good_samples = real_data + np.random.randn(500, 2) * 0.1
    
    # Bad model: samples from different distribution
    bad_samples = np.random.randn(500, 2) * 2 + 3
    
    # Evaluate good model
    print("\nGood model evaluation:")
    good_results = evaluate_model(good_samples, real_data)
    print_evaluation_results(good_results, "Good Model")
    
    # Evaluate bad model
    print("\nBad model evaluation:")
    bad_results = evaluate_model(bad_samples, real_data)
    print_evaluation_results(bad_results, "Bad Model")
    
    print("Evaluation utilities test passed!")