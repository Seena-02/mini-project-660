"""
Visualization Utilities for Generative Models
==============================================
This module provides plotting functions for training curves,
sample comparisons, and evaluation visualization.

Author: Seena
Course: EE660 - Machine Learning
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
import os


# Set matplotlib style for academic papers
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})


def plot_training_curves(train_losses: List[float],
                         val_losses: Optional[List[float]] = None,
                         title: str = "Training Curve",
                         ylabel: str = "Loss",
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training loss values
        val_losses: List of validation loss values (optional)
        title: Plot title
        ylabel: Y-axis label
        save_path: Path to save figure (optional)
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Train', color='blue', linewidth=1.5)
    
    if val_losses is not None and len(val_losses) > 0:
        ax.plot(epochs, val_losses, label='Validation', color='orange', 
                linewidth=1.5, linestyle='--')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Use log scale if values span multiple orders of magnitude
    if len(train_losses) > 0:
        min_val = min(train_losses)
        max_val = max(train_losses)
        if max_val > 0 and min_val > 0 and max_val / min_val > 100:
            ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    
    return fig


def plot_gan_training_curves(g_losses: List[float],
                             d_losses: List[float],
                             title: str = "GAN Training Curves",
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot GAN generator and discriminator loss curves.
    
    Args:
        g_losses: Generator losses
        d_losses: Discriminator losses
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    epochs = range(1, len(g_losses) + 1)
    ax.plot(epochs, g_losses, label='Generator', color='blue', linewidth=1.5)
    ax.plot(epochs, d_losses, label='Discriminator', color='red', linewidth=1.5)
    
    # Add optimal discriminator loss line (-2*log(2) ≈ -1.386)
    optimal_loss = -2 * np.log(2)
    ax.axhline(y=optimal_loss, color='green', linestyle=':', 
               label=f'Optimal D loss ({optimal_loss:.3f})', alpha=0.7)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    
    return fig


def plot_samples_comparison(real_samples: np.ndarray,
                            generated_samples: np.ndarray,
                            title: str = "Sample Comparison",
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot real and generated samples side by side.
    
    Args:
        real_samples: Real data samples (n, 2)
        generated_samples: Generated samples (n, 2)
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Determine axis limits
    all_samples = np.vstack([real_samples, generated_samples])
    x_min, x_max = all_samples[:, 0].min() - 0.2, all_samples[:, 0].max() + 0.2
    y_min, y_max = all_samples[:, 1].min() - 0.2, all_samples[:, 1].max() + 0.2
    
    # Plot real samples
    axes[0].scatter(real_samples[:, 0], real_samples[:, 1], 
                    s=5, alpha=0.6, c='blue', label='Real')
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_min, y_max)
    axes[0].set_aspect('equal')
    axes[0].set_title('Real Data')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].grid(True, alpha=0.3)
    
    # Plot generated samples
    axes[1].scatter(generated_samples[:, 0], generated_samples[:, 1], 
                    s=5, alpha=0.6, c='red', label='Generated')
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(y_min, y_max)
    axes[1].set_aspect('equal')
    axes[1].set_title('Generated Data')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].grid(True, alpha=0.3)
    
    # Overlay plot
    axes[2].scatter(real_samples[:, 0], real_samples[:, 1], 
                    s=5, alpha=0.5, c='blue', label='Real')
    axes[2].scatter(generated_samples[:, 0], generated_samples[:, 1], 
                    s=5, alpha=0.5, c='red', label='Generated')
    axes[2].set_xlim(x_min, x_max)
    axes[2].set_ylim(y_min, y_max)
    axes[2].set_aspect('equal')
    axes[2].set_title('Overlay')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    
    return fig


def plot_dataset_grid(datasets: Dict[str, np.ndarray],
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot multiple datasets in a grid.
    
    Args:
        datasets: Dictionary mapping dataset names to data arrays
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    n_datasets = len(datasets)
    fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 5))
    
    if n_datasets == 1:
        axes = [axes]
    
    for ax, (name, data) in zip(axes, datasets.items()):
        ax.scatter(data[:, 0], data[:, 1], s=5, alpha=0.6)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title(name)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    
    return fig


def plot_model_comparison_grid(results: Dict[str, Dict[str, np.ndarray]],
                               real_data: Dict[str, np.ndarray],
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a grid comparing multiple models across multiple datasets.
    
    Args:
        results: Nested dict {model_name: {dataset_name: generated_samples}}
        real_data: Dict {dataset_name: real_samples}
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    models = list(results.keys())
    datasets = list(real_data.keys())
    
    n_models = len(models)
    n_datasets = len(datasets)
    
    fig, axes = plt.subplots(n_datasets, n_models + 1, 
                             figsize=(4 * (n_models + 1), 4 * n_datasets))
    
    if n_datasets == 1:
        axes = axes.reshape(1, -1)
    
    for i, dataset_name in enumerate(datasets):
        # Plot real data
        ax = axes[i, 0]
        data = real_data[dataset_name]
        ax.scatter(data[:, 0], data[:, 1], s=3, alpha=0.6, c='blue')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title(f'{dataset_name}\n(Real Data)')
        ax.grid(True, alpha=0.3)
        
        if i == n_datasets - 1:
            ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # Plot each model's generated samples
        for j, model_name in enumerate(models):
            ax = axes[i, j + 1]
            if dataset_name in results[model_name]:
                gen_samples = results[model_name][dataset_name]
                ax.scatter(gen_samples[:, 0], gen_samples[:, 1], 
                          s=3, alpha=0.6, c='red')
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect('equal')
            ax.set_title(f'{dataset_name}\n({model_name})')
            ax.grid(True, alpha=0.3)
            
            if i == n_datasets - 1:
                ax.set_xlabel('x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    
    return fig


def plot_evaluation_bar_chart(eval_results: Dict[str, Dict[str, float]],
                              metric: str,
                              title: Optional[str] = None,
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a bar chart comparing models on a specific metric.
    
    Args:
        eval_results: {model_name: {metric_name: value}}
        metric: Which metric to plot
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    models = list(eval_results.keys())
    values = [eval_results[m].get(metric, 0) for m in models]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bars = ax.bar(models, values, color=['blue', 'red', 'green', 'orange'][:len(models)])
    
    ax.set_ylabel(metric)
    ax.set_title(title or f'Model Comparison: {metric}')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    
    return fig


def create_all_figures(figure_dir: str = 'figures'):
    """
    Create the figures directory if it doesn't exist.
    
    Args:
        figure_dir: Path to figures directory
    """
    os.makedirs(figure_dir, exist_ok=True)


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization utilities...")
    
    # Create test data
    np.random.seed(42)
    real_data = np.random.randn(500, 2)
    gen_data = real_data + np.random.randn(500, 2) * 0.3
    
    # Test training curve plot
    train_losses = np.exp(-np.linspace(0, 3, 100)) + np.random.randn(100) * 0.01
    val_losses = np.exp(-np.linspace(0, 2.5, 100)) + np.random.randn(100) * 0.02
    
    fig = plot_training_curves(train_losses.tolist(), val_losses.tolist(), 
                               title="Test Training Curve")
    plt.close()
    
    # Test sample comparison
    fig = plot_samples_comparison(real_data, gen_data, title="Test Comparison")
    plt.close()
    
    print("Visualization utilities test passed!")
