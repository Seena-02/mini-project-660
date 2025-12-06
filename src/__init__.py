"""
EE660 Mini-Project: Generative Modeling
=======================================

This package contains implementations of VAE and GAN generative models
for 2D toy datasets.

Modules:
    - datasets: Data generation utilities
    - vae: Variational Autoencoder implementation
    - gan: Generative Adversarial Network implementation
    - evaluation: Quantitative evaluation metrics
    - visualization: Plotting utilities
    - main: Main training pipeline

Author: Seena
Course: EE660 - Machine Learning
"""

from .datasets import generate_checkerboard, generate_gaussian_mixture, get_dataset
from .vae import VAE, VAETrainer, create_vae
from .gan import GAN, GANTrainer, create_gan
from .evaluation import evaluate_model, estimate_log_likelihood_kde, compute_mmd
from .visualization import (
    plot_training_curves, 
    plot_gan_training_curves,
    plot_samples_comparison,
    plot_dataset_grid
)

__version__ = '1.0.0'
__author__ = 'Seena'
