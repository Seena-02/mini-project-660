"""
Main Training Script for EE660 Mini-Project
============================================
This script trains VAE and GAN models on Checkerboard and 
Gaussian Mixture datasets, generates all figures, and 
computes evaluation metrics.

Usage:
    python main.py

Author: Seena
Course: EE660 - Machine Learning
"""

import os
import sys
import numpy as np
import torch
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import generate_checkerboard, generate_gaussian_mixture, get_dataset
from vae import VAE, VAETrainer, create_vae
from gan import GAN, GANTrainer, create_gan
from evaluation import evaluate_model, print_evaluation_results
from visualization import (
    plot_training_curves, plot_gan_training_curves,
    plot_samples_comparison, plot_dataset_grid,
    plot_model_comparison_grid, create_all_figures
)


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # Data configuration
    'n_train': 2000,
    'n_val': 500,
    'n_test': 500,
    'random_seed': 42,
    
    # Datasets to use
    'datasets': ['checkerboard', 'gaussian_mixture'],
    
    # VAE configuration
    'vae': {
        'input_dim': 2,
        'hidden_dims': [128, 128],
        'latent_dim': 2,
        'beta': 1.0,
        'lr': 1e-3,
        'n_epochs': 1500,
        'batch_size': 128,
    },
    
    # GAN configuration
    'gan': {
        'input_dim': 2,
        'noise_dim': 8,
        'hidden_dims': [256, 256],
        'lr_g': 2e-4,
        'lr_d': 2e-4,
        'n_epochs': 2000,
        'batch_size': 128,
        'n_critic': 1,
    },
    
    # Output configuration
    'figure_dir': 'figures',
    'results_dir': 'results',
}


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def generate_datasets(config: dict) -> dict:
    """
    Generate all datasets for training and evaluation.
    
    Returns:
        Dictionary containing train/val/test splits for each dataset
    """
    data = {}
    
    for dataset_name in config['datasets']:
        print(f"\nGenerating {dataset_name} dataset...")
        
        # Generate with different seeds for train/val/test
        train = get_dataset(dataset_name, config['n_train'], seed=config['random_seed'])
        val = get_dataset(dataset_name, config['n_val'], seed=config['random_seed'] + 1)
        test = get_dataset(dataset_name, config['n_test'], seed=config['random_seed'] + 2)
        
        data[dataset_name] = {
            'train': train,
            'val': val,
            'test': test
        }
        
        print(f"  Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
    
    return data


def train_vae(train_data: np.ndarray, val_data: np.ndarray, 
              config: dict, device: torch.device) -> tuple:
    """
    Train a VAE model.
    
    Returns:
        (model, trainer, history)
    """
    vae_config = config['vae']
    
    # Create model
    model = create_vae(
        input_dim=vae_config['input_dim'],
        hidden_dims=vae_config['hidden_dims'],
        latent_dim=vae_config['latent_dim'],
        beta=vae_config['beta']
    )
    
    # Create trainer
    trainer = VAETrainer(model, lr=vae_config['lr'], device=str(device))
    
    # Train
    history = trainer.train(
        train_data, val_data,
        n_epochs=vae_config['n_epochs'],
        batch_size=vae_config['batch_size'],
        verbose=True
    )
    
    return model, trainer, history


def train_gan(train_data: np.ndarray, val_data: np.ndarray,
              config: dict, device: torch.device) -> tuple:
    """
    Train a GAN model.
    
    Returns:
        (model, trainer, history)
    """
    gan_config = config['gan']
    
    # Create model
    model = create_gan(
        input_dim=gan_config['input_dim'],
        noise_dim=gan_config['noise_dim'],
        hidden_dims=gan_config['hidden_dims']
    )
    
    # Create trainer
    trainer = GANTrainer(
        model, 
        lr_g=gan_config['lr_g'],
        lr_d=gan_config['lr_d'],
        device=str(device)
    )
    
    # Train
    history = trainer.train(
        train_data, val_data,
        n_epochs=gan_config['n_epochs'],
        batch_size=gan_config['batch_size'],
        n_critic=gan_config['n_critic'],
        verbose=True
    )
    
    return model, trainer, history


def main():
    """Main training and evaluation pipeline."""
    
    print("="*60)
    print("EE660 Mini-Project: Generative Modeling")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup
    set_seed(CONFIG['random_seed'])
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Create output directories
    os.makedirs(CONFIG['figure_dir'], exist_ok=True)
    os.makedirs(CONFIG['results_dir'], exist_ok=True)
    
    # Generate datasets
    print("\n" + "="*60)
    print("Step 1: Generating Datasets")
    print("="*60)
    data = generate_datasets(CONFIG)
    
    # Plot and save dataset visualizations
    real_data_for_plot = {name: d['train'] for name, d in data.items()}
    plot_dataset_grid(real_data_for_plot, 
                      save_path=os.path.join(CONFIG['figure_dir'], 'datasets.png'))
    
    # Store results
    all_results = {
        'VAE': {},
        'GAN': {}
    }
    all_histories = {
        'VAE': {},
        'GAN': {}
    }
    all_eval_metrics = {
        'VAE': {},
        'GAN': {}
    }
    trained_models = {
        'VAE': {},
        'GAN': {}
    }
    
    # Train models on each dataset
    for dataset_name, dataset in data.items():
        print("\n" + "="*60)
        print(f"Step 2: Training on {dataset_name}")
        print("="*60)
        
        train_data = dataset['train']
        val_data = dataset['val']
        test_data = dataset['test']
        
        # =====================
        # Train VAE
        # =====================
        print(f"\n--- Training VAE on {dataset_name} ---")
        vae_model, vae_trainer, vae_history = train_vae(
            train_data, val_data, CONFIG, device
        )
        trained_models['VAE'][dataset_name] = vae_trainer
        all_histories['VAE'][dataset_name] = vae_history
        
        # Generate samples
        vae_samples = vae_trainer.sample(len(test_data))
        all_results['VAE'][dataset_name] = vae_samples
        
        # Plot VAE training curve
        plot_training_curves(
            vae_history['train_loss'],
            vae_history['val_loss'],
            title=f'VAE Training Curve ({dataset_name})',
            ylabel='ELBO Loss',
            save_path=os.path.join(CONFIG['figure_dir'], 
                                   f'vae_training_{dataset_name}.png')
        )
        
        # Plot VAE sample comparison
        plot_samples_comparison(
            test_data, vae_samples,
            title=f'VAE Samples vs Real Data ({dataset_name})',
            save_path=os.path.join(CONFIG['figure_dir'],
                                   f'vae_samples_{dataset_name}.png')
        )
        
        # Evaluate VAE
        print(f"\nEvaluating VAE on {dataset_name}...")
        vae_eval = evaluate_model(vae_samples, test_data, train_data)
        all_eval_metrics['VAE'][dataset_name] = vae_eval
        print_evaluation_results(vae_eval, f"VAE ({dataset_name})")
        
        # =====================
        # Train GAN
        # =====================
        print(f"\n--- Training GAN on {dataset_name} ---")
        gan_model, gan_trainer, gan_history = train_gan(
            train_data, val_data, CONFIG, device
        )
        trained_models['GAN'][dataset_name] = gan_trainer
        all_histories['GAN'][dataset_name] = gan_history
        
        # Generate samples
        gan_samples = gan_trainer.sample(len(test_data))
        all_results['GAN'][dataset_name] = gan_samples
        
        # Plot GAN training curves
        plot_gan_training_curves(
            gan_history['g_loss'],
            gan_history['d_loss'],
            title=f'GAN Training Curve ({dataset_name})',
            save_path=os.path.join(CONFIG['figure_dir'],
                                   f'gan_training_{dataset_name}.png')
        )
        
        # Plot GAN sample comparison
        plot_samples_comparison(
            test_data, gan_samples,
            title=f'GAN Samples vs Real Data ({dataset_name})',
            save_path=os.path.join(CONFIG['figure_dir'],
                                   f'gan_samples_{dataset_name}.png')
        )
        
        # Evaluate GAN
        print(f"\nEvaluating GAN on {dataset_name}...")
        gan_eval = evaluate_model(gan_samples, test_data, train_data)
        all_eval_metrics['GAN'][dataset_name] = gan_eval
        print_evaluation_results(gan_eval, f"GAN ({dataset_name})")
    
    # =====================
    # Create comparison grid
    # =====================
    print("\n" + "="*60)
    print("Step 3: Creating Comparison Figures")
    print("="*60)
    
    plot_model_comparison_grid(
        all_results,
        real_data_for_plot,
        save_path=os.path.join(CONFIG['figure_dir'], 'model_comparison_grid.png')
    )
    
    # =====================
    # Save results
    # =====================
    print("\n" + "="*60)
    print("Step 4: Saving Results")
    print("="*60)
    
    # Save evaluation metrics
    results_summary = {
        'config': {k: v for k, v in CONFIG.items() 
                   if k not in ['figure_dir', 'results_dir']},
        'metrics': all_eval_metrics
    }
    
    with open(os.path.join(CONFIG['results_dir'], 'evaluation_metrics.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save training histories (convert to regular lists)
    histories_to_save = {}
    for model_name, datasets in all_histories.items():
        histories_to_save[model_name] = {}
        for dataset_name, history in datasets.items():
            histories_to_save[model_name][dataset_name] = {
                k: [float(v) for v in vals] for k, vals in history.items()
            }
    
    with open(os.path.join(CONFIG['results_dir'], 'training_histories.json'), 'w') as f:
        json.dump(histories_to_save, f, indent=2)
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    print("\n--- Quantitative Comparison ---")
    print(f"{'Model':<8} {'Dataset':<20} {'Log-Lik':<12} {'MMD':<12} {'Coverage':<10} {'Density':<10}")
    print("-" * 72)
    
    for model_name in ['VAE', 'GAN']:
        for dataset_name in CONFIG['datasets']:
            metrics = all_eval_metrics[model_name][dataset_name]
            print(f"{model_name:<8} {dataset_name:<20} "
                  f"{metrics['log_likelihood']:<12.4f} "
                  f"{metrics['mmd']:<12.6f} "
                  f"{metrics['coverage']:<10.4f} "
                  f"{metrics['density']:<10.4f}")
    
    print("\n" + "="*60)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Figures saved to: {CONFIG['figure_dir']}/")
    print(f"Results saved to: {CONFIG['results_dir']}/")
    print("="*60)
    
    return all_results, all_eval_metrics, all_histories


if __name__ == "__main__":
    results, metrics, histories = main()
