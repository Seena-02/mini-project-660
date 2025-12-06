"""
Variational Autoencoder (VAE) Implementation
=============================================
This module implements a VAE for 2D data generation.

The VAE consists of:
- Encoder: Maps input x to latent distribution parameters (mu, log_var)
- Decoder: Maps latent z to reconstructed x
- Loss: Reconstruction loss + KL divergence

Author: Seena
Course: EE660 - Machine Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, List, Optional, Dict
from tqdm import tqdm


class VAEEncoder(nn.Module):
    """
    Encoder network that maps input x to latent distribution parameters.
    
    Architecture: MLP with hidden layers
    Output: mean and log_variance of latent distribution
    """
    
    def __init__(self, input_dim: int = 2, hidden_dims: List[int] = [128, 128], 
                 latent_dim: int = 2):
        """
        Args:
            input_dim: Dimension of input data
            hidden_dims: List of hidden layer dimensions
            latent_dim: Dimension of latent space
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Output layers for mean and log_variance
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            mu: Mean of latent distribution (batch_size, latent_dim)
            log_var: Log variance of latent distribution (batch_size, latent_dim)
        """
        h = self.shared_layers(x)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var


class VAEDecoder(nn.Module):
    """
    Decoder network that maps latent z to reconstructed x.
    
    Architecture: MLP with hidden layers
    """
    
    def __init__(self, latent_dim: int = 2, hidden_dims: List[int] = [128, 128], 
                 output_dim: int = 2):
        """
        Args:
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of output data
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Build decoder layers
        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            z: Latent tensor of shape (batch_size, latent_dim)
            
        Returns:
            x_recon: Reconstructed input of shape (batch_size, output_dim)
        """
        return self.layers(z)


class VAE(nn.Module):
    """
    Variational Autoencoder combining encoder and decoder.
    
    The VAE learns to:
    1. Encode input x to a latent distribution q(z|x)
    2. Sample z from q(z|x) using reparameterization trick
    3. Decode z to reconstruct x
    
    Loss = Reconstruction Loss + β * KL Divergence
    """
    
    def __init__(self, input_dim: int = 2, hidden_dims: List[int] = [128, 128],
                 latent_dim: int = 2, beta: float = 1.0):
        """
        Args:
            input_dim: Dimension of input data
            hidden_dims: List of hidden layer dimensions
            latent_dim: Dimension of latent space
            beta: Weight for KL divergence term (beta-VAE)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        
        self.encoder = VAEEncoder(input_dim, hidden_dims, latent_dim)
        self.decoder = VAEDecoder(latent_dim, list(reversed(hidden_dims)), input_dim)
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * epsilon
        
        This allows gradients to flow through the sampling operation.
        
        Args:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
            
        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode, sample, decode.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            x_recon: Reconstructed input
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var
    
    def loss_function(self, x: torch.Tensor, x_recon: torch.Tensor, 
                      mu: torch.Tensor, log_var: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss: Reconstruction + β * KL Divergence
        
        Reconstruction loss: MSE between x and x_recon
        KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0, I)
        
        Args:
            x: Original input
            x_recon: Reconstructed input
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
            
        Returns:
            Dictionary containing total loss and components
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        
        # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def sample(self, n_samples: int, device: torch.device = None) -> torch.Tensor:
        """
        Generate samples from the learned distribution.
        
        Samples z ~ N(0, I) and decodes to x.
        
        Args:
            n_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            samples: Generated samples of shape (n_samples, input_dim)
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Sample from prior p(z) = N(0, I)
        z = torch.randn(n_samples, self.latent_dim, device=device)
        
        # Decode
        with torch.no_grad():
            samples = self.decoder(z)
        
        return samples
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input tensor
            
        Returns:
            mu, log_var: Latent distribution parameters
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to output space.
        
        Args:
            z: Latent tensor
            
        Returns:
            Decoded output
        """
        return self.decoder(z)


class VAETrainer:
    """
    Trainer class for VAE with logging and checkpointing.
    """
    
    def __init__(self, model: VAE, lr: float = 1e-3, device: str = 'auto'):
        """
        Args:
            model: VAE model to train
            lr: Learning rate
            device: Device to train on ('auto', 'cuda', 'mps', 'cpu')
        """
        self.model = model
        self.lr = lr
        
        # Set device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_recon_losses = []
        self.train_kl_losses = []
    
    def train(self, train_data: np.ndarray, val_data: Optional[np.ndarray] = None,
              n_epochs: int = 1000, batch_size: int = 128, 
              verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the VAE.
        
        Args:
            train_data: Training data array of shape (n_samples, 2)
            val_data: Validation data array (optional)
            n_epochs: Number of training epochs
            batch_size: Batch size
            verbose: Whether to show progress bar
            
        Returns:
            Dictionary containing training history
        """
        # Create data loaders
        train_tensor = torch.FloatTensor(train_data)
        train_dataset = TensorDataset(train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_data is not None:
            val_tensor = torch.FloatTensor(val_data).to(self.device)
        
        # Training loop
        iterator = tqdm(range(n_epochs), desc="Training VAE") if verbose else range(n_epochs)
        
        for epoch in iterator:
            self.model.train()
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            n_batches = 0
            
            for batch in train_loader:
                x = batch[0].to(self.device)
                
                # Forward pass
                x_recon, mu, log_var = self.model(x)
                losses = self.model.loss_function(x, x_recon, mu, log_var)
                
                # Backward pass
                self.optimizer.zero_grad()
                losses['loss'].backward()
                self.optimizer.step()
                
                epoch_loss += losses['loss'].item()
                epoch_recon += losses['recon_loss'].item()
                epoch_kl += losses['kl_loss'].item()
                n_batches += 1
            
            # Record average loss
            avg_loss = epoch_loss / n_batches
            avg_recon = epoch_recon / n_batches
            avg_kl = epoch_kl / n_batches
            self.train_losses.append(avg_loss)
            self.train_recon_losses.append(avg_recon)
            self.train_kl_losses.append(avg_kl)
            
            # Validation loss
            if val_data is not None:
                self.model.eval()
                with torch.no_grad():
                    x_recon, mu, log_var = self.model(val_tensor)
                    val_losses = self.model.loss_function(val_tensor, x_recon, mu, log_var)
                    self.val_losses.append(val_losses['loss'].item())
            
            # Update progress bar
            if verbose and isinstance(iterator, tqdm):
                if val_data is not None:
                    iterator.set_postfix({
                        'train_loss': f'{avg_loss:.4f}',
                        'val_loss': f'{self.val_losses[-1]:.4f}'
                    })
                else:
                    iterator.set_postfix({'train_loss': f'{avg_loss:.4f}'})
        
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'recon_loss': self.train_recon_losses,
            'kl_loss': self.train_kl_losses
        }
    
    def sample(self, n_samples: int) -> np.ndarray:
        """
        Generate samples from trained model.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Generated samples as numpy array
        """
        self.model.eval()
        samples = self.model.sample(n_samples, self.device)
        return samples.cpu().numpy()


def create_vae(input_dim: int = 2, hidden_dims: List[int] = [128, 128],
               latent_dim: int = 2, beta: float = 1.0) -> VAE:
    """
    Factory function to create a VAE model.
    
    Args:
        input_dim: Dimension of input data
        hidden_dims: List of hidden layer dimensions
        latent_dim: Dimension of latent space
        beta: Weight for KL divergence term
        
    Returns:
        VAE model instance
    """
    return VAE(input_dim, hidden_dims, latent_dim, beta)


if __name__ == "__main__":
    # Test VAE implementation
    print("Testing VAE implementation...")
    
    # Create model
    vae = create_vae(input_dim=2, hidden_dims=[128, 128], latent_dim=2)
    print(f"VAE created with {sum(p.numel() for p in vae.parameters())} parameters")
    
    # Test forward pass
    x = torch.randn(32, 2)
    x_recon, mu, log_var = vae(x)
    print(f"Forward pass: x={x.shape} -> x_recon={x_recon.shape}, mu={mu.shape}")
    
    # Test loss computation
    losses = vae.loss_function(x, x_recon, mu, log_var)
    print(f"Losses: total={losses['loss']:.4f}, recon={losses['recon_loss']:.4f}, kl={losses['kl_loss']:.4f}")
    
    # Test sampling
    samples = vae.sample(100)
    print(f"Sampling: {samples.shape}")
    
    print("VAE implementation test passed!")
