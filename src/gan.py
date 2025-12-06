"""
Generative Adversarial Network (GAN) Implementation
====================================================
This module implements a GAN for 2D data generation.

The GAN consists of:
- Generator: Maps noise z to generated samples x
- Discriminator: Classifies samples as real or fake
- Training: Minimax game between generator and discriminator

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


class Generator(nn.Module):
    """
    Generator network that maps noise z to data space.
    
    Architecture: MLP with hidden layers and tanh output activation
    to constrain outputs to [-1, 1] range matching our data.
    """
    
    def __init__(self, noise_dim: int = 2, hidden_dims: List[int] = [128, 128],
                 output_dim: int = 2):
        """
        Args:
            noise_dim: Dimension of input noise vector
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of output data
        """
        super().__init__()
        
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        
        # Build generator layers
        layers = []
        prev_dim = noise_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(hidden_dim),
            ])
            prev_dim = hidden_dim
        
        # Output layer with tanh to bound outputs
        layers.extend([
            nn.Linear(prev_dim, output_dim),
            nn.Tanh()
        ])
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through generator.
        
        Args:
            z: Noise tensor of shape (batch_size, noise_dim)
            
        Returns:
            x_fake: Generated samples of shape (batch_size, output_dim)
        """
        return self.layers(z)


class Discriminator(nn.Module):
    """
    Discriminator network that classifies samples as real or fake.
    
    Architecture: MLP with hidden layers and sigmoid output
    """
    
    def __init__(self, input_dim: int = 2, hidden_dims: List[int] = [128, 128]):
        """
        Args:
            input_dim: Dimension of input data
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        self.input_dim = input_dim
        
        # Build discriminator layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
            ])
            prev_dim = hidden_dim
        
        # Output layer (logit for binary classification)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through discriminator.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            logit: Discrimination logit of shape (batch_size, 1)
        """
        return self.layers(x)


class GAN(nn.Module):
    """
    Generative Adversarial Network combining generator and discriminator.
    
    The GAN learns through a minimax game:
    - Discriminator tries to distinguish real from fake samples
    - Generator tries to fool the discriminator
    
    Loss functions:
    - D loss: -E[log(D(x_real))] - E[log(1 - D(G(z)))]
    - G loss: -E[log(D(G(z)))] (non-saturating version)
    """
    
    def __init__(self, input_dim: int = 2, noise_dim: int = 2,
                 hidden_dims: List[int] = [128, 128]):
        """
        Args:
            input_dim: Dimension of data
            noise_dim: Dimension of noise vector
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.noise_dim = noise_dim
        
        self.generator = Generator(noise_dim, hidden_dims, input_dim)
        self.discriminator = Discriminator(input_dim, hidden_dims)
    
    def generate(self, n_samples: int, device: torch.device = None) -> torch.Tensor:
        """
        Generate samples from the learned distribution.
        
        Args:
            n_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            samples: Generated samples of shape (n_samples, input_dim)
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Sample noise
        z = torch.randn(n_samples, self.noise_dim, device=device)
        
        # Generate samples
        with torch.no_grad():
            samples = self.generator(z)
        
        return samples
    
    def discriminator_loss(self, x_real: torch.Tensor, 
                           x_fake: torch.Tensor) -> torch.Tensor:
        """
        Compute discriminator loss.
        
        D_loss = -E[log(D(x_real))] - E[log(1 - D(x_fake))]
        
        Using BCE with logits for numerical stability.
        
        Args:
            x_real: Real samples
            x_fake: Generated samples
            
        Returns:
            Discriminator loss
        """
        batch_size = x_real.size(0)
        device = x_real.device
        
        # Labels
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(x_fake.size(0), 1, device=device)
        
        # Discriminator outputs
        d_real = self.discriminator(x_real)
        d_fake = self.discriminator(x_fake.detach())
        
        # BCE loss
        loss_real = F.binary_cross_entropy_with_logits(d_real, real_labels)
        loss_fake = F.binary_cross_entropy_with_logits(d_fake, fake_labels)
        
        return loss_real + loss_fake
    
    def generator_loss(self, x_fake: torch.Tensor) -> torch.Tensor:
        """
        Compute generator loss (non-saturating version).
        
        G_loss = -E[log(D(G(z)))]
        
        Using BCE with logits for numerical stability.
        
        Args:
            x_fake: Generated samples
            
        Returns:
            Generator loss
        """
        batch_size = x_fake.size(0)
        device = x_fake.device
        
        # Generator wants discriminator to think fake samples are real
        real_labels = torch.ones(batch_size, 1, device=device)
        d_fake = self.discriminator(x_fake)
        
        return F.binary_cross_entropy_with_logits(d_fake, real_labels)


class GANTrainer:
    """
    Trainer class for GAN with logging and various training strategies.
    """
    
    def __init__(self, model: GAN, lr_g: float = 2e-4, lr_d: float = 2e-4,
                 betas: Tuple[float, float] = (0.5, 0.999), device: str = 'auto'):
        """
        Args:
            model: GAN model to train
            lr_g: Learning rate for generator
            lr_d: Learning rate for discriminator
            betas: Adam optimizer betas
            device: Device to train on ('auto', 'cuda', 'mps', 'cpu')
        """
        self.model = model
        self.lr_g = lr_g
        self.lr_d = lr_d
        
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
        
        # Separate optimizers for G and D
        self.optimizer_g = torch.optim.Adam(
            self.model.generator.parameters(), lr=lr_g, betas=betas
        )
        self.optimizer_d = torch.optim.Adam(
            self.model.discriminator.parameters(), lr=lr_d, betas=betas
        )
        
        # Training history
        self.g_losses = []
        self.d_losses = []
        self.d_real_acc = []
        self.d_fake_acc = []
    
    def train(self, train_data: np.ndarray, val_data: Optional[np.ndarray] = None,
              n_epochs: int = 1000, batch_size: int = 128,
              n_critic: int = 1, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the GAN.
        
        Args:
            train_data: Training data array of shape (n_samples, 2)
            val_data: Validation data array (optional, used for monitoring)
            n_epochs: Number of training epochs
            batch_size: Batch size
            n_critic: Number of discriminator updates per generator update
            verbose: Whether to show progress bar
            
        Returns:
            Dictionary containing training history
        """
        # Create data loader
        train_tensor = torch.FloatTensor(train_data)
        train_dataset = TensorDataset(train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        iterator = tqdm(range(n_epochs), desc="Training GAN") if verbose else range(n_epochs)
        
        for epoch in iterator:
            self.model.train()
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_d_real_acc = 0.0
            epoch_d_fake_acc = 0.0
            n_batches = 0
            
            for batch in train_loader:
                x_real = batch[0].to(self.device)
                batch_size_curr = x_real.size(0)
                
                # ==================
                # Train Discriminator
                # ==================
                for _ in range(n_critic):
                    # Generate fake samples
                    z = torch.randn(batch_size_curr, self.model.noise_dim, device=self.device)
                    x_fake = self.model.generator(z)
                    
                    # Compute discriminator loss
                    d_loss = self.model.discriminator_loss(x_real, x_fake)
                    
                    # Update discriminator
                    self.optimizer_d.zero_grad()
                    d_loss.backward()
                    self.optimizer_d.step()
                
                # ===============
                # Train Generator
                # ===============
                # Generate new fake samples
                z = torch.randn(batch_size_curr, self.model.noise_dim, device=self.device)
                x_fake = self.model.generator(z)
                
                # Compute generator loss
                g_loss = self.model.generator_loss(x_fake)
                
                # Update generator
                self.optimizer_g.zero_grad()
                g_loss.backward()
                self.optimizer_g.step()
                
                # Record losses
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                
                # Compute discriminator accuracy
                with torch.no_grad():
                    d_real = torch.sigmoid(self.model.discriminator(x_real))
                    d_fake = torch.sigmoid(self.model.discriminator(x_fake))
                    epoch_d_real_acc += (d_real > 0.5).float().mean().item()
                    epoch_d_fake_acc += (d_fake < 0.5).float().mean().item()
                
                n_batches += 1
            
            # Record average losses
            avg_g_loss = epoch_g_loss / n_batches
            avg_d_loss = epoch_d_loss / n_batches
            avg_d_real_acc = epoch_d_real_acc / n_batches
            avg_d_fake_acc = epoch_d_fake_acc / n_batches
            
            self.g_losses.append(avg_g_loss)
            self.d_losses.append(avg_d_loss)
            self.d_real_acc.append(avg_d_real_acc)
            self.d_fake_acc.append(avg_d_fake_acc)
            
            # Update progress bar
            if verbose and isinstance(iterator, tqdm):
                iterator.set_postfix({
                    'G_loss': f'{avg_g_loss:.4f}',
                    'D_loss': f'{avg_d_loss:.4f}',
                    'D_acc': f'{(avg_d_real_acc + avg_d_fake_acc) / 2:.2f}'
                })
        
        return {
            'g_loss': self.g_losses,
            'd_loss': self.d_losses,
            'd_real_acc': self.d_real_acc,
            'd_fake_acc': self.d_fake_acc
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
        samples = self.model.generate(n_samples, self.device)
        return samples.cpu().numpy()


def create_gan(input_dim: int = 2, noise_dim: int = 2,
               hidden_dims: List[int] = [128, 128]) -> GAN:
    """
    Factory function to create a GAN model.
    
    Args:
        input_dim: Dimension of data
        noise_dim: Dimension of noise vector
        hidden_dims: List of hidden layer dimensions
        
    Returns:
        GAN model instance
    """
    return GAN(input_dim, noise_dim, hidden_dims)


if __name__ == "__main__":
    # Test GAN implementation
    print("Testing GAN implementation...")
    
    # Create model
    gan = create_gan(input_dim=2, noise_dim=2, hidden_dims=[128, 128])
    g_params = sum(p.numel() for p in gan.generator.parameters())
    d_params = sum(p.numel() for p in gan.discriminator.parameters())
    print(f"GAN created - Generator: {g_params} params, Discriminator: {d_params} params")
    
    # Test generator forward pass
    z = torch.randn(32, 2)
    x_fake = gan.generator(z)
    print(f"Generator: z={z.shape} -> x_fake={x_fake.shape}")
    
    # Test discriminator forward pass
    x_real = torch.randn(32, 2)
    d_out = gan.discriminator(x_real)
    print(f"Discriminator: x={x_real.shape} -> d={d_out.shape}")
    
    # Test losses
    d_loss = gan.discriminator_loss(x_real, x_fake)
    g_loss = gan.generator_loss(x_fake)
    print(f"Losses: D_loss={d_loss:.4f}, G_loss={g_loss:.4f}")
    
    # Test sampling
    samples = gan.generate(100)
    print(f"Sampling: {samples.shape}")
    
    print("GAN implementation test passed!")
