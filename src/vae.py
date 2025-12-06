"""
Variational Autoencoder (VAE) Implementation
=============================================
Robust implementation for 2D data generation.

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


class VAE(nn.Module):
    """
    Variational Autoencoder for 2D data.
    
    Uses separate encoder and decoder MLPs with proper initialization.
    """
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 256, 
                 latent_dim: int = 16, num_layers: int = 3, beta: float = 0.01):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder: x -> hidden -> (mu, logvar)
        encoder_layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            encoder_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: z -> hidden -> x
        decoder_layers = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            decoder_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        decoder_layers.append(nn.Linear(hidden_dim, input_dim))
        # NO activation on output - data is in [-1, 1] but we use MSE loss
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        # Clamp logvar for stability
        logvar = torch.clamp(logvar, min=-10, max=10)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def loss_function(self, x: torch.Tensor, x_recon: torch.Tensor,
                      mu: torch.Tensor, logvar: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        
        # KL divergence: -0.5 * mean(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss with beta weighting
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def sample(self, n_samples: int, device: torch.device = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(n_samples, self.latent_dim, device=device)
        with torch.no_grad():
            samples = self.decode(z)
        return samples


class VAETrainer:
    """Trainer for VAE with proper logging."""
    
    def __init__(self, model: VAE, lr: float = 1e-3, device: str = 'auto'):
        self.model = model
        
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
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=100
        )
        
        self.train_losses = []
        self.val_losses = []
        self.train_recon_losses = []
        self.train_kl_losses = []
    
    def train(self, train_data: np.ndarray, val_data: Optional[np.ndarray] = None,
              n_epochs: int = 2000, batch_size: int = 64, 
              verbose: bool = True) -> Dict[str, List[float]]:
        
        # Create data loaders
        train_tensor = torch.FloatTensor(train_data)
        train_dataset = TensorDataset(train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_data is not None:
            val_tensor = torch.FloatTensor(val_data).to(self.device)
        
        iterator = tqdm(range(n_epochs), desc="Training VAE") if verbose else range(n_epochs)
        
        for epoch in iterator:
            self.model.train()
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            n_batches = 0
            
            for batch in train_loader:
                x = batch[0].to(self.device)
                
                x_recon, mu, logvar = self.model(x)
                losses = self.model.loss_function(x, x_recon, mu, logvar)
                
                self.optimizer.zero_grad()
                losses['loss'].backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += losses['loss'].item()
                epoch_recon += losses['recon_loss'].item()
                epoch_kl += losses['kl_loss'].item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            avg_recon = epoch_recon / n_batches
            avg_kl = epoch_kl / n_batches
            
            self.train_losses.append(avg_loss)
            self.train_recon_losses.append(avg_recon)
            self.train_kl_losses.append(avg_kl)
            
            # Validation
            if val_data is not None:
                self.model.eval()
                with torch.no_grad():
                    x_recon, mu, logvar = self.model(val_tensor)
                    val_losses = self.model.loss_function(val_tensor, x_recon, mu, logvar)
                    self.val_losses.append(val_losses['loss'].item())
                self.scheduler.step(val_losses['loss'])
            
            if verbose and isinstance(iterator, tqdm):
                if val_data is not None:
                    iterator.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'recon': f'{avg_recon:.4f}',
                        'kl': f'{avg_kl:.4f}',
                        'val': f'{self.val_losses[-1]:.4f}'
                    })
                else:
                    iterator.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'recon_loss': self.train_recon_losses,
            'kl_loss': self.train_kl_losses
        }
    
    def sample(self, n_samples: int) -> np.ndarray:
        self.model.eval()
        samples = self.model.sample(n_samples, self.device)
        return samples.cpu().numpy()


def create_vae(input_dim: int = 2, hidden_dim: int = 256, latent_dim: int = 16,
               num_layers: int = 3, beta: float = 0.01) -> VAE:
    """Factory function to create VAE."""
    return VAE(input_dim, hidden_dim, latent_dim, num_layers, beta)


if __name__ == "__main__":
    print("Testing VAE implementation...")
    
    # Create simple test data (Gaussian mixture)
    np.random.seed(42)
    n_samples = 1000
    centers = np.array([[0.5, 0.5], [-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5]])
    data = []
    for c in centers:
        data.append(np.random.randn(n_samples // 4, 2) * 0.1 + c)
    data = np.vstack(data).astype(np.float32)
    
    # Train VAE
    vae = create_vae(input_dim=2, hidden_dim=128, latent_dim=8, num_layers=2, beta=0.01)
    trainer = VAETrainer(vae, lr=1e-3)
    history = trainer.train(data, val_data=data[:200], n_epochs=500, verbose=True)
    
    # Generate samples
    samples = trainer.sample(500)
    print(f"\nGenerated samples shape: {samples.shape}")
    print(f"Generated samples range: [{samples.min():.3f}, {samples.max():.3f}]")
    print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
    
    print("\nVAE test passed!")