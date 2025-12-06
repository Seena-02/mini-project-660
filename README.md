# EE660 Mini-Project: Generative Modeling

## Overview

This project implements and compares two generative models:
1. **Variational Autoencoder (VAE)**
2. **Generative Adversarial Network (GAN)**

Both models are trained on two 2D toy datasets:
- Checkerboard
- Gaussian Mixture

## Project Structure

```
ee660_project/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── datasets.py          # Dataset generation utilities
│   ├── vae.py               # VAE implementation
│   ├── gan.py               # GAN implementation
│   ├── evaluation.py        # Quantitative evaluation metrics
│   ├── visualization.py     # Plotting utilities
│   └── main.py              # Main training pipeline
├── figures/                  # Generated figures (created after running)
├── results/                  # Training results and metrics (created after running)
├── report/                   # LaTeX report source
│   ├── report.tex           # Main LaTeX file
│   └── *.png                # Figures (copy from figures/)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy, SciPy, Matplotlib, tqdm

## Installation

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Code

### Full Training Pipeline

Run the main training script to train both models on both datasets:

```bash
cd src
python main.py
```

This will:
1. Generate the Checkerboard and Gaussian Mixture datasets
2. Train VAE and GAN on each dataset
3. Generate all figures in `figures/`
4. Save evaluation metrics in `results/`
5. Print a summary of results

### Expected Output

After running, you should see:
- Training progress bars for each model/dataset combination
- Evaluation metrics printed to console
- Figures saved to `figures/` directory
- JSON results saved to `results/` directory

## Model Details

### VAE (Variational Autoencoder)

- **Architecture**: 2-layer MLP encoder/decoder with 128 hidden units
- **Latent dimension**: 2
- **Loss**: Reconstruction (MSE) + β * KL divergence
- **Optimizer**: Adam (lr=1e-3)
- **Training**: 1500 epochs

### GAN (Generative Adversarial Network)

- **Generator**: 2-layer MLP with 256 hidden units, BatchNorm, LeakyReLU
- **Discriminator**: 2-layer MLP with 256 hidden units, Dropout, LeakyReLU
- **Noise dimension**: 8
- **Loss**: Binary cross-entropy (non-saturating GAN loss)
- **Optimizer**: Adam (lr=2e-4, β1=0.5, β2=0.999)
- **Training**: 2000 epochs

## Evaluation Metrics

1. **Log-Likelihood (KDE)**: Average log-likelihood of test data under a KDE fitted to generated samples
2. **MMD**: Maximum Mean Discrepancy between real and generated distributions
3. **Coverage**: Fraction of real samples with nearby generated samples
4. **Density**: Fraction of generated samples near real data manifold

## Generating the Report

The LaTeX report is in `report/report.tex`. To compile:

```bash
cd report
pdflatex report.tex
pdflatex report.tex  # Run twice for references
```

## Device Support

The code automatically detects and uses:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon)
- CPU (fallback)

## Author

Seena  
EE660 - Machine Learning  
University of Southern California
