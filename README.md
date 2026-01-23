# ML_Crystal

## Objectives

- **Simulation**: Generate 2D nucleation trajectories of Lennard-Jones particles under strict 2D confinement.
- **Representation**: Generate SOAP (Smooth Overlap of Atomic Positions) descriptors to represent local atomic environments.
- **Analysis**: Employ Variational Autoencoders (VAE) and Beta-VAEs to learn low-dimensional latent representations of the nucleation process.

## Features

- **OpenMM Simulation**: Setup for 2D confined Lennard-Jones particle systems.
- **SOAP Generation**: `NoteBooks/SOAP_Generation.ipynb` for generating structural descriptors from trajectories.
- **Machine Learning Models**:
    - **VAE**: Custom PyTorch implementation (`Scripts/vae.py`) extending `deeptime` components.
    - **Beta-VAE**: Disentangled representation learning (`Scripts/BETAVAE.py`).
    - **Autoencoder**: Standard AE implementation (`Scripts/ae.py`).
- **Analysis Notebooks**:
    - `NoteBooks/VAE_Deeptime.ipynb`: Reference workflow using `deeptime`.
    - *(Note: `VAE_Analysis.ipynb` and `BetaVAE_Analysis.ipynb` are available for custom analysis using local scripts).*

## Repository Structure

- `Scripts/`: Core Python modules for models and utilities.
    - `vae.py`, `BETAVAE.py`, `ae.py`: Deep learning model definitions.
    - `base_torch.py`, `util.py`: Helper classes and functions.
- `NoteBooks/`: Jupyter notebooks for data generation and analysis.
- `Data/`: Directory for simulation data (trajectories, descriptors).
- `assets/`: Resource files.

## Installation

```bash
# Clone the repository
git clone https://github.com/dmighty007/ML_Crystal.git
cd ML_Crystal

# Install dependencies (Recommended to use a conda environment)
# Core: openmm, numpy, matplotlib
# ML/Analysis: torch, dscribe, deeptime, scikit-learn
pip install torch dscribe deeptime scikit-learn
```

## Usage

1.  **Simulation & Data Generation**: Use OpenMM scripts (to be developed/documented) to generate trajectories.
2.  **Feature Generation**: Run `NoteBooks/SOAP_Generation.ipynb` to create SOAP descriptors from trajectories.
3.  **Model Training**: Use the classes in `Scripts/` (e.g., `VAE`, `BetaVAE`) within notebooks or custom scripts to train on the generated data.
