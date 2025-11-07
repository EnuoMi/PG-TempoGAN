# Physics-Informed tempoGAN for 3D Turbulence Reconstruction

This repository contains the implementation of a physics-informed tempoGAN framework for reconstructing and predicting freely decaying isotropic turbulence.
Our model enhances spatial resolution and temporal coherence of turbulent flow fields while enforcing physical consistency through spectral and divergence-free constraints.


# Overview

Turbulent flows are inherently chaotic and multi-scale, making accurate reconstruction extremely challenging.
We adopt a GAN-based spatio-temporal super-resolution model (tempoGAN) combined with physics-informed loss functions, achieving high-fidelity and physically plausible results for decaying isotropic turbulence.

Key Contributions

3D convolutional tempoGAN for volumetric turbulence super-resolution

Temporal discriminator to ensure time-consistent predictions

Physics-informed losses: spectral energy and divergence-free constraints

Modular code design for training, testing, and visualization

## Directories
Main source code directories:

`.../main/train:` Training script

`.../main/test:`  Inference and evaluation

`.../main/tempoGAN:`  Generator and discriminators definition

`.../main/losses:`  GAN, feature matching, and physics-based losses

`.../data/dns_preprocess:`  DNS data preprocessing pipeline

`.../main/utils:`  Helper functions (normalization, visualization, etc.)

# Data Preparation

The dataset contains 3D DNS velocity fields (U, V, W) for freely decaying isotropic turbulence.
Before training, preprocess the data using the provided data preparation scripts.


# Training

Train the tempoGAN with physics-informed losses:

python train_tempoGAN.py


# Testing & Visualization

Run the testing and visualization scripts:

python test_tempoGAN.py


Outputs include reconstructed 3D velocity fields and optionally their spectral energy comparison with DNS data.

# Model Highlights
Component	Description
Generator (G)	3D U-Net–like architecture designed for volumetric super-resolution of turbulence fields.
Spatial Discriminator (Dₛ)	Evaluates spatial realism of each high-resolution frame.
Temporal Discriminator (Dₜ)	Ensures temporal coherence across consecutive frames by discriminating short frame sequences.
Physics-informed Losses	Incorporate spectral and divergence-free constraints to enforce physical consistency.
Quarter Jumble Strategy	With a small probability, low-resolution inputs are partially shuffled (¼-volume permutation) to prevent the discriminator from becoming overly dominant, maintaining a balanced adversarial game.
<div align="center"> <img src="./assets/Model-Architecture.PNG" width="700"><br> <em>Figure 1. Overall architecture of the physics-informed tempoGAN.</em> </div>

# RESULT

Key Findings

Visual reconstruction quality is similar: Both the original tempoGAN and the PINN-enhanced model produce reconstructions that look nearly identical to the naked eye.
<div align="center"> <img src="./assets/slice1.png" width="700"><br> </div>

High-frequency energy improved with physics loss: While tempoGAN reproduces the general flow structures well, it exhibits noticeable discrepancies in the high-wavenumber range of the energy spectrum. Incorporating the physics-informed loss restores these high-frequency components, bringing the spectrum closer to the true DNS.
<div align="center"> <img src="./assets/2.png" width="700"><br>  </div>
Enhanced physical consistency: PINN-enforced constraints prevent unphysical patterns in the reconstructed fields, ensuring that the predictions better adhere to the underlying physics.

Frame-wise error reduction: Relative L2 error per frame is slightly improved with the PINN-enhanced model, particularly in regions where fine-scale structures dominate.
<div align="center"> <img src="./assets/3.png" width="700"><br> </div>

# Citation

If you find this work useful, please consider citing the corresponding paper:

```
@article{Wang2025PINN-tempoGAN,
    title={Physics-Informed Super-Resolution of Turbulent Flows using a Temporal Generative Adversarial Network},
    author={YUJIE WANG,ZHISONG WANG},
    year={2025}
}




