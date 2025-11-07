🌀 Physics-Informed tempoGAN for 3D Turbulence Reconstruction

This repository contains the implementation of a physics-informed tempoGAN framework for reconstructing and predicting freely decaying isotropic turbulence.
Our model enhances spatial resolution and temporal coherence of turbulent flow fields while enforcing physical consistency through spectral and divergence-free constraints.

🌊 Overview

Turbulent flows are inherently chaotic and multi-scale, making accurate reconstruction extremely challenging.
We adopt a GAN-based spatio-temporal super-resolution model (tempoGAN) combined with physics-informed loss functions, achieving high-fidelity and physically plausible results for decaying isotropic turbulence.

Key Contributions

3D convolutional tempoGAN for volumetric turbulence super-resolution

Temporal discriminator to ensure time-consistent predictions

Physics-informed losses: spectral energy and divergence-free constraints

Modular code design for training, testing, and visualization

📘 Data Preparation

The dataset contains 3D DNS velocity fields (U, V, W) for freely decaying isotropic turbulence.
Before training, preprocess the data using the provided data preparation scripts.

🚀 Training

Train the tempoGAN with physics-informed losses:

python train_tempoGAN.py

🔍 Testing & Visualization

Run the testing and visualization scripts:

python test_tempoGAN.py


Outputs include reconstructed 3D velocity fields and optionally their spectral energy comparison with DNS data.

🧠 Model Highlights
Component	Description
Generator (G)	3D U-Net–like architecture designed for volumetric super-resolution of turbulence fields.
Spatial Discriminator (Dₛ)	Evaluates spatial realism of each high-resolution frame.
Temporal Discriminator (Dₜ)	Ensures temporal coherence across consecutive frames by discriminating short frame sequences.
Physics-informed Losses	Incorporate spectral and divergence-free constraints to enforce physical consistency.
Quarter Jumble Strategy	With a small probability, low-resolution inputs are partially shuffled (¼-volume permutation) to prevent the discriminator from becoming overly dominant, maintaining a balanced adversarial game.
<div align="center"> <img src="./assets/Model-Architecture.PNG" width="700"><br> <em>Figure 1. Overall architecture of the physics-informed tempoGAN.</em> </div>
📜 Citation

If you find this work useful, please cite:

@article{YUJIE_WANG,
  title={Physics-Informed Super-Resolution of Turbulent Flows using a Temporal Generative Adversarial Network},
  author={YUJIE WANG et al.},
  year={2025},
}

