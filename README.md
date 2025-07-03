# GenFrac
Physics-Supervised Autonomous Inverse Fracture Modelling via Generative Artificial Intelligence (conditional diffusion models with 2D UNet as basic framework)

[Guodong Chen](https://scholar.google.com/citations?user=U2YFkAgAAAAJ&hl=zh-TW&oi=ao), [Jiu Jimmy Jiao*](https://scholar.google.com/citations?user=t7zybZUAAAAJ&hl=zh-TW&oi=ao) et.al

In this work, we introduce GenFrac, a pre-trained generative artificial intelligence method for the autonomous inversion of fracture networks in complex subsurface geological formations. In this method, pre-trained denoising diffusion model formulates the inversion as a conditional denoising process from sparse and noisy observational data while incorporating prior geological information, facilitating improved accuracy in parameter estimations. To ensure the accuracy of the inverse generation of fracture networks, physics-supervised procedure is further conducted to pre-screen the promising parameter fields. The work enables generative parameter inversion through the direct generation of static parameters conditioned on the observational data of state parameters, offering broad applicability to related fields including fluid flow, serving both as a surrogate model and nonlinear optimizer.

## Broader application scenarios of subsurface fracture systems
![Workflow of GenFrac](https://github.com/JellyChen7/GenFrac/raw/master/Assets/Fracture_scenarios.jpg "Workflow of GenFrac")

This file provides the instruction of the software operation to reproduce the figures&results of our work “Physics-Supervised Autonomous Inverse Fracture Modelling via Generative Artificial Intelligence”.

## Network architecture
![Architecture of GenFrac](https://github.com/JellyChen7/GenFrac/raw/master/Assets/Diffusion_model.png "Architecture of GenFrac")
