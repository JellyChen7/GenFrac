# GenFrac
Physics-Supervised Autonomous Inverse Fracture Modelling via Generative Artificial Intelligence (conditional diffusion models with 2D UNet as basic framework)

[Guodong Chen](https://scholar.google.com/citations?user=U2YFkAgAAAAJ&hl=zh-TW&oi=ao), [Jiu Jimmy Jiao*](https://scholar.google.com/citations?user=t7zybZUAAAAJ&hl=zh-TW&oi=ao), Zhongzheng Wang, Rong Mao, Tao Yang, & [Qinjun Kang](https://scholar.google.com/citations?user=M8NwAPUAAAAJ&hl=zh-TW&oi=ao)

In this work, we introduce GenFrac, a pre-trained generative artificial intelligence method for the autonomous inversion of fracture networks in complex subsurface geological formations. In this method, pre-trained denoising diffusion model formulates the inversion as a conditional denoising process from sparse and noisy observational data while incorporating prior geological information, facilitating improved accuracy in parameter estimations. To ensure the accuracy of the inverse generation of fracture networks, physics-supervised procedure is further conducted to pre-screen the promising parameter fields. The work enables generative parameter inversion through the direct generation of static parameters conditioned on the observational data of state parameters, offering broad applicability to related fields including fluid flow, serving both as a surrogate model and nonlinear optimizer.

## Broader application scenarios of subsurface fracture systems:
![Workflow of GenFrac](https://github.com/JellyChen7/GenFrac/raw/master/Assets/Fracture_scenarios.jpg "Workflow of GenFrac")

## Network architecture:
![Architecture of GenFrac](https://github.com/JellyChen7/GenFrac/raw/master/Assets/Diffusion_model.png "Architecture of GenFrac")

## Fracture network synthesis and generative inversion results overview for three separated fractures:
![Case1 of GenFrac](https://github.com/JellyChen7/GenFrac/raw/master/Assets/Case1.jpg "Case1 of GenFrac")

## Denoising process:
![Denoising](https://github.com/JellyChen7/GenFrac/raw/master/Assets/Denoising_process.gif "Denoising")

## Performance analysis and comparison of the simulation prediction of generated fracture networks by GenFrac and ground truth:
![Case1_2 of GenFrac](https://github.com/JellyChen7/GenFrac/raw/master/Assets/Case1_2.jpg "Case1_2 of GenFrac")

## GenFrac for fracture network with two parallel fractures intersecting one fracture:
![Case2 of GenFrac](https://github.com/JellyChen7/GenFrac/raw/master/Assets/Case2.jpg "Case2 of GenFrac")

## GenFrac for fracture network with two intersecting perpendicular fractures and a separated fracture:
![Case3 of GenFrac](https://github.com/JellyChen7/GenFrac/raw/master/Assets/Case3.jpg "Case3 of GenFrac")

## Fracture network reconstruction for groundwater flow system of Poshan area in Hong Kong:
![Poshan of GenFrac](https://github.com/JellyChen7/GenFrac/raw/master/Assets/Poshan_case.jpg "Poshan of GenFrac")

## Datasets
data_Binary.mat;
data_observation.mat

## Autonomous Inverse Fracture Modelling
python main.py

## Generative Inversion
python denoising.py

