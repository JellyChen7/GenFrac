import time

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import torch, os, shutil, h5py
from modules import UNet_conditional
from cdm import Diffusion


device = "cpu"
model = UNet_conditional(num_classes=1050, device=device)
diffusion = Diffusion(img_size=64, device=device)

dir = 'DDPM_conditional_20240911_1618_batchsize16_epochs600'



model1 = torch.load("ckpt.pt", map_location=torch.device('cpu'))
model2 = torch.load("ema_ckpt.pt", map_location=torch.device('cpu'))
model3 = torch.load("optim.pt", map_location=torch.device('cpu'))

predicted_noise = model(x_t, t, labels)
