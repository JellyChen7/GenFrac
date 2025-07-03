import time

import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
import numpy as np
import torch, os, shutil, h5py
from modules import UNet_conditional
from cdm import Diffusion


device = "cuda"
model = UNet_conditional(num_classes=550, device=device)
diffusion = Diffusion(img_size=128, device=device)

dir = 'DDPM_conditional_20241122_2238_batchsize48_epochs600'

ckpt = torch.load("./models/{}/ckpt.pt".format(dir), map_location=torch.device('cuda'))
# if os.path.exists('./perm/'):
#     shutil.rmtree('./perm/')
#     os.makedirs('./perm/')
# else:
#     os.makedirs('./perm/')
model.load_state_dict(ckpt)

n_posterior = 50

pred_data = h5py.File('data_Binary.mat','r')
fracture = pred_data['Binary']
fracture = np.array(fracture).transpose((2,1,0))
fracture = fracture[0:10000,:,:]
pred_data.close()

obs_data = h5py.File('data_observation.mat','r')
observation = obs_data['observation']
observation = np.array(observation).transpose((1,0))
observation = observation[0:10000,:]
obs_data.close()

fracture = torch.as_tensor(fracture, dtype=torch.float32)
observation = torch.as_tensor(observation, dtype=torch.float32)


true_obs_data = observation[0,:]
# true_obs_data = np.loadtxt('dobs_norm.txt')
# true_obs_data = true_obs_data.reshape(1, -1)
obs = true_obs_data.unsqueeze(0)
obs = np.repeat(obs, n_posterior, axis=0)

y = torch.FloatTensor(obs).to(device)
# y = torch.FloatTensor(true_obs_data).to(device)
start = time.time()
x1, x_array1 = diffusion.sample(model, n_posterior, y, cfg_scale=3)
print(time.time()-start)
x2, x_array2 = diffusion.sample(model, n_posterior, y, cfg_scale=3)
x = torch.cat((x1, x2), 0)
x_array = torch.cat((x_array1, x_array2), 1)

for i in range(len(x)):
    sampled_perm = x[i, 0].clamp(-1, 1).cpu().numpy()
    plt.imshow(sampled_perm, vmin=-1, vmax=1, cmap='coolwarm')
    plt.savefig('./result/sampled_perm{}.png'.format(i + 1))
    np.savetxt('./result/sampled_perm{}.txt'.format(i + 1), sampled_perm.flatten())

x_array = x_array.clamp(-1, 1).cpu().numpy()
hf = h5py.File('denoising{}.h5'.format(len(x)), 'w')
hf.create_dataset('x_array', data=x_array, dtype='f', compression='gzip')
hf.close()
