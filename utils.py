import os
import torch
from torch.utils.data import DataLoader
#from load_data import load_data
import numpy as np
import h5py

def l2_loss(pred, true):
    loss = torch.sum((pred-true)**2, dim=[1, 2, 3])
    return torch.mean(loss)


def get_train_data(args):

    frac_data = h5py.File('data_Binary.mat','r')
    fracture = frac_data['Binary']
    fracture = np.array(fracture).transpose((2,1,0))
    fracture = fracture[0:args.train_num,np.newaxis,:,:]
    frac_data.close()

    obs_data = h5py.File('data_observation.mat','r')
    observation = obs_data['observation']
    observation = np.array(observation).transpose((1,0))
    observation = observation[0:args.train_num,:]
    obs_data.close()
    
    fracture = torch.as_tensor(fracture, dtype=torch.float32)
    observation = torch.as_tensor(observation, dtype=torch.float32)
    
    dataset = torch.utils.data.TensorDataset(fracture, observation)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
