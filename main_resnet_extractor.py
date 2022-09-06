# Imports
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)

import torchvision
from torchvision.transforms import transforms

from utils import *
from data_loader import DepthDatasetLoader
from model.resnet_for_extractor import Autoencoder 

print(torch.__version__)
print(torchvision.__version__)

def torch2np(x_torch):
    if x_torch is None:
        x_np = None
    else:
        x_np = x_torch.detach().cpu().numpy()
    return x_np

def np2torch(x_np,device='cpu'):
    if x_np is None:
        x_torch = None
    else:
        x_torch = torch.tensor(x_np,dtype=torch.float32,device=device)
    return x_torch


param_names = ('init_lr', 'batch_size', 'weight_decay')
parameters = OrderedDict(
    run = [0.05, 64, 0.001],
)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device: {}".format(device))

m = RunManager()
num_epochs = 300

root_path = root_path = "/home/sangbeom/unity_dataset/depth/"
tf=transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])
depth_dataset = DepthDatasetLoader(root_path=root_path, transform=tf)
train_set, val_set = train_val_split(depth_dataset, 0.1)

for hparams in RunBuilder.get_runs_from_params(param_names, parameters):

    # Instantiate a network model
    ae = Autoencoder().to(device)
    # Construct a DataLoader object with training data
    train_loader = DataLoader(train_set, batch_size=hparams.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=hparams.batch_size, shuffle=True)
    test_loader = DataLoader(val_set, batch_size=10, shuffle=False)
    test_images = next(iter(test_loader))
    test_images = next(iter(val_loader))

    # Define optimizer
    optimizer = optim.SGD(ae.parameters(), lr=hparams.init_lr, momentum=0.9, weight_decay=hparams.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 60, 0.1)
    
    # Setup run instance
    # m.begin_run(hparams, ae, torch.Tensor(test_images).to(device))
    print('Now training model with hyperparameters: init_lr={0}, batch_size={1}, weight_decay={2}'
         .format(hparams.init_lr, hparams.batch_size, hparams.weight_decay))
    
    # Start training loop
    for epoch in range(num_epochs):
        # m.begin_epoch()
        
        # Train the model
        for i, batch in enumerate(train_loader):
            images = batch[:,:,8:328,6:-6]
            images = torch.Tensor(images).to(device)
            # Zero all gradients
            optimizer.zero_grad()
            
            # Calculating the loss
            preds,_ = ae(images)
            loss = F.mse_loss(preds, images)
            # Backpropagate
            loss.backward()

            # Update the weights
            optimizer.step()
            
            if i % 10 == 0:
                with torch.no_grad():
                    val_images  = next(iter(val_loader))
                    val_images = torch.Tensor(val_images[:,:,8:328,6:-6]).to(device)
                    val_preds, _ = ae(val_images)
                    val_loss = F.mse_loss(val_preds, val_images)
                print('Epoch {0}, iteration {1}: train loss {2}, val loss {3}'.format(epoch+1,
                                                                               i*hparams.batch_size,
                                                                               round(loss.item(), 6),
                                                                               round(val_loss.item(), 6)))
                
                val_images = val_images.detach().cpu().numpy()
                val_preds = val_preds.detach().cpu().numpy()
        if (epoch+1)%1==0 or (epoch+1)==num_epochs:
            with torch.no_grad():
                val_images  = next(iter(val_loader))
                val_images = torch.Tensor(val_images[:,:,8:328,6:-6]).to(device)
                val_preds, encoded = ae(val_images)
                val_loss = F.mse_loss(val_preds, val_images)
            # print('Epoch {0}, iteration {1}: train loss {2}, val loss {3}'.format(epoch+1,
            #                                                                 i*hparams.batch_size,
            #                                                                 round(loss.item(), 6),
            #                                                                 round(val_loss.item(), 6)))
            fig, axs = plt.subplots(2,5, figsize=(15,4))
            val_images = val_images.detach().cpu().numpy()
            val_preds = val_preds.detach().cpu().numpy()
            for i in range(5):
                axs[0][i].matshow(np.reshape(val_images[i, :], (320,512)), cmap=plt.get_cmap('gray'))
                axs[1][i].matshow(np.reshape(val_preds[i, :], (320,512)), cmap=plt.get_cmap('gray'))
            plt.savefig("data/resnet_ex_eval{}.png".format(epoch+1))

            plt.figure(figsize=(6,6))
            plt.scatter(torch2np(encoded[:,0]),torch2np(encoded[:,1]), alpha=0.5)
            plt.savefig("data/resnet_ex_encoded{}.png".format(epoch+1))

            plt.figure(figsize=(6,6))
            plt.xlim(-10, 10)
            plt.ylim(-10, 10)
            plt.scatter(torch2np(encoded[:,0]),torch2np(encoded[:,1]), alpha=0.5)
            plt.savefig("data/resnet_ex_fix_encoded{}.png".format(epoch+1))
            torch.save(ae.state_dict(), 'weights/resnet_ex_{}steps.pth'.format(epoch+1))
    # m.end_run()
    print("Model has finished training.\n")
    scheduler.step()
print("Training completed.")