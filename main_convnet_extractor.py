from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import transforms

from utils import *
from data_loader import DepthDatasetLoader
from model.convnet_for_extractor import Autoencoder

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)

print(torch.__version__)
print(torchvision.__version__)

param_names = ('init_lr', 'batch_size', 'weight_decay')
parameters = OrderedDict(
    run = [1e-3, 256, 0.001],
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device: {}".format(device))

m = RunManager()
num_epochs = 1000

root_path = "/home/sangbeom/resnet/data/depth1014/"

depth_dataset = DepthDatasetLoader(root_path=root_path)
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

    print('Now training model with hyperparameters: init_lr={0}, batch_size={1}, weight_decay={2}'
         .format(hparams.init_lr, hparams.batch_size, hparams.weight_decay))
    
    # Start training loop
    for epoch in range(num_epochs):
        # m.begin_epoch()
        print("Epoch: {}".format(epoch+1))
        # Train the model
        for i, batch in enumerate(train_loader):
            train_images = batch.reshape(-1, 1, 96, 192)
            train_images =  Variable(train_images.float().to(device))
            # Zero all gradients
            optimizer.zero_grad()
            
            # Calculating the loss
            preds,_ = ae(train_images)
            loss = F.mse_loss(preds, train_images)
            # Backpropagate
            loss.backward()

            # Update the weights
            optimizer.step()
            
            if i % 100 == 0:
                with torch.no_grad():
                    val_images  = next(iter(val_loader))
                    val_images = val_images.reshape(-1, 1, 96, 192)
                    val_images = Variable(val_images.float().to(device))
                    val_preds,_ = ae(val_images)
                    val_loss = F.mse_loss(val_preds, val_images)
                print('Epoch {0}, iteration {1}: train loss {2}, val loss {3}'.format(epoch+1,
                                                                               i*hparams.batch_size,
                                                                               round(loss.item(), 6),
                                                                               round(val_loss.item(), 6)))
                
                val_images = val_images.detach().cpu().numpy()
                val_preds = val_preds.detach().cpu().numpy()
        if (epoch+1)%100==0 or (epoch+1)==num_epochs:
            with torch.no_grad():
                val_images  = next(iter(val_loader))
                val_images = val_images.reshape(-1, 1, 96, 192)
                val_images = Variable(val_images.float().to(device))
                val_preds, _ = ae(val_images)
                val_loss = F.mse_loss(val_preds, val_images)
            # print('Epoch {0}, iteration {1}: train loss {2}, val loss {3}'.format(epoch+1,
            #                                                                 i*hparams.batch_size,
            #                                                                 round(loss.item(), 6),
            #                                                                 round(val_loss.item(), 6)))
            fig, axs = plt.subplots(2,5, figsize=(15,4))
            val_images = val_images.detach().cpu().numpy()
            val_preds = val_preds.detach().cpu().numpy()
            for i in range(5):
                axs[0][i].matshow(np.reshape(val_images[i, :], (96,192)), cmap=plt.get_cmap('gray'))
                axs[1][i].matshow(np.reshape(val_preds[i, :], (96,192)), cmap=plt.get_cmap('gray'))
            plt.savefig("data/convnet/convnet1026_eval{}.png".format(epoch+1))
            torch.save(ae.encoder.state_dict(), 'weights/convnet/convnet_encoder{}steps.pth'.format(epoch+1))
            torch.save(ae.decoder.state_dict(), 'weights/convnet/convnet_decoder{}steps.pth'.format(epoch+1))

    # m.end_run()
    print("Model has finished training.\n")
    scheduler.step()
print("Training completed.")