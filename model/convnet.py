import torch
from torch import max_pool2d, nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms, datasets, utils
from torch.autograd import Variable
import matplotlib.pyplot as plt 
import wandb 
import numpy as np

class CAE(nn.Module):
    def __init__(self, input_dim=1):
        super(CAE, self).__init__()
        self.encoder = nn.Sequential(
            # First Layer 
            nn.Conv2d(in_channels=input_dim, 
                      out_channels=32, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2),
            # Second Layer
            nn.Conv2d(in_channels=32, 
                      out_channels=64, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, 
                      out_channels=128, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            )

        self.decoder = nn.Sequential(
            # First Layer
            nn.ConvTranspose2d(in_channels=128, 
                               out_channels=64,
                               kernel_size=4, 
                               stride=2,
                               padding=1), 
            nn.ReLU(), 
            # Second Layer
            nn.ConvTranspose2d(in_channels=64, 
                               out_channels=32,
                               kernel_size=4, 
                               stride=2,
                               padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, 
                               out_channels=input_dim,
                               kernel_size=4, 
                               stride=2,
                               padding=1), 
            nn.ReLU(), 
            )

    def init_weights(self, m): 
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x): 
        encode_out = self.encoder(x)
        decode_out = self.decoder(encode_out)
        return decode_out 