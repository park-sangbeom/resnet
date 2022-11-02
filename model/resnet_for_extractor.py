from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)
import torchvision
from torchvision.transforms import transforms


class ResBlock(nn.Module):
    """
    A two-convolutional layer residual block.
    """
    
    def __init__(self, c_in, c_out, k, s=1, p=1, mode='encode'):
        assert mode in ['encode', 'decode'], "Mode must be either 'encode' or 'decode'."
        super(ResBlock, self).__init__()
        if mode == 'encode':
            self.conv1 = nn.Conv2d(c_in, c_out, k, s, p)
            self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1)
            self.relu  = nn.LeakyReLU()
        elif mode == 'decode':
            self.conv1 = nn.ConvTranspose2d(c_in, c_out, k, s, p)
            self.conv2 = nn.ConvTranspose2d(c_out, c_out, 3, 1, 1)
            self.relu = nn.ReLU()
        self.BN = nn.BatchNorm2d(c_out)
        self.resize = s > 1 or (s == 1 and p == 0) or c_out != c_in
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        conv1 = self.BN(self.conv1(x))
        relu = self.dropout(self.relu(conv1))
        conv2 = self.BN(self.conv2(relu))
        if self.resize:
            x = self.BN(self.conv1(x))
        return self.dropout(self.relu(x + conv2))

class Encoder(nn.Module):
    """
    Encoder class, mainly consisting of three residual blocks.
    """
    
    def __init__(self):
        super(Encoder, self).__init__()
        self.init_conv = nn.Conv2d(1, 16, 3, 1, 1) 
        self.BN = nn.BatchNorm2d(16)
        self.rb1 = ResBlock(16, 32, 3, 2, 1, 'encode')
        self.rb2 = ResBlock(32, 32, 3, 2, 1, 'encode') 
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(32*24*48, 1024)
        self.lin2 = nn.Linear(1024, 16)
        self.relu = nn.LeakyReLU()

    def forward(self, inputs):
        init_conv = self.relu(self.BN(self.init_conv(inputs)))
        out = self.rb1(init_conv)
        out = self.rb2(out)
        out = self.flatten(out)
        out = self.relu(self.lin1(out))
        out = self.relu(self.lin2(out))
        return out

class Decoder(nn.Module):
    """
    Decoder class, mainly consisting of two residual blocks.
    """
    
    def __init__(self):
        super(Decoder, self).__init__()
        self.rb1 = ResBlock(32, 32, 3, 2, 0, 'decode') # 16 16 16
        self.rb2 = ResBlock(32, 16, 3, 2, 1, 'decode') # 16 32 32
        self.de_lin1 = nn.Linear(1024, 32*24*48)
        self.de_lin2 = nn.Linear(16, 1024)
        self.out_conv = nn.ConvTranspose2d(16, 1, 2, 1, 1) # 3 32 32
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, inputs):
        out = self.relu(self.de_lin2(inputs))
        out = self.relu(self.de_lin1(out))
        out = out.view(-1, 32, 24, 48)
        out = self.rb1(out)
        out = self.rb2(out)
        out = self.out_conv(out)
        out = self.tanh(out)
        return out

class Autoencoder(nn.Module):
    """
    Autoencoder class, combines encoder and decoder model.
    """
    
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    @property
    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        num_p = sum([np.prod(p.size()) for p in model_parameters])
        return num_p
    
    def forward(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded