import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
torch.set_grad_enabled(True)


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
        elif mode == 'decode':
            self.conv1 = nn.ConvTranspose2d(c_in, c_out, k, s, p)
            self.conv2 = nn.ConvTranspose2d(c_out, c_out, 3, 1, 1)
        self.relu        = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.BN     = nn.BatchNorm2d(c_out)
        self.resize = s > 1 or (s == 1 and p == 0) or c_out != c_in
        self.mode   = mode 
    
    def forward(self, x):
        conv1 = self.BN(self.conv1(x))
        if self.mode == "encode":
            relu = self.leakyrelu(conv1)
        else: 
            relu = self.relu(conv1)
        conv2 = self.BN(self.conv2(relu))
        if self.resize:
            x = self.BN(self.conv1(x))
        if self.mode == "encode":
            out = self.leakyrelu(x + conv2)
        else: 
            out = self.relu(x + conv2)
        return out

class Encoder(nn.Module):
    """
    Encoder class, mainly consisting of three residual blocks.
    """
    
    def __init__(self):
        super(Encoder, self).__init__()
        self.init_conv = nn.Conv2d(1, 16, 3, 1, 1) # 16 32 32
        self.BN = nn.BatchNorm2d(16)
        self.rb1 = ResBlock(16, 16, 3, 2, 1, 'encode') # 16 16 16
        self.rb2 = ResBlock(16, 32, 3, 1, 1, 'encode') # 32 16 16
        self.rb3 = ResBlock(32, 32, 3, 2, 1, 'encode') # 32 8 8
        self.rb4 = ResBlock(32, 48, 3, 1, 1, 'encode') # 48 8 8
        self.rb5 = ResBlock(48, 48, 3, 2, 1, 'encode') # 48 4 4
        self.rb6 = ResBlock(48, 64, 3, 2, 1, 'encode') # 64 2 2
        self.leakyrelu = nn.LeakyReLU()
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(64*12*6, 1024)
        self.lin2 = nn.Linear(1024, 10)

    def forward(self, inputs):
        init_conv = self.leakyrelu(self.BN(self.init_conv(inputs)))
        out = self.rb1(init_conv)
        out = self.rb2(out)
        out = self.rb3(out)
        out = self.rb4(out)
        out = self.rb5(out)
        out = self.rb6(out)
        out = self.flatten(out)
        out = self.leakyrelu(self.lin1(out))
        out = self.leakyrelu(self.lin2(out))
        return out

class Decoder(nn.Module):
    """
    Decoder class, mainly consisting of two residual blocks.
    """
    
    def __init__(self):
        super(Decoder, self).__init__()
        self.delin1 = nn.Linear(10, 1024)
        self.delin2 = nn.Linear(1024, 64*12*6)
        self.rb1 = ResBlock(64, 48, 2, 2, 0, 'decode') # 48 4 4
        self.rb2 = ResBlock(48, 48, 2, 2, 0, 'decode') # 48 8 8
        self.rb3 = ResBlock(48, 32, 3, 1, 1, 'decode') # 32 8 8
        self.rb4 = ResBlock(32, 32, 2, 2, 0, 'decode') # 32 16 16
        self.rb5 = ResBlock(32, 16, 3, 1, 1, 'decode') # 16 16 16
        self.rb6 = ResBlock(16, 16, 2, 2, 0, 'decode') # 16 32 32
        self.out_conv = nn.ConvTranspose2d(16, 1, 3, 1, 1) # 3 32 32
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, inputs):
        out = self.relu(self.delin1(inputs))
        out = self.relu(self.delin2(out))
        out = out.view(-1, 64, 6, 12)
        out = self.rb1(out)
        out = self.rb2(out)
        out = self.rb3(out)
        out = self.rb4(out)
        out = self.rb5(out)
        out = self.rb6(out)
        out_conv = self.out_conv(out)
        output   = self.tanh(out_conv)
        return output

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