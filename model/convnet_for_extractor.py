
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

class Autoencoder(nn.Module):
    """
    Autoencoder class, combines encoder and decoder model.
    """
    
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = ConvNetEncoder()
        self.decoder = ConvNetDecoder()
    
    @property
    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        num_p = sum([np.prod(p.size()) for p in model_parameters])
        return num_p
    
    def forward(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded ,encoded

class ConvNetEncoder(nn.Module):
    def __init__(self):
        super(ConvNetEncoder, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), #1 * 96 * 192 -> 16 * 96 * 192
            nn.BatchNorm2d(16), #16 * 96 * 192
            nn.ReLU(),                 #16 * 96 * 192
            nn.MaxPool2d(2),
            nn.Dropout(0.25))       # 16 * 48 * 96
        self.layer2  = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1), # 16* 48 * 96 -> 32 * 48 * 96
            nn.BatchNorm2d(16), #  32 * 48 * 96
            nn.ReLU(), #  32 * 48 * 96
            # nn.MaxPool2d(2),
            nn.Dropout(0.25))       # 32 * 24 * 48
        self.layer3  = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # 32 * 24 * 48 -> 64 * 24 * 48 
            nn.BatchNorm2d(32), #  64 * 24 * 48 
            nn.ReLU(), #  64 * 24 * 48 
            nn.MaxPool2d(2),
            nn.Dropout(0.25))       # 64 * 12 * 24
        self.layer4  = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), # 64 * 12 * 24 -> 128 * 12 * 24
            nn.BatchNorm2d(32), #  128 * 12 * 24 
            nn.ReLU(), #  128 * 12 * 24 
            # nn.MaxPool2d(2),
            nn.Dropout(0.25))
        self.layer5  = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64 * 12 * 24 -> 128 * 12 * 24
            nn.BatchNorm2d(64), #  128 * 12 * 24 
            nn.ReLU(), #  128 * 12 * 24 
            nn.MaxPool2d(2),
            nn.Dropout(0.25))      
        self.layer6  = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # 64 * 12 * 24 -> 128 * 12 * 24
            nn.BatchNorm2d(64), #  128 * 12 * 24 
            nn.ReLU(), #  128 * 12 * 24 
            # nn.MaxPool2d(2),
            nn.Dropout(0.25))  
        self.layer7  = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 64 * 12 * 24 -> 128 * 12 * 24
            nn.BatchNorm2d(128), #  128 * 12 * 24 
            nn.ReLU(), #  128 * 12 * 24 
            nn.MaxPool2d(2),
            nn.Dropout(0.25)) 
        self.layer8  = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), # 64 * 12 * 24 -> 128 * 12 * 24
            nn.BatchNorm2d(128), #  128 * 12 * 24 
            nn.ReLU(), #  128 * 12 * 24 
            # nn.MaxPool2d(2),
            nn.Dropout(0.25)) 
        self.linlayer = nn.Sequential(
            nn.Linear(128*6*12, 1024),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(1024, 16),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.flatten(out)
        out = self.linlayer(out)
        return out

class ConvNetDecoder(nn.Module):
    def __init__(self):
        super(ConvNetDecoder, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0), 
            nn.BatchNorm2d(64),
            nn.ReLU(),            
            nn.Dropout(0.25))      
        self.layer2  = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.Dropout(0.25))      
        self.layer3  = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(16), 
            nn.ReLU(),
            nn.Dropout(0.25))       
        self.layer4  = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(1), 
            nn.ReLU(),
            nn.Dropout(0.25))  

        self.linlayer = nn.Sequential(
            nn.Linear(16, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 128*6*12),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        
    def forward(self, x):
        out = self.linlayer(x)
        out = out.view(-1, 128, 6, 12)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out