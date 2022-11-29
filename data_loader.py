
import os
import json
import numpy as np 
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt 
import sys
sys.path.append('..')

class UnityDepthDatasetLoader(Dataset):
    def __init__(self, root_path="/home/sangbeom/resnet/data/depth1116_new/", 
                       json_name="depth1116_w_label.json",
                       transform=None):
        self.root_path = root_path 
        self.json_name = json_name 
        self.json_content=[]
        for line in open(self.json_name,'r'):
            self.json_content.append(json.loads(line))
        self.image_lst = self.json_content
        self.transform = transform 

    def __len__(self):
        return len( self.image_lst)
        
    def __getitem__(self, idx):
        image_path = self.image_lst[idx]
        dir = image_path["file_path"]
        # dir = self.root_path+image_path["file_path"]
        #label = torch.from_numpy(np.array(float(image_path["label"])-1.0).astype(np.int64))
        #image = read_image(dir)
        image = np.load(dir)
        if self.transform: 
            image = self.transform(image)
        return image
   