
import os
import json
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
import sys
sys.path.append('..')
print("Done.")

class DepthDatasetLoader(Dataset):
    def __init__(self, root_path="/home/sangbeom/unity_dataset/depth/", 
                       json_name="data_loader_1m.json",
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
        dir = self.root_path+image_path["file_path"]
        image = read_image(dir)
        if self.transform: 
            image = self.transform(image)
        return image
   