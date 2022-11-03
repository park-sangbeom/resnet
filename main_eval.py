import numpy as np
import cv2
import time 
import matplotlib.pyplot as plt 
import torch 
import json 
import torch.nn as nn 
import argparse
import wandb
import sys
from pathlib import Path
BASEDIR = str(Path(__file__).parent)
sys.path.append(BASEDIR)
sys.path.append("..")
from model.utils import torch2np, np2torch, model_freeze
from model.utils_projection import realworld_ortho_proejction, unity_ortho_projection
# from model.convnet_for_extractor import Autoencoder
from model.resnet_for_extractor import Encoder

def main_inference(args):
    print("Start Evaluation: {}".format(args.version))
    # Set random seed 
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  

    feature_extractor = Encoder().to(device=args.device)
    pretrained_dict = torch.load(args.LOADWEIGHT)
    feature_extractor.load_state_dict(pretrained_dict)
    model_freeze(feature_extractor)
    feature_extractor.eval()
    print("Feature Extractor Weight: {} Load.".format(args.LOADWEIGHT))	

    with torch.no_grad():
        image = np.load('./data/realworld/1025_test11_norm.npy')
        image = realworld_ortho_proejction(image)
        image = np2torch(image, device= args.device)
        image = image.reshape(-1, 1, image.shape[1], image.shape[0]) # Image shape [Width x Height] # CNN [-1 Channel Height Width]
        temp = 0 
        for i in range(3):
            c_torch1 = feature_extractor(image) # Worker num x C dim [50x128]
            c_torch1 = torch2np(c_torch1).reshape(-1)
            print("C vector1: ",c_torch1)
            print("Offset:", np.abs(temp-c_torch1))
            temp = c_torch1 

        image2 = np.load('./data/realworld/1025_test10_norm.npy')
        image2 = realworld_ortho_proejction(image2)
        image2 = np2torch(image2, device= args.device)
        image2 = image2.reshape(-1, 1, image2.shape[1], image2.shape[0]) # Image shape [Width x Height] # CNN [-1 Channel Height Width]
        c_torch2 = feature_extractor(image2) # Worker num x C dim [50x128]
        temp2 = 0 
        print("***"*25)
        for i in range(3):
            c_torch2 = feature_extractor(image2) # Worker num x C dim [50x128]
            c_torch2 = torch2np(c_torch2).reshape(-1)
            print("C vector2: ",c_torch2)
            print("Offset:", np.abs(temp2-c_torch2))
            temp2 = c_torch2 
        print("***"*25)
        print("[{}]Cdims:{}".format(args.version, c_torch2.shape))
        print("[{}]Dev:{}".format(args.version, np.sum(np.abs(c_torch1-c_torch2))))
        print("[{}]Avg Dev:{}".format(args.version, np.sum(np.abs(c_torch1-c_torch2))/c_torch2.shape))
        print("[{}]Old Dev:{}".format(args.version,np.abs(np.sum(c_torch1-c_torch2))))
        print("[{}]AVg Old Dev:{}".format(args.version,np.abs(np.sum(c_torch1-c_torch2))/c_torch2.shape))

if __name__=="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device: {}".format(device))
    args   = argparse.ArgumentParser(description= "Parse for ResNet")
    # LOAD WEIGHTS
    # args.add_argument('--LOADWEIGHT', default='./weights/convnet/convnet1031/convnet_encoder700steps.pth', type=str,
    #                   help="ResNet weight path")
    args.add_argument('--version', default='1103', type=str,
                      help="ResNet weight path")
    args.add_argument('--LOADWEIGHT', default='./weights/resnet1103/resnet_encoder300steps.pth', type=str,
                      help="ResNet weight path")
    args.add_argument('--imagepath', default='./demo/data/real_world/1025_test12_norm.npy', type=str,
                      help="Image path")
    # DEVICE
    args.add_argument('--device', default= device,type=str, #cpu cuda:0
                      help='device')
    # Random Seed  
    args.add_argument('--random_seed', default= 42, type=int,
                      help='Random Seed') 
    args = args.parse_args()
    main_inference(args)
