# Imports
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)
from utils import *
import torchvision
from torchvision.transforms import transforms
import wandb 
import argparse
from data_loader import DepthDatasetLoader
from model.resnet_for_extractor import Autoencoder 
from model.utils import torch2np, np2torch, get_runname

def main(args):
    unityimg1 = np.load('./data/depth1116_new/depth1_1.npy')
    unityimg2 = np.load('./data/depth1116_new/depth1_2.npy')
    realimg1 = np.load('./data/depth1116_new/depth10_2.npy')
    realimg2 = np.load('./data/depth1116_new/depth10_100.npy')

    runname = get_runname() if args.runname=='None' else args.runname
    if args.WANDB:
        wandb.init(project = args.pname)
        wandb.run.name = runname  

    root_path = "/home/sangbeom/resnet/data/depth1116_new/"

    depth_dataset = DepthDatasetLoader(root_path=root_path)
    train_set, val_set = train_val_split(depth_dataset, 0.1)

    # Instantiate a network model
    ae = Autoencoder().to(args.device)
    # Construct a DataLoader object with training data
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(val_set, batch_size=10, shuffle=False)

    # Define optimizer
    optimizer = optim.SGD(ae.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.01) # 30, 0.01
    # Setup run instance
    print('Now training model with hyperparameters: init_lr={0}, batch_size={1}, weight_decay={2}'
        .format(args.lr, args.batch_size, args.weight_decay))
    
    # Start training loop
    for epoch in range(args.max_epochs):
        total_loss = 0
        # ae.train()

        # Train the model
        for i, batch in enumerate(train_loader):
            train_images = batch.reshape(-1, 1, 96, 192)
            train_images =  Variable(train_images.float().to(device))
            # Zero all gradients
            optimizer.zero_grad()
            
            # Calculating the loss
            preds = ae(train_images)
            loss  = F.mse_loss(preds, train_images)
            # Backpropagate
            loss.backward()
            total_loss += round(loss.item(), 6)
            # Update the weights
            optimizer.step()
            
            # if i % 100 == 0:
            #     with torch.no_grad():
            #         val_images  = next(iter(val_loader))
            #         val_images = val_images.reshape(-1, 1, 96, 192)
            #         val_images = Variable(val_images.float().to(device))
            #         val_preds = ae(val_images)
            #         val_loss = F.mse_loss(val_preds, val_images)
            #     print('Epoch {0}, iteration {1}: train loss {2}, val loss {3}'.format(epoch+1,
            #                                                                 i*args.batch_size,
            #                                                                 round(loss.item(), 6),
            #                                                                 round(val_loss.item(), 6)))
        if args.WANDB:
            wandb.log({"train loss":total_loss/i}, step=epoch+1)  
        print("Epoch: {}, Train loss: {}".format(epoch+1,total_loss/i))
        scheduler.step()    
        # Evaluation 
        if (epoch+1)%10==0 or (epoch+1)==1 or (epoch+1)==5 or (epoch+1)==args.max_epochs:
            # ae.eval()
            with torch.no_grad():
                val_images  = next(iter(val_loader))
                val_images = val_images.reshape(-1, 1, 96, 192)
                val_images = np2torch(val_images, device=args.device)
                val_preds  = ae(val_images)

                c_vector1= torch2np(ae.encoder(np2torch(realimg1.reshape(-1, 1, 96, 192), device=args.device)))
                c_vector2= torch2np(ae.encoder(np2torch(realimg2.reshape(-1, 1, 96, 192), device=args.device)))
                c_vector3= torch2np(ae.encoder(np2torch(unityimg1.reshape(-1, 1, 96, 192), device=args.device)))
                c_vector4= torch2np(ae.encoder(np2torch(unityimg2.reshape(-1, 1, 96, 192), device=args.device)))
            real_dev = np.sum(np.abs(c_vector1-c_vector2))
            unity_dev = np.sum(np.abs(c_vector3-c_vector4))
            if args.WANDB: 
                wandb.log({"unity deviation2":real_dev,
                           "unity deviation":unity_dev}, step=epoch+1)  

            fig, axs = plt.subplots(2,5, figsize=(15,4))
            val_images = torch2np(val_images)
            val_preds  = torch2np(val_preds)
            for i in range(5):
                axs[0][i].matshow(np.reshape(val_images[i, :], (96,192)))
                axs[1][i].matshow(np.reshape(val_preds[i, :], (96,192)))
            plt.suptitle("Resnet[Encoder: Resblock 3th, ReLU][Decoder: Resblock 3th, Shallow]", fontsize=15)
            plt.savefig("data/resnet/{}/{}_eval{}.png".format(args.runname,args.runname,epoch+1))
            torch.save(ae.encoder.state_dict(), 'weights/{}/resnet_encoder{}steps.pth'.format(args.runname,epoch+1))
            torch.save(ae.decoder.state_dict(), 'weights/{}/resnet_decoder{}steps.pth'.format(args.runname,epoch+1))


    print("Training completed.")

if __name__=="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device: {}".format(device))
    args   = argparse.ArgumentParser(description= "Parse for ResNet")
    # LOG 
    args.add_argument("--WANDB", default = True, action="store_true",
                      help="Use WANDB")
    args.add_argument('--pname', default= 'feature_extractor',type=str, 
                      help='wandb project name')
    args.add_argument('--runname', default='resnet1116_new_bigger_10', type=str,
                      help="wandb runname")
    # DEVICE 
    args.add_argument("--device", default=device, type=str,
                      help="Device")
    # PARAMETERS 
    args.add_argument('--max_epochs', default= 300, type=int,
                      help='Total epochs for training') 
    args.add_argument('--lr', default= 0.1, type=float, #0.1
                      help='learning rate')
    args.add_argument('--batch_size', default= 256, type=float,
                      help='batch size')
    args.add_argument('--weight_decay', default= 0.0001, type=float, #0.0001
                      help='weight_decay')
    args = args.parse_args()
    main(args)
