import numpy as np
import matplotlib.pyplot as plt  
import wandb 
import json
import argparse 
from pathlib import Path
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from data_loader import UnityDepthDatasetLoader
from model.ae_resnet import Autoencoder 
from model.utils import torch2np, np2torch

def main(args):
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #Set logger
    if args.WANDB:
        wandb.init(project = args.pname)
        wandb.run.name = args.runname
    
    SAVE_WEIGHT_PATH = "../data/weights/"+args.runname
    SAVE_IMAGE_PATH  = "../data/eval/"+args.runname
    SAVE_args_PATH   = "../data/args/"+args.runname

    Path.mkdir(SAVE_WEIGHT_PATH,exist_ok=True)
    Path.mkdir(SAVE_IMAGE_PATH,exist_ok=True)
    Path.mkdir(SAVE_args_PATH,exist_ok=True)

    # Save args
    args_file_name = str(SAVE_args_PATH/"args.json")
    args_file = json.dumps(vars(args))
    with open(args_file_name,"w") as args_json: args_json.write(args_file)
    
    # Load Dataset
    dataset = UnityDepthDatasetLoader(root_path=args.data_path,
                                    json_name=args.json_path)
    
    dataset_size = len(dataset)
    train_size   = int(dataset_size*0.8)
    val_size     = int(dataset_size*0.1)
    test_size    = dataset_size--train_size-val_size
    # Split dataset 
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    # Instantiate a network model
    model = Autoencoder().to(args.device)
    # Construct a DataLoader object with training data
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_set, batch_size=10, shuffle=False)

    # Define Optimizer 
    optimizer = optim.SGD(ae.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.01) # 30, 0.01
    
    # Evaluation Image 
    unityimg1 = np.load('./data/depth1129/depth1_1.npy')
    unityimg2 = np.load('./data/depth1129/depth1_2.npy')
    unityimg3 = np.load('./data/depth1129/depth10_2.npy')
    unityimg4 = np.load('./data/depth1129/depth10_100.npy')

    print('Now training model with hyperparameters: init_lr={}, batch_size={}, weight_decay={}'.format(args.lr, args.batch_size, args.weight_decay))  

    for epoch in range(args.max_epochs):
        model.train()
        epoch_loss = 0 
        for idx, batch in enumerate(train_loader):
            train_images = batch.reshape(-1, 1, 96, 192) # Unity Image Size
            train_images = np2torch(train_images, device=args.device)

            recon_images = model(train_images)
            recon_loss   = F.mse_loss(recon_images, train_images)
            # Backprob 
            recon_loss.backward()
            # Update the weights
            optimizer.step()      
            epoch_loss += round(recon_loss.item(), 6)
        if args.WANDB:
            wandb.log({"train loss":epoch_loss/(idx+1)}, step=epoch+1)   
        scheduler.step()    

        # Evaluation 
        if (epoch+1)%10==0 or (epoch+1)==1 or (epoch+1)==5 or (epoch+1)==args.max_epochs:
            model.eval()
            with torch.no_grad():
                val_images  = next(iter(val_loader))
                val_images = val_images.reshape(-1, 1, 96, 192)
                val_images = np2torch(val_images, device=args.device)
                val_preds  = model(val_images)

                c_vector1= torch2np(model.encoder(np2torch(unityimg1.reshape(-1, 1, 96, 192), device=args.device)))
                c_vector2= torch2np(model.encoder(np2torch(unityimg2.reshape(-1, 1, 96, 192), device=args.device)))
                c_vector3= torch2np(model.encoder(np2torch(unityimg3.reshape(-1, 1, 96, 192), device=args.device)))
                c_vector4= torch2np(model.encoder(np2torch(unityimg4.reshape(-1, 1, 96, 192), device=args.device)))
            unity_dev  = np.sum(np.abs(c_vector1-c_vector2))
            unity_dev2 = np.sum(np.abs(c_vector3-c_vector4))
            if args.WANDB: 
                wandb.log({"unity deviation":unity_dev,
                           "unity deviation2":unity_dev2}, step=epoch+1)  

            fig, axs = plt.subplots(2,5, figsize=(15,4))
            val_images = torch2np(val_images)
            val_preds  = torch2np(val_preds)
            for i in range(5):
                axs[0][i].matshow(np.reshape(val_images[i, :], (96,192)))
                axs[1][i].matshow(np.reshape(val_preds[i, :], (96,192)))
            plt.suptitle("Resnet[Encoder: Resblock 3th, ReLU][Decoder: Resblock 3th, Shallow]", fontsize=15)
            plt.savefig("data/resnet/{}/{}_eval{}.png".format(args.runname,args.runname,epoch+1))
            torch.save(model.encoder.state_dict(), 'weights/{}/resnet_encoder{}steps.pth'.format(args.runname,epoch+1))
            # torch.save(model.decoder.state_dict(), 'weights/{}/resnet_decoder{}steps.pth'.format(args.runname,epoch+1))

  

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
    args.add_argument('--data_path', default='resnet1116_new_bigger_10', type=str,
                      help="wandb runname")
    args.add_argument('--json_path', default='resnet1116_new_bigger_10', type=str,
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