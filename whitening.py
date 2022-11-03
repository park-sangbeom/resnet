import numpy as np
import matplotlib.pyplot as plt 
import torch
import argparse
import json
from data_loader import DepthDatasetLoader
from model.utils import torch2np, np2torch, whitening
from model.resnet_for_extractor import Encoder 

def main(args):
    root_path = "/home/sangbeom/resnet/data/depth1014/"
    depth_dataset = DepthDatasetLoader(root_path=root_path)
    # Set random seed 
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  

    print("Feature Extractor Weight: {} Load.".format(args.LOADWEIGHT))	
    feature_extractor = Encoder().to(device=args.device)
    pretrained_dict = torch.load(args.LOADWEIGHT)
    feature_extractor.load_state_dict(pretrained_dict)
    print(feature_extractor.load_state_dict(pretrained_dict))
    feature_extractor.eval() 
    c_vector_lst1d = []
    c_vector_lst2d = []
    json_path = "c_vector.json"
    json_path_whiten1d = "c_vector_whiten1d.json"    
    json_path_whiten2d = "c_vector_whiten2d.json"    

    for idx, depth in enumerate(depth_dataset):
        print("Index: {}".format(idx+1))
        c_vector = torch2np(feature_extractor(np2torch(depth.reshape(-1, 1, 96, 192), device=args.device)))
        c_vector_lst2d.append(c_vector)
        c_vector_lst1d.append(c_vector.reshape(-1))
        c_vector_content = {"idx":idx+1,
                            "c_vector2d":c_vector.tolist(),
                            "c_vector1d":c_vector.reshape(-1).tolist()}
        with open(json_path, "a") as f:
            f.write(json.dumps(c_vector_content)+'\n')

    print("Feature extraction end.")
    print("Total num: {}".format(len(c_vector_lst2d)))

    c_vector_arr1d = np.array(c_vector_lst1d)
    c_vector_arr2d = np.array(c_vector_lst2d)
    np.save("c_vector_arr2d.npy",c_vector_arr2d)
    np.save("c_vector_arr2d.npy",c_vector_arr1d)

    whitened1d, x_mean1d, x_std1d = whitening(c_vector_arr1d)
    whitened2d, x_mean2d, x_std2d = whitening(c_vector_arr2d)

    whitened1d_content = {"whitened1d":whitened1d.tolist(),
                        "x_mean1d":x_mean1d.tolist(),
                        "x_std1d":x_std1d.tolist(),
                        "c_vector_arr1d":c_vector_arr1d.tolist()}

    whitened2d_content = {"whitened2d":whitened2d.tolist(),
                        "x_mean2d":x_mean2d.tolist(),
                        "x_std2d":x_std2d.tolist(),
                        "c_vector_arr2d":c_vector_arr2d.tolist()}


    with open(json_path_whiten1d, "a") as f:
        f.write(json.dumps(whitened1d_content)+'\n')
    with open(json_path_whiten2d, "a") as f:
        f.write(json.dumps(whitened2d_content)+'\n')
    print("End.")


if __name__=="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device: {}".format(device))
    args   = argparse.ArgumentParser(description= "Parse for Reach Task")
    # LOAD WEIGHTS
    args.add_argument('--LOADWEIGHT', default='./weights/resnet1101_3/resnet_encoder50steps.pth', type=str,
                      help="ResNet weight path")
    # DEVICE
    args.add_argument('--device', default= device,type=str, #cpu cuda:0
                      help='device')
    # Random Seed  
    args.add_argument('--random_seed', default= 42, type=int,
                      help='Random Seed') 
    args = args.parse_args()
    main(args)
