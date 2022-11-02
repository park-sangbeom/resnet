import numpy as np
from datetime import datetime
import torch 

def torch2np(x_torch):
    if x_torch is None:
        x_np = None
    else:
        x_np = x_torch.detach().cpu().numpy()
    return x_np
    
def np2torch(x_np,device='cpu'):
    if x_np is None:
        x_torch = None
    else:
        x_torch = torch.tensor(x_np,dtype=torch.float32,device=device)
    return x_torch

def rad2deg(rad):
    """ Radian to Degree """
    return rad*180/np.pi

def get_runname():
    now = datetime.now()
    format = "%m%d:%H%M"
    runname = now.strftime(format)
    return runname