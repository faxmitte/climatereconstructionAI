import torch
import torch.nn.functional as F
from torch import nn
from .utils import conv_variance


class GaussLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.Gauss = nn.GaussianNLLLoss()


    def forward(self, mask, output, gt):
        loss_dict = {
            'gauss': 0.0
        } 

        loss_dict['gauss'] += self.Gauss(output[:,0,:,:],gt,output[:,1,:,:]**2)

        return loss_dict