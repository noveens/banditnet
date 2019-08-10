import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from utils import *

class CustomLoss(torch.nn.Module):
    def __init__(self, hyper_params):
        super(CustomLoss,self).__init__()
        self.hyper_params = hyper_params
        self.lamda = hyper_params['lamda']

    def forward(self, output, action, delta, prop):
        risk = 1.0 - delta

        loss = (risk - self.lamda) * (output[range(action.size(0)), action] / prop)
            
        return torch.mean(loss)
