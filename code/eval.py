import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

from utils import *

def evaluate(model, criterion, reader, hyper_params):

    metrics = {}
    total_batches = 0.0
    total_loss = FloatTensor([ 0.0 ])
    correct, total = LongTensor([ 0 ]), 0.0
    control_variate = FloatTensor([ 0.0 ])
    ips = FloatTensor([ 0.0 ])

    model.eval()

    for x, y, action, delta, prop in reader.iter():
        output = model(x)
        output = F.softmax(output, dim = 1)

        total_loss += criterion(output, action, delta, prop).data

        control_variate += torch.mean(output[range(action.size(0)), action] / prop).data
        ips += torch.mean((delta * output[range(action.size(0)), action]) / prop).data

        _, predicted = torch.max(output.data, 1)
        total += y.size(0)
        correct += predicted.eq(y.data).sum().data
        
        total_batches += 1.0

    metrics['loss'] = round(float(total_loss) / total_batches, 4)
    metrics['Acc'] = round(100.0 * float(correct) / float(total), 4)
    metrics['CV'] = round(float(control_variate) / total_batches, 4)
    metrics['SNIPS'] = round(float(ips) / float(control_variate), 4)

    return metrics
