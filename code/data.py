import torch
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable

from utils import *

class DataLoader():
    def __init__(self, hyper_params, x, delta, prop = None, action = None):
        self.x = x
        self.delta = delta
        self.prop = prop
        self.action = action
        self.bsz = hyper_params['batch_size']
        self.hyper_params = hyper_params

    def __len__(self):
        return len(self.x)
        
    def iter(self, eval = False):
        x_batch, y_batch, action, delta, all_delta, prop, all_prop = [], [], [], [], [], [], []
        data_done = 0
        for ind in tqdm(range(len(self.x))):

            if self.prop[ind][self.action[ind]] < 0.001: continue # Overflow issues, Sanity check

            x_batch.append(self.x[ind].reshape(3, 32, 32))
            y_batch.append(np.argmax(self.delta[ind]))

            # Pick already chosen action
            choice = self.action[ind]
            action.append(choice)

            delta.append(self.delta[ind][choice])
            prop.append(self.prop[ind][choice])
            
            all_delta.append(self.delta[ind])
            all_prop.append(self.prop[ind])
            
            data_done += 1

            if len(x_batch) == self.bsz:
                if eval == False:
                    yield Variable(FloatTensor(x_batch)), Variable(LongTensor(y_batch)), Variable(LongTensor(action)), \
                    Variable(FloatTensor(delta)), Variable(FloatTensor(prop))
                else:
                    yield Variable(FloatTensor(x_batch)), Variable(LongTensor(y_batch)), Variable(LongTensor(action)), \
                    Variable(FloatTensor(delta)), Variable(FloatTensor(prop)), all_prop, all_delta
                
                x_batch, y_batch, action, delta, all_delta, prop, all_prop = [], [], [], [], [], [], []

def readfile(path, hyper_params):
    x, delta, prop, action = [], [], [], []
    
    data = load_obj(path)
    
    for line in data:    
        x.append(line[:3072])
        delta.append(line[3072:3082])
        prop.append(line[3082:3092])
        action.append(int(line[-1]))

    return np.array(x), np.array(delta), np.array(prop), np.array(action)

def load_data(hyper_params):

    path  = '../data/cifar-10-batches-py/bandit_data' 
    path += '_sampled_' + str(hyper_params['num_sample'])

    x_train, delta_train, prop_train, action_train = readfile(path + '_train', hyper_params)
    x_test, delta_test, prop_test, action_test = readfile(path + '_test', hyper_params)
    x_val, delta_val, prop_val, action_val = readfile(path + '_val', hyper_params)

    # Shuffle train set
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)

    x_train = x_train[indices]
    delta_train = delta_train[indices]
    prop_train = prop_train[indices]
    action_train = action_train[indices]

    trainloader = DataLoader(hyper_params, x_train[:hyper_params['train_limit']], delta_train[:hyper_params['train_limit']], prop_train[:hyper_params['train_limit']], action_train[:hyper_params['train_limit']])
    testloader = DataLoader(hyper_params, x_test, delta_test, prop_test, action_test)
    valloader = DataLoader(hyper_params, x_val, delta_val, prop_val, action_val)

    return trainloader, testloader, valloader
