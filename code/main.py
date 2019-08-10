import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import datetime as dt
import time
from tensorboardX import SummaryWriter
writer = None

from model import ModelCifar
from data import load_data
from eval import evaluate
from loss import CustomLoss
from utils import *

def train(model, criterion, optimizer, reader, hyper_params):
    model.train()
    
    metrics = {}
    total_batches = 0.0
    total_loss = FloatTensor([ 0.0 ])
    correct, total = LongTensor([ 0 ]), 0.0
    control_variate = FloatTensor([ 0.0 ])
    ips = FloatTensor([ 0.0 ])

    for x, y, action, delta, prop in reader.iter():
        
        # Empty the gradients
        model.zero_grad()
        optimizer.zero_grad()
    
        # Forward pass
        output = model(x)
        output = F.softmax(output, dim = 1)
        
        # Backward pass
        loss = criterion(output, action, delta, prop)
        loss.backward()
        optimizer.step()

        # Log to tensorboard
        writer.add_scalar('train loss', loss.data, total_batches)

        # Metrics evaluation
        total_loss += loss.data
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

def main(hyper_params = None, return_model=False):
    # If custom hyper_params are not passed, load from hyper_params.py
    if hyper_params is None: from hyper_params import hyper_params
    else: print("Using passed hyper-parameters..")

    # Initialize a tensorboard writer
    global writer
    path = hyper_params['tensorboard_path']
    writer = SummaryWriter(path)

    # Train It..
    train_reader, test_reader, val_reader = load_data(hyper_params)

    file_write(hyper_params['log_file'], "\n\nSimulation run on: " + str(dt.datetime.now()) + "\n\n")
    file_write(hyper_params['log_file'], "Data reading complete!")
    file_write(hyper_params['log_file'], "Number of train batches: {:4d}".format(len(train_reader)))
    file_write(hyper_params['log_file'], "Number of test batches: {:4d}".format(len(test_reader)))

    model = ModelCifar(hyper_params)
    if is_cuda_available: model.cuda()

    criterion = CustomLoss(hyper_params)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=hyper_params['lr'], momentum=0.9, weight_decay=hyper_params['weight_decay']
    )

    file_write(hyper_params['log_file'], str(model))
    file_write(hyper_params['log_file'], "\nModel Built!\nStarting Training...\n")

    best_metrics_train = None
    best_metrics_test = None

    try:
        for epoch in range(1, hyper_params['epochs'] + 1):
            epoch_start_time = time.time()
            
            # Training for one epoch
            metrics = train(model, criterion, optimizer, train_reader, hyper_params)
            
            string = ""
            for m in metrics: string += " | " + m + ' = ' + str(metrics[m])
            string += ' (TRAIN)'

            best_metrics_train = metrics

            # Calulating the metrics on the validation set
            metrics = evaluate(model, criterion, test_reader, hyper_params)
            string2 = ""
            for m in metrics: string2 += " | " + m + ' = ' + str(metrics[m])
            string2 += ' (TEST)'

            best_metrics_test = metrics

            ss  = '-' * 89
            ss += '\n| end of epoch {:3d} | time: {:5.2f}s'.format(epoch, (time.time() - epoch_start_time))
            ss += string
            ss += '\n'
            ss += '-' * 89
            ss += '\n| end of epoch {:3d} | time: {:5.2f}s'.format(epoch, (time.time() - epoch_start_time))
            ss += string2
            ss += '\n'
            ss += '-' * 89
            file_write(hyper_params['log_file'], ss)
            
            for metric in metrics: writer.add_scalar('Test_metrics/' + metric, metrics[metric], epoch - 1)
            
    except KeyboardInterrupt: print('Exiting from training early')

    writer.close()

    if return_model == True: return model
    return best_metrics_train, best_metrics_test

if __name__ == '__main__':
    main()
