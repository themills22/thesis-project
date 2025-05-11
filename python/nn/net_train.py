import argparse
import datetime as dt
import math
import numpy as np
import os
import python.parser_helpers as ph
import torch
import torch.nn as nn
import torch.optim as optim

from python.dataset.graph.file_dataset import FileDataset as GraphDataset
from python.nn.graph_net import GraphNet
from scipy.special import comb
from torchvision.transforms import v2

def create_graph_settings(paths, size, input_norm_cap):
    size = int(comb(size, 2))
    dataset = GraphDataset(paths, size, input_norm_cap)
    model = GraphNet(size)
    return size, dataset, model

def train(model, device, train_loader, optimizer, loss_function, print_batch_interval=None):
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        if print_batch_interval and batch_index % print_batch_interval == 0:
            print('Batch index: {}, Loss: {}'.format(batch_index, loss.item()))
            
def test(model, device, test_loader, loss_function, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_function(output, target)
        test_loss /= len(test_loader.dataset)
    print('Test: Epoch: {}, Average loss: {:.4f}'.format(epoch, test_loss))
    return test_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', help='The batch size to test and train with.', type=int, default=64)
    parser.add_argument('--type', help='The model type: graph or matrix.', required=True, choices=['graph'])
    parser.add_argument('--size', help='The network node size or matrix dimension.', required=True, type=int)
    parser.add_argument('--data-folder', help='A directory where the files to train on live.', action='append', \
        required=True, type=ph.is_valid_file)
    parser.add_argument('--model-folder', help='The directory to save pytorch model files.', required=True, type=ph.is_valid_file)
    parser.add_argument('--model-to-load', help='The model weights to start with', type=ph.is_valid_file)
    parser.add_argument('--epochs', help='The number of epochs to run.', type=int, default=100)
    parser.add_argument('--epoch-save', help='The number of epochs to go through before saving.', type=int, default=10)
    parser.add_argument('--print-interval', help='How many batches to run through before printing the current stats.', type=int)
    parser.add_argument('--input-norm-cap', help='What input parameters to discard based off their norm.', type=float)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': True
    }
    test_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': True
    }
    if device == 'cuda':
        cuda_kwargs = {
            'num_workers': 1,
            'pin_memory': True
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
        
    size, dataset, model = create_graph_settings(args.data_folder, args.size, args.input_norm_cap)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    
    if args.model_to_load:
        model.load_state_dict(torch.load(args.model_to_load))
    model = model.to(device)
    optimizer = optim.Adadelta(model.parameters(), 1.0)
    loss = nn.MSELoss()
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, loss, args.print_interval)
        test_loss = test(model, device, test_loader, loss, epoch)
        if epoch % args.epoch_save == 0:
            file_name = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.pt')
            torch.save(model.state_dict(), os.path.join(args.model_folder, file_name))
            
if __name__ == '__main__':
    main()