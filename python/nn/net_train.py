from python.dataset.matrix.permutate import Permutate
from python.dataset.matrix.file_dataset import FileDataset as MatrixDataSet
from python.dataset.graph.file_dataset import FileDataset as GraphDataset
from python.nn.matrix_net import MatrixNet
from python.nn.power_flow_net import GraphNet
from scipy.special import comb
from torchvision.transforms import v2

import argparse
import datetime as dt
import numpy as np
import os
import python.parser_helpers as ph
import torch
import torch.nn as nn
import torch.optim as optim

def is_valid_model_type(value):
    return value == 'graph' or value == 'matrix'

def create_type_settings(type, size, train_files, test_files, rng):
    is_graph = type == 'graph'
    size = comb(size, 2) if is_graph else size
    transform = None
    if not is_graph:
        transform = v2.Compose([
            Permutate(size, rng)
        ])
    
    entries_per_file = 1000 if is_graph else 100
    create_dataset = lambda files: GraphDataset(files, entries_per_file, size, 100, transform) \
        if is_graph else  MatrixDataSet(files, entries_per_file, size, 100, transform)
    train_dataset = create_dataset(train_files)
    test_dataset = create_dataset(test_files)
    
    model = GraphNet(size) if is_graph else MatrixNet(size)
    return size, transform, train_dataset, test_dataset, model



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
            print('epoch: {}, iteration: {}, Loss: {}'.format(batch_index, loss.item()))
            
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
    parser.add_argument('--type', help='The model type: graphor matrix.', required=True, type=is_valid_model_type)
    parser.add_argument('--size', help='The network node size or matrix dimension.', required=True, type=int)
    parser.add_argument('--data-folder', help='The directory where the files to train on live.', required=True, type=ph.is_valid_file)
    parser.add_argument('--model-folder', help='The directory to save pytorch model files.', required=True, type=ph.is_valid_file)
    parser.add_argument('--model-to-load', help='The model weights to start with', type=ph.is_valid_file)
    parser.add_argument('--epochs', help='The number of epochs to run.', type=int, default=100)
    parser.add_argument('--print-interval', help='How many batches to run through before printing the current stats.', type=int)
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
        
    rng = np.random.default_rng()
    _, _, files = next(os.walk(args.data_folder), (None, None, []))
    files = [os.path.join(args.data_folder, file) for file in files]
    rng.shuffle(files)
    

    train_count = int(0.8 * len(files))
    train_files = files[0:train_count]
    test_files = files[train_count:]
    size, transform, train_dataset, test_dataset, model = create_type_settings(args.type, args.size, train_files, test_files, rng)
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
        if epoch % args.test_duration == 0:
            file_name = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.pt')
            torch.save(model.state_dict(), os.path.join(args.model_folder, file_name))
            
if __name__ == '__main__':
    main()