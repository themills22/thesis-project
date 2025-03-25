import datetime as dt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from python.dataset.matrix.file_dataset import FileDataset
from python.dataset.matrix.permutate import Permutate
from torchvision.transforms import v2

class NN(nn.Module):
    def __init__(self, size):
        super().__init__()
        total = size ** 3
        self.model = nn.Sequential(
            nn.Linear(total + size, 4 * total),
            nn.ReLU(),
            nn.Linear(4 * total, 4 * total),
            nn.ReLU(),
            nn.Linear(4 * total, 4 * total),
            nn.ReLU(),
            nn.Linear(4 * total, 4 * total),
            nn.ReLU(),
            nn.Linear(4 * total, 4 * total),
            nn.ReLU(),
            nn.Linear(4 * total, 4 * total),
            nn.ReLU(),
            nn.Linear(4 * total, 4 * total),
            nn.ReLU(),
            nn.Linear(4 * total, 4 * total),
            nn.ReLU(),
            nn.Linear(4 * total, 4 * total),
            nn.ReLU(),
            nn.Linear(4 * total, 4 * total),
            nn.ReLU(),
            nn.Linear(4 * total, 4 * total),
            nn.ReLU(),
            nn.Linear(4 * total, 4 * total),
            nn.ReLU(),
            nn.Linear(4 * total, 4 * total),
            nn.ReLU(),
            nn.Linear(4 * total, 4 * total),
            nn.ReLU(),
            nn.Linear(4 * total, 4 * total),
            nn.ReLU(),
            nn.Linear(4 * total, 4 * total),
            nn.ReLU(),
            nn.Linear(4 * total, 4 * total),
            nn.ReLU(),
            nn.Linear(4 * total, 4 * total),
            nn.ReLU(),
            nn.Linear(4 * total, 4 * total),
            nn.ReLU(),
            nn.Linear(4 * total, 1),
        )
        self.size = size
        
    def forward(self, x):
        return self.model(x)
    
def train(model, device, train_loader, optimizer, loss_function, epoch):
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        # if batch_index % 500 == 0:
        #     print('epoch: {}, iteration: {}, Loss: {}'.format(epoch, batch_index, loss.item()))
            
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_kwargs = {
        'batch_size': 64,
        'shuffle': True
    }
    test_kwargs = {
        'batch_size': 64,
        'shuffle': True
    }
    if device == 'cuda':
        cuda_kwargs = {
            'num_workers': 1,
            'pin_memory': True
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    n = 8
    rng = np.random.default_rng()
    transform = v2.Compose([
        Permutate(n, rng)
    ])
    data_folder = 'D:\\deep-reinforcement-learning\\thesis-project\\data\\scaled\\{}'.format(n)
    model_folder = 'D:\\deep-reinforcement-learning\\thesis-project\\model\\{}'.format(n)
    files = next(os.walk(data_folder), (None, None, []))[2]
    files = [os.path.join(data_folder, file) for file in files]
    rng.shuffle(files)
    
    train_count = int(0.8 * len(files))
    train_files = files[0:train_count]
    train_dataset = FileDataset(train_files, 100, n, 100, transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_files = files[train_count:]
    test_dataset = FileDataset(test_files, 100, n, 100, transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model_to_load = 'D:\\deep-reinforcement-learning\\thesis-project\\model\\8\\2025-02-25-19-30-39.pt '
    model = NN(n)
    if model_to_load:
        model.load_state_dict(torch.load(model_to_load))
    model = model.to(device)
    optimizer = optim.Adadelta(model.parameters(), 1.0)
    loss = nn.MSELoss()
    for epoch in range(1, 201):
        train(model, device, train_loader, optimizer, loss, epoch)
        test_loss = test(model, device, test_loader, loss, epoch)
        if epoch % 10 == 0:
            file_name = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.pt')
            torch.save(model.state_dict(), os.path.join(model_folder, file_name))
    
if __name__ == '__main__':
    main()