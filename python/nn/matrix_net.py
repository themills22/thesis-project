import torch.nn as nn

class MatrixNet(nn.Module):
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