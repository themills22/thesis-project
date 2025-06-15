import torch.nn as nn

class MatrixNet(nn.Module):
    """Neural net class for the power-flow matrix representation."""
    
    def __init__(self, size):
        """Initializes the matrix NN.

        Args:
            size : The system dimension.
        """
        
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
        """Forwards the input to the model.

        Args:
            x : The input to the model.

        Returns:
            The ouput of the model.
        """
        
        return self.model(x)