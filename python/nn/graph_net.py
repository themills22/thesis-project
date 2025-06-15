import torch.nn as nn

class GraphNet(nn.Module):
    """Neural net class for the power-flow graph representation."""
    
    def __init__(self, size):
        """Initializes the graph NN.

        Args:
            size : The number of parameters for the system.
        """
        
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(size, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Hardtanh(-100, 100)
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