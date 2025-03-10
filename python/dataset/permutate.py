import numpy as np

class Permutate(object):
    """Permutes the system of matrices for training."""
    
    def __init__(self, dimension, rng):
        """Initializes the Permute.

        Args:
            dimension : The dimension of the systems to permute.
            rng : The numpy RNG.
        """
        
        self.dimension = dimension
        self.flattened_size = dimension ** 2
        self.rng = rng
        
    def __call__(self, sample):
        """Permutes the given system.

        Args:
            sample : The sample to permute.

        Returns:
            The permuted system.
        """
        
        system, solution = sample
        permutation = self.rng.permutation(np.arange(self.dimension))
        copy_system = system.clone().detach()
        for i in range(self.dimension):
            start = i * self.flattened_size
            permutation_start = permutation[i] * self.flattened_size
            system[start:start + self.flattened_size] = copy_system[permutation_start:permutation_start + self.flattened_size]
        return system, solution