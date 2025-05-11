import math
import numpy as np
import os
import torch

from bisect import bisect_right
from torch.utils.data import Dataset

class FileDataset(Dataset):
    """Dataset wrapper over a list of files where each file contains multiple systems."""
    
    def __init__(self, paths, size, input_norm_cap=None, transform=None):
        """Initializes the FileDataset.

        Args:
            data_files : The list of file names. 
            entries_per_file : The number of systems in each file.
            size : The node count of each of the graphs.
            lru_file_capacity : How many files, and their systems, to keep in an LRU cache in memory
            transform (optional): The pytorch transformer. Defaults to None.
        """

        self.size = size
        self.input_norm_cap = input_norm_cap if input_norm_cap else math.inf
        self.transform = transform if transform else self._identity
        files = []
        for path in paths:
            path_files = [os.path.join(path, file) for file in next(os.walk(path), (None, None, []))[2]] \
                if isinstance(path, str) else path
            files.extend(path_files)
        
        self.items = []
        for file in files:
            with np.load(file) as npz_file:
                systems, solution_counts = npz_file['systems'], npz_file['solution_counts']
                for system, solution_count in zip(systems, solution_counts):
                    if np.linalg.norm(system) < self.input_norm_cap:
                        self.items.append((system, solution_count))
        
    def __len__(self):
        """Gets the number of all the systems contained in the dataset.

        Returns:
            The number of all the systems contained in the dataset.
        """
        
        return len(self.items)
    
    def __getitem__(self, index):
        """Gets the system and solution count at the given index.

        Args:
            index : The index.

        Returns:
            The system and solution at the index.
        """
        
        system, solution_count = self.items[index]
        return self.transform((torch.from_numpy(system).float(), torch.tensor([solution_count]).float()))
        
    def _identity(self, x):
        return x