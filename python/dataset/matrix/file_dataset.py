import numpy as np
import os
import torch

from python.dataset.lru_cache import LRUCache
from torch.utils.data import Dataset

# TODO: make this class match the matrix dataset one; can probably do some class inheritance as well :)
class FileDataset(Dataset):
    """Dataset wrapper over a list of files where each file contains multiple systems."""
    
    def __init__(self, data_files, system_mapping, entries_per_file, dimension, lru_file_capacity, transform=None):
        """Initializes the FileDataset.

        Args:
            data_files : The list of file names. 
            entries_per_file : The number of systems in each file.
            dimension : The dimension size of each of the systems.
            lru_file_capacity : How many files, and their systems, to keep in an LRU cache in memory
            transform (optional): The pytorch transformer. Defaults to None.
        """
        
        if isinstance(data_files, str):
            self.files = next(os.walk(data_files), (None, None, []))[2]
            self.files = [os.path.join(data_files, file) for file in self.files]
        else:
            self.files = data_files
        self.system_mapping = system_mapping
        self.entries_per_file = entries_per_file
        self.dimension = dimension
        self.cache = LRUCache(lru_file_capacity)
        self.transform = transform
        
    def __len__(self):
        """Gets the number of all the systems contained in the dataset.

        Returns:
            The number of all the systems contained in the dataset.
        """
        
        return len(self.system_mapping)
    
    def __getitem__(self, index):
        """Gets the system and solution count at the given index.

        Args:
            index : The index.

        Returns:
            The system and solution at the index.
        """
        inputs, solution_counts = None, None
        file_index = int(index / self.entries_per_file)
        item = self.cache.get(file_index)
        if item is None:
            npz_file = np.load(self.files[file_index])
            systems, solution_points, solution_counts = npz_file['systems'], npz_file['solutions'], npz_file['solution_counts']
            inputs = [torch.from_numpy(np.concatenate((system.flatten(), solution_point))).float() \
                for system, solution_point in zip(systems, solution_points)]
            solution_counts = [torch.tensor([solution]).float() for solution in solution_counts] 
            self.cache.put(file_index, (inputs, solution_counts))
        else:
            inputs, solution_counts = item
        
        input, solution = inputs[index % self.entries_per_file], solution_counts[index % self.entries_per_file]
        if self.transform:
            input, solution = self.transform((input, solution))
        return input, solution