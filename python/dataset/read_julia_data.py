import numpy as np
import re

def read_file(file_name, dimension):
    """Reads the systems and solution counts from the given file.

    Args:
        file_name : The name of the file.
        dimension : The dimension of the systems in the file.

    Returns:
        The systems and solution counts.
    """
    
    # abuse regex to get all floats and integers out of line
    float_regex = re.compile(r'[-]?\d+\.\d+')
    int_regex = re.compile(r'\d+')
    with open(file_name) as file:
        text = file.readlines()

    # can then use knowledge of given dimension to figure out how many systems there are
    all_values = float_regex.findall(text[0])
    num_systems = len(all_values) / (dimension ** 3)
    systems = np.array([float(value) for value in all_values]).reshape((int(num_systems), dimension, dimension, dimension))
    
    all_values = int_regex.findall(text[1])
    solutions = np.array([int(value) for value in all_values])
    return (systems, solutions)