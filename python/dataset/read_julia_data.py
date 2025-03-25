from scipy.special import comb

import numpy as np
import re

def read_file(file_name):
    """Reads the systems and solution counts from the given file.

    Args:
        file_name : The name of the file.

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
    systems = np.array([float(value) for value in all_values])
    
    all_values = int_regex.findall(text[1])
    solutions = np.array([int(value) for value in all_values])
    return (systems, solutions)

def read_matrix_file(file_name, dimension):
    """Reads the matrix systems and solution counts from the given file.

    Args:
        file_name : The name of the file.
        dimension : The dimension of the systems in the file.

    Returns:
        The systems and solution counts.
    """
    
    # can then use knowledge of given dimension to figure out how many systems there are
    systems, solutions = read_file(file_name)
    num_systems = int(len(systems) / (dimension ** 3))
    systems = systems.reshape((num_systems, dimension, dimension, dimension))
    return (systems, solutions)

def read_power_flow_file(file_name, size):
    """Reads the power flow systems and solution counts from the given file.

    Args:
        file_name : The name of the file.
        dimension : The dimension of the systems in the file.

    Returns:
        The systems and solution counts.
    """
    
    # can then use knowledge of given dimension to figure out how many systems there are
    systems, solutions = read_file(file_name)
    edge_count = int(comb(size - 1, 2))
    num_systems = int(len(systems) / edge_count)
    systems = systems.reshape((num_systems, edge_count))
    return (systems, solutions)