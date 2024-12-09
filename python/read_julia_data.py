import numpy as np
import re

# abuse regex to get all floats and integers out of line
float_regex = re.compile(r'[-]?\d+\.\d+')
int_regex = re.compile(r'\d+')
def read_file(file, dimension):
  text = file.readlines()

  # can then use knowledge of given dimension to figure out how many systems there are
  all_values = float_regex.findall(text[0])
  num_systems = len(all_values) / (dimension ** 3)
  systems = np.array([float(value) for value in all_values]).reshape((int(num_systems), dimension, dimension, dimension))

  all_values = int_regex.findall(text[1])
  solutions = np.array([int(value) for value in all_values])
  return (systems, solutions)

with open('drive/MyDrive/computing-with-quadratics/data/3_dim.txt') as file:
  systems, solutions = read_file(file, 3)
  approximated_solutions = np.array([approximate_solution(system) for system in systems])
  # compare solutions and approximated_solutions to check if approximator "works"