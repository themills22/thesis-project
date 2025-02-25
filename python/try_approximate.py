import approximator
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import parser_helpers as ph
import python.dataset.read_julia_data as rjd
import time
import tqdm

# point of this file does not reflect the file name
# this file tries to see how reliably a system can projected onto the K_c(X) set

file_path = 'D:\\deep-reinforcement-learning\\thesis-project\\matrices\\10_dim.txt'
rng = np.random.default_rng()
systems, system_solutions = rjd.read_file(file_path, 10)
n = systems.shape[1]
results = {}
for i in tqdm.tqdm(range(systems.shape[0]), total=systems.shape[0]):
    results[i] = {
        'count' : int(system_solutions[i])
    }
    scaled_system, scaled_solutions = approximator.scale_system(systems[i])
    for p in np.linspace(1.1, 3.0, 10):
        results[i][p] = {}
        for _ in range(100):
            point = rng.normal(0, 1, n)
            perturbation_factor = 1 / (n ** p)
            projected_system_tuples = approximator.project_system(scaled_system, perturbation_factor, \
                scaled_solutions, point, rng, 100)
            for j in range(n):
                attempts, projected  = projected_system_tuples[j]
                if projected is not None:
                    continue
                if j not in results[i][p]:
                    results[i][p][j] = 0
                results[i][p][j] += 1
with open('try.json', 'w') as file:
    file.write(json.dumps(results, indent=4))