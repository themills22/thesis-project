from juliacall import Main as jl
from stable_baselines3 import TD3

import argparse
import juliacall
import matplotlib.pyplot as plt
import numpy as np
import os
import python.parser_helpers as ph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', help='The path of the model.', required=True, type=ph.is_valid_file)
    parser.add_argument('--num-systems', help='The number of systems to improve on.', required=True, type=int)
    parser.add_argument('--num-improvements', help='The number of improvements to make to a randomly generated system.',
                        required=True, type=lambda value: ph.check_greater_than_int(value, 0))
    
    args = parser.parse_args()
    rng = np.random.default_rng()
    
    policy = TD3.load(args.model_path).policy
    state_space = policy.observation_space
    size = state_space.shape[0]
    average_count = _get_average_gaussian_count(size)
    print('average gaussian count: {}'.format(average_count))
    
    jl.seval("using PowerFlow: judge_matrix_systems")
    all_counts = []
    systems = np.zeros((1 + args.num_improvements, size, size, size))
    starting_systems = rng.normal(0, 1, (args.num_systems, size, size, size))
    i = 0
    while i < len(starting_systems):
        systems[0] = starting_systems[i]
        for j in range(args.num_improvements):
            action, _ = policy.predict(systems[j])
            systems[j + 1] = np.clip(systems[j] + action, state_space.low, state_space.high)
        
        counts = jl.judge_matrix_systems(systems)
        if counts is not None:
            all_counts.append(np.array(counts[1:]))
            print('counts: {}'.format(counts))
            i += 1
        else:
            continue
    print('average: {}'.format(np.array(all_counts).mean()))
    
def _get_average_gaussian_count(size):
    result = np.power(2, size / 2)
    result *= np.power(size, -1 / 2)
    return result

if __name__ == '__main__':
    main()