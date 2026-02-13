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
    parser.add_argument('--num-improvements', help='The number of improvements to make to a randomly generated system.',
                        required=True, type=lambda value: ph.check_greater_than_int(value, 0))
    
    args = parser.parse_args()
    rng = np.random.default_rng()
    systems = rng.uniform(-1, 1, (1 + args.num_improvements, 10, 10, 10))
    
    policy = TD3.load(args.model_path).policy
    state_space = policy.observation_space
    size = state_space.shape[0]
    systems = np.zeros((1 + args.num_improvements, size, size, size)) 
    systems[0] = rng.uniform(-1, 1, (size, size, size))
    for i in range(args.num_improvements):
        action, _ = policy.predict(systems[i])
        systems[i + 1] = np.clip(systems[i] + action, state_space.low, state_space.high)
    jl.seval("using PowerFlow: judge_matrix_systems")
    counts = jl.judge_matrix_systems(systems)
    average_count = _get_average_gaussian_count(size)
    print('average gaussian count: {}'.format(average_count))
    print('counts: {}'.format(counts))
    
def _get_average_gaussian_count(size):
    result = np.power(2, size / 2)
    result *= np.power(size, -1 / 2)
    return result

if __name__ == '__main__':
    main()