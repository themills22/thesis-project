import approximator as ap
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import python.parser_helpers as ph
import time
import tqdm

from typing import Generator

def enumerate_systems(args, rng):
    total_dimension = (args.dimension, args.dimension, args.dimension)
    def enumerate_from_int():
        yield from ((rng.normal(0, 1, total_dimension), -1) for _ in range(args.system_count))
        
    def enumerate_from_files():
        files = []
        for path in args.data_path:
            if os.path.isdir(path):
                path_files = [os.path.join(path, file) for file in next(os.walk(path), (None, None, []))[2]]
                files.extend(path_files)
            else:
                files.append(path)
        for file in files:
            with np.load(file) as npz_file:
                for system, count in zip(npz_file['systems'], npz_file['solution_counts']):
                    if system.shape != total_dimension:
                        raise ValueError('The system\'s shape, {}, does not match the supplied dimension, {}'.format(system.shape, total_dimension))
                    yield system, count
                    
    return enumerate_from_files() if args.data_path else enumerate_from_int()

def normalize(data):
    return (data - min(data))/(max(data) - min(data))
    
def main():
    is_positive_int = lambda value: ph.check_greater_than_int(value, 0)
    is_none_or_non_negative_int = lambda value: not value or ph.check_greater_than_int(value, -1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimension', help='The dimension of the systems. Must be at least three.', \
        required=True, type=lambda value: ph.check_greater_than_int(value, 2))
    parser.add_argument('--perturb', help='The perturbation factor to use.', required=True, type=float)
    parser.add_argument('--point-count', help='The number of points to sample.', required=True, \
        type=is_positive_int)
    parser.add_argument('--matrix-count', help='The number of matrices to sample.', required=True, \
        type=is_positive_int)
    parser.add_argument('--system-count', help='The number of systems to generate and approximate. Use --data-path if you want to approximate NPZ files.', \
        type=lambda value: ph.check_greater_than_int(value, 0))
    parser.add_argument('--data-path', help='Directories or files to approximate.', action='append', \
        type=ph.is_valid_file)
    parser.add_argument('--results-folder', help='The directory to save the files.', required=True, type=ph.is_valid_file)
    parser.add_argument('--seed', help='The seed to use.', type=is_none_or_non_negative_int)
    
    args = parser.parse_args()
    if args.perturb >= (1 / args.dimension):
        raise ValueError('The perturbation factor >= (1 / n), {}'.format(1 / args.dimension))
    rng = np.random.default_rng(args.seed)
    results = []
    i = 1
    for system, count in enumerate_systems(args, rng):
        start = time.time()
        scaled_system, scaled_solutions = ap.scale_system(system)
        system_approximation = ap.approximate(args.dimension, args.perturb, args.point_count, args.matrix_count, rng, scaled_system, scaled_solutions)
        results.append((count, system_approximation))
        print('System {}, {}'.format(i, time.time() - start))
        i += 1
    indices = [i for i in range(len(results))]
    counts, approximations = map(list, zip(*sorted(results)))
    plt.title('Approximating')
    plt.ylabel('Count')
    plt.plot(indices, normalize(np.array(approximations)), color='green', label='Aprroximation')
    if counts[0] is not None:
        plt.plot(indices, normalize(np.array(counts)), color='blue', label='Actual count')
    if args.results_folder:
        file_path = '{}-{}-{}-{}'.format(args.dimension, args.perturb, args.point_count, args.matrix_count)
        file_path = os.path.join(args.results_folder, file_path)
        plt.savefig('{}.png'.format(file_path))
        np.savez('{}.npz'.format(file_path), actual_counts=counts, approximations=approximations)
    else:
        plt.show()

if __name__ == '__main__':
    # cProfile.run('main()', 'cool-file2')
    main()