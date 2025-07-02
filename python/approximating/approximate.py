import approximator
import argparse
import cProfile
import matplotlib.pyplot as plt
import numpy as np
import os
import python.parser_helpers as ph

from itertools import product

def enumerate_systems(args, rng):
    total_dimension = (args.dimension, args.dimension, args.dimension)
    def enumerate_from_int():
        yield from ((-1, rng.normal(0, 1, total_dimension)) for _ in range(args.system_count))
        
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
                    
    systems_generator = enumerate_from_files() if args.data_path else enumerate_from_int()
    i = 0
    for system, count in systems_generator:
        yield approximator.SystemCache(system, count) 
        i += 1
        if i % 10 == 0:
            return
        
def main():
    is_positive_int = lambda value: ph.check_greater_than_int(value, 0)
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
    parser.add_argument('--compare-coercion', help='Compare the coercion values for all points.', action='store_true')
    parser.add_argument('--results-folder', help='The directory to save the files.', required=True, type=ph.is_valid_file)
    
    args = parser.parse_args()
    if args.perturb >= (1 / args.dimension):
        raise ValueError('The perturbation factor >= (1 / n), {}'.format(1 / args.dimension))
    rng = np.random.default_rng()
    total_dimension = (args.dimension, args.dimension, args.dimension)
    coercion_values = [] if args.compare_coercion else None
    results = []
    for system_cache in enumerate_systems(args, rng):
        system_approximation = 0
        for x in (rng.normal(0, 1, args.dimension) for _ in range(args.point_count)):
            w_sum = 0
            for random_system in (args.perturb * rng.normal(0, 1, total_dimension) for _ in range(args.matrix_count)):
                random_system_cache = approximator.RandomSystemCache(system_cache, random_system, x)
                if args.compare_coercion:
                    coercion_values.extend(random_system_cache.get_coercion_matrix())
                approximation, _, w = random_system_cache.get_approximation()
                w_sum += w.prod()
                system_approximation += approximation
        print(w_sum / args.matrix_count)
        system_approximation *= (1 / args.point_count) * (1 / args.matrix_count)
        results.append((system_cache.count, system_approximation))
    indices = [i for i in range(len(results))]
    counts, approximations = map(list, zip(*sorted(results)))
    approximations = 20 * np.array(approximations)
    print(approximations)
    print(counts)
    plt.title('Approximating')
    plt.ylabel('Count')
    plt.plot(indices, approximations, color='green', label='Aprroximation')
    if counts[0] is not None:
        plt.plot(indices, counts, color='blue', label='Actual count')
    plt.show()


if __name__ == '__main__':
    # cProfile.run('main()', 'cool-file2')
    main()