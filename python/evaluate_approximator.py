import python.approximating.approximator as ap
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import parser_helpers as ph
import time

from juliacall import Main as jl
from tqdm import tqdm

def _normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def _evaluate(rng, num_systems, size, perturb, point_count, matrix_count):
    systems = rng.normal(0, 1, (num_systems, size, size, size))
    results = np.zeros((num_systems, 4))
    julia_start = time.time()
    results[:, 0] = np.array(jl.judge_matrix_systems(systems))
    julia_time_taken = time.time() - julia_start

    for i in tqdm(range(num_systems)):
        scale_start = time.time()    
        scaled_system, scaled_solutions = ap.scale_system(systems[i])
        scale_time_taken = time.time() - scale_start
        
        approximation_start= time.time()
        approximation = ap.approximate(size, perturb, point_count, matrix_count, rng, scaled_system, scaled_solutions)
        approximation_time_taken = time.time() - approximation_start
        
        results[i, 1:4] = [approximation, scale_time_taken, approximation_time_taken]
        
    return julia_time_taken, results

def evaluate(args):
    jl.seval("using PowerFlow: judge_matrix_systems")
    seed_sequence = np.random.SeedSequence(args.seed)
    rng = np.random.default_rng(seed_sequence)
    julia_time_taken, results = _evaluate(rng, args.num_systems, args.size, args.perturb, args.point_count, args.matrix_count)
    np.savez(args.results_path, julia_time=np.array([julia_time_taken]), results=results)
    
def evaluate_time(args):
    jl.seval("using PowerFlow: judge_matrix_systems")
    seed_sequence = np.random.SeedSequence(args.seed)
    rng = np.random.default_rng(seed_sequence)
    for size in args.sizes:
        perturb = 1 / (2 * size)
        point_count = 5 * (size ** 4)
        matrix_count = 2 * (size ** 2)
        julia_time_taken, results = _evaluate(rng, args.num_systems, size, perturb, point_count, matrix_count)
        print('Results for size {}:'.format(size))
        print('\tActual time: {}'.format(julia_time_taken / args.num_systems))
        print('\tApproximate time: {}'.format(results[:, 3].mean()))
    
def process_results(args):
    julia_time_taken, results = None, None
    with np.load(args.results_path) as data:
        julia_time_taken, results = data['julia_time'][0], data['results']
    indices = [x for x in range(1, len(results) + 1)]
    julia_results, approximate_results = zip(*sorted(zip(results[:, 0], results[:, 1])))
    julia_results, approximate_results = np.array(julia_results), np.array(approximate_results)
    normalized_julia_results, normalized_approximate_results = _normalize(julia_results), _normalize(approximate_results)
    
    print('Julia time average: {}'.format(julia_time_taken / len(results)))
    print('Approximate time average: {}'.format(results[:, 3].mean()))
    print('Scale time average: {}'.format(results[:, 2].mean()))
    
    plt.plot(indices, normalized_julia_results, label='Actual real solution count', linestyle='--')
    plt.plot(indices, normalized_approximate_results, label='Approximate real solution count')
    plt.xlabel('System')
    plt.ylabel('Normalized count')
    plt.title('Real solution count comparison')
    plt.legend()
    if args.plot_path is not None:
        plt.savefig(args.plot_path, dpi=300)
        
    if args.display:
        plt.show()

def main():
    is_positive_int = lambda value: ph.check_greater_than_int(value, 0)
    is_none_or_non_negative_int = lambda value: not value or ph.check_greater_than_int(value, -1)
    is_valid_matrix_int = lambda value: ph.check_greater_than_int(value, 2)
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    
    subparser = subparsers.add_parser('evaluate', help='Run the approximator comparison')
    subparser.add_argument('--num-systems', help='The number of systems to evaluate on.', required=True, type=is_positive_int)
    subparser.add_argument('--size', help='The size of the matrix systems to generate.', required=True, type=is_valid_matrix_int)
    subparser.add_argument('--perturb', help='The perturbation factor to use', required=True, type=float)
    subparser.add_argument('--point-count', help='The number of points to sample.', required=True, type=is_positive_int)
    subparser.add_argument('--matrix-count', help='The number of entries to sample from the K_c set.', required=True, type=is_positive_int)
    subparser.add_argument('--results-path', help='The npz file to save the comparison results to.', required=True, type=str)
    subparser.add_argument('--seed', help='The seed to use.', type=is_none_or_non_negative_int)
    subparser.set_defaults(func=evaluate)
    
    subparser = subparsers.add_parser('evaluate-time', help='Compare time taken to "count" solutions.')
    subparser.add_argument('--num-systems', help='The number of systems to evaluate on.', required=True, type=is_positive_int)
    subparser.add_argument('--sizes', nargs='+', help='The size of the matrix systems to generate.', required=True, type=int)
    subparser.add_argument('--seed', help='The seed to use.', type=is_none_or_non_negative_int)
    subparser.set_defaults(func=evaluate_time)
    
    subparser = subparsers.add_parser('process-results', help='Process the comparison results')
    subparser.add_argument('--results-path', help='The comparison results file.', required=True, type=ph.is_valid_file)
    subparser.add_argument('--plot-path', help='The path to save the plot to.', required=False, type=str)
    subparser.add_argument('--display', help='Whether to display the plot.', action='store_true')
    subparser.set_defaults(func=process_results)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()