from python.scaling.scaler import Scaler

import argparse
import itertools as it
import numpy as np
import os
import python.parser_helpers as ph
import time
import tqdm

is_positive_int = lambda value: ph.check_greater_than_int(value, 0)
is_none_or_non_negative_int = lambda value: not value or ph.check_greater_than_int(value, -1)
is_positive_float = lambda value: ph.check_greater_than_float(value, 0)
parser = argparse.ArgumentParser()
parser.add_argument('--start-size', help='The system size to start with. Must be at least 3.', required=True, type=lambda value: ph.check_greater_than_int(value, 2))
parser.add_argument('--stop-size', help='The system size to stop with. Must be >= --start-size.', required=True, type=is_positive_int)
parser.add_argument('--step', help='Step size to use when going to system of next size.', required=True, type=is_positive_int)
parser.add_argument('--attempts', help='The number of systems of a specific size to generate.', required=True, type=is_positive_int)
parser.add_argument('--directory', help='The directory to save the results to.', required=True, type=ph.is_valid_file)
parser.add_argument('--seed', help='The seed to use.', type=is_none_or_non_negative_int)
parser.add_argument('--newton-maxiter', help='The maximum number of iterations to pass to the Newton-CG minimizer', type=is_positive_int, default=5)
parser.add_argument('--ignore-newton', help='Whether to ignore the Newton-CG improvement.', action='store_true')
parser.add_argument('--display', help='Display graph of the results.', type=bool)

args = parser.parse_args()
if args.stop_size < args.start_size:
    raise ValueError("The stop-size <= start-size. {args.stop_size} <= {args.start_size}")

seed_sequence = np.random.SeedSequence(args.seed)
rng = np.random.default_rng(seed_sequence)
n_range = np.arange(args.start_size, args.stop_size + 1, args.step)
indices = {n_range[i]:i for i in range(n_range.shape[0])}
bfgs_results = np.zeros((n_range.shape[0], args.attempts, 4))
newton_results = np.zeros((n_range.shape[0], args.attempts, 5))
newton_options = {
    'xtol' : 1e-10,
    'maxiter' : args.newton_maxiter,
    }

def get_results(scaler, scaler_type, x0):
    # Note that the Newton-CG method has precision issues and frequently fails :(
    # Luckily, the result is still usable as its x value comes from the BFGS scaling.
    start = time.time()
    result, exp, T = scaler.scale_optimized_bfgs() if scaler_type == 'BFGS' \
        else scaler.scale_newton_cg(x0, newton_options, True)
    end = time.time()
    scaled_system = np.zeros(scaler.system.shape)
    traces = np.zeros(scaler.n)
    for j in range(scaler.n):
        scaled_system[j] = exp[j] * (T @ scaler.system[j] @ T)
        traces[j] = np.trace(scaled_system[j])
    summation = np.sum(scaled_system, axis=0)
    time_taken = end - start
    traces_diff_norm = np.linalg.norm(traces - np.ones(scaler.n))
    summation_norm = np.linalg.norm(summation - np.identity(scaler.n))
    radius = np.linalg.norm(exp)
    return result, time_taken, traces_diff_norm, summation_norm, radius

for n, i in tqdm.tqdm(it.product(n_range, range(args.attempts)), total=bfgs_results.shape[0] * bfgs_results.shape[1]):
    system = rng.normal(0, 1, (n, n, n))
    psd_system = np.array([A.T @ A for A in system])
    scaler = Scaler(psd_system)
    
    result, time_taken, traces_diff_norm, summation_norm, radius = get_results(scaler, 'BFGS', None)
    bfgs_results[indices[n], i] = np.array([time_taken, traces_diff_norm, summation_norm, radius])
    
    if not args.ignore_newton:
        result, time_taken, traces_diff_norm, summation_norm, radius = get_results(scaler, 'Newton-CG', result.x)
        newton_results[indices[n], i] = np.array([time_taken, traces_diff_norm, summation_norm, radius, \
            1 if result.success or result.nit == args.newton_maxiter else 0])

file = 'bfgs.{0}.{1}.{2}.{3}.{4}'.format(args.start_size, args.stop_size, args.step, \
    args.attempts, seed_sequence.entropy)
np.savez(os.path.join(args.directory, file), time_taken=bfgs_results[:, :, 0], traces_diff_norm=bfgs_results[:, :, 1], \
    summation_norm=bfgs_results[:, :, 2], radius=bfgs_results[:, :, 3])

if not args.ignore_newton:
    file = 'newton-cg.{0}.{1}.{2}.{3}.{4}'.format(args.start_size, args.stop_size, args.step, \
        args.attempts, seed_sequence.entropy)
    np.savez(os.path.join(args.directory, file), time_taken=newton_results[:, :, 0], traces_diff_norm=newton_results[:, :, 1], \
        summation_norm=newton_results[:, :, 2], radius=newton_results[:, :, 3], successes=newton_results[:, :, 4])