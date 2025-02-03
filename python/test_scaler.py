import argparse
import itertools as it
import matplotlib.pyplot as plt
import numpy as np
import os
import parser_helpers as ph
import scaler
import time
import tqdm

is_positive_int = lambda value: ph.check_greater_than_int(value, 0)
is_none_or_non_negative_int = lambda value: not value or ph.check_greater_than_int(value, -1)
parser = argparse.ArgumentParser()
parser.add_argument('--start-size', help='The system size to start with. Must be at least 3.', required=True, type=lambda value: ph.check_greater_than_int(value, 2))
parser.add_argument('--stop-size', help='The system size to stop with. Must be >= --start-size.', required=True, type=is_positive_int)
parser.add_argument('--step', help='Step size to use when going to system of next size.', required=True, type=is_positive_int)
parser.add_argument('--attempts', help='The number of systems of a specific size to generate.', required=True, type=is_positive_int)
parser.add_argument('--directory', help='The directory to save the results to.', required=True, type=ph.is_valid_dir)
parser.add_argument('--seed', help='The seed to use.', type=is_none_or_non_negative_int)
parser.add_argument('--display', help='Display graph of the results.', type=bool)

args = parser.parse_args()
if args.stop_size < args.start_size:
    raise ValueError("The stop-size <= start-size. {args.stop_size} <= {args.start_size}")

# things to average:
# * time taken to minimize scaler function
# * "accuracy" of traces

# * "accuracy" of identity
seed_sequence = np.random.SeedSequence(args.seed)
rng = np.random.default_rng(seed_sequence)
n_range = np.arange(args.start_size, args.stop_size + 1, args.step)
indices = {n_range[i]:i for i in range(n_range.shape[0])}
results = np.zeros((n_range.shape[0], args.attempts, 4))
for n, i in tqdm.tqdm(it.product(n_range, range(args.attempts)), total=results.shape[0] * results.shape[1]):
    system = rng.normal(0, 1, (n, n, n))
    psd_system = np.array([A.T @ A for A in system])
    
    start = time.time()
    exp, T, success = scaler.scale_hyperplane_reduced_system(psd_system)
    end = time.time()
    if not success:
        # figure out something smarter to do here
        print('System failed on iteration n={0}, i={1}'.format(n, i))
        continue
    
    scaled_system = np.zeros(system.shape)
    traces = np.zeros(n)
    for j in range(n):
        scaled_system[j] = exp[j] * (T @ psd_system[j] @ T)
        traces[j] = np.trace(scaled_system[j])
    summation = np.sum(scaled_system, axis=0)
    
    time_taken = end - start
    traces_diff_norm = np.linalg.norm(traces - np.ones(n))
    summation_norm = np.linalg.norm(summation - np.identity(n))
    radius = np.linalg.norm(exp)
    results[indices[n], i] = np.array([time_taken, traces_diff_norm, summation_norm, radius])

file = '{0}.{1}.{2}.{3}.{4}.npy'.format(args.start_size, args.stop_size, args.step, args.attempts, seed_sequence.entropy)
file = os.path.join(args.directory, file)
np.save(file, results)

averaged_results = np.average(results, axis=1)

figure, (time_plot, trace_plot, identity_plot, radius_plot) = plt.subplots(4)
figure.suptitle('Scaler results with attempts={0}'.format(args.attempts))

time_plot.set_title('Average time taken to minimize')
time_plot.set(xlabel='System size', ylabel='Time (s)')
time_plot.plot(n_range, averaged_results[:, 0])

trace_plot.set_title('Average distance of traces-vector from 1-vector')
trace_plot.set(xlabel='System size', ylabel='Distance')
trace_plot.plot(n_range, averaged_results[:, 1])

identity_plot.set_title('Average distance of scaled-summation matrix from identity matrix')
identity_plot.set(xlabel='System size', ylabel='Distance')
identity_plot.plot(n_range, averaged_results[:, 2])

radius_plot.set_title('Average radius of scaler solution point')
radius_plot.set(xlabel='System size', ylabel='radius')
radius_plot.plot(n_range, averaged_results[:, 3])

file = '{0}.{1}.{2}.{3}.{4}.png'.format(args.start_size, args.stop_size, args.step, args.attempts, seed_sequence.entropy)
file = os.path.join(args.directory, file)
plt.savefig(file)
if args.display:
    plt.show()