import itertools as it
import matplotlib.pyplot as plt
import numpy as np
import scaler
import time
import tqdm

initial_n = 3
final_n = 100
step = 1
attempts = 20

# things to average:
# * time taken to minimize scaler function
# * "accuracy" of traces
# * "accuracy" of identity
n_range = np.arange(initial_n, final_n + 1, step)
rng = np.random.default_rng(1234567)
results = np.zeros((n_range.shape[0], attempts, 3))
for n, i in tqdm.tqdm(it.product(n_range, range(attempts)), total=results.shape[0] * results.shape[1]):
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
    results[n - initial_n, i] = np.array([time_taken, traces_diff_norm, summation_norm])

averaged_results = np.average(results, axis=1)

figure, (time_plot, trace_plot, identity_plot) = plt.subplots(3)
figure.suptitle('Scaler results with attempts={0}'.format(attempts))

time_plot.set_title('Average time taken to minimize')
time_plot.set(xlabel='System size', ylabel='Time (s)')
time_plot.plot(n_range, averaged_results[:, 0])

trace_plot.set_title('Average distance of traces-vector from 1-vector')
trace_plot.set(xlabel='System size', ylabel='Distance')
trace_plot.plot(n_range, averaged_results[:, 1])

identity_plot.set_title('Average distance of scaled-summation matrix from identity matrix')
identity_plot.set(xlabel='System size', ylabel='Distance')
identity_plot.plot(n_range, averaged_results[:, 2])

plt.show()