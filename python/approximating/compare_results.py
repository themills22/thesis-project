import python.approximating.approximator as approximator
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import parser_helpers as ph
import read_julia_data as rjd
import time
import tqdm

is_positive_int = lambda value: ph.check_greater_than_int(value, 0)
is_none_or_non_negative_int = lambda value: not value or ph.check_greater_than_int(value, -1)
is_valid_matrix_int = lambda value: ph.check_greater_than_int(value, 2)
parser = argparse.ArgumentParser()
parser.add_argument('--perturbation-factor', help='The perturbation factor to use', required=True, type=float)
parser.add_argument('--point-count', help='The number of points to sample.', required=True, type=is_positive_int)
parser.add_argument('--level-set-count', help='The number of entries to sample from the K_c set.', required=True, type=is_positive_int)
parser.add_argument('--matrix-file', help='The file that contains the matrices to compare against.', required=True, type=ph.is_valid_file)
parser.add_argument('--matrix-size', help='The size of the matrix system to pull from the matrix directory', required=True, type=is_valid_matrix_int)
parser.add_argument('--save-directory', help='The directory to save the comparison results to.', required=True, type=ph.is_valid_file)
parser.add_argument('--seed', help='The seed to use.', type=is_none_or_non_negative_int)
parser.add_argument('--display', help='Display graph of the results.', type=bool)
parser.add_argument('--perturbation-description', \
    help='The description of the perturbation factor to use in place of the perturbation factor in the saved file.', \
    type=str)

# args = parser.parse_args()
# perturbation_bound = 1 / args.matrix_size
# if args.perturbation_factor >= perturbation_bound:
#     raise ValueError("The perturbation factor >= (1 / n). {args.perturbation_factor} >= {perturbation_bound}")

seed_sequence = np.random.SeedSequence()
rng = np.random.default_rng(seed_sequence)
file = 'D:\\deep-reinforcement-learning\\thesis-project\\matrices\\10_dim.txt'
systems, solution_counts = rjd.read_file(file, 10)
system_indices = list(range(systems.shape[0]))
results = {}
def approximate_system(index, perturbation_factor):  
    scale_start = time.time()    
    scaled_system, scaled_solutions = approximator.scale_system(systems[index])
    scale_time_taken = time.time() - scale_start
    
    approximation_start= time.time()
    approximation = approximator.approximate(scaled_system, scaled_solutions, perturbation_factor, \
        100, 100, rng, 10000)
    approximation_time_taken = time.time() - approximation_start
    
    results[index] = {
        'actual' : int(solution_counts[index]),
        'approximation' : approximation,
        'scale_time' : scale_time_taken,
        'approximation_time' : approximation_time_taken
    }
    
n = 10
approximate_system(0, 1 / (n ** 1.3))
approximate_system(8, 1 / (n ** 1.3))
approximate_system(10, 1 / (n ** 1.3))
# approximate_system(11, 1 / (n ** 1.3))
with open('results.json', 'w') as file:
    file.write(json.dumps(results, indent=4))

# for i in tqdm.tqdm(system_indices):
#     print('index: {0}'.format(i))
#     system = systems[i]
    
#     scale_start = time.time()    
#     scaled_system, scaled_solutions = approximator.scale_system(system)
#     scale_time_taken = time.time() - scale_start
    
#     approximation_start= time.time()
#     approximation = approximator.approximate(scaled_system, scaled_solutions, args.perturbation_factor, \
#         args.point_count, args.level_set_count, rng)
#     approximation_time_taken = time.time() - approximation_start

# figure, (actual_plot, approximation_plot, scale_time_plot, approximation_time_plot) = plt.subplots(4)
# figure.suptitle('Approximation results with point-count={0}, level-set-count={1}' \
#     .format(args.point_count, args.level_set_count))

# actual_plot.set_title('Actual reals solution count')
# actual_plot.set(xlabel='Matrix index', ylabel='Count of reals solutions')
# actual_plot.plot(system_indices, results[:, 0])

# approximation_plot.set_title('Approximated reals solution count')
# approximation_plot.set(xlabel='Matrix index', ylabel='Approximate count of reals solutions')
# approximation_plot.plot(system_indices, results[:, 1])

# scale_time_plot.set_title('Time taken to scale')
# scale_time_plot.set(xlabel='Matrix index', ylabel='Time (s)')
# scale_time_plot.plot(system_indices, results[:, 2])

# approximation_time_plot.set_title('Time taken to approximate')
# approximation_time_plot.set(xlabel='Matrix index', ylabel='Time (s)')
# approximation_time_plot.plot(system_indices, results[:, 3])

# perturbation_description = args.perturbation_description \
#     if not args.perturbation_description else np.format_float_scientific(args.perturbation_factor)
# results_file = '{0}.{1}.{2}.{3}.{4}.npy'.format(perturbation_description, args.point_count, args.level_set_count, \
#     args.matrix_size, seed_sequence.entropy)
# results_file = os.path.join(args.save_directory, results_file)
# np.save(results_file, results)

# plot_file = '{0}.{1}.{2}.{3}.{4}.png'.format(perturbation_description, args.point_count, args.level_set_count, \
#     args.matrix_size, seed_sequence.entropy)
# plot_file = os.path.join(args.save_directory, plot_file)
# plt.savefig(plot_file)

# json_data = {
#     'perturbation_factor': args.perturbation_factor,
#     'point_count': args.point_count,
#     'level_set_count': args.level_set_count,
#     'matrix_file': args.matrix_file,
#     'matrix_size': args.matrix_size,
#     'seed_entropy': seed_sequence.entropy,
#     'results_file': results_file,
#     'plot_file': plot_file
# }
# json_file = '{0}.{1}.{2}.{3}.{4}.json'.format(perturbation_description, args.point_count, args.level_set_count, \
#     args.matrix_size, seed_sequence.entropy)
# json_file = os.path.join(args.save_directory, json_file)
# if args.display:
#     plt.show()