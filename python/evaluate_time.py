import argparse
import numpy as np
import python.approximating.approximator as ap
import time

def get_time_taken(function):
    start = time.time()
    value = function()
    end = time.time()
    return value, end - start

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--system-sizes', nargs='+', help='Sizes of the systems to evaluate', required=True, type=int)
    parser.add_argument('--system-count', help='Number of systems to evaluate for each system size', required=True, type=int)
    parser.add_argument('--sample-size', help='Number of samples to evaluate for each system size', required=True, type=int)
    
    args = parser.parse_args()
    rng = np.random.default_rng()
    for system_size in args.system_sizes:
        print('System size {}'.format(system_size))
        
        total_dimension = (system_size, system_size, system_size)
        scale_times = []
        point_times = []
        matrix_times = []
        for i in range(args.system_count):
            print('\tSystem index {}'.format(i + 1))
            
            system = rng.normal(0, 1, total_dimension)
            try:
                value, time_taken = get_time_taken(lambda: ap.scale_system(system))
                print('\t\tScaling time: {}'.format(time_taken))
                scale_times.append(time_taken)
                
                perturb = 1 / (system_size ** 2)
                scaled_system, scaled_solutions = value
                _, time_taken = get_time_taken(lambda: ap.approximate(system_size, perturb, args.sample_size, 1, rng, scaled_system, scaled_solutions))
                print('\t\tPoint time: {}'.format(time_taken))
                point_times.append(time_taken)
                
                _, time_taken = get_time_taken(lambda: ap.approximate(system_size, perturb, 1, args.sample_size, rng, scaled_system, scaled_solutions))
                print('\t\tMatrix time: {}'.format(time_taken))
                matrix_times.append(time_taken)
            except:
                print('\t\tException occurred')
        print('\tAverage scaling time: {}'.format(np.mean(scale_times)))
        print('\tAverage point time: {}'.format(np.mean(point_times)))
        print('\tAverage matrix time: {}'.format(np.mean(matrix_times)))

if __name__ == '__main__':
    main()