import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import python.parser_helpers as ph

def get_bfgs_results(file):
    npz_file = np.load(file)
    return npz_file['time_taken'], npz_file['traces_diff_norm'], npz_file['summation_norm']

def get_newton_results(file):
    npz_file = np.load(file)
    return npz_file['time_taken'], npz_file['traces_diff_norm'], npz_file['summation_norm'], npz_file['successes']

def get_info_from_file(file):
    file = os.path.basename(file)
    info = file.split('.')
    optimization_method = info[0]
    system_sizes = np.arange(int(info[1]), int(info[2]) + 1, int(info[3]))
    attempts = int(info[4])
    seed_sequence = np.random.SeedSequence(int(info[5]))
    return optimization_method, system_sizes, attempts, seed_sequence

def get_averages(axis, *args):
    averages = np.array([np.average(arg, axis) for arg in args])
    return tuple(averages)

def display_plots(system_sizes, time_taken, traces_diff_norm, summation_norm):
    def display_plot(title, y_label, data):
        plt.suptitle(title)
        plt.xlabel('System size')
        plt.ylabel(y_label)
        plt.plot(system_sizes, data)
        plt.show()
        plt.clf()
    
    display_plot('Average time taken to minimize', 'Time (s)', time_taken)
    display_plot('Average distance of traces-vector from 1-vector', 'Distance', traces_diff_norm)
    display_plot('Average distance of scaled-summation matrix from identity matrix', 'Distance', summation_norm)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bfgs', help='The BFGS optimization file.', required=True, type=ph.is_valid_file)
    parser.add_argument('--newton-cg', help='The Newton-CG optimization file.', required=False, type=ph.is_valid_file)
    parser.add_argument('--display', help='Display the BFGS optimization graphs', type=bool)
    parser.add_argument('--output-file', help='File path to save Newton results csv file.', required=False, type=str)
    args = parser.parse_args()
    
    _, system_sizes, attempts, _ = get_info_from_file(args.bfgs)
    bfgs_time_taken, bfgs_traces_diff_norm, bfgs_summation_norm = get_bfgs_results(args.bfgs)
    if args.display:
        bfgs_time_taken_averages, bfgs_traces_diff_norm_averages, bfgs_summation_norm_averages = get_averages(1, bfgs_time_taken, bfgs_traces_diff_norm, bfgs_summation_norm)
        display_plots(system_sizes, bfgs_time_taken_averages, bfgs_traces_diff_norm_averages, bfgs_summation_norm_averages)
    
    if not args.newton_cg:
        return
    
    newton_file = 'D:\\deep-reinforcement-learning\\thesis-project\\results\\test-scaler\\newton-cg.10.250.20.50.226502390332954714580540253289295598598.npz'
    newton_time_taken, newton_traces_diff_norm, newton_summation_norm, newton_successes = get_newton_results(newton_file)
    
    headers = ['system_size', 'bfgs_time_taken', 'bfgs_traces_diff_norm', 'bfgs_summation_diff_norm', \
        'newton_time_taken', 'newton_traces_diff_norm', 'newton_summation_diff_norm', 'success_rate']
    results = np.zeros((len(system_sizes), len(headers)))
    for i in range(len(system_sizes)):
        results[i, 0] = system_sizes[i]
        systems_of_interest = (newton_successes[i] == 1).nonzero()[0]
        bfgs_tt, bfgs_tdn, bfgs_sn = bfgs_time_taken[i, systems_of_interest], bfgs_traces_diff_norm[i, systems_of_interest], bfgs_summation_norm[i, systems_of_interest]
        bfgs_tta, bfgs_tdna, bfgs_sdna = get_averages(None, bfgs_tt, bfgs_tdn, bfgs_sn)
        results[i, 1:4] = [bfgs_tta, bfgs_tdna, bfgs_sdna]
        
        newton_tt, newton_tdn, newton_sn = newton_time_taken[i, systems_of_interest], newton_traces_diff_norm[i, systems_of_interest], newton_summation_norm[i, systems_of_interest]
        newton_tta, newton_tdna, newton_sdna, success_rate = get_averages(None, newton_tt, newton_tdn, newton_sn, newton_successes[i])
        results[i, 4:8] = [newton_tta, newton_tdna, newton_sdna, success_rate]
    
    output_file = args.output_file if args.output_file else 'newton.csv'
    df = pd.DataFrame(results)
    df.to_csv(output_file, header=headers, index=False)

if __name__ == '__main__':
    main()