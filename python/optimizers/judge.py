import argparse
import numpy as np
import os
import python.parser_helpers as ph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+', help='The paths to judge.', type=ph.is_valid_file)
    parser.add_argument('--count-cutoff', help='The number of guessed solutions a system must meet to be saved.', \
        required=True, type=int)
    
    args = parser.parse_args()
    files = []
    for path in args.paths:
        if os.path.isdir(path):
            files.extend([os.path.join(path, file) for file in next(os.walk(path), (None, None, []))[2]])
        else:
            files.append(path)
    
    improved_count = 0
    for file in files:
        counts = None
        hooray = []
        with np.load(file) as npz_file:
            counts = zip(npz_file['initial_counts'], npz_file['solution_counts'], npz_file['guess_counts'])
        
        for initial_count, final_count, guess_count in counts:
            improved_count += 1 if final_count > initial_count else 0
            if final_count >= args.count_cutoff:
                hooray.append((initial_count, final_count, guess_count))
        
        if improved_count > 0:
            print('Improved {} entries in file {}'.format(improved_count, file))
            
        if len(hooray) > 0:
            print('>= {} found in file {}\n{}'.format(args.count_cutoff, file, hooray))

if __name__ == '__main__':
    main()