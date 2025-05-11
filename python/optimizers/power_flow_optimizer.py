import argparse
import datetime as dt
import math
import numpy as np
import os
import python.parser_helpers as ph
import scipy.optimize as opt
import torch
import warnings

from collections import namedtuple
from python.nn.graph_net import GraphNet
from scipy.optimize import OptimizeResult
from scipy.special import comb
from tqdm import tqdm

class PowerFlowNetOptimizer:
    def __init__(self, model):
        self.model = model
        
    def f_and_jac(self, x):
        self.model.zero_grad()
        x = torch.from_numpy(x).float().requires_grad_(True)
        output = self.model.forward(x)
        output.backward()
        return -output.item(), -x.grad.detach().cpu().numpy()
    
def optimize(optimizer, x0, options):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            result = opt.minimize(optimizer.f_and_jac, x0, method='BFGS', jac=True, options=options)
            return OptimizeResult(x=result.x, success=result.success, status=result.status, message=result.message, \
                fun=-result.fun, jac=-result.jac, nfev=result.nfev, njev=result.njev, nit=result.nit)
        except RuntimeWarning:
            return OptimizeResult(x=np.zeros(1), success=False, fun=0)
    
def main():
    Results = namedtuple('Results', ['initial_systems', 'final_systems', 'solution_counts'])
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', help='The network node size or matrix dimension.', required=True, type=int)
    parser.add_argument('--results-folder', help='The directory where the files to train on live.', required=True, type=ph.is_valid_file)
    parser.add_argument('--model-to-load', help='The model weights to start with.', required=True, type=ph.is_valid_file)
    parser.add_argument('--count-cutoff', help='The number of guessed solutions a system must meet to be saved.', \
        required=True, type=int)
    parser.add_argument('--improved-system-cutoff', help='The number of systems to save in a single file.', type=int, default=10000)
    parser.add_argument('--input-norm-cap', help='What input parameters to discard based off their norm.', type=float)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_kwargs = {
        'batch_size': 64,
        'shuffle': True
    }
    if device == 'cuda':
        cuda_kwargs = {
            'num_workers': 1,
            'pin_memory': True
        }
        model_kwargs.update(cuda_kwargs)

    model = GraphNet(int(comb(args.size, 2)))
    model.load_state_dict(torch.load(args.model_to_load))
    
    optimize_options = {
        'gtol' : 1e-4,
        'maxiter' : 1000
    }
    
    rng = np.random.default_rng()
    optimizer = PowerFlowNetOptimizer(model)
    results = Results(np.zeros((args.improved_system_cutoff, model.size)), np.zeros((args.improved_system_cutoff, model.size)), np.zeros(args.improved_system_cutoff))
    current_system = 0
    input_norm_cap = args.input_norm_cap if args.input_norm_cap else math.inf
    def loop_count():
        while current_system < args.improved_system_cutoff:
            yield
    for _ in tqdm(loop_count()):
        input = rng.standard_normal(size=model.size, dtype=np.float32)
        result = optimize(optimizer, input, optimize_options)
        system, count = result.x, result.fun
        if count <= args.count_cutoff or np.linalg.norm(system) >= input_norm_cap:
            continue
        results.initial_systems[current_system] = input
        results.final_systems[current_system] = system
        results.solution_counts[current_system] = count
        current_system += 1
        
    file_name = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.npz')
    np.savez(os.path.join(args.results_folder, file_name), systems=results.final_systems, solution_counts=results.solution_counts, \
        initial_systems=results.initial_systems)
            
if __name__ == '__main__':
    main()