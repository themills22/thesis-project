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
from itertools import combinations
from python.nn.graph_net import GraphNet
from scipy.optimize import OptimizeResult
from scipy.special import comb
from tqdm import tqdm

class PowerFlowNetOptimizer:
    def __init__(self, model):
        self.model = model
        
    def f(self, x):
        self.model.zero_grad()
        x = torch.from_numpy(x).float()
        return -self.model.forward(x).item()
    
    def jac(self, x):
        self.model.zero_grad()
        x = torch.from_numpy(x).float().requires_grad_(True)
        output = self.model.forward(x)
        output.backward()
        return -x.grad.detach().cpu().numpy()
    

# todo: just make this constraint quadratic x_1^2 + ... + x_n^2 <= up^2
class Constraints:
    def __init__(self, upper_bound):
        self.last_x = (None, -1)
        self.last_norm = (None, -1)
        self.last_jac = (None, -1)
        self.constraints = [opt.NonlinearConstraint(self._f, 0, upper_bound, jac=self._jac, hess=self._hess)]
        
    def _get_version(self, x):
        """Gets the version of cached values to grab based off equality of x with last_x.

        Args:
            x (N - 1) array_like: The point to compare against last_x.

        Returns:
            The version of cached values to use.
        """
        
        return self.last_x[1] if self.last_x[0] is not None and all(x == self.last_x[0]) \
            else self.last_x[1] + 1
            
    
    def _get_x(self, version, x):
        """Gets the cached value of x to use or calculates it.

        Args:
            version (int) : The version to check against.
            x (N - 1) array_like : The x to use to compute the new x if required.

        Returns:
            (N) array_like : The x to use.
        """
        
        last_x, last_version = self.last_x
        if last_version == version:
            return last_x
        self.last_x = (x, version)
        return x
    
    def _get_norm(self, version, x):
        last_norm, last_version = self.last_norm
        if last_version == version:
            return last_norm
        new_norm = np.linalg.norm(x)
        self.last_norm = (new_norm, version)
        return new_norm
    
    def _get_jac(self, version, x, norm):
        last_jac, last_version = self.last_jac
        if last_version == version:
            return last_jac
        new_jac = x / norm
        self.last_jac = (new_jac, version)
        return new_jac
    
    def _f(self, x):
        version = self._get_version(x)
        x = self._get_x(version, x)
        return self._get_norm(version, x)
    
    def _jac(self, x):
        version = self._get_version(x)
        x = self._get_x(version, x)
        norm = self._get_norm(version, x)
        return self._get_jac(version, x, norm)
    
    def _hess(self, x, v):
        version = self._get_version(x)
        x = self._get_x(version, x)
        norm = self._get_norm(version, x)
        jac = self._get_jac(version, x, norm)
        hess = np.zeros((len(x), len(x)))
        for i in range(len(x)):
            hess[i, i] = norm - 2 * norm * jac[i]
        for i, j in combinations(range(len(x)), 2):
            hess[i, j] = -1 * norm * jac[i] * jac[j]
            hess[j, i] = hess[i, j]
        return v[0] * hess
    
def optimize(optimizer, constraints, x0, options):
    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=RuntimeWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        try:
            result = opt.minimize(optimizer.f, x0, method='trust-constr', jac=optimizer.jac, hess=opt.SR1(), \
                constraints=constraints, options=options)
            return OptimizeResult(x=result.x, success=result.success, status=result.status, message=result.message, \
                fun=-result.fun, jac=-result.jac[0][0], nfev=result.nfev, njev=result.njev, nit=result.nit)
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
    parser.add_argument('--input-norm-cap', help='What input parameters to discard based off their norm.', required=True, type=float)
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
        'xtol' : 1e-4,
        'gtol' : 1e-4,
        'maxiter' : 100
    }
    
    rng = np.random.default_rng()
    optimizer = PowerFlowNetOptimizer(model)
    constraints = Constraints(args.input_norm_cap).constraints
    results = Results(np.zeros((args.improved_system_cutoff, model.size)), np.zeros((args.improved_system_cutoff, model.size)), np.zeros(args.improved_system_cutoff))
    current_system = 0
    def loop_count():
        while current_system < args.improved_system_cutoff:
            yield
    for _ in tqdm(loop_count()):
        input = rng.standard_normal(size=model.size, dtype=np.float32)
        result = optimize(optimizer, constraints, input, optimize_options)
        system, count = result.x, result.fun
        if count <= args.count_cutoff:
            continue
        results.initial_systems[current_system] = input
        results.final_systems[current_system] = system
        results.solution_counts[current_system] = count
        current_system += 1
        if current_system % 1000 == 0:
            print('Good systems found: {}'.format(current_system))
        
    file_name = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.npz')
    np.savez(os.path.join(args.results_folder, file_name), systems=results.final_systems, solution_counts=results.solution_counts, \
        initial_systems=results.initial_systems)
            
if __name__ == '__main__':
    main()