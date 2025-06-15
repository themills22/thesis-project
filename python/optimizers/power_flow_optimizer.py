import argparse
import datetime as dt
import numpy as np
import os
import python.parser_helpers as ph
import scipy.optimize as opt
import torch
import warnings

from collections import namedtuple
from multiprocessing import Pool
from python.nn.graph_net import GraphNet
from scipy.optimize import OptimizeResult
from scipy.special import comb

Results = namedtuple('Results', ['initial_systems', 'final_systems', 'solution_counts'])
PoolOptions = namedtuple('PoolOptions', ['args', 'device', 'optimize_options', 'constraints', 'systems_cap', 'systems_cutoff'])

class PowerFlowNetFunction:
    """Auxiliary class to use for optimizing over a power-flow NN."""
    
    def __init__(self, model):
        """Initializes the function instance.

        Args:
            model : The model to use.
        """
        
        self.model = model
        
    def f(self, x):
        """Runs given input on the model.

        Args:
            x : The input.

        Returns:
            The output of the model.
        """
        
        self.model.zero_grad()
        x = torch.from_numpy(x).float()
        return -self.model.forward(x).item()
    
    def jac(self, x):
        """Gets the jacobian of the model for the given input.

        Args:
            x : The input.

        Returns:
            The jacobian of the model.
        """
        
        self.model.zero_grad()
        x = torch.from_numpy(x).float().requires_grad_(True)
        output = self.model.forward(x)
        output.backward()
        return -x.grad.detach().cpu().numpy()
    
    def f_and_jac(self, x):
        """Runs given input on the model and gets the output and jacobian.

        Args:
            x : The input.

        Returns:
            The output and jacobian of the model.
        """
        
        self.model.zero_grad()
        x = torch.from_numpy(x).float().requires_grad_(True)
        output = self.model.forward(x)
        output.backward()
        return -output.item(), -x.grad.detach().cpu().numpy()
    

class Constraints:
    """Represents a norm-constraint for an input."""
    
    def __init__(self, upper_bound):
        """Initializes the contstraints.

        Args:
            upper_bound : The bound for the norm-constraint.
        """
        self.last_x = (None, -1)
        self.last_norm_squared = (None, -1)
        self.last_jac = (None, -1)
        self.hess = None
        self.constraints = [opt.NonlinearConstraint(self._f, 0, upper_bound, jac=self._jac, hess=self._hess)]
        
    def _get_version(self, x):
        """Gets the version of cached values to grab based off equality of x with last_x.

        Args:
            x (N) array_like: The point to compare against last_x.

        Returns:
            The version of cached values to use.
        """
        
        return self.last_x[1] if self.last_x[0] is not None and all(x == self.last_x[0]) \
            else self.last_x[1] + 1
            
    
    def _get_x(self, version, x):
        """Gets the cached value of x to use or calculates it.

        Args:
            version (int) : The version to check against.
            x (N) array_like : The x to use to compute the value.

        Returns:
            (N) array_like : The x to use.
        """
        
        last_x, last_version = self.last_x
        if last_version == version:
            return last_x
        self.last_x = (x, version)
        return x
    
    def _get_norm_squared(self, version, x):
        """Gets the cached value of norm squared to use or calculates it.

        Args:
            version (int) : The version to check against.
            x (N) array_like : The x to use to compute the value.

        Returns:
            The norm squared.
        """
        
        last_norm_squared, last_version = self.last_norm_squared
        if last_version == version:
            return last_norm_squared
        new_norm_squared = np.linalg.norm(x) ** 2
        self.last_norm_squared = (new_norm_squared, version)
        return new_norm_squared
    
    def _get_jac(self, version, x):
        """Gets the cached value of the jacobian to use or calculates it.

        Args:
            version (int) : The version to check against.
            x (N) array_like : The x to use to compute the value.

        Returns:
            (N) array_like : The jacobian to use.
        """
        
        last_jac, last_version = self.last_jac
        if last_version == version:
            return last_jac
        new_jac = 2 * x
        self.last_jac = (new_jac, version)
        return new_jac
    
    def _f(self, x):
        """Computes the value of the function.

        Args:
            x (N) array_like : The x to use to compute the value.

        Returns:
            The norm-squared function value.
        """
        
        version = self._get_version(x)
        x = self._get_x(version, x)
        return self._get_norm_squared(version, x)
    
    def _jac(self, x):
        """Computes the value of the jacobian.

        Args:
            x (N) array_like : The x to use to compute the value.

        Returns:
            The jacobian of the norm-squared function.
        """
        
        version = self._get_version(x)
        x = self._get_x(version, x)
        return self._get_jac(version, x)
    
    def _hess(self, x, v):
        """Computes the value of the hessian.

        Args:
            x (N) array_like : The x to use to compute the value.
            v array_like : The magnitude (I don't really know what this is :)).

        Returns:
            The hessian of the norm-squared function.
        """
        
        if self.hess is None:
            self.hess = 2 * np.identity(len(x))
        return v[0] * self.hess
    
def save_npz_file(pool_id, results, folder):
    """Saves results to an NPZ file.

    Args:
        pool_id : The pool ID of the caller.
        results : The results to save.
        folder : The folder to save to.
    """
    
    file_name = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '-{}.npz'.format(pool_id)
    np.savez(os.path.join(folder, file_name), systems=results.final_systems, solution_counts=results.solution_counts, \
        initial_systems=results.initial_systems)
    
def optimize_constrained(optimizer, constraints, x0, options):
    """Optimizes the optimizer with the given constraints.

    Args:
        optimizer : The optimizer.
        constraints : The constraints.
        x0 : The initial point.
        options : The optimization options.

    Returns:
        The optimization results.
    """
    
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

def optimize(optimizer, x0, options):
    """Optimizes the optimizer unconstrainted.

    Args:
        optimizer : The optimizer.
        x0 : The initial point.
        options : The optimization options.

    Returns:
        The optimization results.
    """
    
    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=RuntimeWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        try:
            result = opt.minimize(optimizer.f_and_jac, x0, method='BFGS', jac=True, options=options)
            return OptimizeResult(x=result.x, success=result.success, status=result.status, message=result.message, \
                fun=-result.fun, jac=-result.jac, nfev=result.nfev, njev=result.njev, nit=result.nit)
        except RuntimeWarning:
            return OptimizeResult(x=np.zeros(1), success=False, fun=0)
        
def pool_main(options, pool_id):
    """The main function for each pool.

    Args:
        options : The options to use.
        pool_id : The pool ID.

    Returns:
        The optimization results.
    """
    
    model = GraphNet(int(comb(options.args.size, 2)))
    model.load_state_dict(torch.load(options.args.model_to_load))
    # model = model.to(options.device)
    
    rng = np.random.default_rng()
    optimizer = PowerFlowNetFunction(model)
    minimize = lambda input: optimize_constrained(optimizer, options.constraints, input, options.optimize_options) \
        if options.args.input_norm_squared_cap else optimize(optimizer, input, options.optimize_options)
    results = Results(np.zeros((options.systems_cutoff, model.size)), np.zeros((options.systems_cutoff, model.size)), np.zeros(options.systems_cutoff))
    current_system = 0
    while current_system < options.systems_cap:
        input = rng.normal(size=model.size, scale=10000)
        input = np.array(input, dtype=np.float32)
        result = minimize(input)
        system, count = result.x, result.fun
        if count <= options.args.count_cutoff:
            continue
        system_index = current_system % options.systems_cutoff
        results.initial_systems[system_index] = input
        results.final_systems[system_index] = system
        results.solution_counts[system_index] = count
        current_system += 1
        if current_system % options.systems_cutoff == 0:
            save_npz_file(pool_id, results, options.args.results_folder)
            results = Results(np.zeros((options.systems_cutoff, model.size)), np.zeros((options.systems_cutoff, model.size)), np.zeros(options.systems_cutoff))
    end_index = current_system % options.systems_cutoff
    return Results(results.initial_systems[0:end_index], results.final_systems[0:end_index], results.solution_counts[0:end_index])
            
def main():
    """Sets up the options to pass to the pool_main function."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', help='The network node size or matrix dimension.', required=True, type=int)
    parser.add_argument('--results-folder', help='The directory where the files to train on live.', required=True, type=ph.is_valid_file)
    parser.add_argument('--model-to-load', help='The model weights to start with.', required=True, type=ph.is_valid_file)
    parser.add_argument('--count-cutoff', help='The number of guessed solutions a system must meet to be saved.', \
        required=True, type=int)
    parser.add_argument('--systems-per-file', help='The number of systems to save in a single file.', type=int, default=10000)
    parser.add_argument('--total-systems', help='The number of files to create.', required=True, type=int)
    parser.add_argument('--input-norm-squared-cap', help='What input parameters to discard based off their norm. Uses BFGS if not provided.', type=float)
    parser.add_argument('--cpu-count', help='The number of CPUs to use for multiprocessing. Defaults to os.cpu_count() if not provided.', type=int)
    args = parser.parse_args()
    
    optimize_options = {
        'gtol' : 1e-4,
        'maxiter' : 100
    }
    
    if args.input_norm_squared_cap:
        optimize_options['xtol'] = 1e-4
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    constraints = Constraints(args.input_norm_squared_cap).constraints if args.input_norm_squared_cap else None # todo: this is wrong, there needs to be constraints instance for pool as the instance relies on cached values
    cpu_count = args.cpu_count if args.cpu_count else os.cpu_count()
    per_process_systems_count = int(args.total_systems / cpu_count)
    all_options = [PoolOptions(args, device, optimize_options, constraints, per_process_systems_count, args.systems_per_file) \
        for _ in range(cpu_count - 1)]
    last_count = per_process_systems_count + (args.total_systems - (per_process_systems_count * cpu_count))
    all_options.append(PoolOptions(args, device, optimize_options, constraints, last_count, args.systems_per_file))
    all_arguments = [tuple for tuple in zip(all_options, range(1, cpu_count + 1))]
    with Pool(cpu_count) as p:
        pool_results = p.starmap(pool_main, all_arguments)
        pool_results = Results( \
            np.vstack([result.initial_systems for result in pool_results]), \
            np.vstack([result.final_systems for result in pool_results]), \
            np.hstack([result.solution_counts for result in pool_results]))
        if len(pool_results.initial_systems) > 0:
            save_npz_file(0, pool_results, args.results_folder) # this save operation is allowed to ignore the systems-per-file setting because I'm lazy

        
if __name__ == '__main__':
    main()