from python.scaling.doubly_stochastic_function import DoublyStochasticFunction

import numpy as np
import scipy as sp
import scipy.optimize as opt

class Scaler:
    """Scales a tuple of PSD matrices s.t. they are doubly-stochastic.
    """
    
    def __init__(self, system):
        """Initializes the Scaler.

        Args:
            system (N, N, N) array_like: The tuple of PSD matrices. 
        """
        
        self.n = system.shape[0]
        self.system = system
        
    def _scale_from_result(self, result):
        """Calculates the scaling values based off the result.

        Args:
            result (OptimizationResult): The optimization result from a minimization method.

        Returns:
            The optimization result and the scalars and T matrix that is needed to scale the system.
        """
        
        point = np.zeros(self.n)
        point[0:-1] = result.x
        point[-1] = np.sum(-1 * result.x)
        exp = np.exp(point)
        S = np.zeros((self.n, self.n))
        for i in range(self.n):
            S += exp[i] * self.system[i]
        sqrt_S = sp.linalg.sqrtm(S)
        return result, exp, np.linalg.inv(sqrt_S)
    
    def scale_optimized_bfgs(self, x0=None, options=None):
        """Minimizes the scaler function using BFGS on the hyperplane and returns relevant
        information needed to scale the given system to doubly-stochastic. Uses a more optimized
        function for BFGS minimization.

        Args:
            x0: (N - 1) array_like: The initial point to give to the minimizer. Defaults to origin.
            options: dictionary: The options to pass to the minimizer.

        Returns:
            The optimization result and the scalars and T matrix that is needed to scale the system.
        """
        
        jac_hess_p = DoublyStochasticFunction(self.system)
        x0 = np.zeros(self.n - 1) if x0 is None else x0
        result = opt.minimize(jac_hess_p.f_and_jac, x0, method='BFGS', jac=True, options=options)
        return self._scale_from_result(result)
    
    def scale_unoptimized_bfgs(self, x0=None, options=None):
        """Minimizes the scaler function using BFGS on the hyperplane and returns relevant
        information needed to scale the given system to doubly-stochastic.

        Args:
            x0: (N - 1) array_like: The initial point to give to the minimizer. Defaults to origin.
            options: dictionary: The options to pass to the minimizer.

        Returns:
            The optimization result and the scalars and T matrix that is needed to scale the system.
        """
        
        jac_hess_p = DoublyStochasticFunction(self.system)
        x0 = np.zeros(self.n - 1) if x0 is None else x0
        result = opt.minimize(jac_hess_p.f, x0, method='BFGS', jac=jac_hess_p.jac, options=options)
        return self._scale_from_result(result)
    
    def scale_newton_cg(self, x0=None, options=None, use_hess_p=False):
        """Minimizes the scaler function using Newton-CG on the hyperplane and returns relevant
        information needed to scale the given system to doubly-stochastic.

        Args:
            x0: (N - 1) array_like: The initial point to give to the minimizer. Defaults to origin.
            options: dictionary: The options to pass to the minimizer.

        Returns:
            The optimization result and the scalars and T matrix that is needed to scale the system.
        """
        
        jac_hess_p = DoublyStochasticFunction(self.system)
        x0 = np.zeros(self.n - 1) if x0 is None else x0
        result = opt.minimize(jac_hess_p.f, x0, method='Newton-CG', jac=jac_hess_p.jac, hessp=jac_hess_p.hess_p, options=options) if use_hess_p \
            else opt.minimize(jac_hess_p.f, x0, method='Newton-CG', jac=jac_hess_p.jac, hess=jac_hess_p.hess, options=options)
        return self._scale_from_result(result)

    def scale_system_bfgs(self, x0=None, options=None):
        """Scales the PSD system using the optimized BFGS. That way you don't have to scale the system yourself.

        Args:
            x0: (N - 1) array_like: The initial point to give to the minimizer. Defaults to origin.
            options: dictionary: The options to pass to the minimizer.

        Returns:
            The scaled system and scaled solutions.
        """
        
        exp, T, success = self.scale_optimized_bfgs(x0, options)
        if not success:
            return None
        solution_sum = np.sum(exp)
        solution_scale_factor = self.n / solution_sum
        scaled_solutions = solution_scale_factor * exp
        scaled_system = np.zeros(self.system.shape)
        for i in range(self.n):
            scaled_system[i] = exp[i] * (T @ self.system[i] @ T)
        return scaled_system, scaled_solutions