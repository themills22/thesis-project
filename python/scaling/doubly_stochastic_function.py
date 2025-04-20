import itertools as it
import numpy as np

class DoublyStochasticFunction:
    """Contains the methods to use for doubly-stochastic scaling.
    """
    
    def __init__(self, system):
        """Initializes the DoublyStochasticSystem

        Args:
            system (N, N, N) array_like: The system of PSD matrices.
        """
        
        self.n = system.shape[0]
        self.system = system
        self.last_x = (None, -1)
        self.last_exp_x = (None, -1)
        self.last_system_S = (None, -1)
        self.last_S = (None, -1)
        self.last_inverse_S = (None, -1)
        self.last_system_V = (None, -1)
        self.last_f = (None, -1)
        self.last_jac = (None, -1)
        self.last_hess = (None, -1)
        
    def _get_version(self, x):
        """Gets the version of cached values to grab based off equality of x with last_x.

        Args:
            x (N - 1) array_like: The point to compare against last_x.

        Returns:
            The version of cached values to use.
        """
        
        return self.last_x[1] if self.last_x[0] is not None and all(x == self.last_x[0][0:-1]) \
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
        new_x = np.zeros(self.n)
        new_x[0:-1] = x
        new_x[-1] = np.sum(-1 * x)
        self.last_x = (new_x, version)
        return new_x
    
    def _get_exp_x(self, version, x):
        """Gets the cached value of exp_x to use or calculates it.

        Args:
            version (int) : The version to check against.
            x (N) array_like : The x to use to compute the new exp_x if required.

        Returns:
            (N) array_like : The exp_x to use.
        """
        
        last_exp_x, last_version = self.last_exp_x
        if last_version == version:
            return last_exp_x
        new_exp_point = np.exp(x)
        self.last_exp_x = (new_exp_point, version)
        return new_exp_point
    
    def _get_system_S(self, version, exp_x):
        """Gets the cached value of system_S to use or calculates it.

        Args:
            version (int) : The version to check against.
            exp_x (N) array_like: The exp_x to use to compute the new system_S if required.

        Returns:
            (N, N, N) array_like : The system_S to use.
        """
        
        last_system_S, last_version = self.last_system_S
        if last_version == version:
            return last_system_S
        new_system_S = np.zeros(self.system.shape)
        for i in range(self.n):
            new_system_S[i] = exp_x[i] * self.system[i]
        self.last_system_S = (new_system_S, version)
        return new_system_S
    
    def _get_S(self, version, system_S):
        """Gets the cached value of S to use or calculates it.

        Args:
            version (int) : The version to check against.
            system_S (N, N, N) array_like : The system_S to use to compute the new S if required.

        Returns:
            (N, N) array_like : The S to use.
        """
        
        last_S, last_version = self.last_S
        if last_version == version:
            return last_S
        new_S = np.sum(system_S, 0)
        self.last_S = (new_S, version)
        return new_S
    
    def _get_inverse_S(self, version, S):
        """Gets the cached value of inverse_S to use or calculates it.

        Args:
            version (int) : The version to check against.
            S (N, N) array_like : The S to use to compute the new inverse_S if required.

        Returns:
            (N, N) array_like : The inverse_S to use.
        """
        
        last_inverse_S, last_version = self.last_inverse_S
        if last_version == version:
            return last_inverse_S
        new_inverse_S = np.linalg.inv(S)
        self.last_inverse_S = (new_inverse_S, version)
        return new_inverse_S
    
    def _get_system_V(self, version, system_S, inverse_S):
        """Gets the cached value of system_V to use or calculates it.

        Args:
            version (int) : The version to check against.
            system_S (N, N, N) array_like : The system_S to use to compute the new S if required.
            inverse_S (N, N) array_like : The inverse_S to use to compute the new system_V if required.

        Returns:
            (N - 1, N - 1, N - 1) array_like : The system_V to use.
        """
        
        last_system_V, last_version = self.last_system_V
        if last_version == version:
            return last_system_V
        system_E = system_S[0:-1] - system_S[-1]
        new_system_V = inverse_S @ system_E
        self.last_system_V = (new_system_V, version)
        return new_system_V
    
    def f(self, x):
        """Computes the value of the doubly-stochastic scaling function.

        Args:
            x (N - 1) array_like : The x to compute the value for.

        Returns:
            The value of the function at x.
        """
        
        version = self._get_version(x)
        last_f, last_version = self.last_f
        if last_version == version:
            return last_f
        x = self._get_x(version, x)
        exp_x = self._get_exp_x(version, x)
        system_S = self._get_system_S(version, exp_x)
        S = self._get_S(version, system_S)
        sign, absolute_logdet = np.linalg.slogdet(S)
        new_f = sign * absolute_logdet
        self.last_f = (new_f, version)
        return new_f
    
    def jac(self, x):
        """Computes the gradient of the doubly-stochastic scaling function.

        Args:
            x (N - 1) array_like : The x to compute the gradient for.

        Returns:
            (N - 1) array_like : The gradient of the function at x.
        """
        
        version = self._get_version(x)
        last_jac, last_version = self.last_jac
        if last_version == version:
            return last_jac
        x = self._get_x(version, x)
        exp_x = self._get_exp_x(version, x)
        system_S = self._get_system_S(version, exp_x)
        S = self._get_S(version, system_S)
        inverse_S = self._get_inverse_S(version, S)
        system_V = self._get_system_V(version, system_S, inverse_S)
        new_jac = np.array([np.trace(system_V[i]) for i in range(self.n - 1)])
        self.last_jac = (new_jac, version)
        return new_jac
    
    def hess(self, x):
        """Computes the hessian of the doubly-stochastic scaling function.

        Args:
            x (N - 1) array_like : The x to compute the gradient for.

        Returns:
            (N - 1, N - 1) array_like : The hessian of the function at x.
        """
        
        version = self._get_version(x)
        last_hess, last_version = self.last_hess
        if last_version == version:
            return last_hess
        x = self._get_x(version, x)
        exp_x = self._get_exp_x(version, x)
        system_S = self._get_system_S(version, exp_x)
        S = self._get_S(version, system_S)
        inverse_S = self._get_inverse_S(version, S)
        system_V = self._get_system_V(version, system_S, inverse_S)
        c = np.trace(inverse_S @ system_S[-1])
        hess = np.array([c - np.trace(system_V[i] @ system_V[i]) for i in range(self.n - 1)])
        hess = np.diag(hess)
        for i, j in it.combinations(np.arange(self.n - 1), 2):
            t = -1 * np.trace(system_V[i] @ system_V[j])
            hess[i][j] = c - t
            hess[j][i] = c - t
        self.last_hess = (hess, version)
        return hess
    
    def hess_p(self, x, p):
        """Computes the hessian-p multiplication of the doubly-stochastic scaling function.

        Args:
            x (N - 1) array_like : The x to compute the gradient for.
            p (N - 1) array_like : The p vector to multiply the hessian against.

        Returns:
            (N - 1) array_like : The hessian-p multiplication of the function at x.
        """
        
        # just going to assume that x and p are always different than a previous iteration
        version = self._get_version(x)
        x = self._get_x(version, x)
        exp_x = self._get_exp_x(version, x)
        system_S = self._get_system_S(version, exp_x)
        S = self._get_S(version, system_S)
        inverse_S = self._get_inverse_S(version, S)
        system_V = self._get_system_V(version, system_S, inverse_S)
        c = np.trace(inverse_S @ system_S[-1])
        hess_p = np.full(self.n - 1, c * np.sum(p))
        for i in range(self.n - 1):
            hess_p[i] += -1 * p[i] * np.trace(system_V[i] @ system_V[i])
        for i, j in it.combinations(np.arange(self.n - 1), 2):
            t = -1 * np.trace(system_V[i] @ system_V[j])
            hess_p[i] += p[j] * t
            hess_p[j] += p[i] * t
        return hess_p
    
    def f_and_jac(self, x):
        """Computes the value and gradient of the doubly-stochastic scaling function. This function
        gets to skip comparison against the last_x that all the other functions need.

        Args:
            x (N - 1): The x to compute the gradient for.

        Returns:
            (N - 1) array_like : The gradient of the function at x.
        """
        
        version = self.last_x[1] + 1
        x = self._get_x(version, x)
        exp_x = self._get_exp_x(version, x)
        system_S = self._get_system_S(version, exp_x)
        S = self._get_S(version, system_S)
        inverse_S = self._get_inverse_S(version, S)
        system_V = self._get_system_V(version, system_S, inverse_S)
        sign, absolute_logdet = np.linalg.slogdet(S)
        f = sign * absolute_logdet
        jac = np.array([np.trace(system_V[i]) for i in range(self.n - 1)])
        self.last_f = (f, version)
        self.last_jac = (jac, version)
        return f, jac