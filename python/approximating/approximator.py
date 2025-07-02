import itertools as it
import numpy as np
import scipy as sp
import tqdm

from itertools import product
from python.scaling.scaler import Scaler

def scale_system(system):
    """Scales the given system of matrices to fit into the approximation scheme.

    Args:
        system (N, N, N) array_like: The system of matrices.

    Raises:
        ValueError: The system cannot be scaled.

    Returns:
        The scaled system of matrices and the scaled solutions.
    """
    n = system.shape[0]
    psd_system = np.array([matrix.T @ matrix for matrix in system])
    scaler = Scaler(psd_system)
    result, exp, T = scaler.scale_optimized_bfgs()
    if not result.success:
        raise ValueError('The given system cannot be scaled.')
    solution_sum = np.sum(exp)
    solution_scale_factor = n / solution_sum
    scaled_system = np.zeros(system.shape)
    scaled_solutions = np.zeros(n)
    for i in range(n):
        scaled_solutions[i] = solution_scale_factor * exp[i]
        scaled_system[i] = np.sqrt(scaled_solutions[i]) * (system[i] @ T)
    return scaled_system, scaled_solutions

def get_system_diagonals(system):
    dimension = len(system)
    system_diagonals = np.zeros((dimension, dimension))
    for i, j in product(range(dimension), range(dimension)):
        system_diagonals[i, j] = system[i, j, j]
    return system_diagonals

def approximate(system, system_solutions, perturbation_factor, point_count, level_set_count, rng, fail_cap):
    """Approximates the system with the given system.

    Args:
        system (N, N, N) array_like: The system of matrices.
        system_solutions (N) array_like: The solutions to the system of matrices.
        perturbation_factor (float64): The perturbation factor.
        point_count (integer): The number of points to sample.
        level_set_count (integer): The number of elements from the K_c "level set" to sample.
        rng (RNG): The RNG to use for sampling.

    Returns:
        float64: The approximation to the number of solutions the system possesses.
        Is not currently scaled by 2^n.
    """
    n = system.shape[0]
    sum = 0
    for _ in tqdm.tqdm(range(point_count), total=point_count):
        point = rng.normal(0, 1, n)
        for _ in range(level_set_count):
            # attempts = 0
            projected_system_tuples = project_system(system, perturbation_factor, system_solutions, point, rng, fail_cap)
            # while any(projected_matrix is None for _, projected_matrix in projected_system_tuples) and attempts < 10:
            #     attempts += 1
            #     point = rng.normal(0, 1, n)
            #     projected_system_tuples = project_system(system, perturbation_factor, system_solutions, point, rng, fail_cap)
            if any(projected_matrix is None for _, projected_matrix in projected_system_tuples):
                raise ValueError('System failed mega time')
            A = np.zeros((n, n))
            for i in range(n):
                projected_matrix = projected_system_tuples[i][1]
                A[i] = projected_matrix.T @ (projected_matrix @ point)
            sum += np.abs(np.linalg.det(A))
    return (2 ** n) * (sum / (point_count * level_set_count)) # note that I'm not scaling by 2^n here
        
def project_system(system, perturbation_factor, system_solutions, point, rng, fail_cap):
    """Performs the K_c(point) "level set" projection.

    Args:
        system (N, N, N) array_like: The system of matrices to project.
        system_solutions (N) array_like: The solutions to the system of matrices.
        point (N) array_like: The point in the K_c(point) set.
        rng (RNG): The RNG to use.

    Raises:
        ValueError: The projection scheme leads to a matrix with complex elements.

    Returns:
        (N, N, N): The projected system of matrices to perform the approximation
        calculation on.
    """
    n = system.shape[0]
    point_norm = np.dot(point, point)
    projected_system_tuples = [] 
    for i in range(n):
        projected_matrix_tuple = _try_project_matrix(system[i], perturbation_factor, system_solutions[i], \
            point, point_norm, rng, fail_cap)
        projected_system_tuples.append(projected_matrix_tuple)
    return projected_system_tuples

def _try_project_matrix(matrix, perturbation_factor, solution, point, point_norm, rng, fail_cap):
    random_matrix = matrix + perturbation_factor * rng.normal(0, 1, matrix.shape)
    for i in range(fail_cap):
        projected_matrix = _project_matrix(random_matrix, solution, point, point_norm, rng)
        if projected_matrix is not None:
            return (i, projected_matrix)
        random_matrix = matrix + perturbation_factor * rng.normal(0, 1, matrix.shape)
    return (fail_cap, None)

def _project_matrix(matrix, solution, point, point_norm, rng):
    D = matrix @ point
    d = np.dot(D, D) - solution
    B = matrix.T + matrix
    b = np.dot(point, B @ point) / point_norm
    value = ((b ** 2) / 4) - (d / point_norm)
    if value < 0:
        return None
    l = -1 * (b / 2) + rng.choice([-1, 1]) * np.sqrt(value)
    projected_matrix = np.array(matrix)
    projected_matrix[np.diag_indices(projected_matrix.shape[0])] += l
    return projected_matrix

class SystemCache:
    def __init__(self, system, count):
        self.count = count
        self.dimension = len(system)
        self.B_scaled_system, self.scaled_solutions = scale_system(system)
        self.B_psd_system_diagonals = [None for _ in range(self.dimension)]
        
    def get_B_diagonals(self, i):
        if self.B_psd_system_diagonals[i] is not None:
            return self.B_psd_system_diagonals[i]
        self.B_psd_system_diagonals[i] = np.array([np.dot(B[:, i], B[:, i]) for B in self.B_scaled_system])
        return self.B_psd_system_diagonals[i]

class RandomSystemCache:
    def __init__(self, system_cache, random_system, x):
        self.system_cache = system_cache
        self.A_system = self.system_cache.B_scaled_system + random_system 
        self.A_psd_system_diagonals = [None for _ in range(self.system_cache.dimension)]
        
        self.x = x
        self.partial_results = np.array([A.T @ (A @ x) for A in self.A_system])
        self.results = x.T @ self.partial_results.T
        
        self.x_squared = np.power(x, 2)
        self.x_squared_inverse = 1 / self.x_squared
        self.special_index = self._get_special_index()
        self.G = [None for _ in range(self.system_cache.dimension)]
        
    def _get_special_index(self):
        sorted = np.sort(self.x_squared)
        start_index = next((i for i in range(self.system_cache.dimension) if sorted[i] >= 1 / 2), None)
        if start_index is None:
            return self.system_cache.dimension - 1
        entry_count = self.system_cache.dimension - start_index
        median_index = start_index + (entry_count // 2 - (1 if entry_count % 2 == 0 else 0))
        return median_index
        valid = self.x_squared >= 1 / 2
        if all(~valid):
            return self.x_squared.argmax()
        valid_values = np.sort(np.extract(valid, self.x_squared))
        median = valid_values[len(valid_values) // 2 - (1 - len(valid_values) % 2)]
        return next(i for i in range(self.system_cache.dimension) if self.x_squared[i] == median)
    
    def _get_A_diagonals(self, i):
        if self.A_psd_system_diagonals[i] is not None:
            return self.A_psd_system_diagonals[i]
        self.A_psd_system_diagonals[i] = np.array([np.dot(A[:, i], A[:, i]) for A in self.A_system])
        return self.A_psd_system_diagonals[i]
    
    def get_g(self, i):
        if self.G[i] is not None:
            return self.G[i]
        x_A_diagonals = self._get_A_diagonals(i) * self.x_squared[i]
        self.G[i] = self.x_squared_inverse[i] * (self.system_cache.scaled_solutions - self.results + x_A_diagonals)
        return self.G[i]
    
    def get_coercion_vector(self, i):
        g = self.get_g(i)
        g_B_diagonals_diff = np.pow(g - self.system_cache.get_B_diagonals(i), 2)
        A_B_diagonals_diff = np.power(self._get_A_diagonals(i) - self.system_cache.get_B_diagonals(i), 2)
        w = (-g_B_diagonals_diff + A_B_diagonals_diff) / 2
        for i in range(self.system_cache.dimension):
            if w[i] < -50:
                w[i] = -50
            elif w[i] > 20:
                w[i] = 20
        w = np.exp(w)
        return self.x[i], w, g, g_B_diagonals_diff, A_B_diagonals_diff
        
    def get_coercion_matrix(self):
        yield from (self.get_coercion_vector(i) for i in range(self.system_cache.dimension))
            
    def get_approximation(self):
        _, w, _, _, _ = self.get_coercion_vector(self.special_index)
        log_gradient = 1 / self.x[self.special_index]
        D_x_G = np.array(self.partial_results) 
        D_x_G[:, self.special_index] += (self.system_cache.scaled_solutions - self.results) * log_gradient
        D_x_G *= -2 * self.x_squared_inverse[self.special_index]
        return np.abs(np.linalg.det(D_x_G)) * w.prod(), D_x_G, w