import itertools as it
import numpy as np
import scipy as sp

from itertools import product
from numba import njit
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

@njit
def get_system_diagonals(system, i):
    dimension = len(system)
    diagonals = np.zeros(dimension)
    for j, A in zip(range(dimension), system):
        a = np.ascontiguousarray(A[:, i])
        diagonals[j] = np.dot(a, a)
    return diagonals

# def approximate(system, system_solutions, perturbation_factor, point_count, level_set_count, rng, fail_cap):
#     """Approximates the system with the given system.

#     Args:
#         system (N, N, N) array_like: The system of matrices.
#         system_solutions (N) array_like: The solutions to the system of matrices.
#         perturbation_factor (float64): The perturbation factor.
#         point_count (integer): The number of points to sample.
#         level_set_count (integer): The number of elements from the K_c "level set" to sample.
#         rng (RNG): The RNG to use for sampling.

#     Returns:
#         float64: The approximation to the number of solutions the system possesses.
#         Is not currently scaled by 2^n.
#     """
#     n = system.shape[0]
#     sum = 0
#     for _ in tqdm.tqdm(range(point_count), total=point_count):
#         point = rng.normal(0, 1, n)
#         for _ in range(level_set_count):
#             # attempts = 0
#             projected_system_tuples = project_system(system, perturbation_factor, system_solutions, point, rng, fail_cap)
#             # while any(projected_matrix is None for _, projected_matrix in projected_system_tuples) and attempts < 10:
#             #     attempts += 1
#             #     point = rng.normal(0, 1, n)
#             #     projected_system_tuples = project_system(system, perturbation_factor, system_solutions, point, rng, fail_cap)
#             if any(projected_matrix is None for _, projected_matrix in projected_system_tuples):
#                 raise ValueError('System failed mega time')
#             A = np.zeros((n, n))
#             for i in range(n):
#                 projected_matrix = projected_system_tuples[i][1]
#                 A[i] = projected_matrix.T @ (projected_matrix @ point)
#             sum += np.abs(np.linalg.det(A))
#     return (2 ** n) * (sum / (point_count * level_set_count)) # note that I'm not scaling by 2^n here
        
# def project_system(system, perturbation_factor, system_solutions, point, rng, fail_cap):
#     """Performs the K_c(point) "level set" projection.

#     Args:
#         system (N, N, N) array_like: The system of matrices to project.
#         system_solutions (N) array_like: The solutions to the system of matrices.
#         point (N) array_like: The point in the K_c(point) set.
#         rng (RNG): The RNG to use.

#     Raises:
#         ValueError: The projection scheme leads to a matrix with complex elements.

#     Returns:
#         (N, N, N): The projected system of matrices to perform the approximation
#         calculation on.
#     """
#     n = system.shape[0]
#     point_norm = np.dot(point, point)
#     projected_system_tuples = [] 
#     for i in range(n):
#         projected_matrix_tuple = _try_project_matrix(system[i], perturbation_factor, system_solutions[i], \
#             point, point_norm, rng, fail_cap)
#         projected_system_tuples.append(projected_matrix_tuple)
#     return projected_system_tuples

# def _try_project_matrix(matrix, perturbation_factor, solution, point, point_norm, rng, fail_cap):
#     random_matrix = matrix + perturbation_factor * rng.normal(0, 1, matrix.shape)
#     for i in range(fail_cap):
#         projected_matrix = _project_matrix(random_matrix, solution, point, point_norm, rng)
#         if projected_matrix is not None:
#             return (i, projected_matrix)
#         random_matrix = matrix + perturbation_factor * rng.normal(0, 1, matrix.shape)
#     return (fail_cap, None)

# def _project_matrix(matrix, solution, point, point_norm, rng):
#     D = matrix @ point
#     d = np.dot(D, D) - solution
#     B = matrix.T + matrix
#     b = np.dot(point, B @ point) / point_norm
#     value = ((b ** 2) / 4) - (d / point_norm)
#     if value < 0:
#         return None
#     l = -1 * (b / 2) + rng.choice([-1, 1]) * np.sqrt(value)
#     projected_matrix = np.array(matrix)
#     projected_matrix[np.diag_indices(projected_matrix.shape[0])] += l
#     return projected_matrix

class PointCache:
    def __init__(self, x):
        self.x = x
        self.dimension = len(self.x)
        self.x_squared = np.power(x, 2)
        self.special_index = self._get_special_index()
        self.x_squared_inverse = 1 / self.x_squared
        
    def _get_special_index(self):
        sorted = np.sort(self.x_squared)
        start_index = next((i for i in range(self.dimension) if sorted[i] >= 1 / 2), None)
        if start_index is None:
            return np.abs(self.x).argmax()
        entry_count = self.dimension - start_index
        median_index = start_index + (entry_count // 2 - (1 if entry_count % 2 == 0 else 0))
        return next((i for i in range(self.dimension) if sorted[median_index] == self.x_squared[i]))
    
class SystemCache:
    def __init__(self, system, count):
        self.count = count
        self.dimension = len(system)
        self.B_scaled_system, self.scaled_solutions = scale_system(system)
        self.B_psd_system_diagonals = [None for _ in range(self.dimension)]
        
    def get_B_diagonals(self, i):
        if self.B_psd_system_diagonals[i] is not None:
            return self.B_psd_system_diagonals[i]
        self.B_psd_system_diagonals[i] = get_system_diagonals(self.B_scaled_system, i)
        return self.B_psd_system_diagonals[i]
    
class RandomSystemCache:
    def __init__(self, system_cache, random_system, random_system_output):
        self.system_cache = system_cache
        self.A_system = np.add(self.system_cache.B_scaled_system, random_system, out=random_system_output)
        self.A_psd_system_diagonals = [None for _ in range(self.system_cache.dimension)]
        self.A_B_diagonals_diff = [None for _ in range(self.system_cache.dimension)]
        
    def get_A_diagonals(self, i):
        if self.A_psd_system_diagonals[i] is not None:
            return self.A_psd_system_diagonals[i]
        self.A_psd_system_diagonals[i] = get_system_diagonals(self.A_system, i)
        return self.A_psd_system_diagonals[i]
    
    def get_A_B_diagonals_diff(self, i):
        if self.A_B_diagonals_diff[i] is not None:
            return self.A_B_diagonals_diff[i]
        B_diagonals = self.system_cache.get_B_diagonals(i)
        A_diagonals = self.get_A_diagonals(i)
        diff = A_diagonals - B_diagonals
        self.A_B_diagonals_diff[i] = np.power(diff, 2, out=diff)
        return self.A_B_diagonals_diff[i]

class ApproximatorCache:
    def __init__(self, random_system_cache, point_cache):
        self.random_system_cache = random_system_cache
        self.point_cache = point_cache
        
        # self.partial_results = np.array([A.T @ (A @ self.point_cache.x) for A in self.random_system_cache.A_system])
        self.partial_results = self.random_system_cache.A_system @ self.point_cache.x
        for i in range(len(self.partial_results)):
            self.partial_results[i] @= self.random_system_cache.A_system[i]
        self.results = self.point_cache.x.T @ self.partial_results.T
        self.results_difference = self.random_system_cache.system_cache.scaled_solutions - self.results
        self.partial_results, self.results, self.results_difference = \
            ApproximatorCache._initialize_results(self.random_system_cache.A_system, self.point_cache.x, self.random_system_cache.system_cache.scaled_solutions)
        self.G = [None for _ in range(self.random_system_cache.system_cache.dimension)]
    
    @njit
    def _initialize_results(A_system, x, scaled_solutions):
        dimension = len(x)
        partial_results = np.zeros((dimension, dimension))
        for i in range(len(x)):
            partial_results[i] = A_system[i].T @ (A_system[i] @ x)
        results = x.T @ partial_results.T
        results_difference = scaled_solutions - results
        return partial_results, results, results_difference
    
    @njit
    def _get_g(A_diagonals, results_difference, x_squared, x_squared_inverse):
        return x_squared_inverse * (results_difference + (A_diagonals * x_squared))
    
    @njit
    def _get_log_w(g, A_B_diagonals_diff, B_diagonals):
        log_w = -1 * np.power(g - B_diagonals, 2)
        return (log_w + A_B_diagonals_diff) / 2
    
    @njit
    def _get_approximation(special_index, x, x_squared_inverse, log_w, partial_results, scaled_solutions, results):
        log_gradient = 1 / x[special_index]
        D_x_G = partial_results
        D_x_G[:, special_index] += (scaled_solutions - results) * log_gradient
        D_x_G *= -2 * x_squared_inverse[special_index]
        sign, logdet = np.linalg.slogdet(D_x_G)
        weight = log_w.sum()
        if weight < -10:
            weight = -10
        elif weight > 10:
            weight = 10
        weight = np.exp(weight)
        approximation = np.exp(logdet) * weight
        return approximation, log_gradient, logdet, weight
    
    def get_g(self, i):
        if self.G[i] is not None:
            return self.G[i]
        self.G[i] = ApproximatorCache._get_g(self.random_system_cache.get_A_diagonals(i), self.results_difference, self.point_cache.x_squared[i], self.point_cache.x_squared_inverse[i])
        return self.G[i]
    
    def get_log_w(self, i):
        g = self.get_g(i)
        A_B_diagonals_diff = self.random_system_cache.get_A_B_diagonals_diff(i)
        B_diagonals = self.random_system_cache.system_cache.get_B_diagonals(i)
        return ApproximatorCache._get_log_w(g, A_B_diagonals_diff, B_diagonals)
        
    # note that calling this method once will modify the partial_results we calculated earlier
    def get_approximation(self):
        log_w = self.get_log_w(self.point_cache.special_index)
        log_gradient = 1 / self.point_cache.x[self.point_cache.special_index]
        approximation, log_gradient, logdet, weight = ApproximatorCache._get_approximation(self.point_cache.special_index, self.point_cache.x, \
            self.point_cache.x_squared_inverse, log_w, self.partial_results, self.random_system_cache.system_cache.scaled_solutions, self.results)
        if approximation > 1000:
            print('Found suspect approximation {}\n\tlog_gradient={}\n\tdet={}\n\tweight={}\n\tx={}'.format(approximation, log_gradient, np.exp(logdet), weight, self.point_cache.x))
        return approximation