import numpy as np

from collections import namedtuple
from numba import njit
from python.scaling.scaler import Scaler

PointCache = namedtuple('PointCache', ['x', 'x_squared', 'x_squared_inverse', 'special_index'])
SystemCache = namedtuple('SystemCache', ['solution_count', 'B_scaled_system', 'scaled_solutions', 'A_system'])

@njit
def _get_special_index(x_squared):
    dimension = len(x_squared)
    sorted = np.sort(x_squared)
    start_index = None
    for i in range(dimension):
        if sorted[i] >= 1 / 2:
            start_index = i
            break
    if start_index is None:
        return np.abs(x_squared).argmax()
    entry_count = dimension - start_index
    median_index = start_index + (entry_count // 2 - (1 if entry_count % 2 == 0 else 0))
    for i in range(dimension):
        if sorted[median_index] == x_squared[i]:
            return i
    raise ValueError('this shouldn\'t happen, there should be an exact match')

@njit
def _get_results(point_cache : PointCache, system_cache : SystemCache):
    dimension = len(point_cache.x)
    partial_results = np.zeros((dimension, dimension))
    for i in range(len(point_cache.x)):
        partial_results[i] = system_cache.A_system[i].T @ (system_cache.A_system[i] @ point_cache.x)
    results = point_cache.x.T @ partial_results.T
    results_difference = system_cache.scaled_solutions - results
    return partial_results, results, results_difference

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

@njit
def create_point_cache(x):
    x_squared = np.power(x, 2)
    x_squared_inverse = 1 / x_squared
    special_index = _get_special_index(x_squared)
    return PointCache(x=x, x_squared=x_squared, x_squared_inverse=x_squared_inverse, special_index=special_index)

@njit
def create_system_cache(solution_count, B_scaled_system, scaled_solutions, random_system):
    A_system = B_scaled_system + random_system
    return SystemCache(solution_count=solution_count, B_scaled_system=B_scaled_system, scaled_solutions=scaled_solutions, \
        A_system=A_system)
    
@njit
def approximate(point_cache : PointCache, system_cache : SystemCache):
    B_psd_system_diagonals = get_system_diagonals(system_cache.B_scaled_system, point_cache.special_index)
    A_psd_system_diagonals = get_system_diagonals(system_cache.A_system, point_cache.special_index)
    A_B_diagonals_diff = np.power(A_psd_system_diagonals - B_psd_system_diagonals, 2)
    partial_results, results, results_difference = _get_results(point_cache, system_cache)
    
    special_squared = point_cache.x_squared[point_cache.special_index]
    special_squared_inverse = point_cache.x_squared_inverse[point_cache.special_index]
    g = results_difference
    g += A_psd_system_diagonals * special_squared
    g *= special_squared_inverse
    
    g_B_diagonals_diff = g
    g -= B_psd_system_diagonals
    g_B_diagonals_diff = np.power(g_B_diagonals_diff, 2)
    log_w = g_B_diagonals_diff
    log_w *= -1
    log_w += A_B_diagonals_diff
    log_w /= 2
    
    log_gradient = 1 / point_cache.x[point_cache.special_index]
    D_x_G = partial_results
    D_x_G[:, point_cache.special_index] += (system_cache.scaled_solutions - results) * log_gradient
    D_x_G *= -2 * special_squared_inverse
    _, logdet = np.linalg.slogdet(D_x_G)
    weight = log_w.sum()
    if weight < -10:
        weight = -10
    elif weight > 10:
        weight = 10
    weight = np.exp(weight)
    approximation = np.exp(logdet) * weight
    return approximation, log_gradient, logdet, weight