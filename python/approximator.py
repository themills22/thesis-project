import itertools as it
import numpy as np
import scaler
import scipy as sp
import tqdm

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
    psd_system = np.array([A.T @ A for A in system])
    exp, T, success = scaler.scale_hyperplane_reduced_system(psd_system)
    if not success:
        raise ValueError('The given system cannot be scaled.')
    solution_sum = np.sum(exp)
    solution_scale_factor = n / solution_sum
    scaled_system = np.zeros(system.shape)
    scaled_solutions = np.zeros(n)
    for i in range(n):
        scaled_solutions[i] = solution_scale_factor * exp[i]
        scaled_system[i] = np.sqrt(scaled_solutions[i]) * (system[i] @ T)
    return scaled_system, scaled_solutions

def approximate(system, system_solutions, perturbation_factor, point_count, level_set_count, rng):
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
    points = rng.normal(0, 1, (point_count, n))
    sum = 0
    for point, _ in tqdm.tqdm(it.product(points, range(level_set_count)), total=point_count * level_set_count):
        random_matrices = rng.normal(0, 1, system.shape)
        random_system = system + perturbation_factor * random_matrices
        projected_system = _project(random_system, system_solutions, point, rng)
        A = np.zeros((n, n))
        for i in range(n):
            A[i] = projected_system[i].T @ (projected_system[i] @ point)
        sum += np.abs(np.linalg.det(A))
    return sum / point_count # note that I'm not scaling by 2^n here
        
def _project(system, system_solutions, point, rng):
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
    projected_system = np.zeros(0, 1, system.shape)
    for i in range(n):
        D = system[i] @ point
        d = np.dot(D, D) - system_solutions[i]
        B = system[i].T + system[i]
        b = np.dot(point, B @ point) / point_norm
        root = np.sqrt(((b ** 2) / 4) - (d / point_norm))
        if np.isnan(root):
            raise ValueError('The given system cannot be projected.')
        l = -1 * (b / 2) + rng.choice([-1, 1]) * root
        projected_system[i] = np.array(system[i])
        projected_system[i][np.diag_indices(n)] += l
    return projected_system