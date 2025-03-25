import itertools as it
import numpy as np
import python.scaling.scaler as scaler
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
    exp, T, success = scaler.scale_hyperplane_reduced_system_bfgs(psd_system)
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