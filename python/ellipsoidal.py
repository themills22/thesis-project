import common
import numpy as np

# mostly ripped from some Boyd lecture notes: https://web.stanford.edu/class/ee364b/lectures/ellipsoid_method_notes.pdf
def ellipsoid(ellipsoid_matrix, center, delta, oracle):
    n = ellipsoid_matrix.shape[0]
    oracle_result = oracle(center, delta)
    best_result = common.EllipsoidResult(oracle_result.answer, oracle_result.value, center, oracle_result.gradient)
    while not oracle_result.answer and not inside(ellipsoid_matrix, delta):
        gradient = oracle_result.gradient
        gradient = (1 / np.sqrt(gradient.T @ ellipsoid_matrix @ gradient)) * gradient
        center = center - (1 / (n + 1)) * (ellipsoid_matrix @ gradient)
        b = ellipsoid_matrix @ gradient
        ellipsoid_adjust = (2 / (n + 1)) * np.outer(b, b.T)
        ellipsoid_matrix = (np.power(n, 2) / (np.power(n, 2) - 1)) * (ellipsoid_matrix - ellipsoid_adjust)
        oracle_result = oracle(center, delta)
        if oracle_result.value < best_result.value:
            # note that it is possible for oracle to return True answer while function_value >= best_function_value
            # we still choose to leave the best_function_value be and exit the ellipsoid algorithm
            best_result = common.EllipsoidResult(oracle_result.answer, oracle_result.value, center, oracle_result.gradient)
    return common.EllipsoidResult(oracle_result.answer, best_result.value, best_result.point, best_result.gradient)

# lazy inside check as I am completely ignoring the orientation of the ellipsoid
def inside(ellipsoid_matrix, delta):
    """The laziest "is this ellipsoid contained in this box?" function you've ever seen.
    Completely ignores the orientation of the ellipsoid and just checks lengths of semiaxes.

    Args:
        ellipsoid_matrix: (N, N) array_like: The PSD matrix that defines the ellipsoid
        delta float64: The length of the box's sides.

    Returns:
        bool: Whether the ellipsoid "fits" inside the box
    """
    # is it sqrt of eigenvalues of inverse of sqrt of eigenvales?
    eigenvalues = np.sqrt(np.linalg.eigvalsh(ellipsoid_matrix))
    return all(eigenvalue < delta / 2 for eigenvalue in eigenvalues)