import numpy as np
import scaler
import time

# mostly ripped from some Boyd lecture notes: https://web.stanford.edu/class/ee364b/lectures/ellipsoid_method_notes.pdf
def ellipsoid(ellipsoid_matrix, center, delta, oracle):
    n = ellipsoid_matrix.shape[0]
    answer, function_value, gradient_value = oracle(center, delta)
    best_center, best_function_value, best_gradient_value = center, function_value, gradient_value
    while not answer and not inside(ellipsoid_matrix, delta):
        gradient_value = (1 / np.sqrt(gradient_value.T @ ellipsoid_matrix @ gradient_value)) * gradient_value
        center = center - (1 / (n + 1)) * (ellipsoid_matrix @ gradient_value)
        b = ellipsoid_matrix @ gradient_value
        ellipsoid_adjust = (2 / (n + 1)) * np.outer(b, b.T)
        ellipsoid_matrix = (np.power(n, 2) / (np.power(n, 2) - 1)) * (ellipsoid_matrix - ellipsoid_adjust)
        answer, function_value, gradient_value = oracle(center, delta)
        if function_value < best_function_value:
            # note that it is possible for oracle to return True answer while function_value >= best_function_value
            # we still choose to leave the best_function_value be and exit the ellipsoid algorithm
            best_center, best_function_value, best_gradient_value = center, function_value, gradient_value
    return answer, best_center, best_function_value, best_gradient_value

# lazy inside check as I am completely ignoring the orientation of the ellipsoid
def inside(ellipsoid_matrix, delta):
    # is it sqrt of eigenvalues of inverse of sqrt of eigenvales?
    eigenvalues = np.sqrt(np.linalg.eigvalsh(ellipsoid_matrix))
    return all(eigenvalue < delta for eigenvalue in eigenvalues)

n = 50
rng = np.random.default_rng(12345)
system = rng.normal(0, 1, (n, n, n))
psd_system = np.array([B.T @ B for B in system])
start = time.time()
R = scaler.radius(n, psd_system.dtype) * np.identity(n - 1)
oracle = lambda point, delta: scaler.scaler_reduced_oracle(psd_system, point, delta)
answer, best_point, best_value, best_gradient = ellipsoid(R, np.zeros(n - 1), 1.0e-8, oracle)
end = time.time()
print('method: Ellipsoidal')
print('n: {0}'.format(n))
print('answer: {0}'.format(answer))
print('minimum: {0}'.format(best_value))
print('time taken: {0}'.format(end - start))

# oracle = lambda point, delta: scaler.lagrange_oracle(psd_system, point, delta)
# answer, best_point, best_value, best_gradient = ellipsoid(R, np.zeros(n + 1), 1.0e-8, oracle)
# print('answer: {answer}, best_value: {best_value}, best_point: {best_point}, best_gradient: {best_gradient}')