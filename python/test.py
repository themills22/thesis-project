import numpy as np
import scaler
import time

def projection(point):
    n = point.shape[0]
    ones = np.ones(n)
    adjustment = (ones.dot(point) / n) * ones
    return point - adjustment

def gradient_descent(evaluator, point, gamma, tolerance, project_pls):
    value, gradient = evaluator(point)
    last_value = 1.0e8
    iteration = 0
    while np.abs(value - last_value) > tolerance:
        point = point - gamma * gradient
        if project_pls:
            point = projection(point)
        last_value = value
        value, gradient = evaluator(point)
        iteration += 1
    return value, point, iteration

n = 50
rng = np.random.default_rng(12345)
system = rng.normal(0, 1, (n, n, n))
psd_system = np.array([B.T @ B for B in system])
# points = rng.normal(0, 1, (10, n))
# for i in range(10):
#     scaler_value, scaler_gradient = scaler.scaler(psd_system, points[i])
#     actual_system = np.zeros(psd_system.shape)
#     for j in range(n):
#         actual_system[j] = np.exp(points[i, j]) * psd_system[j]
#     actual_sum_inverse = np.linalg.inv(np.sum(actual_system, 0))
#     actual_gradient = np.zeros(n)
#     for j in range(n):
#         actual_gradient[j] = np.exp(points[i, j]) * np.linalg.trace(psd_system[j] @ actual_sum_inverse)
#     print('scaler{0}: {1}'.format(i, scaler_gradient))
#     print('actual{0}: {1}'.format(i, actual_gradient))
#     print('')

start = time.time()
evaluator = lambda point: scaler.scaler_reduced(psd_system, point)
minimizer, point, iteration = gradient_descent(evaluator, np.zeros(n - 1), 0.01, 1e-8, False)
end = time.time()
print('method: GD')
print('n: {0}'.format(n))
print('answer: {0}'.format(True))
print('minimum: {0}'.format(minimizer))
print('time taken: {0}'.format(end - start))