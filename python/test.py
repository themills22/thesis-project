import common
import ellipsoidal
import numpy as np
import scaler
import time

def gradient_descent(evaluator, point, gamma, tolerance):
    result = evaluator(point)
    last_result = common.Result(1.0e8, np.zeros(point.shape[0]))
    iteration = 0
    while np.abs(result.value - last_result.value) > tolerance:
        point = point - gamma * result.gradient
        last_result = result
        result = evaluator(point)
        iteration += 1
    return result

def test_ellipsoid(system):
    start = time.time()
    R = scaler.radius(n, psd_system.dtype) * np.identity(n - 1)
    oracle = lambda point, delta: scaler.scaler_hyperplane_reduced_oracle(psd_system, point, delta)
    result = ellipsoidal.ellipsoid(R, np.zeros(n - 1), 1.0e-8, oracle)
    end = time.time()
    print('method: Ellipsoidal')
    print('\tn: {0}'.format(system.shape[0]))
    print('\tanswer: {0}'.format(result.answer))
    print('\tminimum: {0}'.format(result.value))
    print('\ttime taken: {0}'.format(end - start))
    
def test_gradient_descent(system):
    start = time.time()
    evaluator = lambda point: scaler.scaler_hyperplane_reduced(psd_system, point)
    result = gradient_descent(evaluator, np.zeros(n - 1), 0.01, 1e-8)
    end = time.time()
    print('method: GD')
    print('\tn: {0}'.format(system.shape[0]))
    print('\tanswer: {0}'.format(True))
    print('\tminimum: {0}'.format(result.value))
    print('\ttime taken: {0}'.format(end - start))

n = 15
rng = np.random.default_rng(12345)
system = rng.normal(0, 1, (n, n, n))
psd_system = np.array([B.T @ B for B in system])
test_ellipsoid(psd_system)
test_gradient_descent(psd_system)