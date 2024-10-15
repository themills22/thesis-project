import numpy as np

# ripped from Barvinok
def scaler(system, point, include_gradient=True):
    system_S = np.zeros(system.shape)
    exp_point = np.exp(point)
    for i in range(system.shape[0]):
        system_S[i] = exp_point[i] * system[i]
    S = np.sum(system_S, 0)
    sign, absolute_logdet = np.linalg.slogdet(S)
    function_value = sign * absolute_logdet
    gradient_value = None
    if include_gradient:
        gradient_value = np.zeros(system.shape[0])
        inverse_S = np.linalg.inv(S)
        for i in range(system.shape[0]):
            gradient_value[i] = exp_point[i] * np.trace(system[i] @ inverse_S)
    return (function_value, gradient_value)

# linear reduction of the original scaler that eliminates x_n (x_n = x_1 + ... + x_{n-1})
def scaler_reduced(system, point, include_gradient=True):
    n = system.shape[0]
    exp_point = np.zeros(n)
    exp_point[0:n - 1] = np.exp(point)
    exp_point[n - 1] = np.exp(np.sum(-1 * point))
    system_S = np.zeros(system.shape)
    for i in range(n):
        system_S[i] = exp_point[i] * system[i]
    S = np.sum(system_S, 0)
    sign, absolute_logdet = np.linalg.slogdet(S)
    value = sign * absolute_logdet
    gradient = None
    if include_gradient:
        gradient = np.zeros(n - 1)
        inverse_S = np.linalg.inv(S)
        for i in range(n - 1):
            gradient[i] = np.trace(inverse_S @ (system_S[i] - system_S[n - 1]))
    return value, gradient

def scaler_reduced_oracle(system, point, delta):
    value, gradient = scaler_reduced(system, point)
    answer = all((-1 * (delta / 2)) <= value and value <= (delta / 2) for value in gradient)
    return answer, value, gradient

def lagrange_hyperplane(point, l, include_gradients=True):
    l_vector = l * np.ones(point.shape[0])
    function_value = np.dot(l_vector, point)
    point_gradient = None
    l_gradient = None
    if include_gradients:
        point_gradient = l_vector
        l_gradient = np.sum(point)
    return (function_value, point_gradient, l_gradient)

# last index of point is the lagrange multiplier
def lagrange_scaler(system, point, include_gradient=True):
    scaler_value, scaler_gradient = scaler(system, point, include_gradient)
    l_index = point.shape[0] - 1
    hyper_value, hyper_point_gradient, hyper_l_gradient = lagrange_hyperplane(point[0:l_index], point[l_index], include_gradient)
    function_value = scaler_value + hyper_value
    gradient_value = None
    if include_gradient:
        gradient_value = np.zeros(point.shape[0])
        gradient_value[0:l_index] = scaler_gradient + hyper_point_gradient
        gradient_value[l_index] = hyper_l_gradient
    return (function_value, gradient_value)

def lagrange_oracle(system, point, delta):
    scaler_value, scaler_gradient = lagrange_scaler(system, point)
    answer = all((delta / 2) <= value and value <= (delta / 2) for value in scaler_gradient)
    return answer, scaler_value, scaler_gradient

# ripped from Gurvitz and others: file:///C:/Users/mvinc/Documents/UTSA/thesis/mixed-discriminant-approximation.pdf (yes, I know you can't access this :))
# assuming dtype is a flaot type
def radius(n, dtype):
    max_value = np.finfo(dtype).max
    n_power = np.power(n, 3 / 2)
    return n_power * (np.log(n_power) + np.log(max_value))