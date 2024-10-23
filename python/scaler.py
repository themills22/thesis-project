import common
import numpy as np
import scipy as sp
import scipy.optimize as opt

# ripped from Barvinok
def scaler(system, point, include_gradient=True):
    """The default "scaler" function from Barvinok f(x) = log(det(x_1 * Q_1 + ... + x_n * Q_n))

    Args:
        system: (N, N, N) array_like: The PSD system of matrices to use in equation.
        point: (N) array_like: The point to calculate the equation for.
        include_gradient (bool, optional): Whether to calculate the gradient. Defaults to True.

    Returns:
        (Result): The result of the scaler function.
    """
    system_S = np.zeros(system.shape)
    exp_point = np.exp(point)
    for i in range(system.shape[0]):
        system_S[i] = exp_point[i] * system[i]
    S = np.sum(system_S, 0)
    sign, absolute_logdet = np.linalg.slogdet(S)
    value = sign * absolute_logdet
    gradient = None
    if include_gradient:
        gradient = np.zeros(system.shape[0])
        inverse_S = np.linalg.inv(S)
        for i in range(system.shape[0]):
            gradient[i] = exp_point[i] * np.trace(system[i] @ inverse_S)
    return common.Result(value, gradient)

# linear reduction of the original scaler that eliminates x_n (x_n = -x_1 - ... - x_{n-1})
def scaler_hyperplane_reduced(system, point, include_gradient=True):
    """The scaler function on the hyperplane x_1 + ... + x_n = 0. Only need to provide
    the first n-1 points as this function has eliminated the x_n point through a linear
    equality constraint elimination (x_n = -x_1 - ... - x_{n-1}).

    Args:
        system: (N, N, N) array_like: The PSD system of matrices to use in equation.
        point: (N - 1) array_like: The point to calculate the equation for.
        include_gradient (bool, optional): Whether to calculate the gradient. Defaults to True.

    Returns:
        Result: The result of the scaler function.
    """
    n = system.shape[0]
    exp_point = np.zeros(n)
    exp_point[0:-1] = np.exp(point)
    exp_point[-1] = np.exp(np.sum(-1 * point))
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
    return common.Result(value, gradient)

def scaler_hyperplane_reduced_oracle(system, point, delta):
    """The separation oracle to use for scaler-hyperplane-reduced function.

    Args:
        system: (N, N, N) array_like: The PSD system of matrices to use in equation.
        point: (N - 1) array_like: The point to calculate the equation for.
        delta: float64: The length of the box's sides.

    Returns:
        OracleResult: The result of the separation oracle.
    """
    value, gradient = scaler_hyperplane_reduced(system, point)
    answer = all((-1 * (delta / 2)) <= value and value <= (delta / 2) for value in gradient)
    return common.OracleResult(answer, value, gradient)

def scale_hyperplane_reduced_system(system, initial_point=None):
    n = system.shape[0]
    initial_point = np.zeros(n - 1) if initial_point is None else initial_point
    f = lambda point: scaler_hyperplane_reduced(system, point)
    result = opt.minimize(f, initial_point, method='BFGS', jac=True)
    if not result.success:
        return None, None, result.success
    exp = np.zeros(n)
    exp[0:-1] = np.exp(result.x)
    exp[-1] = np.exp(np.sum(-1 * result.x))
    S = np.zeros((n, n))
    for i in range(n):
        S += exp[i] * system[i]
    sqrt_S = sp.linalg.sqrtm(S)
    return exp, np.linalg.inv(sqrt_S), result.success

# ripped from Gurvitz and others: file:///C:/Users/mvinc/Documents/UTSA/thesis/mixed-discriminant-approximation.pdf (yes, I know you can't access this :));
# given that our matrix entries are not going to be integer entries I had to fiddle with the Hadamard's inequality to get the following;
# assuming dtype is a float type;
def radius(n, dtype):
    """The bounded radius of where a minimum can lie w.r.t. the scaler function in the hyperplane.

    Args:
        n : The size of the system.
        dtype : The numpy dtype.

    Returns:
        float64: The bounded radius.
    """
    max_value = np.finfo(dtype).max
    n_power = np.power(n, 3 / 2)
    return n_power * (np.log2(n_power) + np.log2(max_value))