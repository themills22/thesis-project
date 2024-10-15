import argparse
import math
import numpy as np
import scipy as sp

# helper command-line parsing functions
def check_positive_int(value):
    parsed_value = int(value)
    if (parsed_value <= 0):
        raise argparse.ArgumentTypeError("{value} is an invalid positive int value")
    return parsed_value

def check_positive_float(value):
    parsed_value = float(value)
    if (parsed_value <= 0):
        raise argparse.ArgumentTypeError("{value} is an invalid positive float value")
    return parsed_value

# helper functions for projecting X matrices
def project(B, x, gamma, X, alpha, threshold, iteration_cap):
    def evaluate(X):
        T = B + (gamma * X)
        v = T @ x
        functionValue = np.dot(v, v) - 1
        gradientValue = 2 * gamma * functionValue
        gradientValue = gradientValue * np.outer(T @ x, x)
        return functionValue ** 2, gradientValue
    
    i = 0
    diff = 1.e10
    f = 1.e10
    while i < iteration_cap and diff > threshold:
        temp_f, g = evaluate(X)
        X = X - alpha * g
        
        # can only calculate diff once we've calculated f at least once
        if (i > 0):
            diff = abs(temp_f - f)
            
        f = temp_f
        i += 1
        
    return X

# helper functions for the approximation process
def generate_system(rng, n):
    system_B = rng.normal(0, 1, (n, n, n))
    # this is where the code that "fixes" all the B's would be
    return system_B

def generate_point_samples(rng, n, num_samples):
    return rng.normal(0, 1, (num_samples, n))

def generate_matrix_samples(rng, x, system_B, num_samples):
    n = system_B.shape[0]
    matrix_samples = rng.normal(0, 1, (num_samples, n, n, n))
    # this is where the code that projects the matrix samples would be
    return matrix_samples

def calculate_linear(x, system_B, system_X, gamma):
    combined_system = system_B + (gamma * system_X)
    return np.array([np.dot(combined.T, np.dot(combined, x)) for combined in combined_system])

parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="The seed to use for RNG.", required=True, type=int)
parser.add_argument("--n", help="The size of the system to approximate.", required=True, type=check_positive_int)
parser.add_argument("--num-points", help="The number of points to generate and approximate with.", required=True, type=check_positive_int)
parser.add_argument("--num-matrices", help="The number of gaussian matrices to generate and approximate with.", required=True, type=check_positive_int)
parser.add_argument("--gamma", help="The perturbation factor of the generated gaussian matrices.", required=True, type=check_positive_float)
parser.add_argument("--alpha", help="The learning rate to use for the gradient descent projection.", required=False, type=float, default=0.01)
parser.add_argument("--threshold", help="The difference threshold to use for the gradient descent projection.", required=False, type=float, default=1.e-8)
parser.add_argument("--iteration-cap", help="The iteration cap of the gradient descent projection", required=False, type=float, default=10000)

args = parser.parse_args()
rng = np.random.default_rng(args.seed)

sum = 0
system_B = generate_system(rng, args.n)
point_samples = generate_point_samples(rng, args.n, args.num_points)
for point_sample in point_samples:
    matrix_samples = generate_matrix_samples(rng, point_sample, system_B, args.num_matrices)
    for matrix_sample in matrix_samples:
        matrix_sample = np.array([project(system_B[i], point_sample, args.gamma, matrix_sample[i], args.alpha, args.threshold, args.iteration_cap) for i in range(args.n)])
        linear = calculate_linear(point_sample, system_B, matrix_sample, args.gamma)
        sum += abs(np.linalg.det(linear))
        
result = sum / args.num_points

# would be nice to print this out to nice, serializable object to run tests on later :)
print(result)


# attempts to "normalize" B and X
# system_B = rng.normal(0, 1, (args.n, args.n, args.n))
# system_B = np.array([B / math.sqrt(np.trace(B.T @ B)) for B in system_B])
# print(np.array([math.sqrt(np.trace(B.T @ B)) for B in system_B]))

# sum_B = sum(B.T @ B for B in system_B)
# inverse_sum_B_root = np.linalg.inv(sp.linalg.sqrtm(sum_B))
# print(inverse_sum_B_root.T @ inverse_sum_B_root)
# system_B = np.array([B @ inverse_sum_B_root for B in system_B])
# print(np.array([math.sqrt(np.trace(B.T @ B)) for B in system_B]))

# system_X = rng.normal(0, 1, (args.n, args.n, args.n))