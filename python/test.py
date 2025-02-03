import cvxpy as cp
import itertools as it
import numpy as np
import scaler

n = 3
rng = np.random.default_rng(12345)
system = rng.normal(0, 1, (n, n, n))
psd_system = np.array([A.T @ A for A in system])

x = cp.Variable(n, pos=True)
objective = cp.det