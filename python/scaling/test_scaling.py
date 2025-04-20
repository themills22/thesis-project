from python.scaling.scaler import Scaler

import numpy as np

# point of this file is to test that the "portion" that the diagonal takes up of the scaled matrices
# doesn't blow up as we add PSD matrices to the scaled systems

n = 10
rng = np.random.default_rng()
system = rng.normal(0, 1, (n, n, n))
psd_system = np.array([A.T @ A for A in system])
for i in range(50):
    scaler = Scaler(psd_system)
    psd_system, _ = scaler.scale_system_bfgs()
    partial_norm = np.linalg.norm(psd_system.diagonal()) ** 2
    total_norm = np.linalg.norm(psd_system) ** 2
    print('portion norm: {}'.format(partial_norm / total_norm))
    move = rng.normal(0, 1, (n, n, n))
    move = np.array([A.T @ A for A in move])
    psd_system += move