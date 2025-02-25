import numpy as np
import python.scaler as scaler

# point of this file is to test that the "portion" that the diagonal takes up of the scaled matrices
# doesn't blow up as we add PSD matrices to the scaled systems

n = 10
rng = np.random.default_rng()
system = rng.normal(0, 1, (n, n, n))
psd_system = np.array([A.T @ A for A in system])
for i in range(50):
    psd_system = scaler.scale_system(psd_system)
    partial_norm = np.linalg.norm(psd_system.diagonal()) ** 2
    total_norm = np.linalg.norm(psd_system) ** 2
    print('portion norm: {}'.format(partial_norm / total_norm))
    move = rng.normal(0, 1, (n, n, n))
    move = np.array([A.T @ A for A in move])
    psd_system += move