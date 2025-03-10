import numpy as np
import os
import python.dataset.read_julia_data as rjd
import python.scaler as scaler

# scales the julia data files; assumes systems have yet to be transposed
unscaled_folder = 'D:\\deep-reinforcement-learning\\thesis-project\\matrices\\data-unscaled\\8'
unscaled_files = next(os.walk(unscaled_folder), (None, None, []))[2]

scaled_folder = 'D:\\deep-reinforcement-learning\\thesis-project\\matrices\\data-scaled\\8'
for file in unscaled_files:
    systems, solutions = rjd.read_file(os.path.join(unscaled_folder, file), 8)
    psd_systems = np.array([np.array([A.T @ A for A in system]) for system in systems])
    write_systems = np.array([scaler.scale_system(psd_system) for psd_system in psd_systems])
    file, _ = os.path.splitext(file)
    np.savez(os.path.join(scaled_folder, file), systems=write_systems, solutions=solutions)