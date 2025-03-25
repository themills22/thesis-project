import numpy as np
import os
import python.dataset.read_julia_data as rjd
import python.scaling.scaler as scaler

# scales the julia data files; assumes systems have yet to be transposed
unscaled_folder = 'D:\\deep-reinforcement-learning\\thesis-project\\data\\unscaled\\8'
unscaled_files = next(os.walk(unscaled_folder), (None, None, []))[2]

scaled_folder = 'D:\\deep-reinforcement-learning\\thesis-project\\data\\scaled\\8'
for file in unscaled_files:
    systems, solution_counts = rjd.read_matrix_file(os.path.join(unscaled_folder, file), 8)
    psd_systems = np.array([np.array([A.T @ A for A in system]) for system in systems])
    psd_systems = np.array([np.array([A / np.linalg.norm(A) for A in psd_system]) for psd_system in psd_systems])
    write_systems = [scaler.scale_system(psd_system) for psd_system in psd_systems]
    write_solutions = np.array([solutions for _, solutions in write_systems])
    write_systems = np.array([systems for systems, _ in write_systems])
    file, _ = os.path.splitext(file)
    np.savez(os.path.join(scaled_folder, file), systems=write_systems, solutions=write_solutions, solution_counts=solution_counts)