import numpy as np
import os
import python.dataset.read_julia_data as rjd

# "beatutifies" the julia power flow files: just moves them into npz file :)
ugly_folder = 'D:\\deep-reinforcement-learning\\thesis-project\\data\\power-flow-ugly\\4'
ugly_file = next(os.walk(ugly_folder), (None, None, []))[2]

beatiful_folder = 'D:\\deep-reinforcement-learning\\thesis-project\\data\\power-flow\\4'
for file in ugly_file:
    systems, solution_counts = rjd.read_power_flow_file(os.path.join(ugly_folder, file), 4)
    file, _ = os.path.splitext(file)
    np.savez(os.path.join(beatiful_folder, file), systems=systems, solution_counts=solution_counts)