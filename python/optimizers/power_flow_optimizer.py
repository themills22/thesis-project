from python.nn.power_flow_net import GraphNet
from scipy.special import comb
from tqdm import tqdm

import datetime as dt
import numpy as np
import os
import scipy.optimize as opt
import torch

class PowerFlowNetOptimizer:
    def __init__(self, model):
        self.model = model
        
    def f_and_jac(self, x):
        self.model.zero_grad()
        x = torch.from_numpy(x).float().requires_grad_(True)
        output = self.model.forward(x)
        output.backward()
        return output.item(), x.grad.detach().cpu().numpy()
    
def optimize(optimizer, x0, options):
    return opt.minimize(optimizer.f_and_jac, x0, method='BFGS', jac=True, options=options)
    
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_kwargs = {
        'batch_size': 64,
        'shuffle': True
    }
    if device == 'cuda':
        cuda_kwargs = {
            'num_workers': 1,
            'pin_memory': True
        }
        model_kwargs.update(cuda_kwargs)

    model_to_load = 'D:\\deep-reinforcement-learning\\thesis-project\\model\\power-flow\\4\\2025-04-20-21-11-14.pt'
    model = GraphNet(int(comb(4, 2)))
    model.load_state_dict(torch.load(model_to_load))
    
    optimize_options = {
        'gtol' : 1e-4,
        'maxiter' : 1000
    }
    
    data_folder = 'D:\\deep-reinforcement-learning\\thesis-project\\data\\power-flow\\4\\guesses'
    rng = np.random.default_rng()
    optimizer = PowerFlowNetOptimizer(model)
    count_cutoff = 8
    systems_per_file = 1000
    current_systems = np.zeros((systems_per_file, model.size))
    current_counts = np.zeros(systems_per_file)
    current_system = 0
    num_files_cap = 1000
    num_files = 0
    def loop_count():
        while num_files < num_files_cap:
            yield
    for _ in tqdm(loop_count()):
        input = rng.standard_normal(size=model.size, dtype=np.float32)
        result = optimize(optimizer, input, optimize_options)
        system, count = result.x, result.fun
        if count <= count_cutoff:
            continue
        current_systems[current_system], current_counts[current_system] = system, count
        current_system += 1
        if current_system % 100 == 0:
            print('num files: {}, current system: {}'.format(num_files, current_system))
        if current_system == systems_per_file:
            file_name = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.npz')
            np.savez(os.path.join(data_folder, file_name), systems=current_systems, solution_counts=current_counts)
            current_system = 0
            num_files += 1
            
if __name__ == '__main__':
    main()