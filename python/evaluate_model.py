from itertools import cycle
from juliacall import Main as jl
from python.rl.power_flow_matrices import PowerFlowMatrices
from stable_baselines3 import TD3
from tqdm import tqdm

import argparse
import json
import juliacall
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import python.parser_helpers as ph

class EllipseModelActor:
    def __init__(self, model_path, model_id):
        model = TD3.load(model_path)
        self.id = model_id
        self.policy = model.policy
        self.observation_space = self.policy.observation_space
    
    def julia_systems(self, systems):
        return systems
        
    def get_next_system(self, system):
        action, _ = self.policy.predict(system)
        return np.clip(system + action, self.observation_space.low, self.observation_space.high)
    
class RandomEllipseActor:
    def __init__(self, rng, size):
        self.id = 'random'
        self.rng = rng
        self.size = size
        
    def julia_systems(self, systems):
        return systems
    
    def get_next_system(self, system):
        return self.rng.normal(0, 1, system.shape)
    
class PowerFlowModelActor:
    def __init__(self, model, graph, model_id):
        self.policy = model.policy
        self.observation_space = self.policy.observation_space
        
        self.graph = graph
        self.sorted_edges = sorted(self.graph.edges)
        self.matrices = PowerFlowMatrices(len(self.graph), self.sorted_edges)
        self.id = model_id
    
    def julia_systems(self, systems):
        julia_systems = np.zeros((len(systems),) + self.matrices.matrix_systems.shape)
        for i in range(len(systems)):
            self.matrices.update(systems[i])
            julia_systems[i] = self.matrices.matrix_systems
        return self.julia_systems 
    
    def get_next_system(self, system):
        action, _ = self.policy.predict(system)
        location = np.clip(system + action, self.observation_space.low, self.observation_space.high)
        return location
    
class RandomPowerFlowActor:
    def __init__(self, graph, rng):
        self.rng = rng
        self.graph = graph
        self.matrices = PowerFlowMatrices(len(self.graph), sorted(self.graph.edges))
        self.id = 'random'
        
    def julia_systems(self, systems):
        julia_systems = np.zeros((len(systems),) + self.matrices.matrix_systems.shape)
        for i in range(len(systems)):
            self.matrices.update(systems[i])
            julia_systems[i] = self.matrices.matrix_systems
        return self.julia_systems 
    
    def get_next_system(self, system):
        action = self.rng.uniform(-0.01, 0.01, len(self.graph.edges))
        location = np.clip(system + action, -1, 1)
        self.matrices.update(location)
        return location

def _evaluate(cutoff, global_root_counts, initial_systems, actors):
    data = {}
    for actor in actors:
        data[actor.id] = {
            'root_counts': {},
            'runs': {},
            'repetitions': len(initial_systems),
            'cutoff': cutoff
        }
        
        for root_count in global_root_counts:
            data[actor.id]['root_counts'][root_count] = []
        
    i = 1
    for system in tqdm(initial_systems):
        for actor in actors:
            counts = []
            systems = np.zeros((len(initial_systems),) + system.shape)
            for j in range(cutoff):
                system = actor.get_next_system(system)
                systems[j] = system
            
            julia_systems = actor.julia_systems(systems)
            counts = jl.judge_matrix_systems(julia_systems)
            counts = [count for count in counts]
            
            root_counts = [root_count for root_count in global_root_counts]    
            for count, j in zip(counts, range(1, cutoff + 1)):
                exceeded_root_counts = [root_count for root_count in root_counts if count > root_count]
                for exceeded_root_count in exceeded_root_counts:
                    data[actor.id]['root_counts'][exceeded_root_count].append(j)
                    root_counts.remove(exceeded_root_count)
        
            data[actor.id]['runs'][i] = counts
        
        i += 1
    
    return data

def evaluate_ellipse(args):
    rng = np.random.default_rng()
    actors = [RandomEllipseActor(rng, args.size)]
    for model_path, model_id in zip(args.model_paths, args.model_ids):
        actors.append(EllipseModelActor(model_path, model_id))
    initial_systems = rng.uniform(-1, 1, (args.repetitions, args.size, args.size, args.size))
    jl.create_matrix_system(args.size)
    data = _evaluate(args.cutoff, args.root_counts, initial_systems, actors)
    with open('{}.json'.format(args.results_path), 'w') as file:
        json.dump(data, file)
        
    np.savez('{}.npz'.format(args.results_path), initial_systems=initial_systems)
        
def evaluate_power_flow(args):
    rng = np.random.default_rng()
    graph = nx.read_adjlist(args.graph_path)
    actors = [RandomPowerFlowActor(graph, rng), PowerFlowModelActor(TD3.load(args.model_path), graph, args.model_id)]
    size = len(graph.edges)
    initial_systems = rng.uniform(-1, 1, (args.repetitions, size))
    data = _evaluate(args.cutoff, args.root_counts, initial_systems, actors)
    with open(args.results_path, 'w') as file:
        json.dump(data, file)
        
    np.savez('{}.npz'.format(args.results_path), initial_systems=initial_systems)
        
def process_results(args):
    data = None
    with open(args.results_path, 'r') as file:
        data = json.load(file)
    
    averages = {}
    for actor_id in data:
        counts_sum = 0
        runs_total = 0
        for i in data[actor_id]['runs']:
            counts = data[actor_id]['runs'][i]
            counts_sum += sum(counts)
            runs_total += len(counts)
        
        averages[actor_id] = counts_sum / runs_total
    
    print(averages)
    for actor_id in data:
        print('Actor {} tries:'.format(actor_id))
        for root_count in data[actor_id]['root_counts']:
            tries = np.array(data[actor_id]['root_counts'][root_count])
            print('\t{}: {} median, {} successes'.format(root_count, np.median(tries), len(tries)))
    run_ids = [run_id for run_id in data[next(iter(data))]['runs']]
    lines = ["-","--","-.",":"]
    line_cycler = cycle(lines)
    for run_id in run_ids:
        for actor_id in data:
            run = data[actor_id]['runs'][run_id]
            run_length = [r for r in range(1, len(run) + 1)]
            plt.plot(run_length, run, label=actor_id, linestyle=next(line_cycler))
            
        plt.xlabel('System')
        plt.ylabel('Real solution count')
        plt.title('Real solution count')
        plt.legend()
        plt.savefig('{}.{}.png'.format(args.plot_path, run_id), dpi=300)
        plt.clf()

def main():
    greater_than_check = lambda value: ph.check_greater_than_int(value, 0)
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    
    subparser = subparsers.add_parser('ellipse', help='Evaluate an ellipse model.')
    subparser.add_argument('--model-paths', nargs='+', help='The path of the model.', required=True, type=ph.is_valid_file)
    subparser.add_argument('--model-ids', nargs='+', help='The IDs of the models.', required=True, type=str)
    subparser.add_argument('--size', help='The size of the systems to generate.', required=True, type=greater_than_check)
    subparser.add_argument('--root-counts', nargs='+', help='The root counts to search for in improving a system.', required=True, type=greater_than_check)
    subparser.add_argument('--cutoff', help='The limit of how many systems to generate.', required=True, type=greater_than_check)
    subparser.add_argument('--repetitions', help='How many times to repeat the experiment.', required=True, type=greater_than_check)
    subparser.add_argument('--results-path', help='Where to save the results.', required=True, type=str)
    subparser.set_defaults(func=evaluate_ellipse)
    
    subparser = subparsers.add_parser('power-flow', help='Evaluate a power flow model.')
    subparser.add_argument('--model-path', help='The path of the model.', required=True, type=ph.is_valid_file)
    subparser.add_argument('--model-id', help='The model ID.', required=True, type=str)
    subparser.add_argument('--root-counts', nargs='+', help='The root counts to search for in improving a system.', required=True, type=greater_than_check)
    subparser.add_argument('--cutoff', help='The limit of how many systems to generate.', required=True, type=greater_than_check)
    subparser.add_argument('--repetitions', help='How many times to repeat the experiment.', required=True, type=greater_than_check)
    subparser.add_argument('--graph-path', help='The file path for the graph.', required=True, type=ph.is_valid_file)
    subparser.add_argument('--results-path', help='Where to save the results.', required=True, type=str)
    subparser.set_defaults(func=evaluate_power_flow)
    
    subparser = subparsers.add_parser('process-results', help='Process the JSON results file.')
    subparser.add_argument('--results-path', help='Where to read the results from.', required=True, type=str)
    subparser.add_argument('--plot-path', help='Where to save the run plots to.', required=True, type=str)
    subparser.set_defaults(func=process_results)
    
    args = parser.parse_args()
    jl.seval("using PowerFlow: judge_matrix_systems, create_matrix_system")
    args.func(args)
    
def _get_average_gaussian_count(size):
    result = np.power(2, size / 2)
    result *= np.power(size, -1 / 2)
    return result

if __name__ == '__main__':
    main()
