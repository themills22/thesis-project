from juliacall import Main as jl
from python.rl.power_flow_matrices import PowerFlowMatrices
from stable_baselines3 import TD3

import argparse
import json
import juliacall
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import python.parser_helpers as ph

class EllipseModelActor:
    def __init__(self, model):
        self.policy = model.policy
        self.observation_space = self.policy.observation_space
    
    def julia_system(self, system):
        transformed_system = np.zeros((1,) + system.shape)
        transformed_system[0] = system
        return transformed_system
        
    def initialize(self):
        return self.observation_space.sample()
    
    def get_next_system(self, system):
        action, _ = self.policy.predict(system)
        return np.clip(system + action, self.observation_space.low, self.observation_space.high)
    
class RandomEllipseActor:
    def __init__(self, rng, size):
        self.rng = rng
        self.size = size
        
    def julia_system(self, system):
        transformed_system = np.zeros((1,) + system.shape)
        transformed_system[0] = system
        return transformed_system
    
    def initialize(self):
        return self.rng.uniform(-1, 1, (self.size, self.size, self.size))
    
    def get_next_system(self, system):
        return self.rng.uniform(-1, 1, system.shape)
    
class PowerFlowModelActor:
    def __init__(self, model, graph):
        self.policy = model.policy
        self.observation_space = self.policy.observation_space
        
        self.graph = graph
        self.sorted_edges = sorted(self.graph.edges)
        self.matrices = PowerFlowMatrices(len(self.graph), self.sorted_edges)
    
    def julia_system(self, system):
        transformed_system = np.zeros((1,) + self.matrices.matrix_systems.shape)
        transformed_system[0] = self.matrices.matrix_systems
        return transformed_system
    
    def initialize(self):
        location = self.observation_space.sample()
        self.matrices.update(location)
        return location
    
    def get_next_system(self, system):
        action, _ = self.policy.predict(system)
        location = np.clip(system + action, self.observation_space.low, self.observation_space.high)
        self.matrices.update(location)
        return location
    
class RandomPowerFlowActor:
    def __init__(self, graph, rng):
        self.rng = rng
        self.graph = graph
        self.matrices = PowerFlowMatrices(len(self.graph), sorted(self.graph.edges))
        
    def julia_system(self, system):
        transformed_system = np.zeros((1,) + self.matrices.matrix_systems.shape)
        transformed_system[0] = self.matrices.matrix_systems
        return transformed_system
    
    def initialize(self):
        location = self.rng.uniform(-1, 1, len(self.graph))
        self.matrices.update(location)
        return location
    
    def get_next_system(self, system):
        location = self.rng.uniform(-1, 1, len(self.graph))
        self.matrices.update(location)
        return location

def _evaluate(repetitions, cutoff, global_root_counts, actor):
    data = {
        'root_counts': {},
        'runs': {},
        'repetitions': repetitions,
        'cutoff': cutoff
    }
    for root_count in global_root_counts:
        data['root_counts'][root_count] = []
        
    for i in range(1, repetitions + 1):
        counts = []
        root_counts = [root_count for root_count in global_root_counts]
        system = actor.initialize()
        for j in range(1, cutoff + 1):
            system = actor.get_next_system(system)
            count = jl.judge_matrix_systems(actor.julia_system(system))[0]
            counts.append(count)
            exceeded_root_counts = [root_count for root_count in root_counts if count > root_count]
            for exceeded_root_count in exceeded_root_counts:
                data['root_counts'][exceeded_root_count].append(j)
                root_counts.remove(exceeded_root_count)
        
        data['runs'][i] = counts
    
    return data

def evaluate_ellipse_model(args):
    actor = EllipseModelActor(TD3.load(args.model_path))
    data = _evaluate(args.repetitions, args.cutoff, args.root_counts, actor)
    with open(args.results_path, 'w') as file:
        json.dump(data, file)
        
def evaluate_ellipse_random(args):
    rng = np.random.default_rng()
    actor = RandomEllipseActor(rng, args.size)
    data = _evaluate(args.repetitions, args.cutoff, args.root_counts, actor)
    with open(args.results_path, 'w') as file:
        json.dump(data, file)
        
def evaluate_power_flow_model(args):
    actor = PowerFlowModelActor(TD3.load(args.model_path), nx.read_adjlist(args.graph_path))
    data = _evaluate(args.repetitions, args.cutoff, args.root_counts, actor)
    with open(args.results_path, 'w') as file:
        json.dump(data, file)
        
def evaluate_power_flow_random(args):
    rng = np.random.default_rng()
    actor = RandomPowerFlowActor(nx.read_adjlist(args.graph_path), rng)
    data = _evaluate(args.repetitions, args.cutoff, args.root_counts, actor)
    with open(args.results_path, 'w') as file:
        json.dump(data, file)
        
def process_results(args):
    data = None
    with open(args.results_path, 'r') as file:
        data = json.load(file)
    
    counts_sum = 0
    runs_total = 0
    for i in data['runs']:
        counts = data['runs'][i]
        counts_sum += sum(counts)
        runs_total += len(counts)
        
    average = counts_sum / runs_total
    average_iterations = {}
    for i in data['root_counts']:
        average_iterations[i] = np.array(data['root_counts']).mean()
        
    print('repetitions: {}'.format(data['repetitions']))
    print('cutoff: {}'.format(data['cutoff']))
    print('average root count: {}'.format(average))
    for i in sorted(average_iterations):
        print('average iteration for {} count: {}'.format(i, average_iterations[i]))

def main():
    greater_than_check = lambda value: ph.check_greater_than_int(value, 0)
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    
    subparser = subparsers.add_parser('ellipse-model', help='Evaluate an ellipse model.')
    subparser.add_argument('--model-path', help='The path of the model.', required=True, type=ph.is_valid_file)
    subparser.add_argument('--root-counts', nargs='+', help='The root counts to search for in improving a system.', required=True, type=greater_than_check)
    subparser.add_argument('--cutoff', help='The limit of how many systems to generate.', required=True, type=greater_than_check)
    subparser.add_argument('--repetitions', help='How many times to repeat the experiment.', required=True, type=greater_than_check)
    subparser.add_argument('--results-path', help='Where to save the results.', required=True, type=str)
    subparser.set_defaults(func=evaluate_ellipse_model)
    
    subparser = subparsers.add_parser('ellipse-random', help='Evaluate randomly generated ellipses.')
    subparser.add_argument('--size', help='The size of the systems to generate.', required=True, type=greater_than_check)
    subparser.add_argument('--root-counts', nargs='+', help='The root counts to search for in improving a system.', required=True, type=greater_than_check)
    subparser.add_argument('--cutoff', help='The limit of how many systems to generate.', required=True, type=greater_than_check)
    subparser.add_argument('--repetitions', help='How many times to repeat the experiment.', required=True, type=greater_than_check)
    subparser.add_argument('--results-path', help='Where to save the results.', required=True, type=str)
    subparser.set_defaults(func=evaluate_ellipse_random)
    
    subparser = subparsers.add_parser('power-flow-model', help='Evaluate a power flow model.')
    subparser.add_argument('--model-path', help='The path of the model.', required=True, type=ph.is_valid_file)
    subparser.add_argument('--root-counts', nargs='+', help='The root counts to search for in improving a system.', required=True, type=greater_than_check)
    subparser.add_argument('--cutoff', help='The limit of how many systems to generate.', required=True, type=greater_than_check)
    subparser.add_argument('--repetitions', help='How many times to repeat the experiment.', required=True, type=greater_than_check)
    subparser.add_argument('--graph-path', help='The file path for the graph.', required=True, type=ph.is_valid_file)
    subparser.add_argument('--results-path', help='Where to save the results.', required=True, type=str)
    subparser.set_defaults(func=evaluate_power_flow_model)
    
    subparser = subparsers.add_parser('power-flow-random', help='Evaluate randomly generated power flows.')
    subparser.add_argument('--root-counts', nargs='+', help='The root counts to search for in improving a system.', required=True, type=greater_than_check)
    subparser.add_argument('--cutoff', help='The limit of how many systems to generate.', required=True, type=greater_than_check)
    subparser.add_argument('--repetitions', help='How many times to repeat the experiment.', required=True, type=greater_than_check)
    subparser.add_argument('--graph-path', help='The file path for the graph.', required=True, type=ph.is_valid_file)
    subparser.add_argument('--results-path', help='Where to save the results.', required=True, type=str)
    subparser.set_defaults(func=evaluate_power_flow_random)
    
    subparser = subparsers.add_parser('process-results', help='Process the JSON results file.')
    subparser.add_argument('--results-path', help='Where to read the results from.', required=True, type=str)
    subparser.set_defaults(func=process_results)
    
    args = parser.parse_args()
    jl.seval("using PowerFlow: judge_matrix_systems")
    args.func(args)
    
def _get_average_gaussian_count(size):
    result = np.power(2, size / 2)
    result *= np.power(size, -1 / 2)
    return result

if __name__ == '__main__':
    main()