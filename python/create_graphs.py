import argparse
import matplotlib.pyplot as plt
import networkx as nx
import os
import tempfile
import uuid

from mpi4py import MPI
from python.rl.environments import PowerFlowSystemEnv

def main():
    comm = MPI.COMM_WORLD
    if comm.Get_size() > 1:
        if comm.Get_rank() == 0:
            print("This script is not meant to be run with multiple processes.")
        return

    parser = argparse.ArgumentParser()
    parser.add_argument('--size', help='Size of the graph', required=True, type=int)
    parser.add_argument('--edge-probability', help='Probability of edge creation', required=True, type=float)
    parser.add_argument('--num-graphs', help='Number of graphs to generate', required=True, type=int)
    parser.add_argument('--output-directory', help='Directory to output files', required=True, type=str)
    parser.add_argument('--show', help='Whether to show the generated graphs', action='store_true')
    
    args = parser.parse_args()
    paths = []
    for i in range(args.num_graphs):
        graph = None
        for _ in range(100):
            with tempfile.TemporaryFile(delete_on_close=False) as fp:
                fp.close()
                random_graph = nx.fast_gnp_random_graph(args.size, args.edge_probability)
                nx.write_adjlist(random_graph, fp.name)
                env = PowerFlowSystemEnv(fp.name, 1/ (args.size ** 2), 1, 1, 0.01)
                env.reset()
                _, reward, _, _, _ = env.step(env.action_space.sample())
                if reward == -100:
                    continue
                graph = random_graph
                break
            
        if graph is None:
            print(f"Failed to generate a solvable graph {i} after 100 attempts, skipping.")
            continue
        path = os.path.join(args.output_directory, f"graph-{str(uuid.uuid4())}.adjlist")
        nx.write_adjlist(graph, path)
        paths.append(path)
        print(f"Graph {i} saved to {path}")
        
    if args.show:
        for path in paths:
            graph = nx.read_adjlist(path)
            nx.draw(graph, with_labels=True)
            print(f"Showing graph from {path}")
            plt.show()
        
if __name__ == "__main__":
    main()