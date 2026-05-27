import argparse
import glob
import matplotlib.pyplot as plt
import networkx as nx
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', help='Directory containing graph files', required=True, type=str)
    args = parser.parse_args()
    
    graph_files = glob.glob(os.path.join(args.directory, "*.adjlist")) +  glob.glob(os.path.join(args.directory, "**/*.adjlist"))
    graphs = [(graph_file, nx.read_adjlist(graph_file)) for graph_file in graph_files]
    graphs.sort(key=lambda g: len(g[1].nodes))
    for graph_file, graph in graphs:
        nx.draw(graph, with_labels=True)
        print(f"Showing graph from {graph_file}")
        plt.show()
        
if __name__ == "__main__":
    main()