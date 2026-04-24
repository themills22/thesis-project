import argparse
import glob
import matplotlib.pyplot as plt
import networkx as nx
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', help='Directory containing graph files', required=True, type=str)
    args = parser.parse_args()
    
    path = os.path.join(args.directory, "**/*.adjlist")
    graph_files = glob.glob(path)
    graphs = [(graph_file, nx.read_adjlist(graph_file)) for graph_file in graph_files]
    graphs.sort(key=lambda g: len(g[1].nodes))
    for graph_file, graph in graphs:
        nx.draw(graph, with_labels=True)
        print(f"Showing graph from {graph_file}")
        plt.show()
        
if __name__ == "__main__":
    main()