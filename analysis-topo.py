# -*- coding: utf-8 -*-
"""
Topological Analysis of Emergent Stable Structures

This script performs the "Next Step" of the research campaign. It loads the
most successful, long-lived cluster identified by the main simulation and
analyzes its network topology in detail.

It compares the emergent structure against a random graph (null model) to
identify non-random properties that may contribute to its stability.

To Run:
1. Ensure a simulation campaign has been completed and the file
   'experiment_outputs_final/best_run_data.pkl' exists.
2. Run this script from the command line: `python analyze_topology.py`
"""

import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def calculate_graph_metrics(graph):
    """Calculates a dictionary of key topological metrics for a given graph."""
    if not graph or graph.number_of_nodes() == 0:
        return {}

    # Find the largest weakly connected component to analyze
    if nx.is_weakly_connected(graph):
        largest_component = graph
    else:
        connected_components = list(nx.weakly_connected_components(graph))
        if not connected_components: return {}
        largest_component = graph.subgraph(max(connected_components, key=len))

    # --- FIX ---
    # Convert the largest component to UNDIRECTED for the shortest path calculation,
    # as the original directed graph is not strongly connected.
    undirected_lcc = largest_component.to_undirected()

    metrics = {
        "Nodes": graph.number_of_nodes(),
        "Edges": graph.number_of_edges(),
        "Density": nx.density(graph),
        "Avg. Clustering Coeff.": nx.average_clustering(undirected_lcc),
        "Avg. Shortest Path (LCC)": nx.average_shortest_path_length(undirected_lcc) if largest_component.number_of_nodes() > 1 else 0,
        "Source Nodes (in-degree=0)": sum(1 for _, d in graph.in_degree() if d == 0),
        "Sink Nodes (out-degree=0)": sum(1 for _, d in graph.out_degree() if d == 0)
    }
    return metrics

def plot_degree_distribution(stable_graph, random_graph, output_dir):
    """Plots the in-degree and out-degree distributions for comparison."""
    fig, axs = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    
    # In-degree
    stable_in_degrees = [d for _, d in stable_graph.in_degree()]
    random_in_degrees = [d for _, d in random_graph.in_degree()]
    max_in_degree = max(max(stable_in_degrees) if stable_in_degrees else [0], max(random_in_degrees) if random_in_degrees else [0])
    bins = range(max_in_degree + 2)
    
    axs[0].hist(stable_in_degrees, bins=bins, alpha=0.7, label='Stable Cluster', density=True, color='skyblue', edgecolor='black')
    axs[0].hist(random_in_degrees, bins=bins, alpha=0.7, label='Random Graph (Control)', density=True, histtype='step', lw=2, color='crimson')
    axs[0].set_title("In-Degree Distribution", fontsize=14)
    axs[0].set_xlabel("In-Degree")
    axs[0].set_ylabel("Probability Density")
    axs[0].legend()
    axs[0].grid(True, linestyle=':')

    # Out-degree
    stable_out_degrees = [d for _, d in stable_graph.out_degree()]
    random_out_degrees = [d for _, d in random_graph.out_degree()]
    max_out_degree = max(max(stable_out_degrees) if stable_out_degrees else [0], max(random_out_degrees) if random_out_degrees else [0])
    bins = range(max_out_degree + 2)

    axs[1].hist(stable_out_degrees, bins=bins, alpha=0.7, label='Stable Cluster', density=True, color='skyblue', edgecolor='black')
    axs[1].hist(random_out_degrees, bins=bins, alpha=0.7, label='Random Graph (Control)', density=True, histtype='step', lw=2, color='crimson')
    axs[1].set_title("Out-Degree Distribution", fontsize=14)
    axs[1].set_xlabel("Out-Degree")
    axs[1].legend()
    axs[1].grid(True, linestyle=':')
    
    fig.suptitle("Comparative Topology: Degree Distribution", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(output_dir, 'degree_distribution_comparison.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nDegree distribution plot saved to '{plot_path}'")


def main_analysis():
    """Main function to load data and run the topological analysis."""
    print("--- Starting Post-Campaign Topological Analysis ---")
    
    output_dir = 'experiment_outputs_final' # Assumes default output dir
    best_run_path = os.path.join(output_dir, 'best_run_data.pkl')

    try:
        with open(best_run_path, 'rb') as f:
            best_run_data = pickle.load(f)
        print(f"Successfully loaded data for best run (Run #{best_run_data['run_id']})")
    except FileNotFoundError:
        print(f"Error: Could not find '{best_run_path}'. Please run the main simulation campaign first.")
        return

    rep_run_hist = best_run_data.get('history', {})
    if not rep_run_hist: print("Error: The history for the best run is empty."); return
        
    stable_cluster_nodes = max(rep_run_hist.keys(), key=lambda k: rep_run_hist[k].get('peak_volume', 0), default=set())
    if not stable_cluster_nodes: print("Error: Could not identify a stable cluster from the best run's history."); return
        
    full_graph = best_run_data['graph']
    stable_cluster_graph = full_graph.subgraph(stable_cluster_nodes).copy() # Use .copy() for a mutable subgraph

    n = stable_cluster_graph.number_of_nodes(); m = stable_cluster_graph.number_of_edges()
    if n == 0 or m == 0: print("Stable cluster graph is empty. Cannot perform analysis."); return
    random_control_graph = nx.gnm_random_graph(n, m, directed=True)
    
    stable_metrics = calculate_graph_metrics(stable_cluster_graph)
    random_metrics = calculate_graph_metrics(random_control_graph)

    print("\n--- Comparative Topological Metrics ---")
    report = pd.DataFrame([stable_metrics, random_metrics], index=['Stable Cluster', 'Random Graph']).T.fillna('N/A')
    for col in report.columns:
        report[col] = report[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    print(report)
    
    plot_degree_distribution(stable_cluster_graph, random_control_graph, output_dir)
    print("\n--- Analysis Complete ---")


if __name__ == '__main__':
    # This block allows the script to be run directly.
    # In a real project, you might import functions from it.
    # We decide which script to run based on a command-line argument for simplicity.
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--analyze':
        main_analysis()
    else:
        # This part is just for context; the user should run the scripts separately.
        # To avoid confusion, we only provide the analysis script content in this file
        # when it's structured this way. The above is the recommended structure.
        # For this response, we'll assume two separate files as instructed.
        print("This is the analysis script. Run it after the main simulation with:")
        print("python analyze_topology.py")

# To make this file runnable, we'll just call main_analysis() directly
if __name__ == '__main__':
    # The above __main__ block is for explanation.
    # For direct execution, we simply call the analysis function.
    main_analysis()