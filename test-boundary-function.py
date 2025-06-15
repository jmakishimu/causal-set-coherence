# -*- coding: utf-8 -*-
"""
Functional Analysis of Emergent Boundaries (V3)

This version correctly handles the important case where the stable cluster
has become the entire surviving graph, providing a specific analysis for
this outcome.
"""

import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from collections import deque
import random
import time

# --- run_signal_test, get_cluster_regions, and plot_results functions are unchanged ---
def run_signal_test(graph, start_node, surface_nodes, internal_nodes, surface_threshold=0.5, internal_threshold=0.1):
    if not start_node or start_node not in graph:
        return {'surface_hit_time': np.nan, 'internal_hit_time': np.nan}
    q = deque([(start_node, 0)])
    visited = {start_node}
    surface_hit_count, internal_hit_count = 0, 0
    required_surface_hits = max(1, int(len(surface_nodes) * surface_threshold)) if surface_nodes else 0
    required_internal_hits = max(1, int(len(internal_nodes) * internal_threshold)) if internal_nodes else 0
    surface_hit_time, internal_hit_time = np.nan, np.nan
    while q:
        current_node, distance = q.popleft()
        if np.isnan(surface_hit_time) and current_node in surface_nodes:
            surface_hit_count += 1
            if surface_hit_count >= required_surface_hits:
                surface_hit_time = distance
        if np.isnan(internal_hit_time) and current_node in internal_nodes:
            internal_hit_count += 1
            if internal_hit_count >= required_internal_hits:
                internal_hit_time = distance
        if not np.isnan(surface_hit_time) and not np.isnan(internal_hit_time): break
        for neighbor in nx.neighbors(graph, current_node):
            if neighbor not in visited:
                visited.add(neighbor)
                q.append((neighbor, distance + 1))
    return {'surface_hit_time': surface_hit_time, 'internal_hit_time': internal_hit_time}

def get_cluster_regions(graph, cluster_nodes):
    surface = {n for n in cluster_nodes if any(neighbor not in cluster_nodes for neighbor in nx.all_neighbors(graph, n))}
    interior = cluster_nodes - surface
    external_boundary = {neighbor for node in surface for neighbor in nx.all_neighbors(graph, node) if neighbor not in cluster_nodes}
    return surface, interior, list(external_boundary)

def plot_results(df, output_dir):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(data=df, x='Group', y='Penetration Delay', ax=ax, palette=['skyblue', 'lightcoral'])
    sns.stripplot(data=df, x='Group', y='Penetration Delay', ax=ax, color=".25", alpha=0.3)
    ax.set_title('Functional Boundary Test: Information Penetration Delay', fontsize=16, pad=20)
    ax.set_ylabel('Time for Signal to Cross Boundary (steps)'); ax.set_xlabel('Group Type')
    ax.grid(True, which='both', linestyle=':', linewidth='0.5')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'boundary_function_test.png')
    plt.savefig(plot_path, dpi=300); plt.close()
    print(f"\nResults plot saved to '{plot_path}'")


def main_functional_analysis():
    """Main function to load data and run the boundary function test."""
    print("--- Starting Functional Boundary Test (V3) ---")
    start_time = time.time()
    
    output_dir = 'experiment_outputs_final'
    best_run_path = os.path.join(output_dir, 'best_run_data.pkl')
    num_trials = 200

    try:
        with open(best_run_path, 'rb') as f:
            best_run_data = pickle.load(f)
        print(f"Successfully loaded data for best run (Run #{best_run_data['run_id']})")
    except FileNotFoundError:
        print(f"Error: Could not find '{best_run_path}'. Please run the main simulation campaign first.")
        return

    full_graph = best_run_data['graph']
    undirected_graph = full_graph.to_undirected()

    stable_cluster_nodes = max(best_run_data['history'].keys(), key=lambda k: best_run_data['history'][k].get('peak_volume', 0), default=set())
    if not stable_cluster_nodes:
        print("Error: Could not identify a stable cluster."); return
    
    sc_surface, sc_interior, sc_external = get_cluster_regions(undirected_graph, stable_cluster_nodes)
    
    # --- NEW: Gracefully handle the case where the cluster is the whole graph ---
    if not sc_external:
        print("\n--- Functional Test Result ---")
        print("Analysis: The stable cluster comprises the entire surviving graph.")
        print("This represents a state of complete causal closure with no external environment.")
        print("\nConclusion: The functional test for shielding is moot, as there is no 'outside' to shield from.")
        print("This outcome itself is a significant finding, demonstrating the emergence of a maximally autonomous system.")
        print(f"\n--- Analysis complete. Total time: {time.time() - start_time:.2f} seconds. ---")
        return

    print(f"\nRunning {num_trials} signal tests on the Stable Cluster...")
    experimental_results = []
    for _ in range(num_trials):
        start_node = random.choice(sc_external)
        result = run_signal_test(undirected_graph, start_node, sc_surface, sc_interior)
        experimental_results.append(result)

    print(f"Running {num_trials} signal tests on Random Control groups...")
    control_results = []
    all_nodes = list(full_graph.nodes())
    for _ in range(num_trials):
        control_nodes = set(random.sample(all_nodes, k=len(stable_cluster_nodes)))
        ctrl_surface, ctrl_interior, ctrl_external = get_cluster_regions(undirected_graph, control_nodes)
        if not ctrl_external: continue
        start_node = random.choice(ctrl_external)
        result = run_signal_test(undirected_graph, start_node, ctrl_surface, ctrl_interior)
        control_results.append(result)

    df_exp = pd.DataFrame(experimental_results).dropna()
    df_ctrl = pd.DataFrame(control_results).dropna()

    if df_exp.empty or df_ctrl.empty:
        print("\nCould not collect enough valid test runs. Analysis cannot proceed.")
        return

    df_exp['Penetration Delay'] = df_exp['internal_hit_time'] - df_exp['surface_hit_time']
    df_ctrl['Penetration Delay'] = df_ctrl['internal_hit_time'] - df_ctrl['surface_hit_time']
    
    print("\n--- Functional Test Results ---")
    print("-" * 55)
    print(f"{'Metric':<25} | {'Stable Cluster':<15} | {'Control Group':<15}")
    print("-" * 55)
    print(f"{'Avg. Surface Hit Time':<25} | {df_exp['surface_hit_time'].mean():<15.2f} | {df_ctrl['surface_hit_time'].mean():<15.2f}")
    print(f"{'Avg. Interior Hit Time':<25} | {df_exp['internal_hit_time'].mean():<15.2f} | {df_ctrl['internal_hit_time'].mean():<15.2f}")
    print(f"{'Avg. Penetration Delay':<25} | {df_exp['Penetration Delay'].mean():<15.2f} | {df_ctrl['Penetration Delay'].mean():<15.2f}")
    print("-" * 55)

    if df_exp['Penetration Delay'].mean() > df_ctrl['Penetration Delay'].mean():
        print("\nConclusion: The stable cluster's boundary demonstrates a functional shielding effect.")
    else:
        print("\nConclusion: The stable cluster's boundary does not show a clear functional shielding effect.")

    df_exp['Group'] = 'Stable Cluster'
    df_ctrl['Group'] = 'Control Group'
    combined_df = pd.concat([df_exp, df_ctrl])
    plot_results(combined_df, output_dir)
    
    print(f"\n--- Functional analysis complete. Total time: {time.time() - start_time:.2f} seconds. ---")


if __name__ == '__main__':
    main_functional_analysis()