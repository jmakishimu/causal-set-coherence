# -*- coding: utf-8 -*-
"""
A Final, Audited, and Memory-Optimized Causal Set Model for Emergent Boundaries

This script implements a definitive scientific experiment based on multiple
rounds of expert feedback. It tests the hypothesis that stable, bounded 
structures (proto-Markov Blankets) emerge from a universe of pure causality 
governed by simple, local rules.

V5.1 Update: Saves the best run's graph data using pickle for post-hoc analysis.
"""
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, qmc
from sklearn.utils import resample
import time
import os
import json
import itertools
import pickle
from collections import defaultdict

# --- All analysis and plotting functions are unchanged ---
def _minimize_surface(graph, cluster_nodes, initial_surface):
    minimal_surface = set(initial_surface)
    interior = cluster_nodes - minimal_surface
    if not interior: return minimal_surface
    for node_to_test in initial_surface:
        if node_to_test not in minimal_surface: continue
        potential_surface = minimal_surface - {node_to_test}
        is_still_separator = True
        for pred in graph.predecessors(node_to_test):
            if pred in interior:
                for succ_of_pred in graph.successors(pred):
                    if succ_of_pred not in cluster_nodes:
                        is_still_separator = False; break
            if not is_still_separator: break
        if is_still_separator: minimal_surface.remove(node_to_test)
    return minimal_surface

def analyze_clusters(graph):
    cluster_props = {}
    try: scc = list(nx.weakly_connected_components(graph))
    except Exception: return {}
    for nodes in scc:
        volume = len(nodes)
        if volume < 2: continue
        initial_surface = {n for n in nodes if any(succ not in nodes for succ in graph.successors(n)) or any(pred not in nodes for pred in graph.predecessors(n))}
        minimal_surface_nodes = _minimize_surface(graph, nodes, initial_surface)
        surface_size = len(minimal_surface_nodes)
        coherence = volume / (surface_size + 1e-9)
        cluster_props[frozenset(nodes)] = {'volume': volume, 'surface': surface_size, 'coherence': coherence}
    return cluster_props

def validate_markov_blanket_properties(graph, cluster_nodes, config):
    print("\n--- Performing Post-Hoc Boundary Validation on Most Stable Cluster ---")
    if not cluster_nodes: print("Result: No stable cluster found to validate."); return
    initial_surface = {n for n in cluster_nodes if any(s not in cluster_nodes for s in graph.successors(n)) or any(p not in cluster_nodes for p in graph.predecessors(n))}
    minimal_surface = _minimize_surface(graph, cluster_nodes, initial_surface)
    internal_nodes = list(cluster_nodes - minimal_surface)
    exterior_nodes = list({neighbor for node in minimal_surface for neighbor in itertools.chain(graph.predecessors(node), graph.successors(node))} - cluster_nodes)
    if not internal_nodes or not exterior_nodes: print("Result: Cluster has no distinct interior/exterior to test."); return
    leaky_paths_found, total_paths_checked = 0, 0
    sample_size = config['analysis']['leak_check_sample_size']; cutoff = config['analysis']['leak_check_path_cutoff']
    for source_list, target_list in [(internal_nodes, exterior_nodes), (exterior_nodes, internal_nodes)]:
        for _ in range(sample_size):
            if not source_list or not target_list: continue
            source, target = random.choice(source_list), random.choice(target_list)
            try:
                paths = list(nx.all_simple_paths(graph, source=source, target=target, cutoff=cutoff))
                if paths: total_paths_checked += len(paths); leaky_paths_found += sum(1 for path in paths if not any(node in minimal_surface for node in path))
            except (nx.NodeNotFound, nx.NetworkXError): continue
    print(f"Result: Sampled {sample_size*2} node pairs with path cutoff {cutoff}.")
    if total_paths_checked > 0:
        leak_ratio = leaky_paths_found / total_paths_checked
        print(f"Found {leaky_paths_found} leaky paths among {total_paths_checked} simple paths (Leakage: {leak_ratio:.2%}).")
        print("Conclusion:", "The surface appears to be a robust causal separator." if leak_ratio == 0 else "The boundary is 'leaky'.")
    else: print("Result: No simple paths found between sampled node pairs.")

def calculate_bootstrap_correlation(df, x_col, y_col, n_samples=1000):
    if df.shape[0] < 2: return np.nan, (np.nan, np.nan)
    correlations = []
    for _ in range(n_samples):
        sample_df = resample(df, replace=True)
        if sample_df.shape[0] >= 2:
            corr, _ = spearmanr(sample_df[x_col], sample_df[y_col])
            if np.isfinite(corr): correlations.append(corr)
    if not correlations: return np.nan, (np.nan, np.nan)
    return np.mean(correlations), (np.percentile(correlations, 2.5), np.percentile(correlations, 97.5))

def plot_primary_analysis(df, output_dir, config):
    print("\n--- Generating Plot 1: Primary Analysis (Lifespan vs. Coherence) ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 9))
    exp_df = df[df['b_reinforce'] > 0]; ctrl_df = df[df['b_reinforce'] == 0]
    if not exp_df.empty: sns.scatterplot(x='peak_coherence', y='lifespan', data=exp_df, ax=ax, alpha=0.7, hue='b_reinforce', size='peak_volume', sizes=(30, 500), palette=config['plotting']['primary_plot_palette'], edgecolor='black', linewidth=0.5)
    if not ctrl_df.empty: ax.scatter(ctrl_df['peak_coherence'], ctrl_df['lifespan'], c='gray', marker='x', s=50, label='Control (b_reinforce=0)', alpha=0.6)
    valid_df = exp_df[exp_df['lifespan'] > 1].dropna(subset=['peak_coherence', 'lifespan'])
    if len(valid_df) > 1:
        mean_corr, (ci_lower, ci_upper) = calculate_bootstrap_correlation(valid_df, 'peak_coherence', 'lifespan', n_samples=config['analysis']['bootstrap_samples'])
        stats_text = (fr"Spearman $\rho$ [95% CI]:" fr"\n  {mean_corr:.3f}  [{ci_lower:.3f}, {ci_upper:.3f}]" fr"\nClusters (N): {len(valid_df)}")
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12, va='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
    ax.set_title('Primary Hypothesis Test: Cluster Lifespan vs. Structural Coherence', fontsize=16, pad=20)
    ax.set_xlabel(r'Peak Coherence Ratio ($\mathcal{C} = V/S_{min}$)', fontsize=12); ax.set_ylabel(r'Lifespan ($\tau$, in simulation steps)', fontsize=12)
    ax.set_xscale('log'); ax.set_yscale('log')
    handles, labels = ax.get_legend_handles_labels()
    if handles: ax.legend(handles, labels, title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1]); plt.savefig(os.path.join(output_dir, 'Lifespan_vs_Coherence.png'), dpi=config['plotting']['dpi']); plt.close(fig)
    print("Plot saved.")
    if len(valid_df) > 1:
        print(f"\n--- HYPOTHESIS VALIDATION ---")
        if ci_lower > 0.05: print(f"Conclusion: Hypothesis strongly supported. 95% CI [{ci_lower:.3f}, {ci_upper:.3f}] is robustly positive.")
        else: print(f"Conclusion: Hypothesis not supported by this data. 95% CI [{ci_lower:.3f}, {ci_upper:.3f}] includes zero.")

def plot_representative_run(df, run_id, output_dir, config):
    print(f"\n--- Generating Plot 2: Time Series for Representative Run {run_id} ---")
    run_df = df[df['run_id'] == run_id].copy()
    if run_df.empty: return
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.scatterplot(data=run_df, x='birth_step', y='lifespan', size='peak_volume', hue='peak_coherence', palette=config['plotting']['representative_plot_palette'], ax=ax, edgecolor='black', alpha=0.7, sizes=(50, 600))
    ax.set_title(f'Cluster Evolution for Representative Run #{int(run_id)}', fontsize=16, pad=20)
    plt.tight_layout(rect=[0, 0, 0.85, 1]); plt.savefig(os.path.join(output_dir, 'Representative_Run_TimeSeries.png'), dpi=config['plotting']['dpi']); plt.close(fig)
    print("Plot saved.")

def plot_stable_cluster_example(graph, nodes, output_dir, config):
    print("\n--- Generating Plot 3: Example of a Stable Cluster Structure ---")
    if not nodes: return
    subgraph = graph.subgraph(nodes)
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(subgraph, iterations=50)
    nx.draw(subgraph, pos, with_labels=False, node_size=50, node_color='skyblue', width=0.5, arrowsize=5)
    plt.title(f"Structure of a Long-Lived Cluster (V={subgraph.number_of_nodes()})", fontsize=16)
    plt.savefig(os.path.join(output_dir, 'Stable_Cluster_Example.png'), dpi=config['plotting']['dpi']); plt.close()
    print("Plot saved.")

def plot_parameter_space(df, output_dir, config):
    print("\n--- Generating Plot 4: Parameter Space Heatmap ---")
    exp_df = df[df['b_reinforce'] > 0]
    if exp_df.empty or len(exp_df['p_instantiate'].unique()) < 2 or len(exp_df['b_reinforce'].unique()) < 2: print("Warning: Not enough diverse data for heatmap."); return
    try:
        heatmap_data = pd.pivot_table(exp_df, values='lifespan', index=['b_reinforce'], columns=['p_instantiate'], aggfunc=lambda x: x.max().mean())
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(heatmap_data, ax=ax, cmap=config['plotting']['heatmap_palette'], annot=True, fmt='.0f', linewidths=.5)
        ax.set_title('Parameter Space Analysis: Avg. Max Lifespan', fontsize=16, pad=20)
        plt.savefig(os.path.join(output_dir, 'Parameter_Space_Heatmap.png'), dpi=config['plotting']['dpi']); plt.close(fig)
        print("Plot saved.")
    except Exception as e: print(f"Could not generate heatmap: {e}")

# --- Core Simulation Class ---
class CausalSetSimulator:
    def __init__(self, params, run_id):
        self.params = params; self.run_id = run_id
        self.G = nx.DiGraph(); self.next_node_id = 0
        self.active_clusters = {}; self.finalized_log = []

    def _initialize_universe(self):
        node1, node2 = self.next_node_id, self.next_node_id + 1; self.next_node_id += 2
        self.G.add_node(node1, creation_step=0, energy=self.params['initial_energy'])
        self.G.add_node(node2, creation_step=0, energy=self.params['initial_energy'])
        self.G.add_edge(node1, node2)
        initial_clusters = analyze_clusters(self.G)
        for nodes, props in initial_clusters.items():
            self.active_clusters[nodes] = {'run_id': self.run_id, 'birth_step': 0, 'last_seen_step': 0, 'peak_volume': props['volume'], 'peak_coherence': props['coherence']}

    def _log_cluster_dynamics(self, step):
        current_clusters = analyze_clusters(self.G)
        next_active_clusters = {}
        parent_map = {}
        for cur_key in current_clusters.keys():
            best_match_key, best_jaccard = None, 0.0
            for prev_key in self.active_clusters.keys():
                if not cur_key.isdisjoint(prev_key):
                    jaccard = len(cur_key.intersection(prev_key)) / len(cur_key.union(prev_key))
                    if jaccard > best_jaccard: best_jaccard, best_match_key = jaccard, prev_key
            if best_match_key and best_jaccard >= self.params['jaccard_threshold']: parent_map[cur_key] = best_match_key
        child_counts = defaultdict(list)
        for child_key, parent_key in parent_map.items(): child_counts[parent_key].append(child_key)
        true_parent_map = {}
        for parent_key, child_keys in child_counts.items():
            main_descendant = max(child_keys, key=lambda k: current_clusters[k]['volume'])
            true_parent_map[main_descendant] = parent_key
        processed_parents = set()
        for cur_key, cur_props in current_clusters.items():
            parent_key = true_parent_map.get(cur_key)
            if parent_key:
                hist_entry = self.active_clusters[parent_key]
                next_active_clusters[cur_key] = {'run_id': self.run_id, 'birth_step': hist_entry['birth_step'], 'last_seen_step': step, 'peak_volume': max(hist_entry['peak_volume'], cur_props['volume']), 'peak_coherence': max(hist_entry['peak_coherence'], cur_props['coherence'])}
                processed_parents.add(parent_key)
            else: next_active_clusters[cur_key] = {'run_id': self.run_id, 'birth_step': step, 'last_seen_step': step, 'peak_volume': cur_props['volume'], 'peak_coherence': cur_props['coherence']}
        for prev_key, hist_entry in self.active_clusters.items():
            if prev_key not in processed_parents:
                lifespan = hist_entry['last_seen_step'] - hist_entry['birth_step']
                if lifespan > 1: self.finalized_log.append({'run_id': hist_entry['run_id'], 'birth_step': hist_entry['birth_step'], 'death_step': hist_entry['last_seen_step'], 'lifespan': lifespan, 'peak_volume': hist_entry['peak_volume'], 'peak_coherence': hist_entry['peak_coherence']})
        self.active_clusters = next_active_clusters

    def step(self, current_step):
        nodes_with_energy = [n for n, data in self.G.nodes(data=True) if data['energy'] > self.params['parent_propagate_energy_cost']]
        for parent_node in nodes_with_energy:
            if random.random() < self.params['p_instantiate']:
                self.G.nodes[parent_node]['energy'] -= self.params['parent_propagate_energy_cost']
                new_node = self.next_node_id; self.next_node_id += 1
                self.G.add_node(new_node, creation_step=current_step, energy=self.params['initial_energy'])
                self.G.add_edge(parent_node, new_node)
        nodes_to_remove = []
        for node, data in list(self.G.nodes(data=True)):
            data['energy'] += self.G.out_degree(node) * self.params['b_reinforce']
            data['energy'] -= self.params['d_fade']
            if data['energy'] <= 0: nodes_to_remove.append(node)
        self.G.remove_nodes_from(nodes_to_remove)
        self._log_cluster_dynamics(current_step)

    def run(self):
        self._initialize_universe()
        max_steps = int(self.params.get('max_steps', 500)); node_limit = int(self.params.get('max_graph_nodes_hard_limit', 50000))
        halted = False; debug_enabled = self.params.get('debug_enabled', False); debug_interval = int(self.params.get('debug_log_interval', 50))
        if debug_enabled: print(f"\n--- DEBUG MODE ON (Run {self.run_id}) ---")
        for step in range(1, max_steps + 1):
            if self.G.number_of_nodes() > node_limit:
                if debug_enabled: print(f"\nHALTED (Step {step}): Node count ({self.G.number_of_nodes()}) exceeded limit ({node_limit}).")
                halted = True; break
            if not self.G.nodes():
                if debug_enabled: print(f"DEBUG (Step {step}): Universe empty. Halting run.")
                break
            self.step(step)
            if debug_enabled and step % debug_interval == 0: print(f"  [Step {step:04d}] Nodes: {self.G.number_of_nodes():<6} | Edges: {self.G.number_of_edges():<7} | Active Clusters: {len(self.active_clusters):<5}")
        final_status = "Halted_NodeLimit" if halted else "Completed"
        for hist_entry in self.active_clusters.values():
            lifespan = hist_entry['last_seen_step'] - hist_entry['birth_step']
            if lifespan > 1: self.finalized_log.append({'run_id': hist_entry['run_id'], 'birth_step': hist_entry['birth_step'], 'death_step': hist_entry['last_seen_step'], 'lifespan': lifespan, 'peak_volume': hist_entry['peak_volume'], 'peak_coherence': hist_entry['peak_coherence'], 'final_status': final_status})
        for entry in self.finalized_log:
            if 'final_status' not in entry: entry['final_status'] = "Completed"
            entry.update({k:v for k,v in self.params.items() if k not in entry})
        return self.finalized_log, self.G, self.active_clusters

# --- Main Campaign Execution ---
def generate_lhs_param_sets(param_space, n_samples):
    keys = list(param_space.keys()); sampler = qmc.LatinHypercube(d=len(keys), seed=42)
    sample = sampler.random(n=n_samples)
    l_bounds = [v[0] for v in param_space.values()]; u_bounds = [v[1] for v in param_space.values()]
    scaled_sample = qmc.scale(sample, l_bounds, u_bounds)
    return [dict(zip(keys, s)) for s in scaled_sample]

def main():
    start_time = time.time()
    try:
        with open('config.json', 'r') as f: config = json.load(f)
    except FileNotFoundError: print("Error: `config.json` not found. Exiting."); return
    output_dir = config['output_directory']; os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'campaign_log.csv')
    if os.path.exists(log_path): os.remove(log_path)
    param_sets = generate_lhs_param_sets(config['parameter_space'], config['experimental_design']['lhs_samples'])
    control_params = param_sets[0].copy(); control_params['b_reinforce'] = 0.0
    param_sets.append(control_params)
    print(f"--- Starting Campaign: {config.get('campaign_name', 'Default')} ---")
    best_run_data = {'max_lifespan': -1, 'graph': None, 'history': None, 'run_id': -1}
    run_counter = 0; is_first_write = True
    for i, params in enumerate(param_sets):
        print(f"\nRunning Set {i+1}/{len(param_sets)}: {params}")
        for _ in range(config['experimental_design']['trials_per_param_set']):
            run_counter += 1; print(f"  Trial (Run: {run_counter})...", end='', flush=True)
            run_params = {**params, **config['constants'], **config['debug']}
            simulator = CausalSetSimulator(run_params, run_counter)
            run_log, final_graph, final_active_history = simulator.run()
            if run_log:
                current_max_lifespan = max((entry['lifespan'] for entry in run_log), default=-1)
                if current_max_lifespan > best_run_data['max_lifespan']:
                    best_run_data.update({'max_lifespan': current_max_lifespan, 'graph': final_graph, 'history': final_active_history, 'run_id': run_counter})
                df_run = pd.DataFrame(run_log)
                df_run.to_csv(log_path, mode='a', header=is_first_write, index=False)
                is_first_write = False
            print(" Done.")
            del simulator, final_graph, final_active_history, run_log
    print(f"\nRaw campaign data saved to '{log_path}'")
    
    # *** NEW: Save the best_run_data dictionary using pickle ***
    best_run_path = os.path.join(output_dir, 'best_run_data.pkl')
    with open(best_run_path, 'wb') as f:
        pickle.dump(best_run_data, f)
    print(f"Best run's graph data saved to '{best_run_path}'")

    try:
        df = pd.read_csv(log_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("\nCampaign finished, but no data was logged. Cannot generate plots."); return
    plot_primary_analysis(df, output_dir, config)
    if not df.empty and best_run_data['run_id'] != -1:
        rep_run_id = best_run_data['run_id']
        print(f"\nLongest-lived cluster was found in Run #{rep_run_id}, using it for representative plots.")
        plot_representative_run(df, rep_run_id, output_dir, config)
        rep_run_graph, rep_run_hist = best_run_data['graph'], best_run_data['history']
        stable_cluster_nodes = set()
        if rep_run_hist: stable_cluster_nodes = max(rep_run_hist.keys(), key=lambda k: rep_run_hist[k].get('peak_volume', 0), default=set())
        if stable_cluster_nodes:
            plot_stable_cluster_example(rep_run_graph, stable_cluster_nodes, output_dir, config)
            validate_markov_blanket_properties(rep_run_graph, stable_cluster_nodes, config)
        plot_parameter_space(df, output_dir, config)
    print(f"\n--- Campaign Finished. Total execution time: {time.time() - start_time:.2f} seconds. ---")

if __name__ == '__main__':
    main()