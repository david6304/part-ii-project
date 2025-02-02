import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
    )
import pandas as pd
from typing import Dict
import networkx as nx

def calculate_base_performance(preds: pd.Series, true: pd.Series, task="classification") -> Dict:
    """Calculate accuracy, precision, recall, and F1 score"""
    if task == "classification":
        return {
            "accuracy": accuracy_score(true, preds),
            "precision": precision_score(true, preds, average="macro"),
            "recall": recall_score(true, preds, average="macro"),
            "f1": f1_score(true, preds, average="macro")
        }
    elif task == "regression":
        return {
            "mse": mean_squared_error(true, preds),
            "mae": mean_absolute_error(true, preds)
        }
    else:
        raise ValueError(f"Unsupported task: {task}")
        

def evaluate_all_nodes(
    model_class,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    nodes: list
) -> Dict[str, Dict]:
    """Evaluate model on all nodes in the graph"""
    results = {}
    for node in nodes:
        model = model_class(target_node=node)
        model.train(train_data, val_data)
        preds = model.predict(test_data)
        results[node] = calculate_base_performance(
            preds,
            test_data[node]
        )
    return results

def structural_hamming_distance(
    true_adj: pd.DataFrame,
    learned_adj: pd.DataFrame,
    allow_undirected: bool = False
) -> dict:
    """Unified SHD calculation with detailed breakdown"""
    # Align columns and indices
    nodes = true_adj.columns.tolist()
    true_adj = true_adj.loc[nodes, nodes]
    learned_adj = learned_adj.loc[nodes, nodes]
    
    # Calculate mismatches
    dir_diff = (true_adj != learned_adj).sum().sum()
    
    if allow_undirected:
        sym_true = true_adj | true_adj.T
        sym_learned = learned_adj | learned_adj.T
        undir_diff = (sym_true != sym_learned).sum().sum()
    else:
        undir_diff = 0
    
    return {
        "directed_shd": dir_diff,
        "undirected_shd": undir_diff,
        "extra_edges": max(0, learned_adj.sum().sum() - true_adj.sum().sum()),
        "missing_edges": max(0, true_adj.sum().sum() - learned_adj.sum().sum())
    }

def sample_value_from_cpd(cpd, evidence):
    
    # Convert the values to a numpy array (if not already).
    values = np.array(cpd.values)
    
    # If there are no parents, just use the marginal distribution.
    if len(cpd.variables) == 1:
        probs = values.flatten()
    else:
        # There are parents. The parent's order is given by cpd.variables[1:].
        parent_vars = cpd.variables[1:]
        # Build a tuple of indices for each parent.
        evidence_indices = []
        for parent in parent_vars:
            # Get the evidence state for this parent.
            state = evidence[parent]
            # Convert to an index using the mapping.
            ev_index = cpd.name_to_no[parent][state]
            evidence_indices.append(ev_index)
        
        # Use tuple indexing: select all entries along the target axis (slice(None))
        # and the specified indices for the parent dimensions.
        probs = values[(slice(None),) + tuple(evidence_indices)]
    
    return np.random.choice(len(probs), p=probs)
    
    
def intervene_and_resample_bnlearn(instance, adj_mat, cpds, intervention_node, intervention_value):
    """
    Given an instance (a pandas Series) sampled from the BN, intervene on intervention_node
    by setting it to intervention_value. Then, resample all descendants of intervention_node
    using the BN's CPDs (found in DAG['CPD']) and the topological order from DAG['model'].
    """
    new_instance = instance.copy()
    new_instance[intervention_node] = intervention_value

    # Get the topological order of the nodes in the BN.
    G = nx.DiGraph(adj_mat)
    
    # Get the descendants of the intervention node.
    descendants = nx.descendants(G, intervention_node)
    
    # Get the topological order of the nodes in the BN.
    topo_order = nx.topological_sort(G)
    
    for node in topo_order:
        if node in descendants:
            # Look up the CPD for 'node' in DAG['CPD'].
            cpd = None
            for d in cpds:
                if d.variable == node:
                    cpd = d
                    break
            if cpd is None:
                continue
            # Build the evidence dictionary from the current values of the node's parents.
            parents = list(G.predecessors(node))
            evidence = {p: int(new_instance[p]) for p in parents}
            new_instance[node] = sample_value_from_cpd(cpd, evidence)
    return new_instance
    
def calculate_cace(models, data: pd.DataFrame, intervention_node: str, adj_mat, cpds) -> float:
    do_1 = []
    do_0 = []
    data = data.to_dict(orient='records')
    for instance in data:
        # Generate two counterfactual instances:
        do_1.append(intervene_and_resample_bnlearn(instance, adj_mat, cpds, intervention_node, 1))
        do_0.append(intervene_and_resample_bnlearn(instance, adj_mat, cpds, intervention_node, 0))

    do_1 = pd.DataFrame(do_1)
    do_0 = pd.DataFrame(do_0)
    
    results = {}
    for model in models:
        preds_1 = model.predict(pd.DataFrame(do_1))
        preds_0 = model.predict(pd.DataFrame(do_0))
        results[model] = {np.mean(preds_1 - preds_0)}
    return results, np.mean(do_1['dysp'] - do_0['dysp'])
