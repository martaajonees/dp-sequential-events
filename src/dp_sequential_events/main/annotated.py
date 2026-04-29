
import pandas as pd
from scipy.stats import gaussian_kde
import numpy as np
import networkx as nx

# Functions
def normalize_rt(group):
    min_t = group["RelTime"].min()
    max_t = group["RelTime"].max()

    if max_t == min_t:
        group["NrmRelTime"] = 0.0
    else:
        group["NrmRelTime"] = ((group["RelTime"] - min_t) / (max_t - min_t))
    return group

def precision(group):        
    r = group["RelTime"].max() - group["RelTime"].min()

    if r == 0:
        return pd.Series(np.full(len(group), 0.01), index=group.index)
    
    p_vals = []
    for rt in group["RelTime"]:
        if rt == group["RelTime"].min():
            precision_real = 1.0  
        else:
            precision_real = 10/60  

        p_norm = precision_real / r

        p_vals.append(p_norm)
    return pd.Series(p_vals, index=group.index)

def build_dafsa_graph(unique_seqs):
    G = nx.MultiDiGraph()
    G.add_node(0)
    node_counter = 1
    
    for seq in unique_seqs:
        current = 0
        for act in seq:
            next_node = None
            for nbr in G.successors(current):
                for key, data in G[current][nbr].items():
                    if data.get('label') == act:
                        next_node = nbr
                        break
                if next_node is not None:
                    break
            
            if next_node is None:
                G.add_node(node_counter)
                G.add_edge(current, node_counter, label=act)
                current = node_counter
                node_counter += 1
            else:
                current = next_node

    changed = True
    while changed:
        changed = False
        nodes = list(G.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                n1 = nodes[i]
                n2 = nodes[j]
                if n1 not in G or n2 not in G:
                    continue
                
                edges1 = sorted([(data['label'], nbr) for nbr in G.successors(n1) for key, data in G[n1][nbr].items()])
                edges2 = sorted([(data['label'], nbr) for nbr in G.successors(n2) for key, data in G[n2][nbr].items()])
                
                if edges1 == edges2:
                    incoming = list(G.predecessors(n2))
                    for pred in incoming:
                        for key, data in list(G[pred][n2].items()):
                            G.add_edge(pred, n1, label=data['label'])
                    G.remove_node(n2)
                    changed = True
                    break
            if changed:
                break
    return G

def estimate_pk(group, delta=0.3, name="PK"):
    t = group["NrmRelTime"].values
    if len(t) < 5:
        group[name] = (1 - delta) / 2
        return group
    
    if np.all(t == t[0]) or np.var(t) == 0:
        group[name] = (1 - delta) / 2
        return group
    try:
        kde = gaussian_kde(t)
        xs = np.linspace(0, 1, 1000)
        cdf_vals = np.cumsum(kde(xs))
        
        if cdf_vals[-1] == 0: # Handle cases where KDE produces all zeros
            group[name] = (1 - delta) / 2
            return group

        cdf_vals /= cdf_vals[-1]  # normalizar

        def cdf(x):
            return np.interp(x, xs, cdf_vals)

        pks = []

        for v, p in zip(t, group["Prec"]):
            low = max(0, v - p)
            high = min(1, v + p)

            pk = cdf(high) - cdf(low)
            pks.append(pk)

        group[name] = pks
    except(np.linalg.LinAlgError, ValueError):
        group[name] = (1 - delta) / 2
    return group

# Main function to create annotated table
def DAFSA_annotated_table(nombre_archivo="../databases/datos_sinteticos.csv"):
    # 1. Load and preprocess the event log
    log = pd.read_csv(nombre_archivo, parse_dates=["Timestamp"])
    log = log.sort_values(["CaseID", "Timestamp"]).reset_index(drop=True) # Sort logs

    # 2. Extract sequences from the log
    grouped = log.groupby("CaseID")
    sequences = grouped["Activity"].apply(lambda x: x.astype(str).tolist()).to_dict()

    sequences = {k: ["START"] + v for k, v in sequences.items()}

    # 3. Create DAFSA from sequences
    unique_seqs = list(set(tuple(seq) for seq in sequences.values()))
    unique_seqs.sort() 
    
    graph = build_dafsa_graph(unique_seqs)

    # State map 
    state_map = {state: i for i, state in enumerate(graph.nodes())}

    # 4. Find start state (root)
    targets = {v for _, v in graph.edges()}
    candidates = [n for n in graph.nodes() if n not in targets]

    if len(candidates) == 0:
        raise ValueError("No root state found.")

    start = candidates[0]

    def next_state(G, current, act):
        for nbr in G.successors(current):
            for key, data in G[current][nbr].items():
                if data.get("label") == act:
                    return nbr
        raise ValueError(f"No transition from {current} with {act}")

    # 6. Build DAFSA-annotated table
    rows = []

    for case_id, group in grouped:
        group = group.sort_values("Timestamp").reset_index(drop=True)
        acts = group["Activity"].tolist()
        times = group["Timestamp"].tolist()

        current = start

        tgt = next_state(graph, current, "START")
        current = tgt

        tgt = next_state(graph, current, acts[0])
        t0 = log["Timestamp"].min()
        first_rel = (times[0] - t0).total_seconds() / 86400  # en días
        rows.append([case_id, acts[0], times[0], state_map[current], state_map[tgt], first_rel])
        current = tgt

        for i in range(1, len(acts)):
            act = acts[i]
            tgt = next_state(graph, current, act)

            rel = (times[i] - times[i-1]).total_seconds() / 60 # in minutes

            rows.append([case_id, act, times[i], state_map[current], state_map[tgt], rel])
            current = tgt

    df = pd.DataFrame(rows, columns=[
        "CaseID", "Activity", "Timestamp", "SrcState", "TgtState", "RelTime"
    ])

    group_cols = ["SrcState", "Activity", "TgtState"]

    # 6. Normalized relative time
    #df = df.groupby(group_cols, group_keys=False).apply(normalize_rt).reset_index(drop=True)
    min_rt = df.groupby(group_cols)["RelTime"].transform("min")
    max_rt = df.groupby(group_cols)["RelTime"].transform("max")

    range_rt = max_rt - min_rt

    df["NrmRelTime"] = np.where(
        range_rt == 0,
        0.0,
        (df["RelTime"] - min_rt) / range_rt
    )
    
    # 7. Precision
    df["Prec"] = df.groupby(group_cols).apply(precision).values

    # 8. Prior Knowledge PK
    df["PK"] = (
        df.groupby(group_cols)
        .apply(lambda g: estimate_pk(g)["PK"])
        .values
    )

    # Round numeric columns 
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(2)

    return df