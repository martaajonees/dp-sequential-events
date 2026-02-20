
import pandas as pd
from dafsa import DAFSA
from scipy.stats import gaussian_kde
import numpy as np

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

def estimate_pk(group, delta=0.3, name="PK"):
    t = group["NrmRelTime"].values
    if len(t) < 5:
        group[name] = (1 - delta) / 2
        return group
    
    kde = gaussian_kde(t)
    xs = np.linspace(0, 1, 1000)
    cdf_vals = np.cumsum(kde(xs))
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
    dafsa = DAFSA(list(sequences.values()))
    graph = dafsa.to_graph() 

    # State map 
    state_map = {state: i for i, state in enumerate(graph.nodes())}

    # 4. Find start state (root)
    targets = {v for _, v in graph.edges()}
    candidates = [n for n in graph.nodes() if n not in targets]

    if len(candidates) == 0:
        raise ValueError("No root state found.")

    start = candidates[0]

    def next_state(G, current, act):
        for nbr in G.neighbors(current):
            if G.get_edge_data(current, nbr).get("label") == act:
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
    df = df.groupby(group_cols, group_keys=False).apply(normalize_rt)
    
    # 7. Precision
    df["Prec"] = df.groupby(group_cols, group_keys=False).apply(precision)

    # 8. Prior Knowledge PK
    df = df.groupby(group_cols, group_keys=False).apply(estimate_pk)

    # Round numeric columns 
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(2)

    # 9. Visualize the DAFSA
    from graphviz import Digraph

    dot = Digraph(engine="dot")
    dot.attr(rankdir="LR", dpi="200")

    dot.attr("node",
            shape="circle",
            style="filled",
            fillcolor="lightgray",
            fontcolor="black")

    dot.attr("edge",
            color="gray40",
            penwidth="0.3",
            arrowsize="0.4")

    for u, v, data in graph.edges(data=True):
        lbl = data.get("label", "")
        dot.edge(str(state_map[u]), str(state_map[v]), label=lbl)

    dot.render("dafsa", format="png", cleanup=True)


    return df