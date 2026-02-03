
import pandas as pd
from dafsa import DAFSA
from scipy.stats import gaussian_kde
import numpy as np

# 1. Load and preprocess the event log
log = pd.read_csv("datos.csv", parse_dates=["Timestamp"])
log = log.sort_values(["CaseID", "Timestamp"]).reset_index(drop=True) # Sort logs

# 2. Extract sequences from the log
grouped = log.groupby("CaseID")
sequences = grouped["Activity"].apply(lambda x: x.astype(str).tolist()).to_dict()

# 3. Create DAFSA from sequences
dafsa = DAFSA(list(sequences.values()))
graph = dafsa.to_graph() 
# State map 
state_map = {state: i for i, state in enumerate(graph.nodes())}

# 4. Find start state (root)
targets = {v for _, v in graph.edges()}
candidates = [n for n in graph.nodes() if n not in targets]

if len(candidates) != 1:
    raise ValueError("There should be exactly one root state.")

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

    for i, act in enumerate(acts):
        tgt = next_state(graph, current, act)

        src = current
        if i == 0:
            rel = 0
        else:
            rel = (times[i] - times[i-1]).total_seconds()

        rows.append([case_id, act, times[i], src, tgt, rel])
        current = tgt

df = pd.DataFrame(rows, columns=[
    "CaseID", "Activity", "Timestamp", "SrcState", "TgtState", "RelTime"
])
print(df)

group_cols = ["SrcState", "Activity", "TgtState"]

# 6. Normalized relative time
def normalize(group):
    r_min = group["RelTime"].min()
    r_max = group["RelTime"].max()

    if r_max == r_min:
        group["NrmRelTime"] = 0.5
        group["Range"] = 1
    else:
        group["NrmRelTime"] = (group["RelTime"] - r_min) / (r_max - r_min)
        group["Range"] = r_max - r_min
    return group

df = df.groupby(group_cols, group_keys=False).apply(normalize)

# 7. Precision
def precision(row):
    if row["RelTime"] == 0:
        return min(1, 86400 / row["Range"]) if row["Range"] > 0 else 1
    else:
        return min(1, 10 / row["Range"]) if row["Range"] > 0 else 1

df["Prec"] = df.apply(precision, axis=1).round(2)

# 8. Prior Knowledge PK
def estimate_pk(group, delta=0.3):
    values = group["NrmRelTime"].values

    if len(values) < 3 or np.all(values == values[0]):
        group["PK"] = (1 - delta) / 2
        return group

    kde = gaussian_kde(values)

    def cdf(x):
        return kde.integrate_box_1d(0, x)

    pks = []
    for v, p in zip(group["NrmRelTime"], group["Prec"]):
        lo = max(0, v - p)
        hi = min(1, v + p)
        pk = cdf(hi) - cdf(lo)
        pks.append(pk)

    group["PK"] = pks
    return group

df = df.groupby(group_cols, group_keys=False).apply(estimate_pk)

print(df)

# 9. Visualize the DAFSA
dafsa.write_figure("dafsa.png", label_nodes=True)