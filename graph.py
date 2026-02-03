
import pandas as pd
from dafsa import DAFSA
from scipy.stats import norm
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
            rel = 0.0
        else:
            rel = (times[i] - times[i-1]).total_seconds()

        rows.append([case_id, act, times[i], state_map[src], state_map[tgt], rel])
        current = tgt

df = pd.DataFrame(rows, columns=[
    "CaseID", "Activity", "Timestamp", "SrcState", "TgtState", "RelTime"
])
print(df)

group_cols = ["SrcState", "Activity", "TgtState"]

# 6. Normalized relative time
def normalize_rt(group):
    min_t = group["RelTime"].min()
    max_t = group["RelTime"].max()

    if max_t == min_t:
        group["NrmRelTime"] = round(0.0, 2)
    else:
        group["NrmRelTime"] = round(((group["RelTime"] - min_t) / (max_t - min_t)), 2)
    return group


df = df.groupby(group_cols, group_keys=False).apply(normalize_rt)
# 7. Precision
def precision(group):
    initial_window = 86400  # 1 día en segundos
    relative_window = 10    # 10 segundos
    
    # Calcular rango
    r_min = group["RelTime"].min()
    r_max = group["RelTime"].max()
    rango = r_max - r_min
    
    # Seleccionamos la ventana correspondiente para cada fila
    ventanas = np.where(group["RelTime"] == 0, initial_window, relative_window)
    
    if rango > 0:
        p = np.minimum(1.0, ventanas / rango)
    else:
        p = 1.0
    
    return pd.Series(p, index=group.index)

df["Prec"] = df.groupby(group_cols, group_keys=False).apply(precision)

# 8. Prior Knowledge PK
def normalize_time(tiempos):
    min_t = min(tiempos)
    max_t = max(tiempos)
    if max_t == min_t:
        return np.zeros_like(tiempos)
    return (tiempos - min_t) / (max_t - min_t)

def estimate_pk(group, delta=0.3):
    valores_norm = normalize_time(group["RelTime"].values)
    group["NrmRelTime"] = valores_norm

    pks = []
    for v, p in zip(group["NrmRelTime"], group["Prec"]):
        # Para p=0 o un rango muy pequeño, usamos el fallback
        if p == 0:
            pk = (1 - delta) / 2
        else:
            pk = norm.cdf(v + p, loc=np.mean(valores_norm), scale=np.std(valores_norm)+1e-6) - \
                 norm.cdf(v - p, loc=np.mean(valores_norm), scale=np.std(valores_norm)+1e-6)
        pks.append(pk)

    group["PK"] = pks
    return group
df = df.groupby(["CaseID"], group_keys=False).apply(estimate_pk)
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].round(2)

print(df)

# 9. Visualize the DAFSA
dafsa.write_figure("dafsa.png", label_nodes=True)