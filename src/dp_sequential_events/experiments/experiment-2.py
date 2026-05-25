from dp_sequential_events.main.main import annotation_and_filtering
from dp_sequential_events.main.main import sampling_and_anonymization

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def get_all_patterns(df, n=None):
    """
    Devuelve los patrones de un registro. 
    Si n es None (por defecto), devuelve todos.
    """
    sequences = (
        df.sort_values("Timestamp")
        .groupby("CaseID")["Activity"]
        .apply(lambda x: "".join(x.astype(str)))
    )
    counts = sequences.value_counts().reset_index()
    counts.columns = ["pattern", "count"]
    
    if n is not None:
        counts = counts.head(n)
        
    return counts


def build_comparison_table(dataset_path, dataset_name,
                            deltas, condition_number, top_n=None):
    """
    Para cada delta, obtiene TODOS los patrones del log original
    y sus frecuencias en el log anonimizado.
    """
    rows = []

    for delta in deltas:
        df_filtered = annotation_and_filtering(
            dataset_path, delta, condition_number, _print=False
        )

        # Patrones originales (le pasamos top_n que ahora será None)
        top_original = get_all_patterns(df_filtered, n=top_n)

        # Log anonimizado
        df_final = sampling_and_anonymization(df_filtered)

        # Frecuencias anonimizadas de TODOS los patrones
        all_anon = get_all_patterns(df_final, n=None)
        anon_dict = dict(zip(all_anon["pattern"], all_anon["count"]))

        total_orig = top_original["count"].sum()
        total_anon = all_anon["count"].sum() if not all_anon.empty else 1

        for _, row in top_original.iterrows():
            pat = row["pattern"]
            freq_orig = row["count"]
            freq_anon = anon_dict.get(pat, 0)

            # Frecuencias relativas
            rel_orig = freq_orig / total_orig if total_orig > 0 else 0
            rel_anon = freq_anon / total_anon if total_anon > 0 else 0

            # Variación porcentual de la frecuencia relativa
            if rel_orig > 0:
                variacion_pct = ((rel_anon - rel_orig) / rel_orig) * 100
            else:
                variacion_pct = 0.0

            rows.append({
                "dataset": dataset_name,
                "delta": delta,
                "pattern": pat,
                "freq_original": freq_orig,
                "freq_anonymized": freq_anon,
                "rel_original": round(rel_orig, 4),
                "rel_anonymized": round(rel_anon, 4),
                "variacion_pct": round(variacion_pct, 2)
            })

    return pd.DataFrame(rows)


def plot_comparison_table(df_table, dataset_name, deltas, ax):
    """
    Genera una tabla visual (heatmap de texto) con:
        filas   = patrones
        columnas = delta
        valores  = variación porcentual de frecuencia
    """
    pivot = df_table.pivot_table(
        index="pattern",
        columns="delta",
        values="variacion_pct",
        aggfunc="mean"
    )

    # Ordenar filas por frecuencia original media (descendente)
    freq_order = (
        df_table.groupby("pattern")["freq_original"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    pivot = pivot.reindex(freq_order)

    # Color: verde si la variación es pequeña, rojo si es grande
    norm_vals = pivot.abs()
    colors = norm_vals.map(
        lambda v: (1.0, max(0.3, 1 - v / 100), max(0.3, 1 - v / 100))
        if v < 50
        else (1.0, 0.3, 0.3)
    )

    ax.axis("off")
    col_labels = [f"δ={d}" for d in pivot.columns]
    row_labels = list(pivot.index)

    cell_text = [
        [f"{pivot.loc[r, c]:+.1f}%" for c in pivot.columns]
        for r in pivot.index
    ]
    cell_colors = [
        [colors.loc[r, c] for c in pivot.columns]
        for r in pivot.index
    ]

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.4, 1.4) # Un poco menos de escala vertical para que quepan todos
    ax.set_title(
        f"Variación de frecuencia de patrones — {dataset_name}",
        fontsize=12, pad=12
    )


if __name__ == "__main__":

    datasets = {
        "Registro 1": "../databases/synthetic_data_reg1.csv",
        "Registro 2": "../databases/synthetic_data_reg2.csv",
        "Registro 3": "../databases/synthetic_data_reg3.csv",
    }

    deltas = [0.20, 0.25, 0.30, 0.35, 0.40]
    condition_number = 1
    
    # ¡CLAVE AQUÍ! Al poner None, procesará absolutamente todos los patrones.
    top_n = None 

    all_tables = {}

    for dataset_name, dataset_path in datasets.items():
        print(f"\nProcesando {dataset_name}...")
        df_table = build_comparison_table(
            dataset_path, dataset_name, deltas, condition_number, top_n
        )
        all_tables[dataset_name] = df_table
        # Opcional: comentar esto si son muchísimos patrones y saturan la consola
        # print(df_table.to_string(index=False))

    # ---------------------------------------------------------------
    # FIGURA: una tabla por registro
    # ---------------------------------------------------------------
    # Ajustamos la altura (de 7 a 15) para que las tablas largas no se corten
    fig, axes = plt.subplots(1, 3, figsize=(22, 15))

    for ax, (dataset_name, df_table) in zip(axes, all_tables.items()):
        plot_comparison_table(df_table, dataset_name, deltas, ax)

    plt.suptitle(
        "Experimento 2: Variación de frecuencia de patrones originales vs. anonimizados",
        fontsize=16, y=0.95
    )
    plt.tight_layout()
    plt.savefig("experimento2_tabla_patrones.png", dpi=300, bbox_inches="tight")
    plt.show()