from dp_sequential_events.main.main import annotation_and_filtering

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def extract_epsilon_stats(df_filtered):
    eps_col = "ϵt"

    if eps_col not in df_filtered.columns:
        raise ValueError(
            f"La columna '{eps_col}' no se encuentra en el DataFrame filtrado. "
            f"Columnas disponibles: {list(df_filtered.columns)}"
        )

    eps = df_filtered[eps_col].dropna()

    return {
        "mean": eps.mean(),
        "median": eps.median(),
        "min": eps.min(),
        "max": eps.max(),
        "std": eps.std()
    }


def extract_epsilon_by_transition(df_filtered):
    eps_col = "ϵt"
    group_cols = ["SrcState", "Activity", "TgtState"]

    grouped = (
        df_filtered
        .groupby(group_cols)[eps_col]
        .mean()
        .reset_index()
        .rename(columns={eps_col: "mean_epsilon_t"})
    )
    return grouped


if __name__ == "__main__":

    datasets = {
        "Registro 1": "../databases/synthetic_data_reg1.csv",
        "Registro 2": "../databases/synthetic_data_reg2.csv",
        "Registro 3": "../databases/synthetic_data_reg3.csv",
    }

    deltas = [0.20, 0.25, 0.30, 0.35, 0.40]
    condition_number = 1

    summary_results = []
    transition_results = []

    for dataset_name, dataset_path in datasets.items():
        print(f"\n{'=' * 60}")
        print(f"DATASET: {dataset_name}")
        print(f"{'=' * 60}")

        for delta in deltas:
            print(f"\n  Delta: {delta}")

            df_filtered = annotation_and_filtering(
                dataset_path, delta, condition_number, _print=False
            )

            # --- Estadísticas globales de epsilon_t ---
            stats = extract_epsilon_stats(df_filtered)
            print(
                f"  ϵt — media: {stats['mean']:.4f} | "
                f"mediana: {stats['median']:.4f} | "
                f"min: {stats['min']:.4f} | "
                f"max: {stats['max']:.4f}"
            )

            summary_results.append({
                "dataset": dataset_name,
                "delta": delta,
                "mean_epsilon_t": round(stats["mean"], 4),
                "median_epsilon_t": round(stats["median"], 4),
                "min_epsilon_t": round(stats["min"], 4),
                "max_epsilon_t": round(stats["max"], 4),
                "std_epsilon_t": round(stats["std"], 4)
            })

            # --- Epsilon_t por transición DAFSA ---
            df_trans = extract_epsilon_by_transition(df_filtered)
            df_trans["dataset"] = dataset_name
            df_trans["delta"] = delta
            transition_results.append(df_trans)

    # Sum up and print summary table
    summary_df = pd.DataFrame(summary_results)
    print("\n\n=== TABLA RESUMEN: ϵt por dataset y delta ===")
    print(summary_df.to_string(index=False))

    transition_df = pd.concat(transition_results, ignore_index=True)

    # Evolution of epsilon_t with delta: line plot with mean and min-max range
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Media de epsilon_t
    sns.lineplot(
        data=summary_df,
        x="delta",
        y="mean_epsilon_t",
        hue="dataset",
        marker="o",
        ax=axes[0]
    )
    axes[0].set_title(
        "Media de $\\epsilon_t$ en función de $\\delta$",
        fontsize=13
    )
    axes[0].set_xlabel("$\\delta$")
    axes[0].set_ylabel("Media de $\\epsilon_t$")

    # Banda de variabilidad: min-max por dataset y delta
    for dataset_name in summary_df["dataset"].unique():
        sub = summary_df[summary_df["dataset"] == dataset_name]
        axes[1].fill_between(
            sub["delta"],
            sub["min_epsilon_t"],
            sub["max_epsilon_t"],
            alpha=0.25,
            label=dataset_name
        )
        axes[1].plot(
            sub["delta"],
            sub["mean_epsilon_t"],
            marker="o",
            linewidth=1.5
        )

    axes[1].set_title(
        "Rango de $\\epsilon_t$ (mín–máx) en función de $\\delta$",
        fontsize=13
    )
    axes[1].set_xlabel("$\\delta$")
    axes[1].set_ylabel("$\\epsilon_t$")
    axes[1].legend(title="Dataset")

    plt.tight_layout()
    plt.savefig("experimento_epsilon_t.png", dpi=300)
    plt.show()

    # ---------------------------------------------------------------
    # FIGURA 2: distribución de ϵt por transición para cada dataset
    # uno por registro, con una línea por delta
    # ---------------------------------------------------------------
    n_datasets = len(datasets)
    fig2, axes2 = plt.subplots(1, n_datasets, figsize=(7 * n_datasets, 5))

    if n_datasets == 1:
        axes2 = [axes2]

    for ax, dataset_name in zip(axes2, datasets.keys()):
        sub = transition_df[transition_df["dataset"] == dataset_name]

        sns.boxplot(
            data=sub,
            x="delta",
            y="mean_epsilon_t",
            ax=ax,
            palette="Blues"
        )
        ax.set_title(
            f"Distribución de $\\epsilon_t$ por transición\n{dataset_name}",
            fontsize=12
        )
        ax.set_xlabel("$\\delta$")
        ax.set_ylabel("$\\epsilon_t$ medio por transición")

    plt.suptitle(
        "Variabilidad de $\\epsilon_t$ entre transiciones DAFSA según $\\delta$",
        fontsize=14,
        y=1.02
    )
    plt.tight_layout()
    plt.savefig("experimento_epsilon_t_boxplot.png", dpi=300, bbox_inches="tight")
    plt.show()