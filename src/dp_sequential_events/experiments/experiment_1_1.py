from dp_sequential_events.main.main import annotation_and_filtering
from dp_sequential_events.main.main import sampling_and_anonymization
from dp_sequential_events.main.main import print_patterns

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def patterns_to_dataframe(patterns, label):
    if isinstance(patterns, pd.DataFrame):
        df = patterns.copy()
        df.columns = ["pattern", f"count_{label}"]
    elif isinstance(patterns, dict):
        df = pd.DataFrame(
            patterns.items(),
            columns=["pattern", f"count_{label}"]
        )
    else:
        raise ValueError(f"Unsupported type for patterns: {type(patterns)}")

    df[f"count_{label}"] = pd.to_numeric(
        df[f"count_{label}"], errors="coerce"
    ).fillna(0)

    return df


def calculate_precision_difference(original_patterns, anonymized_patterns):
    df_original = patterns_to_dataframe(original_patterns, "original")
    df_anonymized = patterns_to_dataframe(anonymized_patterns, "anonymized")

    df = pd.merge(
        df_original, df_anonymized, on="pattern", how="outer"
    ).fillna(0)

    df["precision_original"] = (
        df["count_original"] / df["count_original"].sum()
    )
    df["precision_anonymized"] = (
        df["count_anonymized"] / df["count_anonymized"].sum()
    )
    df["difference"] = abs(
        df["precision_original"] - df["precision_anonymized"]
    )

    return df["difference"].sum()


def count_removed_cases(df_original_log, df_filtered):
    """
    Compara el número de casos únicos antes y después del filtrado.
    df_original_log: DataFrame cargado directamente del CSV (antes de anotar).
    df_filtered: DataFrame devuelto por annotation_and_filtering.
    """
    original_cases = df_original_log["CaseID"].nunique()
    filtered_cases = df_filtered["CaseID"].nunique()
    removed = original_cases - filtered_cases
    removed_pct = (removed / original_cases) * 100 if original_cases > 0 else 0
    return original_cases, filtered_cases, removed, removed_pct


if __name__ == "__main__":

    datasets = {
        "Registro 1": "../databases/synthetic_data_reg1.csv",
        "Registro 2": "../databases/synthetic_data_reg2.csv",
        "Registro 3": "../databases/synthetic_data_reg3.csv",
    }

    deltas = [0.20, 0.25, 0.30, 0.35, 0.40]
    condition_number = 1

    precision_results = []
    removal_results = []

    for dataset_name, dataset_path in datasets.items():
        print(f"\n{'=' * 60}")
        print(f"DATASET: {dataset_name}")
        print(f"{'=' * 60}")

        # Cargamos el log original para poder contar casos antes del filtrado
        df_raw = pd.read_csv(dataset_path, parse_dates=["Timestamp"])

        for delta in deltas:
            print(f"\n  Delta: {delta}")

            # --- Anotación y filtrado ---
            df_filtered = annotation_and_filtering(
                dataset_path, delta, condition_number, _print=False
            )

            # --- Casos eliminados ---
            n_orig, n_filt, n_removed, pct_removed = count_removed_cases(
                df_raw, df_filtered
            )
            print(f"  Casos originales: {n_orig} | "
                  f"Tras filtrado: {n_filt} | "
                  f"Eliminados: {n_removed} ({pct_removed:.1f}%)")

            removal_results.append({
                "dataset": dataset_name,
                "delta": delta,
                "casos_originales": n_orig,
                "casos_filtrados": n_filt,
                "casos_eliminados": n_removed,
                "porcentaje_eliminado": round(pct_removed, 2)
            })

            # --- Patrones originales y anonimizados ---
            original = print_patterns(
                df_filtered,
                "\nPatrones más comunes (original):"
            )
            df_final = sampling_and_anonymization(df_filtered)
            anonymized = print_patterns(
                df_final,
                "\nPatrones más comunes (anonimizado):"
            )

            # --- Diferencia de precisión ---
            precision = calculate_precision_difference(original, anonymized)
            print(f"  Diferencia de precisión: {precision:.4f}")

            precision_results.append({
                "dataset": dataset_name,
                "delta": delta,
                "precision_difference": precision
            })

    # ---------------------------------------------------------------
    # TABLA: casos eliminados por dataset y delta
    # ---------------------------------------------------------------
    removal_df = pd.DataFrame(removal_results)
    print("\n\n=== TABLA: CASOS ELIMINADOS ===")
    print(removal_df.to_string(index=False))

    # ---------------------------------------------------------------
    # FIGURA 1: diferencia de precisión frente a delta
    # ---------------------------------------------------------------
    precision_df = pd.DataFrame(precision_results)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Gráfico de precisión
    sns.lineplot(
        data=precision_df,
        x="delta",
        y="precision_difference",
        hue="dataset",
        marker="o",
        ax=axes[0]
    )
    axes[0].set_title(
        "Impacto de δ sobre la diferencia de precisión",
        fontsize=13
    )
    axes[0].set_xlabel("δ")
    axes[0].set_ylabel("Diferencia de precisión total")

    # Gráfico de casos eliminados
    sns.lineplot(
        data=removal_df,
        x="delta",
        y="porcentaje_eliminado",
        hue="dataset",
        marker="s",
        ax=axes[1]
    )
    axes[1].set_title(
        "Porcentaje de casos eliminados por δ",
        fontsize=13
    )
    axes[1].set_xlabel("δ")
    axes[1].set_ylabel("Casos eliminados (%)")

    plt.tight_layout()
    plt.savefig("experimento1_resultados.png", dpi=300)
    plt.show()