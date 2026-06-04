from dp_sequential_events.main.main import annotation_and_filtering
from dp_sequential_events.main.main import shift_timestamps
from dp_sequential_events.main.case_sampling import (
    case_sampling,
    inject_time_noise,
    reconstruct_timestamps,
    compress_timestamps,
    anonymize_case_ids,
    clean_final_table
)

import pandas as pd


def pipeline_bifurcado(df_filtered, months_shift, days_shift):
    df_sampled, duplication_counter = case_sampling(df_filtered)
    df_noisy = inject_time_noise(df_sampled, duplication_counter)
    df_reconstructed = reconstruct_timestamps(df_noisy)
    df_compressed = compress_timestamps(df_reconstructed)

    # Anonimizamos Case IDs UNA sola vez, antes de bifurcar
    df_anon = anonymize_case_ids(df_compressed)

    # Rama A: sin desplazamiento
    df_no_shift = df_anon.copy()
    df_no_shift = df_no_shift.sort_values("FinalTimestamp").reset_index(drop=True)
    df_no_shift = clean_final_table(df_no_shift)

    # Rama B: con desplazamiento, sobre el mismo df_anon
    df_shifted = shift_timestamps(df_anon.copy(), months_shift, days_shift)
    df_shifted = df_shifted.sort_values("FinalTimestamp").reset_index(drop=True)
    df_shifted = clean_final_table(df_shifted)

    return df_no_shift, df_shifted


def pick_sample_case(df_no_shift, df_with_shift):
    cases_no_shift = set(df_no_shift["CaseID"].unique())
    cases_with_shift = set(df_with_shift["CaseID"].unique())
    common = cases_no_shift & cases_with_shift

    for cid in common:
        if len(df_no_shift[df_no_shift["CaseID"] == cid]) >= 3:
            return cid
    return list(common)[0]


def format_latex_table_highlighted(df, caption, label,
                                    highlight_case=None, n_rows=21):
    df = df.head(n_rows).copy()
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"]
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Abreviamos el UUID para que quepa en la tabla (primeros 8 chars)
    df["CaseID_short"] = df["CaseID"]

    lines = []
    lines.append(r"\begin{table}[!ht]")
    lines.append(r"    \centering")
    lines.append(f"    \\caption{{{caption}}}")
    lines.append(f"    \\label{{{label}}}")
    lines.append(r"    \begin{tabular}{lll}")
    lines.append(r"        \toprule")
    lines.append(
        r"        \textbf{Case ID} & \textbf{Activity} "
        r"& \textbf{Timestamp} \\"
    )
    lines.append(r"        \midrule")

    for _, row in df.iterrows():
        case_id = str(row["CaseID_short"])
        activity = str(row["Activity"])
        timestamp = str(row["Timestamp"])

        if highlight_case and str(row["CaseID"]) == str(highlight_case):
            lines.append(
                f"        \\rowcolor{{blue!15}}"
                f"{case_id} & {activity} & {timestamp} \\\\"
            )
        else:
            lines.append(
                f"        {case_id} & {activity} & {timestamp} \\\\"
            )

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


if __name__ == "__main__":

    dataset_path = "../databases/synthetic_data_reg1.csv"
    delta = 0.3
    condition_number = 1
    months_shift = 2
    days_shift = 2

    print("Ejecutando anotación y filtrado...")
    df_filtered = annotation_and_filtering(
        dataset_path, delta, condition_number, _print=False
    )

    print("Generando ambas versiones con Case IDs compartidos...")
    df_no_shift, df_with_shift = pipeline_bifurcado(
        df_filtered, months_shift, days_shift
    )

    # Seleccionamos el caso a destacar
    selected_case = pick_sample_case(df_no_shift, df_with_shift)
    print(f"\nCaso seleccionado para comparación: {selected_case}")

    # Mostramos sus filas en ambas versiones
    print("\n--- Sin desplazamiento ---")
    print(
        df_no_shift[df_no_shift["CaseID"] == selected_case]
        .to_string(index=False)
    )
    print("\n--- Con desplazamiento ---")
    print(
        df_with_shift[df_with_shift["CaseID"] == selected_case]
        .to_string(index=False)
    )

    # Generamos el LaTeX con el caso destacado en azul
    latex_no_shift = format_latex_table_highlighted(
        df_no_shift,
        caption=(
            f"Registro anonimizado sin desplazamiento temporal "
            f"($\\delta = {delta}$). Las filas sombreadas corresponden "
            f"al caso seleccionado para la comparación."
        ),
        label="tab:anon_no_shift",
        highlight_case=selected_case,
        n_rows=21
    )

    latex_with_shift = format_latex_table_highlighted(
        df_with_shift,
        caption=(
            f"Registro anonimizado con desplazamiento temporal "
            f"($\\delta = {delta}$, máximo 2 meses y 2 días). "
            f"Las filas sombreadas corresponden al mismo caso "
            f"de la tabla anterior."
        ),
        label="tab:anon_with_shift",
        highlight_case=selected_case,
        n_rows=21
    )

    print("\n\n=== LATEX: SIN DESPLAZAMIENTO ===\n")
    print(latex_no_shift)
    print("\n\n=== LATEX: CON DESPLAZAMIENTO ===\n")
    print(latex_with_shift)

    with open(
        "tabla_timestamps_comparacion.tex", "w", encoding="utf-8"
    ) as f:
        f.write("% Requiere \\usepackage[table]{xcolor} en el preámbulo\n\n")
        f.write("% Tabla sin desplazamiento temporal\n")
        f.write(latex_no_shift)
        f.write("\n\n")
        f.write("% Tabla con desplazamiento temporal\n")
        f.write(latex_with_shift)

    print("\nTablas guardadas en tabla_timestamps_comparacion.tex")