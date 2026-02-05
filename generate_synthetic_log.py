import numpy as np
import pandas as pd
import random
from datetime import timedelta

def generate_synthetic_log(df_annotated, n_cases=500, seed=42, output_file="datos_sinteticos.csv"):
    np.random.seed(seed)
    random.seed(seed)

    group_cols = ["SrcState", "Activity", "TgtState"]

    # 1. Estadísticas por transición
    trans_stats = {}
    for k, g in df_annotated.groupby(group_cols):
        mu = g["RelTime"].mean()
        sigma = g["RelTime"].std()
        # Evitar NaN en sigma (por ejemplo, grupo con un solo valor)
        sigma = sigma if not np.isnan(sigma) and sigma > 0 else 1.0
        mu = mu if not np.isnan(mu) else 1.0
        trans_stats[k] = {"mu": mu, "sigma": sigma, "count": len(g)}

    # 2. Posibles estados iniciales
    start_state = df_annotated["SrcState"].min()

    synthetic_rows = []
    base_time = pd.to_datetime("2020-01-01 08:00:00")

    for c in range(1000, 1000 + n_cases):
        current = start_state
        t = base_time + timedelta(minutes=random.randint(0, 1440))  # tiempo inicial aleatorio
        steps = random.randint(3, 6)

        for _ in range(steps):
            options = df_annotated[df_annotated["SrcState"] == current]
            if options.empty:
                break

            row = options.sample(1).iloc[0]
            key = (row["SrcState"], row["Activity"], row["TgtState"])
            stats = trans_stats[key]

            # 3. Generar delta de tiempo robusto
            delta_raw = np.random.normal(stats["mu"], stats["sigma"])
            delta = max(1, int(abs(np.nan_to_num(delta_raw, nan=1.0))))

            # 4. Actualizar timestamp
            t = t + timedelta(minutes=delta)

            # 5. Guardar fila sintética
            synthetic_rows.append([c, row["Activity"], t])

            # 6. Mover al siguiente estado
            current = row["TgtState"]

    # 7. Crear DataFrame y guardar
    synth = pd.DataFrame(synthetic_rows, columns=["CaseID", "Activity", "Timestamp"])
    synth.to_csv(output_file, index=False)
    print(f"Log sintético guardado en {output_file} ({n_cases} casos).")

    return synth