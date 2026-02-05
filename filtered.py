
from annotated import DAFSA_annotated_table, estimate_pk
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter("ignore", FutureWarning)

def epsilon_t(group, delta):
    r = 1.0

    epsilons = []

    for pk in group["New PK"]:
        pk = np.clip(pk, 1e-6, 1 - 1e-6)
        term = (pk / (1 - pk)) * ((1 / delta) + pk - 1)
        if term <= 0 or term >= 1:
            epsilons.append(0.0)
            continue
        epsilon_k = -np.log(term) * (1 / r)

        epsilon_k = max(epsilon_k, 0.0)
    
        epsilons.append(epsilon_k)
    return pd.Series(epsilons, index=group.index)

def DAFSA_filtrated(df_annotated, delta=0.3):

    # 1. Identify cases with condición: PK + delta >= 1
    risky_cases = df_annotated[df_annotated["PK"] + delta >= 1]["CaseID"].unique()

    # 2. Filter the dataframe cases
    df = df_annotated[~df_annotated["CaseID"].isin(risky_cases)].copy()

    # 3. Recalculate PK for the filtered dataframe
    group_cols = ["SrcState", "Activity", "TgtState"]
    df = df.groupby(group_cols, group_keys=False).apply(lambda g: estimate_pk(g, delta=delta, name="New PK"))
    df = df.drop(columns=["PK"])

    # 4. Calculate ϵt for the filtered dataframe
    df["ϵt"] = df.groupby(group_cols, group_keys=False).apply(lambda g: epsilon_t(g, delta))
    df = df.drop(columns=["Prec"])
    df = df.drop(columns=["NrmRelTime"])

    # 5. Round numeric columns 
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(2)

    return df
    

if __name__ == "__main__":
    print("Generating DAFSA-annotated table...")
    df = DAFSA_annotated_table("datos_sinteticos.csv")
    print(df.head(20))
    print("\nFiltering DAFSA-annotated table...")
    df_filtered = DAFSA_filtrated(df)
    print(df_filtered.head(20))


    