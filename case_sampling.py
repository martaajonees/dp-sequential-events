
import numpy as np
import pandas as pd
import uuid
from filtered import filter_main

def laplace_noise(scale):
    return np.random.laplace(loc=0.0, scale=scale)

def case_sampling(df, epsilon_d=0.5):
    df = df.copy()
    df["CaseID"] = df["CaseID"].astype(str)
    
    group_cols = ["SrcState", "Activity", "TgtState"]

    # Count unique cases per transition
    transition_counts = (
        df.groupby(group_cols)["CaseID"]
        .nunique()
        .reset_index(name="count")
    )

    # Sensibility Δf = 1 
    delta_f = 1.0
    scale = delta_f / epsilon_d

    # Map transitions to their cases
    transition_cases = (
        df.groupby(group_cols)["CaseID"]
        .unique()
        .to_dict()
    )

    cases_to_duplicate = []
    cases_to_remove = set()

    for _, row in transition_counts.iterrows():
        key = (row["SrcState"], row["Activity"], row["TgtState"])
        true_count = row["count"]

        noise = laplace_noise(scale)
        noisy_count = int(round(true_count + noise))

        diff = noisy_count - true_count

        available_cases = list(transition_cases.get(key, []))

        if len(available_cases) == 0:
            continue

        if diff > 0:
            sampled = np.random.choice(
                available_cases,
                size=min(diff, len(available_cases)),
                replace=True
            )
            cases_to_duplicate.extend(sampled)

        elif diff < 0:
            sampled = np.random.choice(
                available_cases,
                size=min(abs(diff), len(available_cases)),
                replace=False
            )
            cases_to_remove.update(sampled)

    # Delete cases marked for removal
    df = df[~df["CaseID"].isin(cases_to_remove)].copy()

    # Duplicate cases marked for duplication
    duplicated_rows = []
    duplication_counter = {}

    for cid in cases_to_duplicate:
        case_rows = df[df["CaseID"] == cid].copy()

        if len(case_rows) == 0:
            continue

        duplication_counter[cid] = duplication_counter.get(cid, 0) + 1
        new_id = f"{cid}_dup{duplication_counter[cid]}"
        case_rows["CaseID"] = new_id

        duplicated_rows.append(case_rows)

    if duplicated_rows:
        df = pd.concat([df] + duplicated_rows, ignore_index=True)

    return df, duplication_counter

# Adjust noise based on duplication count
def inject_time_noise(df, duplication_counter):
    df = df.copy()

    # Count duplications per original case
    def adjusted_epsilon(row):
        cid = str(row["CaseID"])
        if "_dup" in cid:
            original = cid.split("_dup")[0]
        else:
            original = cid
        D = duplication_counter.get(original, 0) + 1

        return row["ϵt"] / D if row["ϵt"] > 0 else 0.0

    df["adj_ϵt"] = df.apply(adjusted_epsilon, axis=1)

    noisy_rel_times = []

    for idx, row in df.iterrows():
        eps = row["adj_ϵt"]

        if eps == 0:
            noisy_rel_times.append(row["RelTime"])
            continue

        scale = 1.0 / eps
        noise = np.random.laplace(0, scale)

        noisy_rel_times.append(row["RelTime"] + noise)

    df["NoisyRelTime"] = noisy_rel_times

    return df

# Reconstruct timestamps from noisy relative times
def reconstruct_timestamps(df):
    df = df.copy()

    new_timestamps = []

    for case_id, group in df.groupby("CaseID"):
        group = group.sort_values("Timestamp")

        t0 = group["Timestamp"].min()
        current_time = t0

        for _, row in group.iterrows():
            rel = row["NoisyRelTime"]

            if rel < 0:
                rel = 0

            current_time = current_time + pd.Timedelta(minutes=rel)
            new_timestamps.append(current_time)

    df["AnonTimestamp"] = new_timestamps

    return df

# Compress timestamps to original range
def compress_timestamps(df):
    df = df.copy()

    min_original = df["Timestamp"].min()
    max_original = df["Timestamp"].max()

    min_new = df["AnonTimestamp"].min()
    max_new = df["AnonTimestamp"].max()

    original_span = (max_original - min_original).total_seconds()
    new_span = (max_new - min_new).total_seconds()

    if new_span == 0:
        return df

    factor = original_span / new_span

    compressed = []

    for ts in df["AnonTimestamp"]:
        delta = (ts - min_new).total_seconds()
        new_delta = delta * factor
        compressed_ts = min_original + pd.Timedelta(seconds=new_delta)
        compressed.append(compressed_ts)

    df["FinalTimestamp"] = compressed

    return df

# Anonymize case IDs
def anonymize_case_ids(df):
    df = df.copy()

    new_ids = {
        cid: str(uuid.uuid4())
        for cid in df["CaseID"].unique()
    }

    df["AnonCaseID"] = df["CaseID"].map(new_ids)

    return df

def clean_final_table(df):
    df_final = df[["AnonCaseID", "Activity", "FinalTimestamp"]].copy()

    df_final = df_final.rename(columns={
        "AnonCaseID": "Case ID",
        "Activity": "Activity",
        "FinalTimestamp": "Timestamp"
    })

    df_final["Timestamp"] = df_final["Timestamp"].dt.floor("S")
    df_final = df_final.sort_values("Timestamp").reset_index(drop=True)

    return df_final


if __name__ == "__main__":
    # Step 1: Filter DAFSA-annotated table
    df_filtered = filter_main()

    # Step 2: Case sampling
    df_sampled, duplication_counter = case_sampling(df_filtered)

    # Step 3: Inject time noise
    df_noisy = inject_time_noise(df_sampled, duplication_counter)

    # Step 4: Reconstruct timestamps
    df_reconstructed = reconstruct_timestamps(df_noisy)

    # Step 5: Compress time range
    df_compressed = compress_timestamps(df_reconstructed)

    # Step 6: Anonymize Case IDs
    df_final = anonymize_case_ids(df_compressed)

    # Step 7: Final ordering
    df_final = df_final.sort_values("FinalTimestamp").reset_index(drop=True)

    df = clean_final_table(df_final)
    print("\n Final anonymized log:")
    print(df)
