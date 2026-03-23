
import numpy as np
import pandas as pd
import uuid

def laplace_noise(scale):
    return np.random.laplace(loc=0.0, scale=scale)

def extract_full_patterns(df):
    patterns = (
        df.sort_values(["CaseID", "Timestamp"])
          .groupby("CaseID")["Activity"]
          .apply(lambda x: "".join(x))
          .reset_index(name="Pattern")
    )
    return patterns

def count_pattern_frequencies(patterns):
    pattern_counts = (
        patterns.groupby("Pattern")["CaseID"]
        .count()
        .reset_index(name="count")
    )
    return pattern_counts

def case_sampling(df, epsilon_d=1):
    df = df.copy()
    df["CaseID"] = df["CaseID"].astype(str)

    # Group by patterns
    patterns = extract_full_patterns(df)
    pattern_groups = patterns.groupby("Pattern")["CaseID"].apply(list).to_dict()

    # Apply Laplace noise to counts and determine how many cases to duplicate/remove
    scale = 1.0 / epsilon_d
    duplication_counter = {}
    df_final = df.copy()

    for pattern, case_list in pattern_groups.items():
        true_count = len(case_list)
        noisy_count = int(round(true_count + laplace_noise(scale)))
        noisy_count = max(0, noisy_count)
        diff = noisy_count - true_count

        if diff > 0:
            # Duplicate complex cases selected randomly
            sampled = np.random.choice(case_list, size=diff, replace=True)
            duplicated_rows = []
            for cid in sampled:
                case_rows = df_final[df_final["CaseID"] == cid].copy()
                duplication_counter[cid] = duplication_counter.get(cid, 0) + 1
                new_id = f"{cid}_dup{duplication_counter[cid]}"
                case_rows["CaseID"] = new_id
                duplicated_rows.append(case_rows)
            if duplicated_rows:
                df_final = pd.concat([df_final] + duplicated_rows, ignore_index=True)

        elif diff < 0:
            # Delete cases randomly
            remove_cids = np.random.choice(case_list, size=min(abs(diff), len(case_list)), replace=False)
            df_final = df_final[~df_final["CaseID"].isin(remove_cids)].copy()

    df_final = df_final.sort_values(["CaseID", "Timestamp"]).reset_index(drop=True)

    return df_final, duplication_counter

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
        "AnonCaseID": "CaseID",
        "Activity": "Activity",
        "FinalTimestamp": "Timestamp"
    })

    df_final["Timestamp"] = df_final["Timestamp"].dt.floor("s")
    df_final = df_final.sort_values(["CaseID", "Timestamp"]).reset_index(drop=True)

    return df_final