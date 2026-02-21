
from dp_sequential_events.main.annotated import DAFSA_annotated_table
from dp_sequential_events.main.filtered import DAFSA_filtrated
from dp_sequential_events.main.case_sampling import case_sampling, inject_time_noise, reconstruct_timestamps, compress_timestamps, anonymize_case_ids, clean_final_table
from dp_sequential_events.main.patterns import most_common_patterns
from tabulate import tabulate

def annotation_and_filtering(data_name="../databases/datos_sinteticos.csv", delta=0.3, condition_number=1, _print=True, download_dafsa=True):
    # Annotated table 
    if _print:
        print("Generating DAFSA-annotated table...")
    df = DAFSA_annotated_table(data_name, download_dafsa)
    if _print:
        print(tabulate(df.head(10), headers='keys', tablefmt='grid', showindex=False))

    # Filtered table
    if _print:
        print("\nFiltering DAFSA-annotated table...")
    df_filtered = DAFSA_filtrated(df, delta, condition_number)
    if _print:
        print(tabulate(df_filtered.head(10), headers='keys', tablefmt='grid', showindex=False))

    len_before = len(df)
    len_after = len(df_filtered)
    if _print:
        print(f"\n Cases removed: {len_before - len_after} ({(len_before - len_after) / len_before:.2%})")
    return df_filtered


def sampling_and_anonymization(df_filtered):
    df_sampled, duplication_counter = case_sampling(df_filtered)
    df_noisy = inject_time_noise(df_sampled, duplication_counter)
    df_reconstructed = reconstruct_timestamps(df_noisy)
    df_compressed = compress_timestamps(df_reconstructed)
    
    # Anonymize Case IDs
    df_final = anonymize_case_ids(df_compressed)
    df_final = df_final.sort_values("FinalTimestamp").reset_index(drop=True)

    return clean_final_table(df_final)

def main():
    while True:
        dataset_name = input("\nEnter dataset path: ").strip()
        delta = float(input("Enter delta value: "))
        condition_number = float(input("Enter condition number: "))

        df_filtered = annotation_and_filtering(dataset_name, delta, condition_number)

        repeat = input("\nDo you want to try other values? (y/n): ").strip().lower()
        if repeat != "y":
            break

    df_final = sampling_and_anonymization(df_filtered)

    print("\nFinal anonymized log:")
    print(tabulate(df_final.head(10), headers='keys', tablefmt='grid', showindex=False))

    
def main_patterns():
    dataset_name = input("\nEnter dataset path: ").strip()
    delta = float(input("Enter delta value: "))
    condition_number = float(input("Enter condition number: "))

    df_filtered = annotation_and_filtering(dataset_name, delta, condition_number, False, False)

    patterns_original = most_common_patterns(df_filtered)
    print("\nMost common full patterns in original log:")
    print(tabulate(patterns_original, headers='keys', tablefmt='grid', showindex=False))

    df_final = sampling_and_anonymization(df_filtered)

    patterns_anon = most_common_patterns(df_final)
    print("\nMost common full patterns in anonymized log:")
    print(tabulate(patterns_anon, headers='keys', tablefmt='grid', showindex=False))


if __name__ == "__main__":
    main_patterns()