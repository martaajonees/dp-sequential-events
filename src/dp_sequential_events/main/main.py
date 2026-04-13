
from dp_sequential_events.main.annotated import DAFSA_annotated_table
from dp_sequential_events.main.filtered import DAFSA_filtrated
from dp_sequential_events.main.case_sampling import case_sampling, inject_time_noise, reconstruct_timestamps, compress_timestamps, anonymize_case_ids, clean_final_table
from dp_sequential_events.main.patterns import most_common_patterns
from tabulate import tabulate
import pandas as pd

# --- UTILS ----
def get_user_input(patterns=False):
    while True:
        try: 
            dataset_name = input("\nEnter dataset path: ").strip()
            delta = float(input("Enter delta value: "))
            
            condition_number = float(input("Enter condition number (0-1): "))
            if not (0 <= condition_number <= 1):
                raise ValueError("Condition number must be between 0 and 1")
            if not patterns:
                months = int(input("Enter months shift: "))
                if months < 0:
                    raise ValueError("Months must not be negative. Please try again.")
                    

                days = int(input("Enter days shift: "))
                if days < 0:
                    raise ValueError("Days shift must not be negative. Please try again.")
                
                return dataset_name, delta, condition_number, months, days
            else:
                return dataset_name, delta, condition_number
        
        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")

def print_table(df, title=None):
    if title: 
        print(f"\n{title}")
    print(tabulate(df.head(10), headers='keys', tablefmt='grid', showindex=False))

def print_patterns(df, title):
    patterns = most_common_patterns(df)
    print_table(patterns, title)

# --- MAIN FUNCTIONS ---
def annotation_and_filtering(data_name="../databases/datos_sinteticos.csv", delta=0.3, condition_number=1, _print=True, download_dafsa=True):
    # Annotated table 
    if _print:
        print("Generating DAFSA-annotated table...")
    df = DAFSA_annotated_table(data_name, download_dafsa)

    if _print:
        print_table(df)
        print("\nFiltering DAFSA-annotated table...")
        
    df_filtered = DAFSA_filtrated(df, delta, condition_number)
        
    if _print:
        removed = len(df) - len(df_filtered)
        print(f"\nCases removed: {removed} ({removed / len(df):.2%})")
    return df_filtered

def shift_timestamps(df, months, days):
    df = df.copy()
    # Apply the shift
    df["FinalTimestamp"] = pd.to_datetime(df["FinalTimestamp"])
    df["FinalTimestamp"] = (df["FinalTimestamp"] + pd.DateOffset(months=months, days=days))

    return df

def sampling_and_anonymization(df_filtered, months_shift=0, days_shift=0):
    df_sampled, duplication_counter = case_sampling(df_filtered)
    df_noisy = inject_time_noise(df_sampled, duplication_counter)
    df_reconstructed = reconstruct_timestamps(df_noisy)
    df_compressed = compress_timestamps(df_reconstructed)

    df_shifted = shift_timestamps(df_compressed, months_shift, days_shift)
    
    # Anonymize Case IDs
    df_final = anonymize_case_ids(df_shifted)
    df_final = df_final.sort_values("FinalTimestamp").reset_index(drop=True)

    return clean_final_table(df_final)

def main():
    while True:
        dataset_name, delta, condition_number, months, days = get_user_input()

        df = annotation_and_filtering(dataset_name, delta, condition_number)

        if input("\nDo you want to try other values? (y/n): ").strip().lower() != "y":
            break

    df = sampling_and_anonymization(df, months, days)

    print("\nFinal anonymized log:")
    print(tabulate(df.head(10), headers='keys', tablefmt='grid', showindex=False))

    
def main_patterns():
    dataset_name, delta, condition_number = get_user_input(patterns=True)

    df_filtered = annotation_and_filtering(dataset_name, delta, condition_number, False, False)

    print_patterns(df_filtered, "\nMost common full patterns in original log:")

    df_final = sampling_and_anonymization(df_filtered)

    print_patterns(df_final, "\nMost common full patterns in anonymized log")

if __name__ == "__main__":
    #main_patterns()
    main()