
from dp_sequential_events.main.main import annotation_and_filtering, sampling_and_anonymization

def cli_main():
    dataset_name = input("Enter dataset name : ")
    delta = float(input("Enter delta value: "))
    condition_number = float(input("Enter condition number: "))

    df_filtered = annotation_and_filtering(dataset_name, delta, condition_number)

    input("Do you want to choose other values? (y/n): ")
    if input().lower() == 'y':
        cli_main()
    else:
        df = sampling_and_anonymization(df_filtered)
        print("\n Final anonymized log:")
        print(df)
    
