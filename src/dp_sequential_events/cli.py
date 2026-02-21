
from dp_sequential_events.main.main import annotation_and_filtering, sampling_and_anonymization
from tabulate import tabulate

def cli_main():
    while True:
        dataset_name = input("Enter dataset name : ")
        delta = float(input("Enter delta value: "))
        condition_number = float(input("Enter condition number: "))

        df_filtered = annotation_and_filtering(dataset_name, delta, condition_number)

        repeat = input("Do you want to choose other values? (y/n): ")
        if repeat.lower() == 'n':
            break
            
    df = sampling_and_anonymization(df_filtered)
    print("\n Final anonymized log:")
    print(tabulate(df.head(10), headers='keys', tablefmt='grid', showindex=False))
    
