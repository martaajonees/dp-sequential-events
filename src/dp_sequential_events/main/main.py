
from dp_sequential_events.main.annotated import DAFSA_annotated_table
from dp_sequential_events.main.filtered import DAFSA_filtrated
from dp_sequential_events.main.case_sampling import case_sampling, inject_time_noise, reconstruct_timestamps, compress_timestamps, anonymize_case_ids, clean_final_table
from dp_sequential_events.main.patterns import most_common_patterns
from pathlib import Path
import pandas as pd
from datetime import datetime
import os
import random
import sys

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.status import Status
from InquirerPy import inquirer

console = Console()

# --- UTILS ----
def get_user_input(patterns=False):
    while True:
        try: 
            dataset_name = input("\nEnter dataset path: ").strip()
            if not Path(dataset_name).is_file():
                raise ValueError("File does not exist. Please try again.")
            if not dataset_name.endswith(".csv"):
                raise ValueError("Dataset must be a CSV file")
            
            delta = float(input("Enter delta value (0-1): "))
            if not (0 <= delta < 1):
                raise ValueError("Delta value must be between 0 and 1")
            
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
            console.print(f"[bold red]Invalid input: {e}. Please try again.[/bold red]")

def print_table(df, title=None):
    table = Table(title=title, box=box.ROUNDED, show_lines=True)
    for col in df.columns:
        table.add_column(str(col), justify="center")

    for row in df.head(10).itertuples(index=False):
        table.add_row(*[str(x) for x in row])
    
    console.print(table)

def render_dafsa_tree(graph, start):
    for node in graph.nodes():
        prefix = "[green]●[/]" if node == start else "[white]○[/]"
        
        if graph.degree(node) == 0:
            suffix = "[red](final)[/]"
        else:
            suffix = ""

        transitions = ", ".join(
            f"[cyan]{graph.get_edge_data(node, nbr).get('label','')}[/]→{nbr}"
            for nbr in graph.neighbors(node)
        )

        console.print(f"{prefix} {node} {suffix} :: {transitions}")

def print_patterns(df, title):
    patterns = most_common_patterns(df)
    print_table(patterns, title)
    return patterns

def text_input(message, default=""):
    if is_colab():
        value = input(f"{message} ({default}): ").strip()
        return value if value else default
    else:
        return inquirer.text(
            message=message,
            default=default, 
            qmark="",
            amark=""
        ).execute()

def banner():
    console.print(Panel(
        "\n".join([
            "[bold cyan]DP Sequential Events Tool[/bold cyan]",
            "[grey]Process Mining · Anonymization Engine[/grey]",
            "[dim]University of Cádiz · v1.7.0[/dim]"
        ]),
        box=box.DOUBLE,
        border_style="cyan",
        padding=(1, 8),
        expand=False
    ))

def ask(msg):
    return console.input(f"[bold cyan]{msg}[/bold cyan]")

def get_downloads_folder():
    home = Path.home()
    # Linux / macOS 
    downloads = home / "Downloads"

    # Windows fallback 
    if os.name == "nt":
        downloads = Path(os.path.join(os.environ.get("USERPROFILE", home), "Downloads"))
    return downloads

def select_option(message, choices):
    if is_colab():
        # fallback simple
        print(f"\n{message}")
        for i, choice in enumerate(choices, 1):
            print(f"{i}. {choice}")
        
        while True:
            try:
                idx = int(input("Select option: "))
                if 1 <= idx <= len(choices):
                    return choices[idx - 1]
            except:
                pass
            print("Invalid choice, try again.")
    else:
        return inquirer.select(
            message=message,
            choices=choices,
            qmark="",
            amark=""
        ).execute()

def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

def export_csv(df):
    default_folder = get_downloads_folder()

    folder = text_input(
        message="Enter output folder:",
        default=str(default_folder),
    )
    

    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)

    default_name = f"anonymized_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    filename = text_input("Enter filename:", default_name)

    if not filename.endswith(".csv"):
        filename += ".csv"

    full_path = folder_path / filename

    df.to_csv(full_path, index=False)

    console.print(f"\n[bold green]✔ File saved at:[/] {full_path.resolve()}")

def main_menu():
    return select_option("Select an option:", ["Run full pipeline", "Run patterns-oriented pipeline", "Exit"])

# --- MAIN FUNCTIONS ---
def annotation_and_filtering(data_name, delta=0.3, condition_number=1, _print=True):
    # Annotated table 
    if _print:
        console.rule("[bold green]ANNOTATION")
    with Status("[bold green]Generating DAFSA-annotated table..."):
        df = DAFSA_annotated_table(data_name)

    if _print:
        print_table(df, "Annotated Table")
        console.rule("[bold green]FILTERING")
    
    with Status("[bold green]Filtering DAFSA table..."):
        df_filtered = DAFSA_filtrated(df, delta, condition_number)
        
    if _print:
        print_table(df_filtered, "Filtered Table")

        removed = len(df) - len(df_filtered)
        console.print(
            f"\n[bold yellow]Cases removed:[/bold yellow] {removed} "
            f"([red]{removed / len(df):.2%}[/red])"
        )
    return df_filtered

def shift_timestamps(df, max_months, max_days):
    df = df.copy()
    df["FinalTimestamp"] = pd.to_datetime(df["FinalTimestamp"])

    seed = random.randrange(sys.maxsize)
    random.seed(seed)

    series_act = []
    for _, group in df.groupby("CaseID"):
        original_timestamp = group["FinalTimestamp"]
        m = random.randint(0, max_months)
        d = random.randint(0, max_days)
        shifted_timestamp = original_timestamp + pd.DateOffset(months=m, days=d)
        
        in_december = (original_timestamp.dt.month == 12).any()
        change_year = (original_timestamp.dt.year != shifted_timestamp.dt.year).any()
        if in_december or change_year:
            series_act.append(original_timestamp)
        else:
            series_act.append(shifted_timestamp)
    
    df["FinalTimestamp"] = pd.concat(series_act)
    return df

def sampling_and_anonymization(df_filtered, months_shift=0, days_shift=0):
    with Status("[bold green]Sampling cases..."):
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
        console.clear()
        
        banner()
        choice = main_menu()
        if choice == "Run full pipeline":
            pipeline()
        elif choice == "Run patterns-oriented pipeline":
            patterns()
        else:
            break

def pipeline():
    while True:
        dataset_name, delta, condition_number, months, days = get_user_input()

        df = annotation_and_filtering(dataset_name, delta, condition_number)
        choice = select_option("\nDo you want to try other values?", ["Yes", "No"])
        if choice == "No":
            break

    df = sampling_and_anonymization(df, months, days)

    console.rule("[bold green]FINAL OUTPUT")
    print_table(df, "Final Anonymized Log")

    save = select_option("\nDo you want to save the final CSV?", ["Yes", "No"])

    if save == "Yes":
        export_csv(df)

    console.print("\n[dim]Press ENTER to return to menu...[/dim]")
    input()

    
def patterns():

    dataset_name, delta, condition_number = get_user_input(patterns=True)

    df_filtered = annotation_and_filtering(dataset_name, delta, condition_number, False)

    console.rule("[bold cyan]PATTERNS (ORIGINAL)")
    print_patterns(df_filtered, "\nMost common full patterns in original log:")

    df_final = sampling_and_anonymization(df_filtered)

    console.rule("[bold cyan]PATTERNS (ANONYMIZED)")
    print_patterns(df_final, "\nMost common full patterns in anonymized log")
    
    console.print("\n[dim]Press ENTER to return to menu...[/dim]")
    input()

if __name__ == "__main__":
    main()