
from dp_sequential_events.main.main import annotation_and_filtering
from dp_sequential_events.main.main  import sampling_and_anonymization
from dp_sequential_events.main.main  import print_patterns

import rich
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

console = rich.console.Console()

def calculate_transition_times(df, version_name):
   df_temp = df.copy()
   
   col_time = 'FinalTimestamp' if 'FinalTimestamp' in df_temp.columns else 'Timestamp'
   df_temp[col_time] = pd.to_datetime(df_temp[col_time])
   
   df_temp = df_temp.sort_values(by=['CaseID', col_time])
   
   df_temp['Former Timestamp'] = df_temp.groupby('CaseID')[col_time].shift(1)
   df_temp['Transition Time (min)'] = (df_temp[col_time] - df_temp['Former Timestamp']).dt.total_seconds() / 60.0
   
   df_temp['Version'] = version_name
   
   return df_temp.dropna(subset=['Transition Time (min)'])

if __name__ == "__main__":
   dataset = input("Enter the dataset name: ")
   delta = 0.3
   condition_number = 1
   
   # Original dataset
   df_filtered = annotation_and_filtering(dataset, delta, condition_number, False)
   console.rule("[bold cyan]PATTERNS (ORIGINAL)")
   original = print_patterns(df_filtered, "\nMost common full patterns in original log:")
   df_final = sampling_and_anonymization(df_filtered)
   
   # Anonymized dataset
   console.rule("[bold cyan]PATTERNS (ANONYMIZED)")
   anonymized = print_patterns(df_final, "\nMost common full patterns in anonymized log:")

   # Generar el gráfico
   with console.status("[bold magenta]Calculando tiempos y dibujando gráfico..."):
      # Calculate transition times for both datasets
      orig_times = calculate_transition_times(df_filtered, "Original")
      anon_times = calculate_transition_times(df_final, "Anonymized")

      # Combine the two datasets for plotting
      comparative_df = pd.concat([orig_times, anon_times])

      # Draw the boxplot using Seaborn
      sns.set_theme(style="whitegrid")
      plt.figure(figsize=(8, 6))
      sns.boxplot(
         x="Version", 
         y="Transition Time (min)",
         data=comparative_df, 
         palette="Set2", 
         showfliers=False
      )
      
      plt.title("Comparación de Tiempos de Transición", fontsize=14)
      plt.ylabel("Tiempo entre eventos (Minutos)", fontsize=12)
      plt.xlabel("")
      
      # Guardar en PDF de alta calidad
      plt.savefig("boxplot_tiempos.pdf", bbox_inches='tight')
        
      console.print("\n[bold green]✔[/bold green] Gráfico guardado exitosamente como [bold]boxplot_tiempos.pdf[/bold]")
    
   # Mostrar el gráfico por pantalla
   plt.show()