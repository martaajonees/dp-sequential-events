import pandas as pd
import random
from datetime import datetime, timedelta
import pm4py

# Define the variants and their frequencies
# R1
variants = {
    "A B C E F": 900,      # Aprobado estándar sin uso intensivo de foros
    "A B C D E F": 632,    # Aprobado con alta participación colaborativa en foros
    "A B C G": 350,        # Suspenso por inactividad o falta de entrega de evaluaciones
    "A B C E G": 355,      # Suspenso habiendo entregado las evaluaciones
    "A H": 200,            # Abandono temprano (registro sin llegar a entrar al VLE)
    "A B C H": 500,        # Abandono a mitad de curso (interactúa pero no entrega nada)
    "A B C E H": 321      # Abandono tardío (entrega alguna evaluación antes de retirarse)
}
# R2
# variants = {
#     # Variante de suspenso larga (basada en Figura 3):
#     # Empieza en quiz, pasa por el foro, y termina mirando los materiales (page view / url view) al final.
#     "F G H C D I J B K": 15,  
    
#     # Variante de suspenso corta:
#     # Intenta el quiz directamente, pide ayuda en el foro y cierra el cuestionario.
#     "F G C D I K": 16,        

#     # Ruta 1: Perfil Social / Colaborativo (basada en Figura 4).
#     # Empiezan en el foro, publican, actualizan posts y luego van al examen.
#     "A C D E F G I K L": 25,  
    
#     # Ruta 2: Perfil Orientado al Contenido / Individual.
#     # Leen discusiones del foro pasivamente, consumen el contenido y hacen el examen.
#     "A B J F G I K L": 25,    
    
#     # Ruta 3: Perfil Integral (Alta participación).
#     # Tocan todas las ramas (foro, materiales, y cuestionarios).
#     "A B C D E F G I J K L": 20 
# }
# R3
# variants = {
#     #  Lee las instrucciones, el texto, mira el gráfico, 
#     # lee la pregunta y responde directamente sin dudar.
#     "A B C D G": 350,          

#     # Va directamente al gráfico para hacerse una idea 
#     # global antes de leer la teoría, lee la pregunta y responde.
#     "A C B D G": 200,          

#     # Lee la pregunta primero y luego hace un escaneo 
#     # rápido (E, F) buscando palabras clave o datos visuales para responder.
#     "A D E F G": 150,          

#     # Lee todo, pero duda al ver la pregunta. Alterna 
#     # miradas rápidas y lentas entre el texto (E) y el gráfico (C, F) para relacionarlos.
#     "A B C D E C F G": 90,     

#     # Ignora el texto casi por completo y se queda 
#     # atrapado en un bucle visual intentando descifrar la imagen antes de responder.
#     "A C F C F D G": 40        
# }
# Generate synthetic log data
data = []
case_id = 1000
ini_date = datetime(2020, 1, 1, 8, 0, 0)

for traza, frecuencia in variants.items():
    activities = traza.split()
    for _ in range(frecuencia):
        actual_time = ini_date + timedelta(days=random.randint(0, 30), minutes=random.randint(0, 1440))
        
        for act in activities:
            data.append([case_id, act, actual_time])
            actual_time += timedelta(minutes=random.randint(15, 120))
            
        case_id += 1

# Construct DataFrame
df = pd.DataFrame(data, columns=['CaseID', 'Activity', 'Timestamp'])
df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

# Sort
df = df.sort_values(by=['CaseID', 'Timestamp'])
df.to_csv('../databases/synthetic_data_reg1.csv', index=False)
print("Dataset generated")