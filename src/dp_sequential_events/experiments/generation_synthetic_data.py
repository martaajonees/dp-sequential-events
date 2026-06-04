import pandas as pd
import random
from datetime import datetime, timedelta
import pm4py

# Define the variants and their frequencies
# R1
variants = {
    "A B C E F": 900,     
    "A B C D E F": 632,    
    "A B C G": 350,        
    "A B C E G": 355,     
    "A H": 200,            
    "A B C H": 500,       
    "A B C E H": 321      
}
# R2
# variants = {
#     "F G H C D I J B K": 15,  
#     "F G C D I K": 16,        
#     "A C D E F G I K L": 25,  
#     "A B J F G I K L": 25,    
#     "A B C D E F G I J K L": 20 
# }
# R3
# variants = {
#     "A B C D G": 350,          
#     "A C B D G": 200,          
#     "A D E F G": 150,          
#     "A B C D E C F G": 90,     
#     "A C F C F D G": 40        
# }

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