import pandas as pd
from datetime import datetime, timedelta
import random

def generate_synthetic_data(num_ids=50, interactions_per_id=5, start_date="2020-01-01"):
    activity_paths = [
        ['A', 'B', 'C', 'D', 'E'],
        ['A', 'C', 'D', 'E'],
        ['A', 'B', 'D', 'E'],
        ['A', 'C', 'E']
    ]

    data = []
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")

    for case_id in range(1000, 1000 + num_ids):
        current_time = start_datetime
        path = random.choice(activity_paths)
        for i in range(interactions_per_id):
            activity = path[i % len(path)]  # Cicla si interactions_per_id > longitud del path
            data.append([case_id, activity, current_time.strftime("%Y-%m-%d %H:%M:%S")])
            current_time += timedelta(minutes=random.randint(10, 240))

    df = pd.DataFrame(data, columns=["CaseID", "Activity", "Timestamp"])
    return df

# Uso
df = generate_synthetic_data(num_ids=50, interactions_per_id=20)
df.to_csv("synthetic_data_simple.csv", index=False)
print(df.head(20))