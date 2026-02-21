from collections import Counter
import pandas as pd

def most_common_patterns(df):
    sequences = []

    # Group by CaseID
    for case_id, group in df.groupby('CaseID'):
        # Sort by Timestamp to preserve sequence
        activities = group.sort_values('Timestamp')['Activity'].tolist()
        # Convert the sequence to a string pattern
        pattern_str = ''.join(activities)
        sequences.append(pattern_str)

    # Count frequency of each pattern
    counter = Counter(sequences)

    # Convert to DataFrame
    df_patterns = pd.DataFrame(counter.items(), columns=['Pattern', 'Count'])
    df_patterns = df_patterns.sort_values(by='Count', ascending=False).reset_index(drop=True)

    return df_patterns
