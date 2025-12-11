
import pandas as pd
from dafsa import DAFSA

def make_graph(data_name):
    sequences = []

    with open(data_name, 'r') as f:
        for line in f:
            seq = line.strip()
            if seq:
                sequences.append(seq)
    
    dafsa = DAFSA(sequences)
    return dafsa


dfa = make_graph("datos.txt")
dot_text = dfa.to_dot()

with open("datos.dot", "w") as f:
    f.write(dot_text)