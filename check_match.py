import pandas as pd
from tqdm import tqdm
import numpy as np

def hamming_distance(str1, str2):
    # Ensure the strings are of equal length
    if len(str1) != len(str2):
        raise ValueError("Strings must be of the same length")

    # Calculate the Hamming distance
    distance = 0
    for ch1, ch2 in zip(str1, str2):
        if ch1 != ch2:
            distance += 1

    return distance

window=100
compiled_data=pd.read_parquet("compiled.parquet")
chromosome=open("chromosomes/NC_000001.11.txt").read()

top=150000

filter=[]
#for i in tqdm(range(len(compiled_data))):
for i in tqdm(range(150000)):
    sequence=compiled_data['sequence'][i]
    start_position=compiled_data['start_position'][i]

    if sequence!=chromosome[start_position:start_position+window]:
        print("###")
        print(sequence)
        print(chromosome[start_position:start_position+window])
        filter.append(False)
    else:
        filter.append(True)
    #hammin_distance_to_reference.append(hamming_distance(sequence,chromosome[start_position:start_position+window]))
    #exit()

compiled_data=compiled_data.loc[:top-1].loc[filter]

compiled_data=compiled_data.drop_duplicates('sequence').reset_index(drop=True)

compiled_data.to_parquet("compiled_post_processed.parquet")