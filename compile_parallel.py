import os
import polars as pl
from glob import glob
from tqdm import tqdm
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
from Functions import *
from Scoring import *

parser = argparse.ArgumentParser(description='Deep Learning Hyperparameters')
parser.add_argument('--chromosome', type=str, default="CM044487.1")
args = parser.parse_args()

batch_size = 32
n_gpus = 8
stride = 50
window = 200

os.system('mkdir -p compiled')

def process_file(file, folder, window):
    df = pl.read_parquet(file)
    
    
    filter = np.ones(len(df), dtype=bool)
    start = df['start_position'].to_list()
    sequence = df['sequence'].to_list()
    
    for i, (s_pos, seq) in enumerate(zip(start, sequence)):
        filter[i] = (df[i, 'ok_score'] > 80)
        filter[i] = filter[i] and (df[i, 'cross_pair_confidence'] > 0)
        filter[i] = filter[i] and (calculate_gc_content(seq) < 0.9)
        filter[i] = filter[i] and (calculate_au_content(seq) < 0.9)
        filter[i] = filter[i] and (find_longest_repeating_nucleotide(seq)[1] < 11)
    
    df=df.filter(filter)
    df = get_ok_scores_exclude_singlets(df)

    return df

def parallel_process_files(files, folder, window):
    with Pool(processes=cpu_count()) as pool:
        func = partial(process_file, folder=folder, window=window)
        dfs = list(tqdm(pool.imap(func, files), total=len(files)))
    return dfs

def main():
    folder = args.chromosome
    parquet_files = glob(f"predictions/{folder}/*.parquet")
    
    # Process files in parallel
    dfs = parallel_process_files(parquet_files, folder, window)
    
    # Concatenate all filtered DataFrames
    all_data = pl.concat(dfs, how='vertical_relaxed')
    
    # Write the final concatenated DataFrame to a parquet file
    all_data.write_parquet(f"compiled/{args.chromosome}.parquet".replace("*",'-'))

if __name__ == "__main__":
    main()

