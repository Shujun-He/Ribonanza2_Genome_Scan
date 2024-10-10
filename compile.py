import os
import polars as pl
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
from Functions import *

parser = argparse.ArgumentParser(description='Deep Learning Hyperparameters')
parser.add_argument('--chromosome', type=str, default="NC_000001.11")

args = parser.parse_args()

batch_size=32
n_gpus=8
stride=50
window=200

os.system('mkdir compiled')

ef1s=[]
dfs=[]
cross_pair_confidence=[]

folder=args.chromosome
chromosome=open(f"chromosomes/{args.chromosome}.txt").read()

for f in tqdm(glob(f"predictions/{folder}/*.parquet")):
    df=pl.read_parquet(f)

    df = df.with_columns(pl.col("global_confidence").cast(pl.Float32))
    df = df.with_columns(pl.col("cross_pair_confidence").cast(pl.Float32))
    df = df.with_columns(pl.col("ef1s").cast(pl.Float32))
    df = df.with_columns(pl.lit(folder).alias("chromosome"))

    #filter
    filter=np.ones(len(df))
    start=df['start_position'].to_list()
    sequence=df['sequence'].to_list()
    for i, (s_pos, seq) in enumerate(zip(start,sequence)):
        filter[i]= filter[i] * (seq==chromosome[s_pos:s_pos+window])
        filter[i]= filter[i] * (calculate_gc_content(seq)<0.9)
        filter[i]= filter[i] * (calculate_au_content(seq)<0.9)
        filter[i]= filter[i] * (find_longest_repeating_nucleotide(seq)[1]<11)
    dfs.append(df.filter(filter.astype('bool')))
    #exit()

all_data=pl.concat(dfs)

def openknot_ef1(data):
    crossed_pair_ef1=6.2*data['cross_pair_confidence'].to_numpy()-5.17
    global_ef1=3.66*data['global_confidence'].to_numpy()-2.7
    ok_ef1= (crossed_pair_ef1+global_ef1)/2

    data=data.with_columns(pl.Series("crossed_pair_ef1",crossed_pair_ef1))
    data=data.with_columns(pl.Series("global_ef1",global_ef1))
    data=data.with_columns(pl.Series("ok_ef1",ok_ef1))

    return data

all_data=openknot_ef1(all_data)
all_data=all_data.unique(subset='sequence').sort('ok_ef1',descending=True)

all_data.write_parquet(f"compiled/{args.chromosome}.parquet")

