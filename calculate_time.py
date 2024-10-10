import pandas as pd
from Dataset import *
from Network import *
#from Functions import *
from tqdm import tqdm
from sklearn.model_selection import KFold
import argparse
from sklearn.metrics import mean_squared_error
from accelerate import Accelerator
import time
import json
import yaml
from Scoring import get_scores, get_scores_parallel
from Functions import *
from Bio import SeqIO


parser = argparse.ArgumentParser(description='Deep Learning Hyperparameters')
parser.add_argument('--config_path', type=str, default="configs/pairwise.yaml")
parser.add_argument('--genome_file', type=str, default="../genomes/mus.fna")


args = parser.parse_args()

config = load_config_from_yaml(args.config_path)



start_time = time.time()



for record in SeqIO.parse("../../input/GRCh38_latest_genomic.fna", "fasta"):
    #print(record.id)
    #if record.id=="NC_000002.12":
    print(record.id)
    chromosome=record.id
    sequence=str(record.seq)#.upper().replace('T','U')
    print(len(sequence))
    break
baseline=len(sequence)

total_nts=0
for record in SeqIO.parse(args.genome_file, "fasta"):
    #print(record.id)
    #if record.id=="NC_000002.12":
    print(record.id)
    chromosome=record.id
    sequence=str(record.seq).upper().replace('T','U')
    total_nts+=len(sequence)

total_time=total_nts/baseline*2.5

print(total_time)
