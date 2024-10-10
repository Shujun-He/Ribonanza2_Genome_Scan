import argparse
import numpy as np
from Functions import detect_crossed_pairs
#from hungarian import _hungarian
from arnie_utils import *
from tqdm import tqdm
import pandas as pd
import os
from ok_score import *

#create dummy arnie config
with open('arnie_file.txt','w+') as f:
    f.write("linearpartition: . \nTMP: /tmp")
    
os.environ['ARNIEFILE'] = 'arnie_file.txt'

from arnie.pk_predictors import _hungarian

def get_scores(bpps,seq,theta=0.5):
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'U', 4:'N'}

    n=len(bpps)
    structures=[]
    ef1s=[]
    sequences=[]
    pk_string=[]
    global_confidence=[]
    cross_pair_confidence=[]
    for batch_index in range(n):
        
        sequences.append(''.join(mapping[i] for i in seq[batch_index]))
        bpp=bpps[batch_index]

        pseudo_knot_string=len(sequences[-1])*['.']

        

        if 'N' in sequences[-1]:
            #print(sequences[-1])
            structure='.'*len(sequences[-1])
            crossed_pairs,crossed_pairs_set=[],set()
        else:

            structure, bp_list=_hungarian(bpp,theta=theta,min_len_helix=1)
            crossed_pairs,crossed_pairs_set=detect_crossed_pairs(bp_list)
            for i in crossed_pairs_set:
                pseudo_knot_string[i]='X'
        pseudo_knot_string="".join(pseudo_knot_string)
        pk_string.append(pseudo_knot_string)
        #seq=''.join(mapping[i] for i in seq[batch_index])
        

        if len(crossed_pairs)>0 and len(bp_list)>0:

            gc=np.mean([bpp[j, k] for j, k in bp_list])
            cpc=np.mean([bpp[j, k] for j, k in crossed_pairs])
            global_confidence.append(gc)
            cross_pair_confidence.append(cpc)
            ef1 = gc+cpc
            ef1 = ef1/2
        else:
            ef1 = -1
            #gc=np.mean([bpp[j, k] for j, k in bp_list])
            global_confidence.append(-1)
            cross_pair_confidence.append(-1)

        structures.append(structure)
        ef1s.append(ef1)

    data=pd.DataFrame()
    data['sequence']=sequences
    data['structure']=structures
    data['pseudo_knot_string']=pk_string
    data['ef1s']=ef1s
    data['global_confidence']=global_confidence
    data['cross_pair_confidence']=cross_pair_confidence

    return data

from multiprocessing import Pool


def process_batch(args):
    bpps, seq, batch_index = args
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}

    bpp = bpps[batch_index]
    structure, bp_list = _hungarian(bpp)
    crossed_pairs, crossed_pairs_set = detect_crossed_pairs(bp_list)

    pseudo_knot_string = len(structure) * ['.']
    for i in crossed_pairs_set:
        pseudo_knot_string[i] = 'X'
    pseudo_knot_string = "".join(pseudo_knot_string)

    sequence = ''.join(mapping[i] for i in seq[batch_index])

    if len(crossed_pairs) > 0:
        ef1 = np.mean([bpp[j, k] for j, k in bp_list]) +\
              np.mean([bpp[j, k] for j, k in crossed_pairs])
        ef1 = ef1 / 2
    else:
        ef1 = -1

    return sequence, structure, pseudo_knot_string, ef1

def get_scores_parallel(bpps, seq, pool):
    n = len(bpps)
    #with Pool(num_processes) as pool:
    results = pool.map(process_batch, [(bpps, seq, i) for i in range(n)])

    sequences, structures, pk_strings, ef1s = zip(*results)

    data = pd.DataFrame({
        'sequence': sequences,
        'structure': structures,
        'pseudo_knot_string': pk_strings,
        'ef1s': ef1s
    })

    return data

def get_ok_scores(df):
    # Initialize lists for storing the calculated scores
    classic_scores = []
    crossed_pair_scores = []
    crossed_pair_quality_scores = []
    ok_score = []

    shape=df['SHAPE'].to_numpy()
    structures=df['structure'].to_list()

    # Loop through each shape profile and structure, calculate the scores
    for shape_profile, dbn in zip(shape, structures):
        shape_profile = list(shape_profile)
        classic_score = calculateEternaClassicScore(dbn, shape_profile, 0, 0)
        crossed_pair_score, crossed_pair_quality_score = calculateCrossedPairQualityScore(dbn, shape_profile, 0, 0)
        
        classic_scores.append(classic_score)
        crossed_pair_scores.append(crossed_pair_score)
        crossed_pair_quality_scores.append(crossed_pair_quality_score)
        ok_score.append(classic_score/2.+crossed_pair_quality_score/2.)

    df['classic_score']=classic_scores
    df['crossed_pair_score']=crossed_pair_scores
    df['crossed_pair_quality_score']=crossed_pair_quality_scores
    df['ok_score']=ok_score

    return df


