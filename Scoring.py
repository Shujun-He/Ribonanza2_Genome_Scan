import argparse
import numpy as np
from Functions import detect_crossed_pairs
#from hungarian import _hungarian
#from arnie_utils import *
from tqdm import tqdm
import pandas as pd
import os
from ok_score import *
import polars as pl

#create dummy arnie config
# with open('arnie_file.txt','w+') as f:
#     f.write("linearpartition: . \nTMP: /tmp")
    
os.environ['ARNIEFILE'] = '../arnie_file.txt'

from arnie.pk_predictors import _hungarian
from arnie.utils import post_process_struct, convert_dotbracket_to_bp_list

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

def get_helices(bp_list, allowed_buldge_len=0):
    #bp_list = convert_dotbracket_to_bp_list(s, allow_pseudoknots=True)
    bp_list = bp_list[:]
    helices = []
    current_helix = []
    while bp_list != []:
        current_bp = bp_list.pop(0)
        if current_helix == []:
            current_helix.append(current_bp)
        else:
            in_helix_left = list(range(current_helix[-1][0] + 1, current_helix[-1][0] + allowed_buldge_len + 2))
            in_helix_right = list(range(current_helix[-1][1] - allowed_buldge_len - 1, current_helix[-1][1]))
            if current_bp[0] in in_helix_left and current_bp[1] in in_helix_right:
                current_helix.append(current_bp)
            else:
                helices.append(current_helix)
                current_helix = [current_bp]
    helices.append(current_helix)
    return helices

def get_ok_scores_exclude_singlets(df):
    # Initialize lists for storing the calculated scores
    classic_scores = []
    crossed_pair_scores = []
    crossed_pair_quality_scores = []
    ok_score = []

    shape=df['SHAPE'].to_numpy()
    structures=df['structure'].to_list()

    structures=[post_process_struct(s,0,2) for s in structures]

    df=df.with_columns(pl.Series("structure_no_singlet",structures))

    # cp_helices=[get_helices(detect_crossed_pairs(convert_dotbracket_to_bp_list(s, allow_pseudoknots=True))[0]) for s in structures]

    # min_len_cp_helix=[]
    # for h in cp_helices:
    #     if len(h)>0:
    #         min_len_cp_helix.append(min([len(stem) for stem in h]))
    #     else:
    #         min_len_cp_helix.append(-1)
    min_len_cp_helix=[]
    for s in structures:
        bps=convert_dotbracket_to_bp_list(s, allow_pseudoknots=True)
        crossed_pairs=detect_crossed_pairs(bps)[0]
        crossed_pairs.sort()
        helices=get_helices(crossed_pairs)
        if len(helices)>0:
            min_len_cp_helix.append(min([len(stem) for stem in helices]))
        else:
            min_len_cp_helix.append(-1)

    df=df.with_columns(pl.Series("min_len_cp_helix",min_len_cp_helix))
    # Loop through each shape profile and structure, calculate the scores
    # for shape_profile, dbn in zip(shape, structures):
    #     shape_profile = list(shape_profile)
    #     classic_score = calculateEternaClassicScore(dbn, shape_profile, 0, 0)
    #     crossed_pair_score, crossed_pair_quality_score = calculateCrossedPairQualityScore(dbn, shape_profile, 0, 0)
        
    #     classic_scores.append(classic_score)
    #     crossed_pair_scores.append(crossed_pair_score)
    #     crossed_pair_quality_scores.append(crossed_pair_quality_score)
    #     ok_score.append(classic_score/2.+crossed_pair_quality_score/2.)

    # df = df.with_columns([
    #     pl.Series('classic_score_no_singlet', classic_scores, dtype=pl.Float32),
    #     pl.Series('crossed_pair_score_no_singlet', crossed_pair_scores, dtype=pl.Float32),
    #     pl.Series('crossed_pair_quality_score_no_singlet', crossed_pair_quality_scores, dtype=pl.Float32),
    #     pl.Series('ok_score_no_singlet', ok_score, dtype=pl.Float32)
    # ])

    return df
