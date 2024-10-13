import os
import numpy as np
import json
import yaml
import re

def dedupe_lists(list_of_lists):
    # Step 1: Convert each sublist to a sorted tuple
    tuple_set = {tuple(sorted(sublist)) for sublist in list_of_lists}
    
    # Step 2: Convert the set of tuples back to a list of lists
    deduped_list = [list(tup) for tup in tuple_set]
    
    return deduped_list

def detect_crossed_pairs(bp_list):
    """
    Detect crossed base pairs in a list of base pairs in RNA secondary structure.

    Args:
    bp_list (list of tuples): List of base pairs, where each tuple (i, j) represents a base pair.
    
    Returns:
    list of tuples: List of crossed base pairs.
    """
    crossed_pairs_set = set()
    crossed_pairs = []
    # Iterate through each pair of base pairs
    for i in range(len(bp_list)):
        for j in range(i+1, len(bp_list)):
            bp1 = bp_list[i]
            bp2 = bp_list[j]

            # Check if they are crossed
            if (bp1[0] < bp2[0] < bp1[1] < bp2[1]) or (bp2[0] < bp1[0] < bp2[1] < bp1[1]):
                crossed_pairs.append(bp1)
                crossed_pairs.append(bp2)
                crossed_pairs_set.add(bp1[0])
                crossed_pairs_set.add(bp1[1])
                crossed_pairs_set.add(bp2[0])
                crossed_pairs_set.add(bp2[1])
    return dedupe_lists(crossed_pairs), crossed_pairs_set


# def detect_crossed_pairs(bp_list):
#     """
#     Detect crossed base pairs in a list of base pairs in RNA secondary structure,
#     excluding singlet base pairs.

#     Args:
#     bp_list (list of tuples): List of base pairs, where each tuple (i, j) represents a base pair.
    
#     Returns:
#     list of tuples: List of crossed base pairs.
#     """
#     crossed_pairs = set()


#     # Function to check if a base pair is a singlet
#     def is_singlet(bp, bp_list):
#         for other_bp in bp_list:
#             if other_bp == bp:
#                 continue
#             if (other_bp[0] == bp[0] - 1 or other_bp[0] == bp[1] + 1) and \
#                (other_bp[1] == bp[0] + 1 or other_bp[1] == bp[1] - 1):
#                 return False
#         return True

#     # Iterate through each pair of base pairs
#     for i in range(len(bp_list)):
#         for j in range(i + 1, len(bp_list)):
#             bp1 = bp_list[i]
#             bp2 = bp_list[j]

#             # Check if they are crossed and not singlets
#             if ((bp1[0] < bp2[0] < bp1[1] < bp2[1]) or (bp2[0] < bp1[0] < bp2[1] < bp1[1])) \
#                     and not (is_singlet(bp1, bp_list) or is_singlet(bp2, bp_list)):
#                 crossed_pairs.add(bp1[0])
#                 crossed_pairs.add(bp1[1])
#                 crossed_pairs.add(bp2[0])
#                 crossed_pairs.add(bp2[1])

#     return crossed_pairs
def is_singlet(bp, bp_list):
    for other_bp in bp_list:
        if other_bp == bp:
            continue
        if (other_bp[0] == bp[0] - 1 and other_bp[1] == bp[1] + 1) or \
            (other_bp[0] == bp[0] + 1 and other_bp[1] == bp[1] - 1):
            return False
    return True


def read_bp(file):
    pairs=[]
    paired_positions=set()
    for line in open(file,'r'):
        items=line.split()
        if len(items)==3:
            i, nt, j = line.split()
            if j!='0':
                pairs.append((int(i)-1,int(j)-1))
                #paired_positions.add(int(i)-1)
                #paired_positions.add(int(j)-1)
    #to_remove=set()
    non_singlet_pairs=[]#.append(pair)
    for i,pair in enumerate(pairs):
        if not is_singlet(pair, pairs):
            non_singlet_pairs.append(pair)

    # non_singlet_pairs=[]
    # for i,pair in enumerate(pairs):

    #     non_singlet_pairs.append(pair)
    # pairs=[p if i not in to_remove for i,p in enumerate(pairs)]

    for pair in non_singlet_pairs:
        paired_positions.add(pair[0])
        paired_positions.add(pair[1])

    return non_singlet_pairs, paired_positions



def get_PK_TK(sequence,process_id,linearpartition_path):
    L=len(sequence)
    os.system(f'echo {sequence} | ./{linearpartition_path}/linearpartition -T --threshold 0 > tmp{process_id}.txt 2>&1')
    bp=read_bp(f"tmp{process_id}.txt")
    #structure=convert_bp_list_to_dotbracket(bp,L)
    return bp

def get_ok_score(sequence,process_id,linearpartition_path,reactivity):
    bp, paired_positions=get_PK_TK(sequence,process_id,linearpartition_path)
    crossed_positions=detect_crossed_pairs(bp)
    if len(crossed_positions)>2:
        


        assert len(reactivity)==len(sequence)

        classic_score=0
        ok_score=0

        for i in range(len(sequence)):

            classic_score+=1
            if i in paired_positions and reactivity[i]>0.5:
                classic_score-=1
            elif i not in paired_positions and reactivity[i]<0.125:
                classic_score-=1

            
            if i in crossed_positions:
                ok_score+=1 
                if reactivity[i]>0.5:
                    ok_score-=1

        classic_score=classic_score/len(sequence)
        ok_score=ok_score/len(crossed_positions)

        ok_score=(ok_score+classic_score)/2

        #print(classic_score)
        #print(ok_score)

        return ok_score, bp
    else:
        return 0, bp


def calculate_gc_content(rna_sequence):
    """
    Calculate the GC content of an RNA sequence.

    Parameters:
    rna_sequence (str): The RNA sequence.

    Returns:
    float: The GC content as a percentage.
    """
    gc_count = rna_sequence.count('G') + rna_sequence.count('C')
    total_bases = len(rna_sequence)
    gc_content_percentage = (gc_count / total_bases)
    return gc_content_percentage

def calculate_au_content(rna_sequence):
    """
    Calculate the AU content of an RNA sequence.

    Parameters:
    rna_sequence (str): The RNA sequence.

    Returns:
    float: The AU content as a percentage.
    """
    au_count = rna_sequence.count('A') + rna_sequence.count('U')
    total_bases = len(rna_sequence)
    au_content_percentage = (au_count / total_bases)
    return au_content_percentage

def longest_repeating_subsequence_with_backtrack(str):
    n = len(str)
    # Create and initialize DP table
    dp = [[0 for _ in range(n+1)] for _ in range(n+1)]

    # Fill the DP table
    for i in range(1, n+1):
        for j in range(1, n+1):
            # Check if characters match and are not at the same position
            if str[i-1] == str[j-1] and i != j:
                dp[i][j] = 1 + dp[i-1][j-1]
            else:
                dp[i][j] = max(dp[i][j-1], dp[i-1][j])

    # Backtrack to find the subsequence
    i, j = n, n
    subseq = ""

    while i > 0 and j > 0:
        # Check if this cell is part of the solution
        if dp[i][j] == dp[i-1][j-1] + 1:
            subseq = str[i-1] + subseq
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i-1][j]:
            i -= 1
        else:
            j -= 1

    return dp[n][n], subseq


def find_longest_repeating_nucleotide(rna_sequence):
    if not rna_sequence:
        return None, 0

    max_nucleotide = rna_sequence[0]
    max_count = 1
    current_count = 1

    for i in range(1, len(rna_sequence)):
        if rna_sequence[i] == rna_sequence[i - 1]:
            current_count += 1
        else:
            if current_count > max_count:
                max_count = current_count
                max_nucleotide = rna_sequence[i - 1]
            current_count = 1

    # Check last nucleotide sequence
    if current_count > max_count:
        max_count = current_count
        max_nucleotide = rna_sequence[-1]

    return max_nucleotide, max_count


def count_non_augc_chars(input_string):
    non_augc_count = 0
    for char in input_string:
        if char not in "AUGC":
            non_augc_count += 1
    return non_augc_count



class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries=entries

    def print(self):
        print(self.entries)

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(**config)

def mask_diagonal(matrix, mask_value=0):
    matrix=matrix.copy()
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if abs(i - j) < 4:
                matrix[i][j] = mask_value
    return matrix

def sanitize_chromosome_name(name):
    # Remove or replace characters that are problematic for directory names
    # Keep alphanumeric characters, underscores, and hyphens
    return re.sub(r'[^\w\-]', '_', name)