import polars as pl
from glob import glob
from tqdm import tqdm

# concat all the CSV files in generated_structures
# first 
# 1. sort by ok_score*ef1s and get top 2M
# 2. sort and drop duplicates by sequence with hamming distance 5 to get 1M
# second  
# 1. sort by (ok_score)*(1-ef1s) and get top 2M
# 2. sort and drop duplicates by sequence with hamming distance 5 to get 1M

#ef1s is average confidence of global and cross pair confidence

folder="*_compiled/*parquet"
csv_files = glob(f"{folder}")
print(csv_files)
dfs = [pl.read_parquet(file) for file in tqdm(csv_files, desc="Reading CSV files")]
df = pl.concat(dfs, rechunk=True)
df = df.unique(subset=["sequence"])

# print(df.shape)
# exit()

#df = df.with_columns(pl.Series('ok_score_jaccard', df['ok_score'] * df['jaccard']))
df =   df.with_columns(pl.Series('ok_score_ef1s', df['ok_score'] * df['ef1s']))
df =   df.with_columns(pl.Series('anti_ok_score_ef1s', (100-df['ok_score']) * (1-df['ef1s'])))

print(f"Total sequences before processing: {len(df)}")
df1 = df.sort('ok_score_ef1s', descending=True).head(2000000).sort('sequence')
df2 = df.sort('anti_ok_score_ef1s', descending=True).head(2000000).sort('sequence')

def hamming_distance(seq1, seq2):
    """Calculate the Hamming distance between two sequences."""
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of the same length")
    return sum(el1 != el2 for el1, el2 in zip(seq1, seq2))

def drop_duplicates_with_hamming_distance(df, distance_cutoff=4):
    #only compare with previous sequence 
    unique_sequences = []
    filtered_rows = []

    previous_sequences = None
    for row in df.iter_rows(named=True):
        sequence = row['sequence']

        distance = hamming_distance(sequence, previous_sequences) if previous_sequences else float('inf')

        if distance >= distance_cutoff:
            filtered_rows.append(row)
        previous_sequences = sequence
    return pl.DataFrame(filtered_rows)

df1_unique = drop_duplicates_with_hamming_distance(df1)[:1000000]
df2_unique = drop_duplicates_with_hamming_distance(df2)[:1000000]

# df1_unique = df1_unique.unique(subset=["sequence"])[:1000000]
# df2_unique = df2_unique.unique(subset=["sequence"])[:1000000]

df=pl.concat([df1_unique, df2_unique])
unique_sequences = len(df.unique(subset=["sequence"]))
print(f"Total unique sequences after processing: {unique_sequences}")

#df1_unique.write_csv(f"Genome_scan_RN2.1_1M.csv")
df1_unique.write_parquet(f"Genome_scan_RN2.1_1M.parquet")
#df2_unique.write_csv(f"Genome_scan_RN2.1_anti_1M.csv")
df2_unique.write_parquet(f"Genome_scan_RN2.1_anti_1M.parquet")
print("Data processing complete. Files saved as Genome_scan_RN2.1_1M.parquet and Genome_scan_RN2.1_anti_1M.parquet.")



