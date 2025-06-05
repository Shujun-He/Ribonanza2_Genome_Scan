import polars as pl
from glob import glob
from tqdm import tqdm

dfs=[]
for f in tqdm(glob("homo_compiled/*.parquet")):
    df=pl.read_parquet(f)
    dfs.append(df)

for f in tqdm(glob("Danio_compiled/*.parquet")):
    df=pl.read_parquet(f)
    dfs.append(df)

df=pl.concat(dfs)
#exit()

seq_len=100
df=df.with_columns(pl.Series("structuredness",[(seq_len-s.count('.'))/seq_len for s in df['structure']]))


df=df.sort(["structuredness","ok_score"],descending=[True,True])

#take top 1M and save
df=df[:1000000][['structure','ok_score']]#,'cross_pair_confidence','sequence']]


df.write_csv("top1M_for_rl_generation.csv")
