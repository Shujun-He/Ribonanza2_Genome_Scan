
from glob import glob
import polars as pl
import os
from tqdm import tqdm

files=glob("predictions/NC_000001.11/*.parquet")
print(len(files))

data=[]
for f in tqdm(files):
    df=pl.read_parquet(f)
    filtered_df=df.filter(df['ok_score']>80)
    if len(filtered_df)>0:
        data.append(filtered_df)

df=pl.concat(data,how="vertical_relaxed")
