from glob import glob
import polars as pl
from tqdm import tqdm

all_files=glob("predictions/*/*")

cnt=0
for f in tqdm(all_files):
    cnt+=len(pl.read_parquet(f))

print(cnt)