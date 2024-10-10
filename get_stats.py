import polars as pl
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

parquet_files=glob("compiled/*parquet")

topk=150000
dfs=[]
for f in tqdm(parquet_files):
    df=pl.read_parquet(f)#[:topk]
    dfs.append(df)

dfs=pl.concat(dfs)
dfs=dfs.unique(subset='sequence').sort('ok_ef1',descending=True)[:topk]

dfs.write_parquet("curated/compiled_topk.parquet")
dfs.write_csv("curated/compiled_topk.csv")

plt.subplot(311)
#plt.title('global_confidence',labe;l)
plt.hist(dfs['global_confidence'],bins=30,label='global_confidence',alpha=0.5)
plt.hist(dfs['cross_pair_confidence'],bins=30,label='crossed_pair_confidence',alpha=0.5)
plt.legend()
plt.subplot(312)
#plt.title('crossed_pair_ef1')
plt.hist(dfs['crossed_pair_ef1'],bins=30,label='crossed_pair_ef1',alpha=0.5)
plt.hist(dfs['global_ef1'],bins=30,label='global_ef1',alpha=0.5)
plt.legend()
plt.subplot(313)
#plt.title('ok_ef1')
plt.hist(dfs['ok_ef1'],bins=30,label='ok_ef1')
plt.legend()
plt.tight_layout()
plt.savefig("stats.png",dpi=250)