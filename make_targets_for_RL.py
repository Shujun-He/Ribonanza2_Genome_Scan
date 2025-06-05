import polars as pl
df=pl.read_parquet("compiled/NC-.parquet")


seq_len=240
df=df.with_columns(pl.Series("structuredness",[(seq_len-s.count('.'))/seq_len for s in df['structure']]))


df=df.sort(["structuredness","ok_score"],descending=[True,True])

#take top 1M and save
df=df[:2000000][['structure','ok_score']]#,'cross_pair_confidence','sequence']]


df.write_csv("top2M.csv")
