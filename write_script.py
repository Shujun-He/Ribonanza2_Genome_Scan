import pandas as pd

df=pd.read_csv("sequence_report.tsv",sep='\t')

df=df.loc[df['Role']=='assembled-molecule'].reset_index(drop=True)

with open('inference.sh',"w+") as f:
    for name in df['RefSeq seq accession']:
        f.write(f"accelerate launch inference_human_genome.py --chromosome_file {name}.txt\n")

with open('compile.sh',"w+") as f:
    for name in df['RefSeq seq accession']:
        f.write(f"nohup python compile.py --chromosome {name} > {name}.out & \n")