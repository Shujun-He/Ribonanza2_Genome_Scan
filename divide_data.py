from Bio import SeqIO

#using NC_000001.11 for now
import os

os.system('mkdir chromosomes')


for record in SeqIO.parse("../input/GRCh38_latest_genomic.fna", "fasta"):
    print(record.id)
    sequence_id=record.id
    sequence=str(record.seq).upper().replace('T','U')
    #break

    with open(f"chromosomes/{sequence_id}.txt",'w+') as f:
        f.write(sequence)