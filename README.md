# Ribonanza2_Genome_Scan

How to run: `python inference.py`

command line args are --config_path (included with repo) and --genome_file fasta genome file

You will also need to download weights from Rnet2 alpha and Rnet2-SS  

https://www.kaggle.com/models/shujun717/ribonanzanet2
https://www.kaggle.com/datasets/shujun717/rnet2-alpha-ss-weights

output parquets wiyth stats like SS, OK score are saved to `{genome_name}_predictions` 