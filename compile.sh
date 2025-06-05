#!/bin/bash
#SBATCH --nodes 1
#SBATCH --job-name=interactive-job
#SBATCH --partition cpuq

python compile_parallel.py --organism homo 
python compile_parallel.py --organism Danio
python compile_parallel.py --organism Rattus_norvegicus
python compile_parallel.py --organism mus




