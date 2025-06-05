#!/bin/bash

#SBATCH --job-name=mus_genome_scan
#SBATCH -D .
#SBATCH --output=O-%x.%j
#SBATCH --error=E-%x.%j
#SBATCH --nodes=32                   # Number of nodes
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --gpus-per-node=8
#SBATCH --partition=defq

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
node_list=$(scontrol show hostnames $SLURM_JOB_NODELIST)  # Get all node names

export ACCELERATE_DIR="/lustre/fs0/scratch/shujun/miniconda3/envs/torch/bin/accelerate"
export SCRIPT="inference.py --genome_file ../renamed_genomes/mus.fna"
# export SCRIPT_ARGS=" \
#     --mixed_precision fp16 \
#     --output_dir ${ACCELERATE_DIR}/examples/output \
#     "

# Loop over each node and launch accelerate
i=0
for node in $node_list; do
    export CONFIG_FILE="accelerate_configs/accelerate_config_node_${i}.yaml"
    export LAUNCHER="accelerate launch --config_file ${CONFIG_FILE}"
    export CMD="$LAUNCHER $SCRIPT"

    echo "Launching on node $node with config $CONFIG_FILE"
    
    # Launch the job on the specific node using srun
    srun --nodes=1 --nodelist=$node --ntasks=1 $CMD &

    i=$((i+1))  # Increment node index
done

wait  # Wait for all background processes to complete
