#!/bin/bash
#SBATCH -N 2                              # Number of nodes
#SBATCH -p xxxx                           # Partition
#SBATCH --account=xxxxxxxxx               # Account name
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH --cpus-per-task=8                 # CPU cores per task
#SBATCH --mem=200G                        # Memory per node
#SBATCH --gres=gpu:8                      # Number of GPUs per node
#SBATCH --time=4-00:00:00                 # Maximum job time (4 days)
#SBATCH -o ./Report/%j-slurm.out          # Output log file
# Number of GPUs per node
GPUS_PER_NODE=8

# Logging SLURM environment variables
echo "SLURM_NNODES=${SLURM_NNODES}"
echo "NODELIST=${SLURM_JOB_NODELIST}"
echo "SLURM_NODEID=${SLURM_NODEID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

# Set master node and port for distributed training
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(shuf -i 1024-65535 -n 1)

# Enable error handling and debugging for NCCL and CUDA
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # Enable detailed logging for distributed training

# Directory binding for container
CURRENT_DIR=$(pwd)
HOME_DIR=/mnt/home/$USER
BIND_OPTION="--bind $CURRENT_DIR:$HOME_DIR"

# Pre-launch commands to set environment
PRE_LAUNCH="export TORCH_DISTRIBUTED_TIMEOUT=7200;"

# Launch DeepSpeed ZeRO-3 with Accelerate
LAUNCHER="accelerate launch \
    --num_processes=$((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines=$SLURM_NNODES \
    --machine_rank=$SLURM_NODEID \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    --rdzv_backend c10d \
    --deepspeed_config_file ./ds_config.json \
    --deepspeed_hostfile ./hostfile \
    --deepspeed_multinode_launcher standard \
    --dynamo_backend no \
    --use_deepspeed"

# Training script
CMD="./train.py"

# Clear any residual outputs and start training
clear
srun --wait=60 --kill-on-bad-exit=1 --mpi=pmix bash -c "$PRE_LAUNCH $LAUNCHER $CMD"

# Log the end time
echo "END TIME: $(date)"
