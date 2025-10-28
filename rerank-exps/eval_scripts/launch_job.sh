#!/bin/bash


sweep_id=$1
# Load Python environment
source env/bin/activate
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Print debug information
echo "=== Debug Information ==="
echo "Hostname: $(hostname)"
echo "Current directory: $(pwd)"
echo "Date: $(date)"

# Print GPU information if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    echo -e "\n=== GPU Information ==="
    nvidia-smi
else
    echo "nvidia-smi not found - no GPU information available"
fi

# Print memory information
echo -e "\n=== Memory Information ==="
free -h

# Print CPU information
echo -e "\n=== CPU Information ==="
lscpu | grep "Model name"
lscpu | grep "CPU(s):"

# Print arguments
echo -e "\n=== Script Arguments ==="
echo "Model: $1"
echo "Dataset: $2" 
echo "Subtask: $3"
echo "Number of GPUs: $4"

echo -e "\n=== Environment ==="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "PATH: $PATH"
echo "PYTHONPATH: $PYTHONPATH"

echo "=== End Debug Information ===\n"


model=$1
dataset=$2
subtask=$3
num_gpus=$4
# if num_gpus is not provided, then set it to 1
if [ -z "$num_gpus" ]; then
    num_gpus=1
fi
PROJ_DIR="/gpfs/radev/project/cohan/yz979/tingyu/rank1"
# if subtask is "default" ignore it
echo "$(which python)"
if [ "$subtask" != "default" ]; then
    echo "Running with subtask: $subtask"
    python run_mteb.py -m $model -d $dataset -s $subtask -n $num_gpus 
else
    echo "Running with no subtask"
    python run_mteb.py -m $model -d $dataset -n $num_gpus
fi

# example: bash launch_job.sh jhu-clsp/Rank1-7B BrightRetrieval biology 1
