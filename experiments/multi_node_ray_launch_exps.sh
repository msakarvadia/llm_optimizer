#!/bin/bash
#SBATCH --nodes=1                  # Request exactly 2 physical nodes
#SBATCH --ntasks-per-node=1        # Required for stable Ray setup (1 main driver per node)
#SBATCH --gres=gpu:3             # Request 4 GPUs per node (8 total across the job)
#SBATCH --cpus-per-task=64         # Proportional scaling: 16 CPUs per GPU * 4 GPUs = 12 CPUs
#SBATCH --mem=256000M              # Proportional memory: 32000M per GPU * 4 GPUs = 128000M (Cedar layout)
#SBATCH --time=0-12:00:00             # 12 hours walltime constraint (DD-HH:MM)
#SBATCH --job-name=opro_ray
#SBATCH --output=ray_cluster_%j.out
#SBATCH --error=ray_cluster_%j.err


# --- CTRL-C INTERCEPTOR ---
cleanup() {
    echo -e "\nCtrl-C detected! Cleaning up multi-node Ray cluster..."
    srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES --overlap uv run ray stop --force
    exit 1
}
trap cleanup SIGINT
# -------------------------------

# Extract node hostnames
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}

# Compute Canada compliant IP lookup using -i
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -i | awk '{print $1}')
port=6379
ip_head="$head_node_ip:$port"

echo "Head node IP: $ip_head"

# Use --exact and --overlap so background srun steps don't starve each other
echo "Starting Ray HEAD on $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" --exact --overlap \
    --cpus-per-task="$SLURM_CPUS_ON_NODE" --mem="$SLURM_MEM_PER_NODE" --gres="gpu:$SLURM_GPUS_ON_NODE" \
    uv run ray start --head --node-ip-address="$head_node_ip" --port=$port --block &
sleep 15

# Start Ray Worker Nodes
worker_num=$((SLURM_JOB_NUM_NODES - 1))
echo "Starting $worker_num Ray WORKERS"
for ((i = 1; i < SLURM_JOB_NUM_NODES; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting worker on $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" --exact --overlap \
        --cpus-per-task="$SLURM_CPUS_ON_NODE" --mem="$SLURM_MEM_PER_NODE" --gres="gpu:$SLURM_GPUS_ON_NODE" \
        uv run ray start --address="$ip_head" --block &
done
sleep 15

# Run your Python application
# Capture the 1st command line argument, use fallback if empty
EXP_NAME=${1:-"general_rollout"}

echo "Starting experiment execution: $EXP_NAME"
echo "Submitting Ray application..."
# FIX: Change 'ray://' to 'auto' so Ray targets the GCS port (6379) directly
export RAY_ADDRESS="auto"
source /scratch/mansisak/llm_optimizer/.venv/bin/activate
python -u /scratch/mansisak/llm_optimizer/experiments/experiments.py --experiment_name $EXP_NAME

# Clean up the cluster when done normally
echo "Application finished. Stopping Ray cluster..."
srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES --overlap uv run ray stop --force
