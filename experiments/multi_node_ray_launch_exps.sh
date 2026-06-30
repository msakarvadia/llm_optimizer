#!/bin/bash

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
    uv run ray start --head --node-ip-address="$head_node_ip" --port=$port --block &
sleep 15

# Start Ray Worker Nodes
worker_num=$((SLURM_JOB_NUM_NODES - 1))
echo "Starting $worker_num Ray WORKERS"
for ((i = 1; i < SLURM_JOB_NUM_NODES; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting worker on $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" --exact --overlap \
        uv run ray start --address="$ip_head" --block &
done
sleep 15

# Run your Python application
echo "Submitting Ray application..."
# FIX: Change 'ray://' to 'auto' so Ray targets the GCS port (6379) directly
export RAY_ADDRESS="auto"
uv run python /scratch/mansisak/llm_optimizer/experiments/experiments.py

# Clean up the cluster when done normally
echo "Application finished. Stopping Ray cluster..."
srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES --overlap uv run ray stop --force
