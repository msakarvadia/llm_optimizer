#!/bin/bash

# --- CTRL-C INTERCEPTOR ---
cleanup() {
    echo -e "\nCtrl-C detected! Cleaning up multi-node Ray cluster..."
    # Force stop Ray across all allocated Slurm nodes
    srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES uv run ray stop --force
    exit 1
}
# Trap SIGINT (Ctrl-C) and call the cleanup function
trap cleanup SIGINT
# -------------------------------

# Get the list of nodes allocated to this job
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

# Designate the first node as the Head Node
head_node=${nodes_array[0]}
# Get the IP address of the head node
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
port=6379
ip_head="$head_node_ip:$port"

echo "Head node IP: $ip_head"

# Start the Ray Head Node
echo "Starting Ray HEAD on $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    uv run ray start --head --node-ip-address="$head_node_ip" --port=$port --block &
sleep 10

# Start Ray Worker Nodes
worker_num=$((SLURM_JOB_NUM_NODES - 1))
echo "Starting $worker_num Ray WORKERS"
for ((i = 1; i < SLURM_JOB_NUM_NODES; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting worker on $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        uv run ray start --address="$ip_head" --block &
done
sleep 10

# Run your Python application
echo "Submitting Ray application..."
uv run python /scratch/mansisak/llm_optimizer/experiments/experiments.py

# Clean up the cluster when done normally
echo "Application finished. Stopping Ray cluster..."
srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES uv run ray stop --force
