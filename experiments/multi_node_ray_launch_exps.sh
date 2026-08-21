#!/bin/bash
#SBATCH --nodes=1                  # Request exactly 2 physical nodes
#SBATCH --ntasks-per-node=1        # Required for stable Ray setup (1 main driver per node)
#SBATCH --gres=gpu:3             # Request 4 GPUs per node (8 total across the job)
#SBATCH --cpus-per-task=64         # Proportional scaling: 16 CPUs per GPU * 4 GPUs = 12 CPUs
#SBATCH --mem=480000M              # Proportional memory: 32000M per GPU * 4 GPUs = 128000M (Cedar layout)
#SBATCH --time=0-12:00:00             # 12 hours walltime constraint (DD-HH:MM)
#SBATCH --job-name=opro_ray
#SBATCH --output=ray_cluster_%j.out
#SBATCH --error=ray_cluster_%j.err

# Force a job-scoped RAY_TMPDIR regardless of what the submitting shell's
# environment carried in via sbatch's --export=ALL default. Batch scripts
# don't source ~/.bashrc, so its SLURM_JOB_ID-conditional RAY_TMPDIR never
# re-fires in here -- without this, every sbatch call issued from the same
# shell (e.g. population_dynamics_launch_exps.sh's Job A/B back-to-back)
# inherits the same timestamp-based RAY_TMPDIR from .bashrc's login-shell
# fallback, so their Ray clusters collide on one shared session/discovery
# directory and RAY_ADDRESS=auto can resolve to a different job's cluster.
export RAY_TMPDIR="/scratch/mansisak/r_${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR"

# Without node-local dataset storage, concurrent workers all race
# datasets' cache-dir FileLock over NFS, surfacing as ESTALE/ENOLCK.
export HF_HOME="/tmp/hf_cache_${SLURM_JOB_ID}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "$HF_DATASETS_CACHE"
# Seed from the already-warm NFS cache (single sequential copy, ~10s for
# ~250MB) so workers lock against pre-populated data instead of racing
# to download+build from Hub on an empty local cache.
cp -a /scratch/mansisak/.cache/huggingface/datasets/. "$HF_DATASETS_CACHE/" 2>/dev/null || true


# --- SIGNAL INTERCEPTOR ---
# Covers Ctrl-C (SIGINT) and Slurm's timeout/scancel/preemption signal
# (SIGTERM) Doesn't help
# against a hard SIGKILL (OOM-killer, `scancel -s KILL`) -- nothing
# traps that.
cleanup() {
    echo -e "\nSignal received! Cleaning up multi-node Ray cluster..."
    srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES --overlap uv run ray stop --force
    exit 1
}
trap cleanup SIGINT SIGTERM
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
# Any remaining args (e.g. --task_name, --population_dir, --num_gpus,
# --num_iter, --num_cpus for population-init jobs) are forwarded as-is.
shift || true

echo "Starting experiment execution: $EXP_NAME $*"
echo "Submitting Ray application..."
# FIX: Change 'ray://' to 'auto' so Ray targets the GCS port (6379) directly
export RAY_ADDRESS="auto"
source /scratch/mansisak/llm_optimizer/.venv/bin/activate
python -u /scratch/mansisak/llm_optimizer/experiments/experiments.py --experiment_name $EXP_NAME "$@"

# Clean up the cluster when done normally
echo "Application finished. Stopping Ray cluster..."
srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES --overlap uv run ray stop --force
