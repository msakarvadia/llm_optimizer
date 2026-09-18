#!/bin/bash
cd /scratch/mansisak/llm_optimizer || exit 1
source .venv/bin/activate

# Job-scoped RAY_TMPDIR so back-to-back runs don't collide on one cluster.
export RAY_TMPDIR="/scratch/mansisak/r_${SLURM_JOB_ID:-$$}"
mkdir -p "$RAY_TMPDIR"

# Node-local datasets cache avoids NFS FileLock contention (ESTALE/ENOLCK).
# HF_HOME stays on NFS so warm model weights are still reused.
export HF_HOME="/scratch/mansisak/.cache/huggingface"
export HF_DATASETS_CACHE="/tmp/hf_cache_${SLURM_JOB_ID:-$$}/datasets"
mkdir -p "$HF_DATASETS_CACHE"
# Seed from the warm NFS cache instead of racing to rebuild from Hub.
cp -a /scratch/mansisak/.cache/huggingface/datasets/. "$HF_DATASETS_CACHE/" 2>/dev/null || true

# Skip Hub revision checks -- caches are already warm; a miss should fail
# fast, not silently download, under concurrent load.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Node-local TMPDIR: candidate eval temp files otherwise hit shared NFS
# scratch (~/.bashrc's TMPDIR), causing contention under concurrent evals.
export TMPDIR="/tmp/tmp_${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"

 python experiments/experiments.py --experiment_name population_dynamics_prod_grade --task_name prompt_gsm8k --population_dir /scratch/mansisak/llm_optimizer/llm_optimizer/populations --num_iter 550
#python experiments/experiments.py --experiment_name general_rollout --task_name cantbelate --optimizer_llm deepseek/deepseek-v4-flash --num_iter 550
