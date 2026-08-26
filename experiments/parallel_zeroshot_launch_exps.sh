# --- Parallel zero-shot discovery sweep ---
# Sibling of launch_exps.sh's general_rollout Job A/B groupings, adapted for
# parallel_zeroshot: instead of one sequential trajectory rolled out to a
# token/iteration budget, each task runs N independent single-step
# (num_iter=1) OPRO rollouts in parallel, one per seed, for kincontext and
# GEPA only (DE/GA collapse into kincontext at bank size 1 -- see
# PARALLEL_ZEROSHOT_MUTATORS in generate_experiment_args.py). N per task is
# hardcoded in PARALLEL_ZEROSHOT_NUM_SEEDS (chosen to roughly match how many
# sequential steps a general_rollout run gets through before exhausting
# that task's TASK_TOKEN_BUDGETS entry), so no --num_iter flag is passed
# here -- unlike general_rollout/population_dynamics.
# No kernelbench: it has no TASK_TOKEN_BUDGETS entry / no N assigned.

# Job A: shared-vLLM batch -- same task grouping as launch_exps.sh's Job A
# (harmbench's abliterated optimizer + classifier, prompt's shared model).
sbatch --time=0-02:00:00 --nodes=1 --cpus-per-task=64 --mem=480000M --gres=gpu:4 \
  --job-name=parallel_zeroshot_shared_llm_batch multi_node_ray_launch_exps.sh \
  parallel_zeroshot --task_name prompt #harmbench

# Job B: CPU-only -- gemini-3.7-flash optimizer, API-hosted, no local model.
sbatch --time=0-02:00:00 --nodes=1 --cpus-per-task=64 --mem=480000M --gres=gpu:0 \
  --job-name=parallel_zeroshot_cpu_only_batch multi_node_ray_launch_exps.sh \
  parallel_zeroshot --task_name tsp cantbelate cloudcast

# job C: need to isolate circle packing cus it needs heavy cpu usage for eval
sbatch --time=0-02:00:00 --nodes=1 --cpus-per-task=64 --mem=480000M --gres=gpu:0 \
  --job-name=parallel_zeroshot_circle multi_node_ray_launch_exps.sh \
  parallel_zeroshot --task_name circlepacking
