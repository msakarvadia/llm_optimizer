

# --- Population-dynamics sweep over curated_initial_populations_by_iter_budget/ ---
# Sibling of launch_exps.sh's general_rollout Job A/B/C groupings, adapted for
# population_dynamics: each curated population file gets the same 12-config
# OPRO sub-sweep (3 sampling_strategies x 4 mutators, noise=0, that task's
# general_rollout-default optimizer_llm -- see get_args_for_pop_dynamics /
# POP_DYNAMICS_TASK_MAP in generate_experiment_args.py) instead of
# general_rollout's full 14-combo x benchmark x optimizer_llm grid, so
# results are directly comparable to general_rollout with only the initial
# population varying.
#
# All jobs point --population_dir at the SAME base directory --
# curated_initial_populations_by_iter_budget/ -- since get_args_for_pop_dynamics
# auto-discovers <population_dir>/<task_name>/budget_*/ per task_name and
# unions every budget it finds into one sweep. --task_name accepts multiple
# values exactly like general_rollout does, so grouping tasks into one job
# merges their population sweeps into one Ray cluster / one set of shared
# vLLM servers -- same mechanism as launch_exps.sh's Job A/B/C, just applied
# to population_dynamics instead.

#POP_DIR=/scratch/mansisak/llm_optimizer/curated_initial_populations_by_iter_budget
#POP_DIR=/scratch/mansisak/llm_optimizer/curated_initial_populations_tournament_rejection
POP_DIR=/scratch/mansisak/llm_optimizer/currated_initial_populations_baselines

# Job A: shared-vLLM batch -- same task grouping as launch_exps.sh's Job A
# (harmbench's abliterated optimizer + classifier, prompt's shared model),
# just split across prompt's two benchmarks as separate task_name values.
sbatch --time=0-12:00:00 --nodes=1 --cpus-per-task=64 --mem=100000M --gres=gpu:3 \
  --job-name=pop_dyn_shared_llm_batch multi_node_ray_launch_exps.sh population_dynamics \
  --task_name harmbench prompt_drop prompt_gsm8k --population_dir $POP_DIR --num_iter 50

# Job B: CPU-only -- gemini-3.1-pro optimizer, API-hosted, no local model at all.
sbatch --time=0-12:00:00 --nodes=1 --cpus-per-task=64 --mem=300000M --gres=gpu:0 \
  --job-name=pop_dyn_cpu_only_batch multi_node_ray_launch_exps.sh population_dynamics \
  --task_name cantbelate cloudcast --population_dir $POP_DIR --num_iter 50

# Job C: kernelbench -- real GPU per task; gpu:4 is a concurrency dial (how
# many kernelbench tasks run at once), not a shared-model count -- adjust to
# taste.
#sbatch --time=0-12:00:00 --nodes=1 --cpus-per-task=64 --mem=300000M --gres=gpu:4 \
#  --job-name=pop_dyn_kernelbench_batch multi_node_ray_launch_exps.sh population_dynamics \
#  --task_name kernelbench --population_dir $POP_DIR --num_iter 50
