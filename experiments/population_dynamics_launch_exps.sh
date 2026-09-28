POP_DIR=/scratch/mansisak/llm_optimizer/llm_optimizer/populations
POP_DIR=/scratch/mansisak/llm_optimizer/llm_optimizer/populations

# Job A: shared-vLLM batch -- same task grouping as launch_exps.sh's Job A
# just split across prompt's two benchmarks as separate task_name values.
sbatch --time=0-12:00:00 --nodes=1 --cpus-per-task=64 --mem=480000M --gres=gpu:4 \
  --job-name=pop_dyn_shared_llm_batch multi_node_ray_launch_exps.sh population_dynamics \
  --task_name prompt_gsm8k --population_dir $POP_DIR --num_iter 500 # 4925676 #prompt_drop

# Job B: CPU-only -- gemini-3.1-pro optimizer, API-hosted, no local model at all.
sbatch --time=0-12:00:00 --nodes=1 --cpus-per-task=64 --mem=480000M --gres=gpu:0 \
  --job-name=pop_dyn_cpu_only_batch multi_node_ray_launch_exps.sh population_dynamics \
  --task_name cloudcast cantbelate --population_dir $POP_DIR --num_iter 550 # cloudcast

# Job C: circlepacking -- CPU-only, separate job since its populations/ tree
# and default optimizer_llms differ from Job B's tasks (see launch_exps.sh's
# circle_cpu_only_batch job).
sbatch --time=0-12:00:00 --nodes=1 --cpus-per-task=64 --mem=480000M --gres=gpu:0 \
  --job-name=pop_dyn_circle_cpu_only_batch multi_node_ray_launch_exps.sh population_dynamics \
  --task_name circlepacking --population_dir $POP_DIR --num_iter 550
