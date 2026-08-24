# --- Active experiments (see EXPERIMENT_ARGS_REFACTOR_PLAN.md, Step 5) ---

# Job A: shared-vLLM batch -- per-task inference model defaults now
# apply (harmbench -> OLMo-2 DPO, prompt -> Llama-3.2-1B), plus
# harmbench's abliterated optimizer + classifier -> gpu:4
sbatch --time=0-12:00:00 --nodes=1 --cpus-per-task=64 --mem=300000M --gres=gpu:4 \
  --job-name=shared_llm_batch multi_node_ray_launch_exps.sh general_rollout \
  --task_name  prompt --num_iter 50 # harmbench

# Job B: CPU-only
sbatch --time=0-12:00:00 --nodes=1 --cpus-per-task=64 --mem=300000M --gres=gpu:0 \
  --job-name=cpu_only_batch multi_node_ray_launch_exps.sh general_rollout \
  --task_name cantbelate cloudcast tsp circlepacking --num_iter 150

# Job C: kernelbench -- cuda backend, problem_id 1, real GPU per task.
# gpu:4 is a concurrency dial (how many kernelbench tasks run at once),
# not a shared-model count -- adjust to taste.
#sbatch --time=0-12:00:00 --nodes=1 --cpus-per-task=64 --mem=300000M --gres=gpu:4 \
#  --job-name=kernelbench_batch multi_node_ray_launch_exps.sh general_rollout \
#  --task_name kernelbench --num_iter 50


# OLD EXPEIRMENTS:
# NOTE: the standalone 'cloud' experiment_name was merged into
# general_rollout (see EXPERIMENT_ARGS_REFACTOR_PLAN.md) -- use the
# general_rollout line below instead.
#sbatch --time=0-12:00:00 --nodes=2 --cpus-per-task=64 --mem=300000M --gres=gpu:4 --job-name=perturb multi_node_ray_launch_exps.sh perturb

#sbatch --time=0-12:00:00 --nodes=1 --cpus-per-task=64 --mem=300000M --gres=gpu:4 --job-name=pop_dynamics multi_node_ray_launch_exps.sh population_dynamics

#sbatch --time=0-12:00:00 --nodes=1 --cpus-per-task=64 --mem=300000M --gres=gpu:0 --job-name=pop_init_cantbelate multi_node_ray_launch_exps.sh population_dynamics \
#    --task_name cantbelate --num_iter 500 \
#    --population_dir /scratch/mansisak/llm_optimizer/curated_initial_populations_v2/cantbelate

#sbatch --time=0-12:00:00 --nodes=1 --cpus-per-task=64 --mem=300000M --gres=gpu:0 --job-name=pop_init_cloudcast multi_node_ray_launch_exps.sh population_dynamics \
#    --task_name cloudcast --num_iter 500 \
#    --population_dir /scratch/mansisak/llm_optimizer/curated_initial_populations_v2/cloudcast_pop5

#sbatch --time=0-06:00:00 --nodes=1 --cpus-per-task=64 --mem=300000M --gres=gpu:4 --job-name=pop_init_prompt_drop multi_node_ray_launch_exps.sh population_dynamics \
#    --task_name prompt --num_iter 50 \
#    --population_dir /scratch/mansisak/llm_optimizer/curated_initial_populations_v2/prompt_drop_pop5

#sbatch --time=0-01:00:00 --nodes=1 --cpus-per-task=64 --mem=480000M --gres=gpu:4 --job-name=general_rollout multi_node_ray_launch_exps.sh general_rollout

#sbatch --time=0-01:00:00 --nodes=1 --cpus-per-task=48 --mem=225000M --gres=gpu:3 --job-name=harmbench_smoke_test multi_node_ray_launch_exps.sh general_rollout \
#    --task_name harmbench --inference_model_name meta-llama/Llama-3.2-1B-Instruct
