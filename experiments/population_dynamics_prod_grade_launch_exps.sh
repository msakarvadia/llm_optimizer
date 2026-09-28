
POP_DIR=/scratch/mansisak/llm_optimizer/llm_optimizer/populations

# COMBINED:
# Job A: shared-vLLM batch -- prompt_gsm8k's optimizer_llm
# Job B: CPU-only -- API-hosted optimizer_llms, no local model at all.
# gpu:4 (not gpu:0) -- prompt_gsm8k's optimizer_llm
# (meta-llama/Llama-3.1-8B-Instruct) and inference model
# (allenai/OLMo-2-0425-1B-SFT) are both local vllm models needing real
# GPUs; cloudcast/cantbelate/tsp are API-hosted and don't need any, but
# share this same allocation now that they're merged into one job.
sbatch --time=0-12:00:00 --nodes=1 --cpus-per-task=64 --mem=480000M --gres=gpu:4 \
  --job-name=pop_dyn_prod_grade_cpu_only_batch multi_node_ray_launch_exps.sh population_dynamics_prod_grade \
  --task_name prompt_gsm8k cloudcast cantbelate tsp --population_dir $POP_DIR --num_iter 550

# Job C: circlepacking -- CPU-only, separate job since its populations/ tree
sbatch --time=0-06:00:00 --nodes=1 --cpus-per-task=64 --mem=480000M --gres=gpu:0 \
  --job-name=pop_dyn_prod_grade_circle_cpu_only_batch multi_node_ray_launch_exps.sh population_dynamics_prod_grade \
  --task_name circlepacking --population_dir $POP_DIR --num_iter 550
