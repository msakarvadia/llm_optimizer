
POP_DIR=/project/aip-craffel/mansisak/llm_optimizer/llm_optimizer/populations

# Job A: CPU-only -- cloudcast/cantbelate/tsp use API-hosted optimizer_llms
# and inference models, no local model at all. Kept separate from prompt_gsm8k
# so a failed local vLLM startup there can't kill these configs (experiments.py
# exits if any shared vLLM server fails to come up).
sbatch --time=0-12:00:00 --nodes=1 --cpus-per-task=64 --mem=480000M --gres=gpu:0 \
  --job-name=pop_dyn_prod_grade_cpu_only_batch multi_node_ray_launch_exps.sh population_dynamics_prod_grade \
  --task_name cloudcast cantbelate tsp --population_dir $POP_DIR --num_iter 550

# Job B: shared-vLLM batch -- prompt_gsm8k's optimizer_llm
# (meta-llama/Llama-3.1-8B-Instruct) and inference model
# (allenai/OLMo-2-0425-1B-SFT) are both local vllm models, one
# VLLMServerActor (num_gpus=1) each, so 2 GPUs.
sbatch --time=0-12:00:00 --nodes=1 --cpus-per-task=64 --mem=480000M --gres=gpu:2 \
  --job-name=pop_dyn_prod_grade_shared_llm_batch multi_node_ray_launch_exps.sh population_dynamics_prod_grade \
  --task_name prompt_gsm8k --population_dir $POP_DIR --num_iter 550

# Job C: circlepacking -- CPU-only, separate job since its populations/ tree.
# All 15 configs finished (2026-10-22), so nothing to run.
#sbatch --time=0-06:00:00 --nodes=1 --cpus-per-task=64 --mem=480000M --gres=gpu:0 \
#  --job-name=pop_dyn_prod_grade_circle_cpu_only_batch multi_node_ray_launch_exps.sh population_dynamics_prod_grade \
#  --task_name circlepacking --population_dir $POP_DIR --num_iter 550
