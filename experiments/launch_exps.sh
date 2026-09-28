# Job A: shared-vLLM batch -- per-task inference model defaults now
# apply (harmbench -> OLMo-2 DPO, prompt -> Llama-3.2-1B), plus
# harmbench's abliterated optimizer + classifier -> gpu:4
sbatch --time=0-12:00:00 --nodes=1 --cpus-per-task=64 --mem=480000M --gres=gpu:4 \
  --job-name=shared_llm_batch multi_node_ray_launch_exps.sh general_rollout \
  --task_name  prompt --num_iter 500 # harmbench

# Job B: CPU-only
sbatch --time=0-12:00:00 --nodes=1 --cpus-per-task=64 --mem=480000M --gres=gpu:0 \
  --job-name=cpu_only_batch multi_node_ray_launch_exps.sh general_rollout \
  --task_name cloudcast cantbelate  --num_iter 550 # cloudcast

# Job C: CPU-only
sbatch --time=0-12:00:00 --nodes=1 --cpus-per-task=64 --mem=480000M --gres=gpu:0 \
  --job-name=circle_cpu_only_batch multi_node_ray_launch_exps.sh general_rollout \
  --task_name circlepacking --num_iter 550
