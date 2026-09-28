# Job A: shared-vLLM batch -- same task grouping as launch_exps.sh's Job A
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
