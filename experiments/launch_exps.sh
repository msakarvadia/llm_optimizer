#sbatch --time=0-12:00:00 --nodes=2 --cpus-per-task=64 --mem=300000M --gres=gpu:0 --job-name=cloud multi_node_ray_launch_exps.sh cloud
#sbatch --time=0-12:00:00 --nodes=2 --cpus-per-task=64 --mem=300000M --gres=gpu:4 --job-name=perturb multi_node_ray_launch_exps.sh perturb

#sbatch --time=0-12:00:00 --nodes=1 --cpus-per-task=64 --mem=300000M --gres=gpu:4 --job-name=pop_dynamics multi_node_ray_launch_exps.sh population_dynamics

#sbatch --time=0-12:00:00 --nodes=1 --cpus-per-task=64 --mem=300000M --gres=gpu:0 --job-name=pop_init_cantbelate multi_node_ray_launch_exps.sh population_dynamics \
#    --task_name cantbelate --num_gpus 0 --num_cpus 4 --num_iter 500 \
#    --population_dir /scratch/mansisak/llm_optimizer/curated_initial_populations_v2/cantbelate

#sbatch --time=0-12:00:00 --nodes=1 --cpus-per-task=64 --mem=300000M --gres=gpu:0 --job-name=pop_init_cloudcast multi_node_ray_launch_exps.sh population_dynamics \
#    --task_name cloudcast --num_gpus 0 --num_cpus 4 --num_iter 500 \
#    --population_dir /scratch/mansisak/llm_optimizer/curated_initial_populations_v2/cloudcast_pop5


#sbatch --time=0-06:00:00 --nodes=1 --cpus-per-task=64 --mem=300000M --gres=gpu:4 --job-name=pop_init_prompt_drop multi_node_ray_launch_exps.sh population_dynamics \
#    --task_name prompt --num_gpus 0 --num_cpus 4 --num_iter 50 \
#    --population_dir /scratch/mansisak/llm_optimizer/curated_initial_populations_v2/prompt_drop_pop5

#sbatch --time=0-01:00:00 --nodes=1 --cpus-per-task=64 --mem=480000M --gres=gpu:4 --job-name=general_rollout multi_node_ray_launch_exps.sh general_rollout


sbatch --time=0-01:00:00 --nodes=1 --cpus-per-task=48 --mem=225000M --gres=gpu:3 --job-name=harmbench_smoke_test multi_node_ray_launch_exps.sh general_rollout \
    --task_name harmbench --num_gpus 0 --num_cpus 4 --inference_model_name meta-llama/Llama-3.2-1B-Instruct
