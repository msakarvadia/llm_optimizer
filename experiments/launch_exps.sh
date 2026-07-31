sbatch --time=0-01:00:00 --nodes=1 --cpus-per-task=64 --mem=300000M --gres=gpu:4 --job-name=pop_dynamics multi_node_ray_launch_exps.sh population_dynamics

#sbatch --time=0-01:00:00 --nodes=1 --cpus-per-task=64 --mem=480000M --gres=gpu:4 --job-name=general_rollout multi_node_ray_launch_exps.sh general_rollout
