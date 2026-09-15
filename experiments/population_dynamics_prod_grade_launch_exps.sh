

# --- Population-dynamics sweep over populations/, prod-grade optimizers ---
# Sibling of population_dynamics_launch_exps.sh, adapted for
# population_dynamics_prod_grade: instead of OPRO's 12-config
# sampling-strategy x mutator sub-sweep against every discovered population
# file, each task_name here runs exactly 3 configs (gepa, open_evolve,
# shinka_evolve) against ONE hardcoded (optimizer_llm, init_population_path)
# pairing looked up from POP_DYNAMICS_PROD_GRADE_TRIPLES in
# generate_experiment_args.py.
#
# --task_name must be a key already populated in POP_DYNAMICS_PROD_GRADE_TRIPLES
# (currently: cantbelate, cloudcast, tsp, circlepacking, prompt_gsm8k) --
# anything else raises a KeyError in get_args_for_pop_dynamics_prod_grade.
# No --optimizer_llm flag here: unlike population_dynamics, the optimizer_llm
# is fixed per task_name by the triples table, not swept from the CLI.
#
# --population_dir defaults (inside experiments.py) to the same base dir as
# population_dynamics_launch_exps.sh's POP_DIR -- set explicitly below to
# match that script and to keep this overridable the same way.

POP_DIR=/scratch/mansisak/llm_optimizer/llm_optimizer/populations

# Job A: shared-vLLM batch -- prompt_gsm8k's optimizer_llm
#sbatch --time=0-12:00:00 --nodes=1 --cpus-per-task=64 --mem=480000M --gres=gpu:4 \
#  --job-name=pop_dyn_prod_grade_shared_llm_batch multi_node_ray_launch_exps.sh population_dynamics_prod_grade \
#  --task_name prompt_gsm8k --population_dir $POP_DIR --num_iter 500

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
