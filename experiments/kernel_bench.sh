#!/bin/bash
#SBATCH --job-name=dsl_kernelbench
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=3  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-03:00     # DD-HH:MM:SS


# Simple experiment to see if models can achieve the same level of speed up w/ different DSLs
# this experiment would illuminate any pre-existing model biases and how these affect overall gains
# tasks: kernelbench
# backend: tilelang, cuda, triton
# problem_id: 1, 2, 3, 4, 5
# level: 1
# optimizer_name: opro, gepa
# sampling_strategy_name (only for opro): wheel
# mutator (only for opro): kincontext
# n (only for kincontext): 3, 5,
# max_population_size: 10, 20,
# sampling_strategy_name: wheel
# pruning_strategy: lowest_scoring

module load cuda
cd /scratch/mansisak/llm_optimizer/llm_optimizer

# Static parameters
TASK="kernelbench"
LEVEL=1
PRUNING="lowest_scoring"
SAMPLING="wheel"
NUM_ITER=50 # Adjusted based on your previous config, change if needed
MUTATOR="kincontext"

# Loop through all parameter combinations
for backend in "cuda" "triton"; do #TODO(MS) 'tilelang' --> some issue w/ tilelang rn, need to debug
    for prob_id in 1 2 3 4 5; do
        for pop_size in 10 20; do
            for opt_name in "opro" "gepa"; do

                # 1. Handle OPRO-specific dependencies
                if [ "$opt_name" == "opro" ]; then
                    mutators=("kincontext")
                    n_values=(3 5)
                else
                    # Dummy single-element arrays so loops execute exactly once for non-opro
                    mutators=("none")
                    n_values=("none")
                fi

                for mutator in "${mutators[@]}"; do
                    for n in "${n_values[@]}"; do

                        # Construct base command with universal flags
                        CMD="uv run python main.py \
                            --task_name $TASK \
                            --backend $backend \
                            --problem_id $prob_id \
                            --level $LEVEL \
                            --optimizer_name $opt_name \
                            --max_population_size $pop_size \
                            --pruning_strategy $PRUNING \
                            --num_iter $NUM_ITER"

                        # Append OPRO conditional flags
                        if [ "$opt_name" == "opro" ]; then
                            CMD="$CMD --sampling_strategy_name $SAMPLING --mutator $mutator --n $n"
                        fi

                        # Clean up extra spacing/newlines for a clean terminal output
                        CMD=$(echo $CMD | tr -s ' ')

                        # Execute the experiment
                        echo "------------------------------------------------"
                        echo "Running: $CMD"
                        echo "------------------------------------------------"
                        eval $CMD

                    done
                done
            done
        done
    done
done
