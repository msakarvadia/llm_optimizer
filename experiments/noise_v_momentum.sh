#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=3  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-16:00     # DD-HH:MM:SS


# Simple experiment to understand how OPRO reacts to noise and momentum.'
# noise: 0, 0.1, 0.5, 1.0
# n (only for kincontext): 3, 5, 10, 20,
# max_population_size: 3, 5, 10, 20, 50
# mutator: kincontext (w/ varried noise and n), DE, GA, GEPA
# sampling_strategy_name: random, most_recent, highest_scoring, tournament, wheel
# sampling_prob (only for tournament): 0.5, 0.7, 0.9
# pruning_strategy: oldest, lowest_scoring
# tasks: tweet, kernelbench

module load cuda
cd /scratch/mansisak/llm_optimizer/llm_optimizer

# Static parameters
NUM_ITER=50

# Loop through all parameter combinations
for task in "tweet" "kernelbench"; do
    for pruning in "oldest" "lowest_scoring"; do
        for pop_size in 3 5 10 20 50; do
            for strategy in "random" "most_recent" "highest_scoring" "tournament" "wheel"; do
                for mutator in "kincontext" "DE" "GA" "GEPA"; do
                    for noise in 0 0.1 0.5 1.0; do

                        # 1. Handle Tournament Sampling Dependency
                        if [ "$strategy" == "tournament" ]; then
                            sampling_probs=(0.5 0.7 0.9)
                        else
                            sampling_probs=(0.0) # Dummy value for non-tournament
                        fi

                        for prob in "${sampling_probs[@]}"; do

                            # 2. Handle Kincontext Mutator Dependency
                            if [ "$mutator" == "kincontext" ]; then
                                n_values=(3 5 10 20)
                            else
                                n_values=(0) # Dummy/Default value for other mutators
                            fi

                            for n in "${n_values[@]}"; do

                                # Construct base command
                                CMD="uv run python main.py --optimizer_name opro --task_name $task --pruning_strategy $pruning --max_population_size $pop_size --sampling_strategy_name $strategy --mutator $mutator --noise $noise --num_iter $NUM_ITER"

                                # Append conditional tournament flag
                                if [ "$strategy" == "tournament" ]; then
                                    CMD="$CMD --sampling_prob $prob"
                                fi

                                # Append conditional kincontext flag
                                if [ "$mutator" == "kincontext" ]; then
                                    CMD="$CMD --n $n"
                                fi

                                # Execute the command
                                echo "Running: $CMD"
                                eval $CMD

                            done
                        done
                    done
		done
            done
        done
    done
done
