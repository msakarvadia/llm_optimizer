# Simple experiment to understand how OPRO reacts to noise and momentum.'
# Noise settings: 0, 0.1, 0.5, 1.0
# n settings : 3, 5, 10, 20, -1
cd /scratch/mansisak/llm_optimizer/llm_optimizer

for noise in 0 0.1 0.5 1.0; do
    for n in 3 5 10 20 -1; do
        python main.py --optimizer_name opro --n $n --noise $noise --num_iter 50 --task_name tweet
    done
done
