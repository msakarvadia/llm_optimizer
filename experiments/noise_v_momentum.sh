# Simple experiment to understand how OPRO reacts to noise and momentum.'
# Noise settings: 0, 0.1, 0.5, 1.0
# n settings : 3, 5, 10, 20,
cd /scratch/mansisak/llm_optimizer/llm_optimizer

for noise in 0 0.1 0.5 1.0; do
    for n in 3 5 10 20 0; do
    # Skip experiments where noise is 0 AND n is 20, 10, 5, or 3
    	if [ "$noise" = "0" ] && { [ "$n" = "3" ] || [ "$n" = "5" ] || [ "$n" = "10" ] || [ "$n" = "20" ]; }; then
        	continue
        fi
        python main.py --optimizer_name opro --n $n --noise $noise --num_iter 50 --task_name tweet
    done
done
