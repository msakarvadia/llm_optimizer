## How to run an experiment

- [`main.py`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/main.py): run via `python main.py` (`python main.py --help` to see experiment arguments). Default arguments run a tweet optimization task (to maximize tweet engagement/toxicity) using a modular OPRO harness.

## Optimizers

- [`optimizers`](https://github.com/msakarvadia/llm_optimizer/tree/main/llm_optimizer/optimizers): implements the modular (extended OPRO) and the ShinkaEvolve/OpenEvolve/`optimize_anything` discovery harnesses (referred to here as 'optimizers')

## Tasks

- [`tasks`](https://github.com/msakarvadia/llm_optimizer/tree/main/llm_optimizer/tasks): Implements several discovery tasks (tweet optimization, prompt optimization, math function optimization, GPU kernel discovery, cloud scheduling/data transfer algorithm discovery, circle packing task, traveling salesperson). See dir for more task specific details.

## Misc

- [`utils.py`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/utils.py): various utility functions
- [`curate_populations.py`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/curate_populations.py): script to generate various initial populations with varying token budgets across tasks. We recommend using these default initial population curation parameters: `curation_type`:diversity, `p`: 0.98, `dedup`: max, `population_size`: 15. This script also compiles the results of the parallel zeroshot experiments into a [csv](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/parallel_zeroshot_results.csv).
- [`population_curation_utils.py`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/population_curation_utils.py): helper functions to curate initial populations

