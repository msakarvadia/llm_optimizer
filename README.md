# Initialization Improves LLM-Driven Discovery

**Abstract:** 
Large Language Models (LLMs) have been used for novel discovery of algorithms, theorems, drugs, and other tasks through the use of harnesses that prompt an LLM to iteratively optimize an objective. In this work, we study the relationship between the population of previous iterates and eventual discovery success. We generalize past work on harness design to develop a suite of 12 harnesses called Modular and characterize their performance across 5 diverse discovery tasks, finding that discovery success is brittle and sensitive to harness design. We uncover mode collapse, characterized by a dramatic drop in the diversity of iterates, as a common failure mode. We find that popular state-of-the-art harnesses and diversity-inducing harness interventions, which aim to prolong this collapse, yield inconsistent gains. Our results instead uncover that the performance of early discoveries is predictive of eventual success. We therefore propose a universally applicable intervention that performs an initial stage of parallel exploration in order to initialize subsequent iterative optimization. Our method provides consistent gains across many harnesses and target applications, confirming the importance of initialization in LLM-driven discovery.

<img width="1450" height="303" alt="image" src="https://github.com/user-attachments/assets/9f6c4c19-28e3-45db-a0aa-62e8ce2af160" />

_Initialization raises the floor for discovery under a fixed total token budget. We find that initializing (the Modular) discovery harnesses with a high-performing initial population is beneficial for downstream discovery success across harnesses, tasks, and LLMs._


We give a high-level overview of the code structure in this repository below. More detailed READMEs can be found in every subdirectory with pointers to any external repos we utilized or took inspiration from. If there are any questions or concerns, please feel free to open a github issue or email sakarvadia@uchicago.edu.

## Quick Start

## Harnesses

### OPRO [Modular]

### State of the art

## Discovery Tasks
The paper uses the Can't Be Late, CloudCast, Circle Packing, TSP, and Prompt Optimization tasks. We provide those, and additional, tasks below. Additional details about the tasks and code attributions can be found in [`llm_optimizer/tasks`](https://github.com/msakarvadia/llm_optimizer/tree/main/llm_optimizer/tasks)

- [`Can't Be Late`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/tasks/cant_be_late.py): Discover an algorithm for single spot instance deadline-driven job scheduling.
- [`CloudCast`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/tasks/cloud_cast.py): Discover an algorithm to optimize multi-cloud data transfer cost.
- [`Circle Packing`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/tasks/circle_packing.py): Discover an algorithm to arrange 26 circles of variable diameters inside a unit square without overlap such that cumulative diameters are maximized.
- [`HarmBench` ](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/tasks/harm_bench.py): Optimize a prompt prefix to jailbreak a downstream LLM on a harmful benchmark
- [`Kernel Bench`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/tasks/kernel_bench.py): Write custom GPU kernels to maximize speed and correctness for specified task.
- [`Maximize Function`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/tasks/maximize_function.py): For a simple 2D function, find an x value which maximizes the y value.
- [`Prompt Optimization`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/tasks/prompt_optimization.py): Discover an instruction prompt that enables a downstream LLM to maximize score on a downstream benchmark.
- [`Traveling Salesperson (TSP)`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/tasks/traveling_salesman.py): Discover the shortest path that visits 100 cities (with predefined inter-city distances) exactly once and returns to the starting point.
- [`Tweet Engagement`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/tasks/tweet_engagement.py): Optimize a tweet to maximize an engagement metric. Here, the engagement metric is "[toxicity](https://github.com/unitaryai/detoxify)". This task is really fast to run (only needs a single CPU), but is a bit contrived. Suggest using it for testing purposes.




### How to define your own task

- [`base_task.py`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/tasks/base_task.py): a standard abstraction for how to define a discovery 'task' in a way that is compatible with the discovery harnesses.
- [`tweet_engagement`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/tasks/tweet_engagement.py) and [`maximize_function`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/tasks/maximize_function.py) are both small and simple tasks that should be a good reference for how to instantiate and customize `base_task.py` for your task. 

## Experimental Configuration/Launch Scripts

# Experiment launch scripts

Experiment resources found in [`experiments`](https://github.com/msakarvadia/llm_optimizer/tree/main/experiments)

- [`experiments.py`](https://github.com/msakarvadia/llm_optimizer/blob/main/experiments/experiments.py): usage: `uv run python experiments.py` (will start local ray cluster); or you can call it from `multi_node_ray_launch_exps.sh` (below) (will initiailze the previously launch ray cluster). To see what experiments are available run `python experiments.py -h`.
- [`generate_experiment_args.py`](https://github.com/msakarvadia/llm_optimizer/blob/main/experiments/generate_experiment_args.py): each function in this file generates a set of experiment configurations. These are called in `experiments.py` based on the `--experiment_name` flag you set.
  -  The [`general_rollout`](https://github.com/msakarvadia/llm_optimizer/blob/main/experiments/generate_experiment_args.py#L162): runs 72 experiments per task w/ varying harnesses and harness interventions
  - [`parallel_zeroshot`](https://github.com/msakarvadia/llm_optimizer/blob/main/experiments/generate_experiment_args.py#L365): runs several independent experiments where a single (independant) discovery step is done multiple time (e.g., multiple random seeds). The exact number of seeds varies per task, it was eyeballed to exhaust the [task-specific token budgets](https://github.com/msakarvadia/llm_optimizer/blob/main/experiments/generate_experiment_args.py#L45)
  - [`pop_dynamics`](https://github.com/msakarvadia/llm_optimizer/blob/main/experiments/generate_experiment_args.py#L485): we seed the OPRO harness variants w/ varying initial populations. The initial populations are curated from the [`parallel_zeroshot`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/population_curation_utils.py) script.
  - [`pop_dynamics_prod_grade`](https://github.com/msakarvadia/llm_optimizer/blob/main/experiments/generate_experiment_args.py#L626) are very similar to above, except we just explicitly set the initial populations. Note that these more sophisticated harnesses are not initialized with the full initial population, but instead just the [highest scoring sample](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/optimizers/shinka_evolve.py#L253) from that population.
- [`multi_node_ray_launch_exps.sh`](https://github.com/msakarvadia/llm_optimizer/blob/main/experiments/multi_node_ray_launch_exps.sh): submits a slurm job, starts a ray cluster (adapts to whatever resources are present) and then launches experiment script. usage: `./multi_node_ray_launch_exps.sh` for local execution on whatever resources are present. Or submit to slurm via `sbatch multi_node_ray_launch_exps.sh`. Change the slurm arguments to match your projects/username/cluster etc. You can change the experiment being launched from the command line (e.g., `sbatch multi_node_ray_launch_exps.sh population_dynammics`). can also run this locally via `./multi_node_ray_launch_exps.sh <optional name of experiment>`. Warning: this script assumes all datasets and models are locally cached and disallows online downloads to circumvent HF rate limits.
- Slurm job launch scripts are detailed and provided in [`experiments`](https://github.com/msakarvadia/llm_optimizer/blob/main/experiments/)

## Compiling Data

## Installation

Package management via `uv`.

For local development:
```
git clone https://github.com/msakarvadia/llm_optimizer.git # swap url for clone via SSH
cd llm_optimizer
uv sync # OR uv sync --extra dev # for linters tools for code quality
# to run a python program:
uv run python <name of file> <--optimion args>
# OR source .venv/bin/activate then python <name of file> <--optimion args>
```

## Citation

Please cite this work as:

```bibtex
...
```
