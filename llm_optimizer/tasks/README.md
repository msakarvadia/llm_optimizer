# Discovery Tasks

- [`Can't Be Late`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/tasks/cant_be_late.py): Discover an algorithm for single spot instance deadline-driven job scheduling.
-   - [NSDI'24](https://www.usenix.org/conference/nsdi24/presentation/wu-zhanghao)
    - Code adapted from [here](https://github.com/gepa-ai/gepa/tree/main/examples/adrs/can_be_late)
- [`CloudCast`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/tasks/cloud_cast.py): Discover an algorithm to optimize multi-cloud data transfer cost.
    - [NDSI'24](https://www.usenix.org/conference/nsdi24/presentation/wooders)
    - Code adapted from [here](https://github.com/gepa-ai/gepa/tree/main/examples/adrs/cloudcast)
- [`Circle Packing`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/tasks/circle_packing.py): Discover an algorithm to arrange 26 circles of variable diameters inside
a unit square without overlap such that cumulative diameters are maximized.
    - Code adapted from [here](https://github.com/gepa-ai/gepa/blob/main/examples/circle_packing/utils.py) and [here](https://github.com/algorithmicsuperintelligence/openevolve/tree/main/examples/circle_packing)
- [`HarmBench` ](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/tasks/harm_bench.py): Optimize a prompt prefix to jailbreak a downstream LLM on a harmful benchmark
    - [paper](https://arxiv.org/pdf/2402.04249)
    - Code adapted from [here](https://github.com/centerforaisafety/HarmBench)
- [`Kernel Bench`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/tasks/kernel_bench.py): Write custom GPU kernels to maximize speed and correctness for specified task.
    - [ICML'25](https://openreview.net/forum?id=yeoN1iQT1x)
    - Code adapted from [here](https://github.com/ScalingIntelligence/KernelBench)
    - Specific [KernelBench](https://github.com/msakarvadia/llm_optimizer/tree/main/KernelBench) version uploaded to this repo
- [`Maximize Function`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/tasks/maximize_function.py): For a simple 2D function, find an x value which maximizes the y value
- [`Prompt Optimization`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/tasks/prompt_optimization.py): Discover an instruction prompt that enables a downstream
LLM to maximize score on a downstream benchmark.
    - Makes use of the [LM Eval Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main)
- [`Traveling Salesperson`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/tasks/traveling_salesman.py): Discover the shortest path that visits 100 cities (with
predefined inter-city distances) exactly once and returns to the starting point.
    - Code adapted from [here](https://github.com/google-deepmind/opro/blob/main/opro/optimization/optimize_tsp.py)
- [`Tweet Engagement`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/tasks/tweet_engagement.py): Optimize a tweet to maximize an engagement metric. Here, the engagement metric is "[toxicity](https://github.com/unitaryai/detoxify)". This task is really fast to run (only needs a single CPU), but is a bit contrived. Suggest using it for testing purposes.
-   - Inspired by this paper from [ICML'24](https://arxiv.org/pdf/2402.06627)




## How to define your own task

- [`base_task.py`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/tasks/base_task.py): a standard abstraction for how to define a discovery 'task' in a way that is compatible with the discovery harnesses.
- [`tweet_engagement`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/tasks/tweet_engagement.py) and [`maximize_function`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/tasks/maximize_function.py) are both small and simple tasks that should be a good reference for how to instantiate and customize `base_task.py` for your task. 
