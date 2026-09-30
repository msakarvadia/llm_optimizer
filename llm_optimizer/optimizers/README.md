# Discovery Harnesses (referred to as 'optimizers' in this repo)

[Yang et al. (2024)](https://arxiv.org/abs/2309.03409) proposed to use LLMs to sequentially optimize an objective expressed in natural language by iteratively prompting an LLM with the objective and examples of past solutions. Discovery harnesses instantiate this general workflow to enable better downstream discovery. Generally, a harness samples parent discoveries from an active population of past discoveries. An LLM is then prompted to mutate these parents to generate an improved discovery, with respect to the objective. New discoveries undergo task-specific evaluation and are added to the active population if they meet certain criteria. The active population is periodically pruned when it reaches capacity. 

## `Modular` Harnesses (a generalized version of [OPRO](https://github.com/google-deepmind/opro/tree/main))

We develop a set of harnesses called `Modular` by generalizing the OPRO discovery framework proposed by [Yang et al. (2024)](https://arxiv.org/abs/2309.03409). We minimally modify the OPRO harness to expose two axes of freedom: 1) [the LLM mutation strategy](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/optimizers/opro.py#L141) and 2) the [parent sampling strategy](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/optimizers/opro.py#L722) to enable a broader range of harness behavior. 
- [`Modular`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/optimizers/opro.py): Maintains a population of past iterates in [`SolutionBank`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/optimizers/opro.py#L216), and uses an LLM to sequentially refine past iterates.

- # State of the art Harnesses

 We consider three popular, well-engineered discovery harnesses from the recent literature:
 - [`ShinkaEvolve`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/optimizers/shinka_evolve.py)
 - [`OpenEvolve`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/optimizers/open_evolve.py)
 - [`optimize_anything`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/optimizers/gepa.py)
