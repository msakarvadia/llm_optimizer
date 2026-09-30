# Mutator Prompts for `Modular` Harnesses

These mutators provide an LLM with instructions about the discovery objective and past examples of discovery iterates. The LLM is then instructed to produce a new discovery iterate. We provide a few different versions of mutators below.

- [`base_llm_mutator.py`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/optimizers/llm_mutator_library/base_llm_mutator.py): defines the abstraction that specific mutators must instantiate. The number of LLM inference calls varies across mutators.
- [`K in Context` ](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/optimizers/llm_mutator_library/k_in_context.py): provides LLM w/ K past examples of discovery iterates and asks LLM to produce a better one.
- [`Reflective`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/optimizers/llm_mutator_library/gepa.py): uses the same mutator design as is used by [`optimize_anything`](https://github.com/gepa-ai/gepa)
- [`Differential Evolution`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/optimizers/llm_mutator_library/differential_evolution.py): adopted from this [repo](https://github.com/beeevita/EvoPrompt/tree/main); see corresponding [paper](https://openreview.net/pdf?id=ZG3RaNIsO8)
- [`Genetic Algorithm`](https://github.com/msakarvadia/llm_optimizer/blob/main/llm_optimizer/optimizers/llm_mutator_library/genetic_algorithm.py): adopted from this [repo](https://github.com/beeevita/EvoPrompt/tree/main); see corresponding [paper](https://openreview.net/pdf?id=ZG3RaNIsO8)
