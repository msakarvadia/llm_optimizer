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

## Experimental Configuration/Launch Scripts

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
