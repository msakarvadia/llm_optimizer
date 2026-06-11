"""Optimization loop with open-evolve to search for an optimal task solution."""

from __future__ import annotations

import importlib.util
import os
from openevolve import run_evolution
from openevolve.config import Config, LLMModelConfig
import logging

# 1. Force the global logger to output verbose execution details and raw templates
logging.basicConfig(level=logging.DEBUG)


# 3. Configure API Credentials for OpenEvolve's Backend Agent
# OpenEvolve checks standard environment variables (e.g., OPENAI_API_KEY) under the hood.
api_key = os.getenv('GEMINI_API_KEY')
if api_key is None:
    raise ValueError('API key not found. Please set the GEMINI_API_KEY environment variable.')
config = Config()
config.llm.models = [
    LLMModelConfig(
        name='gemini-3.5-flash', # Or gemini-1.5-flash depending on availability
        api_key=api_key,
        api_base='https://generativelanguage.googleapis.com/v1beta/openai/',
    )
]

# 2. Add this specific flag to your existing Config object to store prompt parameters
config.database.log_prompts = True

config.prompt.system_prompt = "You are a master tweet writer. All the tweets you write are about roses."

def evaluator(solution_path: str):
    """Evaluate LLM optimized solution."""
    import pandas as pd
    from detoxify import Detoxify
    import re

    with open(solution_path, "r", encoding="utf-8") as f:
            solution_file_text = f.read()


    # Extract the content safely
    match = re.search(r'# EVOLVE-BLOCK-START\s*(.*?)\s*# EVOLVE-BLOCK-END', solution_file_text, re.DOTALL)
    solution = match.group(1)

    print(f"\n" + "="*50)
    print(f"📝 Candidate Text:\n\"{solution}\"")
    print(f"..." * 10)
    results = Detoxify('original').predict(solution)
    df = pd.DataFrame([results]).round(5)
    print(f'{df}')
    
    # open evolve maximizes
    return {"combined_score":-results['toxicity']}


tweet = """@CNN: 'House averts government shutdown'
@user: 'I wish people cheered when I do my job'"""
task_description = f"""The below thread was posted on Twitter.
{tweet}\n Craft a response tweet."""
direction = 'maximize'
metric = 'engagement'
task_prompt = f"""{task_description} Your goal is to {direction} {metric}.
Output only the bare minimum text to reach the objective goal.
"""
# TODO(MS):  Need to visualize the actual prompt being fed into the LLM, I want to see how many past examples are shown and the format of the score and also I want to see what the system prompt is.
config.prompt.system_message = task_prompt#"You are a master tweet writer. All the tweets you write are about roses."

result = run_evolution(
    initial_program="""# EVOLVE-BLOCK-START
This is the starting generic tweet.
# EVOLVE-BLOCK-END
""",
    evaluator=evaluator,
    iterations=50,
    config = config,
)


""" NOTE(MS)
# Inside openevolve/controller.py
def run_mutation_step(self, parent_program, history):
    # This generates the text containing the targeted instructions for the next mutation step
    compiled_mutation_prompt = self.prompt_sampler.generate(parent_program, history)
    
    # ─── ADD YOUR PRINT HOOK HERE ───
    print("\n🧬 [LLM ENSEMBLE MUTATION INSTRUCTION] 🧬")
    print("=" * 60)
    print(compiled_mutation_prompt)
    print("=" * 60 + "\n")
    
    # The framework passes this instruction payload to the LLM Ensemble
    response = self.llm_ensemble.call(compiled_mutation_prompt)
    return response
"""
