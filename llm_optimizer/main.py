"""Optimization loop with an llm to search for an optimal task solution."""

from __future__ import annotations

from llm_optimizer.optimizer import LLMOptimizer
from llm_optimizer.solution_bank import SolutionBank
from llm_optimizer.tasks.minimize_function import MinimizeFunction

# instanitate task

# tweet_thread = """@CNN: 'House averts government shutdown'
# @user: 'I wish people cheered when I do my job'"""
# task = TweetEngagement(tweet_thread=tweet_thread)

task = MinimizeFunction()


# instantiate solution bank
solution_bank = SolutionBank()

# instantiate optimizer
n = 5
noise = True
shuffle = False
order = 'ascending'
llm_optimizer = LLMOptimizer(
    task=task,
    solution_bank=solution_bank,
    n=n,
    noise=noise,
    shuffle=shuffle,
    order=order,
)

num_iter = 50
llm_optimizer.optimize(num_iter=num_iter)

experiment_dir = 'temp_results'
solution_bank.save_to_json(experiment_dir)
