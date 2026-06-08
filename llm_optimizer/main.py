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
llm_optimizer = LLMOptimizer(task=task, solution_bank=solution_bank)

llm_optimizer.optimize()
