"""Optimization loop with an llm to search for an optimal task solution."""

from __future__ import annotations

from llm_optimizer.optimizers.opro import OPROOptimizer
from llm_optimizer.tasks.minimize_function import MinimizeFunction

# instanitate task

# tweet_thread = """@CNN: 'House averts government shutdown'
# @user: 'I wish people cheered when I do my job'"""
# task = TweetEngagement(tweet_thread=tweet_thread)

task = MinimizeFunction()


# instantiate optimizer
n = 5
noise = True
shuffle = False
order = 'ascending'
llm_optimizer = OPROOptimizer(
    task=task,
    n=n,
    noise=noise,
    shuffle=shuffle,
    order=order,
)

num_iter = 50
llm_optimizer.optimize(num_iter=num_iter)
