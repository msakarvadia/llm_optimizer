"""Optimization loop with an llm to search for an optimal task solution."""

from __future__ import annotations

from llm_optimizer.optimizers.base_optimizer import Optimizer
from llm_optimizer.optimizers.gepa import GEPAOptimizer
from llm_optimizer.optimizers.open_evolve import OpenEvolveOptimizer
from llm_optimizer.optimizers.opro import OPROOptimizer
from llm_optimizer.tasks.base_task import Task
from llm_optimizer.tasks.maximize_function import MaximizeFunction
from llm_optimizer.tasks.tweet_engagement import TweetEngagement

# instanitate task


task: Task = MaximizeFunction()
tweet_thread = """@CNN: 'House averts government shutdown'
@user: 'I wish people cheered when I do my job'"""
task = TweetEngagement(tweet_thread=tweet_thread)


# instantiate optimizer
n = 5
noise = True
shuffle = False
order = 'ascending'
llm_optimizer: Optimizer = OPROOptimizer(
    task=task,
    n=n,
    noise=noise,
    shuffle=shuffle,
    order=order,
)

llm_optimizer = GEPAOptimizer(
    task=task,
    n=n,
    noise=noise,
    shuffle=shuffle,
    order=order,
)

llm_optimizer = OpenEvolveOptimizer(
    task=task,
    n=n,
    noise=noise,
    shuffle=shuffle,
    order=order,
)

num_iter = 50
llm_optimizer.optimize(num_iter=num_iter)
