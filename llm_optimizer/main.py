"""Optimization loop with an llm to search for an optimal task solution."""

from __future__ import annotations

import argparse

from llm_optimizer.optimizers.base_optimizer import Optimizer
from llm_optimizer.optimizers.gepa import GEPAOptimizer
from llm_optimizer.optimizers.open_evolve import OpenEvolveOptimizer
from llm_optimizer.optimizers.opro import OPROOptimizer
from llm_optimizer.tasks.maximize_function import MaximizeFunction
from llm_optimizer.tasks.tweet_engagement import TweetEngagement

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # optimizer args
    parser.add_argument(
        '--optimizer_name',
        type=str,
        default='opro',
        choices=['opro', 'gepa', 'open_evolve'],
        help="""Name of llm-baed optimizer""",
    )
    parser.add_argument(
        '--n',
        type=int,
        default=5,
        help="""Number of past examples to keep in history
            analog of "momentum" in traditional optimization.
            """,
    )
    parser.add_argument(
        '--noise',
        type=float,
        default=0.0,
        help="""Adds gaussian noise with mean 0
        stddev of noise*stdev(scores).""",
    )

    # NOTE(MS) this is not a rigorous method to compare budgets across
    # strategies....make better
    parser.add_argument(
        '--num_iter',
        type=int,
        default=50,
        help="""Number of rounds of optimization.""",
    )

    # task args
    parser.add_argument(
        '--task_name',
        type=str,
        default='tweet',
        choices=['tweet', 'function'],
        help="""Name of individual task being optimized.""",
    )
    args = parser.parse_args()

    tweet_thread = """@CNN: 'House averts government shutdown'
    @user: 'I wish people cheered when I do my job'"""
    tasks = {
        'tweet': TweetEngagement(tweet_thread=tweet_thread),
        'function': MaximizeFunction(),
    }
    # instanitate task
    task = tasks[args.task_name]

    optimizers: dict[str, type[Optimizer]] = {
        'opro': OPROOptimizer,
        'gepa': GEPAOptimizer,
        'open_evolve': OpenEvolveOptimizer,
    }
    optimizer_class = optimizers[args.optimizer_name]
    # instantiate optimizer
    llm_optimizer = optimizer_class(
        task=task,
        num_past_sol=args.n,
        noise=args.noise,
        shuffle=False,
        order='ascending',
    )

    # optimize
    llm_optimizer.optimize(num_iter=args.num_iter)
