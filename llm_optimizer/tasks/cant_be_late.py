"""Optimize cloud scheduling strategies for 'Can't be late' problem.

NSDI'24: https://www.usenix.org/conference/nsdi24/presentation/wu-zhanghao
Code reference:
https://github.com/gepa-ai/gepa/tree/main/examples/adrs/can_be_late
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from llm_optimizer.tasks.base_task import Task
from llm_optimizer.tasks.cant_be_late_utils.dataset import load_trace_dataset
from llm_optimizer.tasks.cant_be_late_utils.simulation import FAILED_SCORE
from llm_optimizer.tasks.cant_be_late_utils.simulation import get_program_path
from llm_optimizer.tasks.cant_be_late_utils.simulation import run_simulation
from llm_optimizer.tasks.cant_be_late_utils.simulation import (
    simulation_failure_info,
)
from llm_optimizer.tasks.cant_be_late_utils.simulation import (
    simulation_success_info,
)
from llm_optimizer.tasks.cant_be_late_utils.simulation import (
    syntax_failure_info,
)
from llm_optimizer.tasks.cant_be_late_utils.simulation import syntax_is_valid


class CantBeLate(Task):
    """Cloud scheduling algo 'task'."""

    def __init__(
        self,
        metric: str = 'average cost',
        direction: str = 'minimize',
        max_traces: int = 15,
        **kwargs: Any,
    ) -> None:
        """Initialize task."""
        self.rng = random.Random(kwargs['seed'])
        self.task_description = f"""{OPTIMIZATION_OBJECTIVE}\n
        {OPTIMIZATION_BACKGROUND}"""

        self.solution_description = 'cloud scheduling algorithm'
        self.metric = metric
        self.direction = direction
        self.seed_candidate = INITIAL_PROGRAM

        # TODO(MS): load dataset
        dataset_root = (
            Path(__file__).resolve().parent
            / 'cant_be_late_utils'
            / 'simulator'
            / 'real'
        )
        dataset = load_trace_dataset(
            dataset_root=str(dataset_root),
            max_traces_per_split=max_traces,
        )
        print(
            f'Dataset — train: {len(dataset["train"])}, val: {len(dataset["val"])}, test: {len(dataset["test"])}',
        )
        self.train_set = dataset['train']
        self.val_set = dataset['val']

    def evaluate(self, solution: str) -> tuple[float, dict[str, Any]]:
        """Evaluate algorithm."""
        program_path = get_program_path(solution)
        print(f'{program_path=}')

        score = 0.0
        # inplace shuffle
        self.rng.shuffle(self.train_set)
        for example in self.train_set:
            if not syntax_is_valid(program_path):
                return FAILED_SCORE, syntax_failure_info(example)

            success, cost, error, details = run_simulation(
                program_path,
                example['trace_file'],
                example['config'],
            )

            if not success:
                return FAILED_SCORE, simulation_failure_info(error, example)

            score += -cost
        # NOTE(MS): we will only pass simulation success info for a single
        # example; we shuffle trian_set so its random each time.
        return score / len(self.train_set), simulation_success_info(
            -cost,
            example,
            details,
        )


OPTIMIZATION_OBJECTIVE = """Optimize a cloud scheduling strategy for the "Can't Be Late" problem.

The strategy decides when to use SPOT instances (cheap but can be preempted) vs ON_DEMAND
instances (expensive but reliable) to complete a task before its deadline. The goal is to
minimize cost while ensuring the task completes on time.
Output only the executable Python code to accomplish the task."""

OPTIMIZATION_BACKGROUND = """Key information about the problem domain:

- ClusterType.SPOT: Use spot instances (cheap, ~$0.3/hour, but can be preempted at any time)
- ClusterType.ON_DEMAND: Use on-demand instances (expensive, ~$1/hour, but guaranteed availability)
- ClusterType.NONE: Wait without using any instances (no cost, but no progress)
- restart_overhead: Time penalty incurred when switching from one instance type to another
- The strategy MUST ensure task completion before the deadline (hard constraint)
- Lower cost is better (scores are negative, representing cost in dollars)

Evaluation feedback format:
- Timeline format: start-end:TYPE@REGION[progress%] (e.g., "0.0-5.0:S@R0[50%]" means SPOT from hour 0-5 reaching 50% progress)
- Spot availability: S=available, X=unavailable (e.g., "0.0-10.0:S | 10.0-15.0:X" means spot available first 10h, then unavailable)

Optimization targets:
1. Reduce overall cost while maintaining deadline guarantees
2. Make better decisions about when to use SPOT vs ON_DEMAND
3. Handle spot unavailability more intelligently
4. Consider the trade-offs between waiting for spot and using on-demand"""

INITIAL_PROGRAM = """import math
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class EvolveSingleRegionStrategy(Strategy):
    NAME = 'evolve_single_region'

    def __init__(self, args):
        super().__init__(args)

    def reset(self, env, task):
        super().reset(env, task)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        env = self.env

        remaining_task_time = self.task_duration - sum(self.task_done_time)
        if remaining_task_time <= 1e-3:
            return ClusterType.NONE

        remaining_time = self.deadline - env.elapsed_seconds

        if remaining_task_time + self.restart_overhead >= remaining_time:
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT
        else:
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)
"""

# if __name__ == '__main__':
#    optimization_task = CantBeLate(seed=42)
#    score, info = optimization_task.evaluate(optimization_task.seed_candidate)
#    print(score)
#    print(info)
