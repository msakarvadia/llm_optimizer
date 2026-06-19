"""Logic to abstract away LLM-driven optimization."""

from __future__ import annotations

import json
import os
import random
from typing import Any

import numpy as np
from openai import OpenAI

from llm_optimizer.optimizers.base_optimizer import Optimizer
from llm_optimizer.optimizers.llm_mutator_library.differential_evolution import (  # noqa
    DEMutator,
)
from llm_optimizer.optimizers.llm_mutator_library.k_in_context import (
    KInContextMutator,
)
from llm_optimizer.tasks.base_task import Task


class OPROOptimizer(Optimizer):
    """LLM optimizer.

    # base on OPRO: https://arxiv.org/abs/2309.03409
    # code inspired but not exactly implementation of:
    https://github.com/google-deepmind/opro
    state include: task, solution_score pairs, stopping criteria
    """

    def __init__(
        self,
        task: Task,
        noise: bool = False,
        num_past_sol: int = 5,
        num_parallel_search: int = 1,
        **kwargs: Any,
    ) -> None:
        """Init optimizer."""
        self.task = task
        self.solution_bank = SolutionBank()

        api_key = os.getenv('GEMINI_API_KEY')
        if api_key is None:
            raise ValueError(
                'API key not found. Set the MY_API_KEY environment variable.',
            )

        # TODO(MS): make generalizable to other base_urls
        self.client = OpenAI(
            api_key=api_key,
            base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
        )

        # NOTE(MS): variables to manage in-context examples/rewards
        self.n = num_past_sol
        self.noise = noise
        self.num_parallel_search = num_parallel_search
        self.shuffle = kwargs['shuffle']
        self.order = kwargs['order']
        if self.task.seed_candidate:
            score, extra_info = self.task.evaluate(self.task.seed_candidate)
            self.solution_bank.add_solution_score_pair(
                self.task.seed_candidate,
                score,
                extra_info,
            )
        self.mutator = {'kincontext': KInContextMutator(), 'DE': DEMutator()}[
            kwargs['mutator']
        ]

    def optimize(self, num_iter: int = 5) -> None:
        """Optimization loop for task."""
        # TODO(MS): impl convergence criteria

        for _i in range(num_iter):
            solution_bank = self.solution_bank.get_solutions(
                n=self.n,
                order=self.order,
                shuffle=self.shuffle,
                noise=self.noise,
            )
            solution = self.mutator.mutate(solution_bank, self.task)
            score, extra_info = self.task.evaluate(solution)
            self.solution_bank.add_solution_score_pair(
                solution,
                score,
                extra_info,
            )

        # NOTE(MS): temporarily save solution bank
        experiment_dir = (
            f'temp_results_dir/{self.task.solution_description}'
            f'_{self.n}_{self.noise}/'
        )

        experiment_dir.replace('.', '')
        self.solution_bank.save_to_json(experiment_dir)


class SolutionBank:
    """Class to track (solution,score) pairs during llm_optimization."""

    def __init__(self) -> None:
        """Initialize bank to store solution/score pairs."""
        # dict[optimizaiton_iteration (int) :
        #          {'solution':solution, 'score':score, 'metadata':...}]
        self.bank: dict[int, dict[str, Any]] = {}

    def save_to_json(self, path: str) -> None:
        """Add solution/score pairs to bank."""
        # Ensure the directory exists; do nothing if it already does
        os.makedirs(path, exist_ok=True)

        print(self.bank)
        full_path = os.path.join(path, 'solution_bank.json')
        with open(full_path, 'w', encoding='utf-8') as json_file:
            json.dump(self.bank, json_file, indent=4)

    def add_solution_score_pair(
        self,
        solution: str,
        score: float,
        extra_info: dict[str, Any],
    ) -> None:
        """Add solution/score pairs to bank."""
        next_iter = self.__len__()
        self.bank[next_iter] = {}
        self.bank[next_iter]['solution'] = solution
        self.bank[next_iter]['score'] = score
        self.bank[next_iter]['extra_info'] = extra_info

    def get_solutions(
        self,
        n: int = -1,
        order: str = 'ascending',
        shuffle: bool = False,
        noise: bool = False,
    ) -> list[tuple[Any, Any, Any]]:
        """Grab (sub)set of past solutions.

        can also grab associated scores/metadata here

        n : # of solutions to be sampled

        order: ascending/descending/random
          (order of iteration IDs before sampling)

        shuffle: boolean; shuffle order of sampled solutions

        noise: add gaussian noise to the reward
            noise gaussian mean=0, std_dev = of current rewards
        """
        # Generate iteration indices in the requested chronological order
        indices = list(range(len(self.bank)))

        # Gather entries matching the requested sequence
        raw_items = []
        for i in indices:
            raw_items.append(
                (
                    self.bank[i]['solution'],
                    self.bank[i]['score'],
                    self.bank[i]['extra_info'],
                ),
            )

        # Inject Gaussian noise if enabled (ignoring non-numerical scores)
        # NOTE(MS): design decision is the noise is adaptive
        # to the current spread in rewards
        if noise > 0:
            numerical_scores = [float(score) for _, score, _ in raw_items]

            if len(numerical_scores) > 1:
                std_dev = float(np.std(numerical_scores))

                noised_items = []
                for sol, score, extra_info in raw_items:
                    noised_items.append(
                        (
                            sol,
                            float(score) + random.gauss(0.0, noise * std_dev),
                            extra_info,
                        ),
                    )
                raw_items = noised_items

        # NOTE(MS): Remove raw_items that have duplicate solutions
        # To keep the FIRST occurrence, reverse the list before converting:
        # not native to opro
        # NOTE(MS): using since it doesn't make sense to
        # noise the same thing differently
        # raw_items = list(dict(reversed(raw_items)).items())
        # raw_items.reverse()

        # Limit to the most rescent (or random) n samples
        sampled_items = raw_items[-n:]

        # Optional secondary shuffle (primarily for ascending/descending)
        if shuffle:
            random.shuffle(sampled_items)

        return sampled_items

    def __len__(self) -> int:
        """Total # of past solutions generated."""
        return len(self.bank)
