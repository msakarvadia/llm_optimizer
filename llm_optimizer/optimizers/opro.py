"""Logic to abstract away LLM-driven optimization."""

from __future__ import annotations

import json
import os
import random
from typing import Any

import numpy as np
from openai import OpenAI

from llm_optimizer.optimizers.base_optimizer import Optimizer
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

    def get_meta_prompt(self) -> str:
        """Setup meta prompt for optimizer.."""
        task_prompt = (
            f'{self.task.task_description} '
            f'Your goal is to {self.task.direction} {self.task.metric}. '
            f'Output only the bare minimum text to reach the objective goal.'
        )

        meta_prompt = task_prompt
        solution_bank = self.solution_bank.get_solutions(
            n=self.n,
            order=self.order,
            shuffle=self.shuffle,
            noise=self.noise,
        )
        if len(self.solution_bank) > 0:
            example_blocks = []
            for solution, score in solution_bank:
                block = (
                    f'### Past Example\n'
                    f'Solution:\n{solution.strip()}\n'
                    f'{self.task.metric}: {score}'
                )
                example_blocks.append(block)

            example_str = '\n\n'.join(example_blocks)

            meta_prompt = (
                task_prompt
                + f'\nHere are some past examples and the {self.task.metric}'
                f'score they received where the goal is to:'
                f'{self.task.direction} the metric\n\n{example_str}\n\n'
                f'Generate a solution that has as high a score as possible.'
            )

        return meta_prompt

    def prompt_lm(self, prompt: str) -> str:
        """Standard LLM api inference call."""
        response = self.client.chat.completions.create(
            model='gemini-3.5-flash',
            # TODO(MS): maybe pass in a pydantic model for the type of
            # response we want...particularly for code
            # response_format={"type": "json_object"},
            # TODO(MS): maybe do somehitng about system prompt??
            messages=[
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant.',
                },
                {
                    'role': 'user',
                    'content': prompt,
                },
            ],
        )
        # Assuming 'response' is your completed OpenAI API call
        raw_output = response.choices[0].message.content
        if raw_output is not None:
            return raw_output
        else:
            # Handle the error state
            raise ValueError(
                'LM api call returned None instead of a valid string.',
            )
        return raw_output

    def optimize(self, num_iter: int = 5) -> None:
        """Optimization loop for task."""
        # TODO(MS): impl convergence criteria

        for _i in range(num_iter):
            meta_prompt = self.get_meta_prompt()
            solution = self.prompt_lm(meta_prompt)
            score = self.task.evaluate(solution)
            self.solution_bank.add_solution_score_pair(solution, score)
            print(meta_prompt)

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

    def add_solution_score_pair(self, solution: str, score: Any) -> None:
        """Add solution/score pairs to bank."""
        next_iter = self.__len__()
        self.bank[next_iter] = {}
        self.bank[next_iter]['solution'] = solution
        self.bank[next_iter]['score'] = score

    def get_solutions(
        self,
        n: int = -1,
        order: str = 'ascending',
        shuffle: bool = False,
        noise: bool = False,
    ) -> list[tuple[Any, Any]]:
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
            raw_items.append((self.bank[i]['solution'], self.bank[i]['score']))

        # Inject Gaussian noise if enabled (ignoring non-numerical scores)
        # NOTE(MS): design decision is the noise is adaptive
        # to the current spread in rewards
        if noise > 0:
            numerical_scores = [float(score) for _, score in raw_items]

            if len(numerical_scores) > 1:
                std_dev = float(np.std(numerical_scores))

                noised_items = []
                for sol, score in raw_items:
                    noised_items.append(
                        (
                            sol,
                            float(score) + random.gauss(0.0, noise * std_dev),
                        ),
                    )
                raw_items = noised_items

        # NOTE(MS): Remove raw_items that have duplicate solutions
        # To keep the FIRST occurrence, reverse the list before converting:
        # not native to opro
        # NOTE(MS): using since it doesn't make sense to
        # noise the same thing differently
        raw_items = list(dict(reversed(raw_items)).items())
        raw_items.reverse()

        # Limit to the most rescent (or random) n samples
        sampled_items = raw_items[-n:]

        # Optional secondary shuffle (primarily for ascending/descending)
        if shuffle:
            random.shuffle(sampled_items)

        return sampled_items

    def __len__(self) -> int:
        """Total # of past solutions generated."""
        return len(self.bank)
