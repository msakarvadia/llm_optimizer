"""Class to manage (solution,score) pairs produced during LLM Optimization.

may also manage meta-data associated w/ pairs.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

import numpy as np


class SolutionBank:
    """Class to track solution produced during llm_optimization."""

    def __init__(self) -> None:
        """Initialize bank to store solution/score pairs."""
        # dict[optimizaiton_iteration (int) :
        #          {'solution':solution, 'score':score, 'metadata':...}]
        self.bank: dict[int, dict[str, Any]] = {}

    def save_to_json(self, path: str) -> None:
        """Add solution/score pairs to bank."""
        # Ensure the directory exists; do nothing if it already does
        os.makedirs(path, exist_ok=True)

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
        if order == 'descending':
            indices.reverse()
        elif order == 'random':
            random.shuffle(indices)
        # 'ascending' keeps the natural sequential range intact

        # Gather entries matching the requested sequence
        raw_items = []
        for i in indices:
            raw_items.append((self.bank[i]['solution'], self.bank[i]['score']))

        # Inject Gaussian noise if enabled (ignoring non-numerical scores)
        # NOTE(MS): design decision is the noise is adaptive
        # to the current spread in rewards
        if noise:
            numerical_scores = [
                float(score)
                for _, score in raw_items
                if isinstance(score, (int | float))
                and not isinstance(score, bool)
            ]

            if len(numerical_scores) > 1:
                std_dev = float(np.std(numerical_scores))

                noised_items = []
                for sol, score in raw_items:
                    if isinstance(score, (int | float)) and not isinstance(
                        score,
                        bool,
                    ):
                        noised_items.append(
                            (
                                sol,
                                float(score) + random.gauss(0.0, std_dev),
                            ),
                        )
                    else:
                        noised_items.append((sol, score))
                raw_items = noised_items

        # Limit to N samples (safely bounding the request)
        sampled_items = raw_items[:n]

        # Optional secondary shuffle (primarily for ascending/descending)
        if shuffle:
            random.shuffle(sampled_items)

        return sampled_items

    def __len__(self) -> int:
        """Total # of past solutions generated."""
        return len(self.bank)
