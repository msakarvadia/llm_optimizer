"""Class to manage (solution,score) pairs produced during LLM Optimization.

may also manage meta-data associated w/ pairs.
"""

from __future__ import annotations

from typing import Any


class SolutionBank:
    """Class to track solution produced during llm_optimization."""

    def __init__(self) -> None:
        """Initialize bank to store solution/score pairs."""
        # dict[optimizaiton_iteration (int) :
        #          {'solution':solution, 'score':score, 'metadata':...}]
        self.bank: dict[int, dict[str, Any]] = {}

    def save_to_json(self, path: str) -> None:
        """Add solution/score pairs to bank."""

    def add_solution_score_pair(self, solution: str, score: Any) -> None:
        """Add solution/score pairs to bank."""
        next_iter = self.__len__()
        self.bank[next_iter] = {}
        self.bank[next_iter]['solution'] = solution
        self.bank[next_iter]['score'] = score

    def get_solutions(self) -> list[tuple[Any, Any]]:
        """Grab (sub)set of past solutions.

        can also grab associated scores/metadata here
        """
        # TODO(MS): put in fancy retrieval logic here!!
        solution_bank: list[tuple[Any, Any]] = []
        for i in range(self.__len__()):
            solution = self.bank[i]['solution']
            score = self.bank[i]['score']
            solution_bank.append((solution, score))

        return solution_bank

    def __len__(self) -> int:
        """Total # of past solutions generated."""
        print('LEN OF SOLUTION BANK IS:', len(self.bank))
        return len(self.bank)
