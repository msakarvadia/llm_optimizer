"""Logic to abstract away LLM-driven optimization."""

from __future__ import annotations

import json
import os
import random
import shutil
import warnings
from typing import Any

import numpy as np

from llm_optimizer.optimizers.base_optimizer import Optimizer
from llm_optimizer.optimizers.llm_mutator_library.differential_evolution import (  # noqa
    DEMutator,
)
from llm_optimizer.optimizers.llm_mutator_library.genetic_algorithm import (
    GAMutator,
)
from llm_optimizer.optimizers.llm_mutator_library.gepa import GEPAMutator
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
        # Capture arguments and flatten kwargs
        # build experiment path
        self.experiment_dir = f'{kwargs["experiment_dir"]}/'
        print(f'{self.experiment_dir=}')
        os.makedirs(self.experiment_dir, exist_ok=True)
        self.init_population_path = kwargs['init_population_path']
        if os.path.isfile(self.init_population_path):
            for file_name in [
                'long_running_solution_bank.json',
                'current_solution_bank.json',
            ]:
                destination_path = os.path.join(self.experiment_dir, file_name)
                shutil.copy(self.init_population_path, destination_path)

        self.seed = kwargs['seed']
        self.sampling_strategy_name = kwargs['sampling_strategy_name']
        self.sampling_prob = kwargs['sampling_prob']
        self.max_population_size = kwargs['max_population_size']
        self.pruning_strategy = kwargs['pruning_strategy']
        self.truncate_generated_solution = kwargs[
            'truncate_generated_solution'
        ]
        random.seed(self.seed)
        self.rng = np.random.default_rng(seed=self.seed)
        self.task = task
        self.n = num_past_sol
        self.noise = noise
        self.num_parallel_search = num_parallel_search

        self.solution_bank = SolutionBank(
            seed=self.seed,
            max_population_size=self.max_population_size,
            pruning_strategy=self.pruning_strategy,
        )
        self.solution_bank.read_from_checkpoint(self.experiment_dir)

        # NOTE(MS): variables to manage in-context examples/rewards
        if self.task.seed_candidate:
            score, extra_info, val_score = self.task.evaluate(
                self.task.seed_candidate,
            )
            self.solution_bank.add_solution_score_pair(
                self.task.seed_candidate,
                score,
                extra_info,
                val_score,
            )
        self.mutator = {
            'kincontext': KInContextMutator(
                api_key=kwargs['api_key'],
                base_url=kwargs['base_url'],
                model_name=kwargs['model_name'],
            ),
            'DE': DEMutator(
                api_key=kwargs['api_key'],
                base_url=kwargs['base_url'],
                model_name=kwargs['model_name'],
            ),
            'GA': GAMutator(
                api_key=kwargs['api_key'],
                base_url=kwargs['base_url'],
                model_name=kwargs['model_name'],
            ),
            'GEPA': GEPAMutator(
                api_key=kwargs['api_key'],
                base_url=kwargs['base_url'],
                model_name=kwargs['model_name'],
            ),
        }[kwargs['mutator']]

    def optimize(self, num_iter: int = 5) -> None:
        """Optimization loop for task."""
        # TODO(MS): impl convergence criteria

        while self.solution_bank.get_num_total_iterations() <= num_iter:
            solution_bank = self.solution_bank.get_solutions(
                n=self.n,
                sampling_strategy_name=self.sampling_strategy_name,
                noise=self.noise,
                selection_prob=self.sampling_prob,
            )
            solution = self.mutator.mutate(solution_bank, self.task)
            if self.truncate_generated_solution > 0:
                solution = solution[: self.truncate_generated_solution]
            score, extra_info, val_score = self.task.evaluate(solution)
            self.solution_bank.add_solution_score_pair(
                solution,
                score,
                extra_info,
                val_score,
            )
            self.solution_bank.prune_population()
            # some notion of experimental check pointing
            # NOTE(MS): currently done every time...maybe less freq?
            self.solution_bank.save_to_json(self.experiment_dir)


class SolutionBank:
    """Class to track (solution,score) pairs during llm_optimization."""

    def __init__(
        self,
        seed: int,
        max_population_size: int,
        pruning_strategy: str,
    ) -> None:
        """Initialize bank to store solution/score pairs."""
        # dict[optimizaiton_iteration (int) :
        #          {'solution':solution, 'score':score, 'metadata':...}]
        self.bank: dict[int, dict[str, Any]] = {}
        self.never_prune_bank: dict[int, dict[str, Any]] = {}

        self.seed = seed
        random.seed(self.seed)
        self.rng = np.random.default_rng(seed=self.seed)
        self.max_population_size = max_population_size
        self.pruning_strategy = pruning_strategy

    def read_from_checkpoint(self, path: str) -> None:
        """Read current version of the solution banks."""
        # Define the exact file paths as saved in save_to_json
        never_prune_path = os.path.join(
            path,
            'long_running_solution_bank.json',
        )
        bank_path = os.path.join(path, 'current_solution_bank.json')

        # Load long_running_solution_bank if it exists
        if os.path.exists(never_prune_path):
            with open(never_prune_path, encoding='utf-8') as json_file:
                raw_data = json.load(json_file)
                # Convert only the top-level keys to int
                self.never_prune_bank = {
                    int(k): v for k, v in raw_data.items()
                }
            print(f'Loaded never_prune_bank from {never_prune_path}')

        # Load current_solution_bank if it exists
        if os.path.exists(bank_path):
            with open(bank_path, encoding='utf-8') as json_file:
                raw_data = json.load(json_file)
                # Convert only the top-level keys to int
                self.bank = {int(k): v for k, v in raw_data.items()}
            print(f'Loaded bank from {bank_path}')

    def save_to_json(self, path: str) -> None:
        """Add solution/score pairs to bank."""
        # Ensure the directory exists; do nothing if it already does

        def json_serializable_fallback(obj: Any) -> str:
            if isinstance(obj, Exception):
                return repr(
                    obj,
                )  # Converts ValueError("...") into "ValueError('...')"
            raise TypeError(
                f'Object of type {obj.__class__.__name__}',
                ' is not JSON serializable',
            )

        # print(self.bank)
        full_path = os.path.join(path, 'long_running_solution_bank.json')
        with open(full_path, 'w', encoding='utf-8') as json_file:
            json.dump(
                self.never_prune_bank,
                json_file,
                indent=4,
                default=json_serializable_fallback,
            )
        full_path = os.path.join(path, 'current_solution_bank.json')
        with open(full_path, 'w', encoding='utf-8') as json_file:
            json.dump(
                self.bank,
                json_file,
                indent=4,
                default=json_serializable_fallback,
            )

    def prune_population(self) -> None:
        """Remove old population solutions."""
        # if the population is smaller than max, do nothing
        if self.__len__() <= self.max_population_size:
            return

        # Calculate how many items need to be removed
        num_to_remove = self.__len__() - self.max_population_size

        if self.pruning_strategy == 'oldest':
            # Since keys are sequential integers (0, 1, 2...),
            # the oldest are 0 to num_to_remove - 1
            for i in range(num_to_remove):
                if i in self.bank:
                    del self.bank[i]
            # NOTE(MS): I choose not to reindex solutions

        if self.pruning_strategy == 'lowest_scoring':
            # Sort bank keys by their corresponding score value
            # in ascending order
            keys_by_score = sorted(
                self.bank.keys(),
                key=lambda k: self.bank[k]['score'],
            )
            # Identify the keys with the lowest scores to drop
            keys_to_remove = keys_by_score[:num_to_remove]
            # Delete the lowest scoring items directly
            for key in keys_to_remove:
                del self.bank[key]

    def add_solution_score_pair(
        self,
        solution: str,
        score: float,
        extra_info: dict[str, Any],
        val_score: float | None,
    ) -> None:
        """Add solution/score pairs to bank."""
        next_iter = max(self.never_prune_bank, default=0) + 1

        self.bank[next_iter] = {}
        self.bank[next_iter]['solution'] = solution
        self.bank[next_iter]['score'] = score
        self.bank[next_iter]['extra_info'] = extra_info
        self.bank[next_iter]['val_score'] = val_score

        self.never_prune_bank[next_iter] = self.bank[next_iter]
        self.never_prune_bank[next_iter]['active_population'] = list(self.bank)

    def get_solutions(
        self,
        n: int = -1,
        sampling_strategy_name: str = 'most_recent',
        noise: bool = False,
        selection_prob: float = 0.5,
    ) -> list[tuple[Any, Any, Any]]:
        """Grab (sub)set of past solutions.

        can also grab associated scores/metadata here

        n : # of solutions to be sampled


        noise: add gaussian noise to the reward
            noise gaussian mean=0, std_dev = of current rewards
        """
        # Generate iteration indices in the requested chronological order
        indices = sorted(self.bank.keys())

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
        sampled_items = self.sampling_strategy(
            raw_items,
            n,
            sampling_strategy_name,
            selection_prob,
        )

        return sampled_items

    def sampling_strategy(
        self,
        population: list[tuple[Any, Any, Any]],
        n: int,
        sampling_strategy_name: str,
        selection_prob: float,
    ) -> list[tuple[Any, Any, Any]]:
        """Implement population sampling strategy.

        n: number of samples to draw from population
        sampling_strategy_name: type of sampling strategy

        selection_prob: specific for 'tournament'
        see: https://en.wikipedia.org/wiki/Tournament_selection
        """
        sampled_items: list[tuple[Any, Any, Any]] = []
        # Cap k to pop. size to prevent crashing if n > population size
        k = min(n, len(population))

        if selection_prob >= 1.0:
            # default to highest_scoring
            warnings.warn(
                'Default to highest_scoring sampling bc selection_prob >= 1',
                stacklevel=2,
            )
            sampling_strategy_name = 'highest_scoring'
        if sampling_strategy_name == 'most_recent':
            sampled_items = population[-n:]
        if sampling_strategy_name == 'random':
            sampled_items = random.sample(population, k=k)
        if sampling_strategy_name == 'highest_scoring':
            # Sort by score (index 1) in ascending order, then take last n
            sampled_items = sorted(population, key=lambda x: x[1])[-n:]
        if sampling_strategy_name == 'tournament':
            # sort population from high to low score
            ranked_population = sorted(
                population,
                key=lambda x: x[1],
                reverse=True,
            )

            # calculate and normalize sampling weights
            weights = [
                selection_prob * ((1 - selection_prob) ** i)
                for i in range(len(ranked_population))
            ]
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]

            # sample
            chosen_indices = self.rng.choice(
                len(ranked_population),
                size=k,
                replace=False,
                p=normalized_weights,
            )
            sampled_items = [ranked_population[idx] for idx in chosen_indices]
            # NOTE(MS): these samples are not strictly ordered
        if sampling_strategy_name == 'wheel':
            print(f'POPULATION SIZE: {len(population)=}')
            # https://arxiv.org/abs/1109.3627

            # Extract scores (index 1 of the tuple)
            # add tiny value to prevent division by 0
            epsilon = 1e-6
            scores = [
                max(0.0, float(item[1])) + epsilon for item in population
            ]
            total_score = sum(scores)

            # Avoid division by zero if all scores are zero
            if total_score == 0:
                probabilities = [1.0 / len(population)] * len(population)
            else:
                probabilities = [s / total_score for s in scores]

            # Sample without replacement using the calculated probabilities
            chosen_indices = self.rng.choice(
                len(population),
                size=k,
                replace=False,
                p=probabilities,
            )
            sampled_items = [population[idx] for idx in chosen_indices]

        return sampled_items

    def __len__(self) -> int:
        """Total # of current solutions in active pop."""
        return len(self.bank)

    def get_num_total_iterations(self) -> int:
        """Total # of past solutions generated."""
        return len(self.never_prune_bank)
