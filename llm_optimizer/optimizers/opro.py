"""Logic to abstract away LLM-driven optimization."""

from __future__ import annotations

import json
import os
import random
import shutil
import time
import warnings
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from llm_optimizer.optimizers.base_optimizer import Optimizer
from llm_optimizer.optimizers.judge_prompts import CODE_TASK_NAMES
from llm_optimizer.optimizers.judge_prompts import JUDGE_SYSTEM_MSG
from llm_optimizer.optimizers.judge_prompts import JUDGE_USER_MSG
from llm_optimizer.optimizers.judge_prompts import NOVELTY_SYSTEM_MSG
from llm_optimizer.optimizers.judge_prompts import NOVELTY_USER_MSG
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
from llm_optimizer.utils import build_openai_client
from llm_optimizer.utils import prompt_lm


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
        # logic to seed initial population if it exists
        # only if experiment hasn't been run before (i.e. no checkpoint exists)
        if os.path.isfile(self.init_population_path):
            for file_name in [
                'long_running_solution_bank.json',
                'current_solution_bank.json',
            ]:
                destination_path = os.path.join(self.experiment_dir, file_name)
                # if experiment dir already has a checkpoint,
                # skip copying the initial seed population
                # NOTE(MS): we are assuming that if the checkpoint exists,
                # it was started w/ a valid initial seed population,
                # so we don't want to overwrite it
                if os.path.exists(destination_path):
                    # NOTE(MS): don't clobber a checkpoint from a prior/
                    # resumed run with the initial seed population again.
                    print(
                        f'Skipping seed copy, checkpoint already exists: '
                        f'{destination_path}',
                    )
                    continue
                # if the experiment dir doesn't exist,
                # create it and copy the initial seed population
                shutil.copy(self.init_population_path, destination_path)

        self.seed = kwargs['seed']
        self.sampling_strategy_name = kwargs['sampling_strategy_name']
        self.sampling_prob = kwargs['sampling_prob']
        self.max_population_size = kwargs['max_population_size']
        self.pruning_strategy = kwargs['pruning_strategy']
        self.prune_noise = kwargs.get('prune_noise', 0.0)
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
            failed_score=getattr(task, 'failed_score', None),
            embed_model=kwargs.get('embed_model'),
            sim_thresh=kwargs.get('sim_thresh', 0.0),
            llm_judge=kwargs.get('llm_judge'),
            llm_judge_base_url=kwargs.get('llm_judge_base_url'),
            llm_judge_api_key=kwargs.get('llm_judge_api_key'),
            task=task,
            task_name=kwargs.get('task_name'),
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

    def optimize(
        self,
        num_iter: int = 5,
        max_tokens: int | None = None,
    ) -> None:
        """Optimization loop for task.

        max_tokens: optional cumulative token budget (mutator +
            diversity-judge calls, see SolutionBank.cumulative_tokens_spent).
            None (default) disables the check, so num_iter is the sole
            stopping criterion. Checked at the top of the loop only (same
            cadence as num_iter), so a run can overshoot the budget by up
            to one iteration's worth of tokens.
        """
        # TODO(MS): impl convergence criteria

        while self.solution_bank.get_num_total_iterations() <= num_iter and (
            max_tokens is None
            or self.solution_bank.get_cumulative_tokens_spent() <= max_tokens
        ):
            solution_bank = self.solution_bank.get_solutions(
                n=self.n,
                sampling_strategy_name=self.sampling_strategy_name,
                noise=self.noise,
                selection_prob=self.sampling_prob,
            )
            start = time.perf_counter()
            solution, token_usage = self.mutator.mutate(
                solution_bank,
                self.task,
            )
            wallclock_seconds = time.perf_counter() - start
            generation_metadata = {
                **token_usage,
                'wallclock_seconds': wallclock_seconds,
            }
            if self.truncate_generated_solution > 0:
                solution = solution[: self.truncate_generated_solution]
            score, extra_info, val_score = self.task.evaluate(solution)
            self.solution_bank.add_solution_score_pair(
                solution,
                score,
                extra_info,
                val_score,
                generation_metadata,
            )
            self.solution_bank.prune_population(noise=self.prune_noise)
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
        failed_score: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize bank to store solution/score pairs.

        kwargs (all optional, diversity-check specific; see
        add_solution_score_pair): embed_model, sim_thresh, llm_judge,
        llm_judge_base_url, llm_judge_api_key, task, task_name. Kept out
        of the named signature to stay under the 5-arg lint limit --
        GEPA/open_evolve construct SolutionBank without any of these.
        """
        # dict[optimizaiton_iteration (int) :
        #          {'solution':solution, 'score':score, 'metadata':...}]
        self.bank: dict[int, dict[str, Any]] = {}
        self.never_prune_bank: dict[int, dict[str, Any]] = {}

        self.seed = seed
        random.seed(self.seed)
        self.rng = np.random.default_rng(seed=self.seed)
        self.max_population_size = max_population_size
        self.pruning_strategy = pruning_strategy
        # NOTE(MS): task-specific sentinel score for failed evaluations
        # (e.g. -math.inf, -1.0); excluded from noise's std_dev calc
        self.failed_score = failed_score
        # NOTE(MS): running total of generation_metadata['total_tokens']
        # across every candidate ever logged (add_solution_score_pair
        # only -- upsert_solution_score_pair's update-existing-entry path
        # doesn't touch this, since a later re-evaluation shouldn't move
        self.cumulative_tokens_spent = 0

        # NOTE(MS): diversity check (see add_solution_score_pair) -- max
        # cosine similarity (embedding space) a new candidate may have to
        # any active-pool solution before it's rejected. sim_thresh <= 0
        # disables the check entirely, so embedding_model/llm_judge_client
        # are only ever loaded (CPU, can be multi-GB for Qodo-Embed) when
        # actually needed.
        self.sim_thresh = kwargs.get('sim_thresh', 0.0)
        # NOTE(MS): Any -- these hold either None or a loaded
        # SentenceTransformer/OpenAI/Task instance depending on
        # sim_thresh/llm_judge, not a plain string.
        self.embedding_model: Any = None
        self.llm_judge_client: Any = None
        self.llm_judge_model_name = kwargs.get('llm_judge', '')
        # NOTE(MS): task/task_name feed llm_judge_sim's prompt -- task for
        # task_description/solution_description/direction/metric (non-code
        # tasks), task_name to pick the code-task judge template (see
        # judge_prompts.CODE_TASK_NAMES). Not on the Task base class, so
        # threaded through separately from OPROOptimizer.
        self.task: Any = kwargs.get('task')
        self.task_name = kwargs.get('task_name')
        if self.sim_thresh > 0:
            self.embedding_model = SentenceTransformer(
                kwargs.get('embed_model', ''),
                device='cpu',
            )
            if self.llm_judge_model_name:
                self.llm_judge_client = build_openai_client(
                    kwargs.get('llm_judge_api_key', ''),
                    kwargs.get('llm_judge_base_url', ''),
                )

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
            if self.never_prune_bank:
                # NOTE(MS): cumulative_tokens_spent only ever grows, so the
                # highest-iteration entry (not necessarily the numerically
                # largest value, in case of float weirdness -- but in
                # practice these are the same) holds the running total as
                # of the last candidate logged.
                last_entry = self.never_prune_bank[max(self.never_prune_bank)]
                self.cumulative_tokens_spent = last_entry.get(
                    'cumulative_tokens_spent',
                    0,
                )

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

    def prune_population(self, noise: float = 0.0) -> None:
        """Remove old population solutions.

        noise: same adaptive gaussian noise as get_solutions
            (scale factor on std_dev of current scores), applied
            to scores before ranking so 'lowest_scoring' pruning
            isn't fully deterministic/one-sided.
        """
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
            # Sort bank keys by their (possibly noised) score value
            # in ascending order
            keys = list(self.bank.keys())
            items = [(k, self.bank[k]['score'], None) for k in keys]
            noised_items = self.apply_noise_to_scores(items, noise)
            keys_by_score = sorted(noised_items, key=lambda item: item[1])
            # Identify the keys with the lowest (noised) scores to drop
            keys_to_remove = [
                key for key, _, _ in keys_by_score[:num_to_remove]
            ]
            # Delete the lowest scoring items directly
            for key in keys_to_remove:
                del self.bank[key]

    def add_solution_score_pair(
        self,
        solution: str,
        score: float,
        extra_info: dict[str, Any],
        val_score: float | None,
        generation_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add solution/score pairs to bank."""
        next_iter = max(self.never_prune_bank, default=0) + 1

        if generation_metadata:
            self.cumulative_tokens_spent += (
                generation_metadata.get('total_tokens', 0) or 0
            )

        self.bank[next_iter] = {}
        self.bank[next_iter]['solution'] = solution
        self.bank[next_iter]['score'] = score
        self.bank[next_iter]['extra_info'] = extra_info
        self.bank[next_iter]['val_score'] = val_score
        self.bank[next_iter]['generation_metadata'] = generation_metadata
        # NOTE(MS): total tokens spent (across every candidate, this one
        # included) by the time this candidate was created -- pairs with
        # 'score' to plot a tokens-spent-vs-score curve directly off this
        # file, ordered by iteration key. See __init__'s NOTE.
        self.bank[next_iter]['cumulative_tokens_spent'] = (
            self.cumulative_tokens_spent
        )
        # NOTE(MS): # of evaluator calls that have contributed to this
        # entry's current score/extra_info/val_score -- 1 here, bumped by
        # upsert_solution_score_pair on later re-evaluations of the same
        # candidate text (e.g. GEPA's minibatch-check -> full-valset-check
        # -> possible later re-selection as a parent). score/extra_info
        # only ever reflect the *most recent* of those calls, so this is
        # the one place the fact that N calls happened isn't silently
        # lost.
        self.bank[next_iter]['eval_count'] = 1

        self.never_prune_bank[next_iter] = self.bank[next_iter]

        # this is where self.bank[next_iter] gets popped back out if it
        # wasn't diverse enough compared to the current active pool
        # NOTE(MS): sim_thresh == -1 is for exact deduplication only.
        if self.sim_thresh == -1:
            active_pop = [
                sub_dict['solution'] for sub_dict in self.bank.values()
            ]
            active_pop.remove(solution)  # exclude this entry itself
            is_duplicate = solution in active_pop
            self.bank[next_iter]['exact_duplicate_rejected'] = is_duplicate
            if is_duplicate:
                self.bank.pop(next_iter)
        elif self.sim_thresh > 0:
            active_pop = [
                sub_dict['solution'] for sub_dict in self.bank.values()
            ]
            active_pop.remove(solution)
            too_similar, closest_solution, max_similarity = (
                self.eval_cosine_sim(active_pop, solution)
            )
            self.bank[next_iter]['max_similarity'] = max_similarity
            self.bank[next_iter]['llm_judge_rejected'] = None
            if too_similar and self.llm_judge_client:
                too_similar, judging_tokens = self.llm_judge_sim(
                    closest_solution,
                    solution,
                )
                self.bank[next_iter]['llm_judge_rejected'] = too_similar
                # NOTE(MS): bank/never_prune_bank[next_iter] are aliased,
                # so this already updates both -- don't double-apply.
                self.bank[next_iter]['cumulative_tokens_spent'] += (
                    judging_tokens
                )
                self.bank[next_iter]['llm_judge_tokens'] = judging_tokens
                self.cumulative_tokens_spent += judging_tokens
            if too_similar:
                self.bank.pop(next_iter)
        self.never_prune_bank[next_iter]['active_population'] = list(self.bank)

    def eval_cosine_sim(
        self,
        active_solution_pool: list[str],
        solution: str,
    ) -> tuple[bool, str | None, float | None]:
        """Evaluate if a new solution is diverse enough to enter active pool.

        Returns (too_similar, closest_solution, max_similarity); all None
        or False if the pool is empty.
        """
        if not active_solution_pool:
            return False, None, None

        # NOTE(MS): no caching -- embeds the full active pool + candidate
        # in a single batched call every time this runs.
        embeddings = self.embedding_model.encode(
            [*active_solution_pool, solution],
            convert_to_numpy=True,
        )
        pool_vecs, solution_vec = embeddings[:-1], embeddings[-1:]
        sims = cosine_similarity(solution_vec, pool_vecs)[0]
        best_idx = int(sims.argmax())
        sim = float(sims[best_idx])

        too_similar = sim > self.sim_thresh
        closest_solution = (
            active_solution_pool[best_idx] if too_similar else None
        )
        return too_similar, closest_solution, sim

    def llm_judge_sim(
        self,
        closest_solution: str | None,
        solution: str,
    ) -> tuple[bool, int]:
        """Evaluate if a new solution is diverse enough to enter active pool.

        Second opinion on eval_cosine_sim's verdict: prompts
        self.llm_judge_client to compare solution against the single
        closest active-pool match (closest_solution) it flagged.

        Template + NOVEL/NOT_NOVEL vocabulary follow ShinkaEvolve's
        novelty judge: https://github.com/SakanaAI/ShinkaEvolve/blob/main/shinka/prompts/prompts_novelty.py
        Code tasks (judge_prompts.CODE_TASK_NAMES) use that template
        verbatim; other tasks use a task-grounded variant of it.
        """
        if self.task_name in CODE_TASK_NAMES:
            system_msg = NOVELTY_SYSTEM_MSG
            user_msg = NOVELTY_USER_MSG.format(
                language='python',
                existing_code=closest_solution,
                proposed_code=solution,
            )
        else:
            system_msg = JUDGE_SYSTEM_MSG.format(
                solution_description=self.task.solution_description,
                task_description=self.task.task_description,
                direction=self.task.direction,
                metric=self.task.metric,
            )
            user_msg = JUDGE_USER_MSG.format(
                existing_solution=closest_solution,
                proposed_solution=solution,
            )

        try:
            response, token_usage = prompt_lm(
                self.llm_judge_client,
                user_msg,
                model_name=self.llm_judge_model_name,
                system_msg=system_msg,
            )
        except Exception:
            # NOTE(MS): matching ShinkaEvolve's novelty judge --
            # a broken/errored judge call shouldn't block an otherwise-
            # fine candidate, so treat it as novel/diverse
            return False, 0

        verdict = response.strip().upper()
        is_novel = verdict.startswith('NOVEL') or verdict.startswith(
            '**NOVEL**',
        )
        return not is_novel, token_usage['total_tokens']

    def upsert_solution_score_pair(
        self,
        solution: str,
        score: float,
        extra_info: dict[str, Any],
        val_score: float | None,
        generation_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a solution, or update its existing entry if already logged.

        Unlike add_solution_score_pair (always appends a new row),
        this is for optimizers where the *same* candidate text can be
        evaluated multiple times (e.g. GEPA re-evaluating a candidate on
        a full valset right after a minibatch check, or re-evaluating it
        again much later if reselected as a parent) and only one row per
        distinct candidate is wanted in the checkpoint.

        score/extra_info/val_score are overwritten with this call's
        values (the most recent evaluation is treated as authoritative).
        generation_metadata's numeric fields are summed into whatever is
        already stored, rather than replaced, so the row always reflects
        total tokens/time spent across every evaluation of this candidate
        to date. Non-numeric fields (e.g. a boolean flag) are overwritten.
        """
        existing_iter = next(
            (
                i
                for i, entry in self.never_prune_bank.items()
                if entry['solution'] == solution
            ),
            None,
        )
        if existing_iter is None:
            self.add_solution_score_pair(
                solution,
                score,
                extra_info,
                val_score,
                generation_metadata,
            )
            return

        entry = self.never_prune_bank[existing_iter]
        entry['score'] = score
        entry['extra_info'] = extra_info
        entry['val_score'] = val_score
        if generation_metadata:
            merged = dict(entry.get('generation_metadata') or {})
            for key, value in generation_metadata.items():
                if isinstance(value, (int, float)) and not isinstance(
                    value,
                    bool,
                ):
                    merged[key] = merged.get(key, 0) + value
                else:
                    merged[key] = value
            entry['generation_metadata'] = merged
        # NOTE(MS): entry is the same dict object as self.bank[existing_iter]
        # (add_solution_score_pair aliases them, not copies), so if this
        # iter is still in the active/pruned population the mutation above
        # already applies there too. If it's been pruned out already, we
        # deliberately leave it out of self.bank -- pruning already decided
        # it's not part of the "current" view.

    def apply_noise_to_scores(
        self,
        items: list[tuple[Any, Any, Any]],
        noise: float,
    ) -> list[tuple[Any, Any, Any]]:
        """Add gaussian noise to the scores of (solution, score, extra_info).

        noise: scale factor; std_dev of the injected noise is
            noise * std_dev(current scores). Adaptive to the
            current spread in rewards. Scores equal to the task's
            failed_score sentinel (if the task defines one) are
            excluded from the std_dev calculation and left
            untouched, so a run of failed evaluations can't blow up
            the noise applied to every other score.
        """
        if noise <= 0:
            return items

        valid_scores = [
            float(score)
            for _, score, _ in items
            if self.failed_score is None or score != self.failed_score
        ]
        if len(valid_scores) <= 1:
            return items

        std_dev = float(np.std(valid_scores))
        noised_items = []
        for sol, score, extra_info in items:
            if self.failed_score is not None and score == self.failed_score:
                noised_items.append((sol, score, extra_info))
                continue
            noised_items.append(
                (
                    sol,
                    float(score) + random.gauss(0.0, noise * std_dev),
                    extra_info,
                ),
            )
        return noised_items

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

        raw_items = self.apply_noise_to_scores(raw_items, noise)

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
            # Shift by the min score (if negative) instead of flooring at
            # zero, so relative differences among non-positive scores are
            # preserved rather than collapsed to a single weight. No-op
            # when every score is already >= 0 (shift = epsilon in that
            # case, identical to the old max(0, score) + epsilon formula).
            epsilon = 1e-6
            min_score = min(float(item[1]) for item in population)
            shift = max(0.0, -min_score) + epsilon
            scores = [float(item[1]) + shift for item in population]
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

    def get_cumulative_tokens_spent(self) -> int:
        """Total tokens spent across every candidate generated so far."""
        return self.cumulative_tokens_spent
