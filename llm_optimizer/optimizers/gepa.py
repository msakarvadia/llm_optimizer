"""Logic to abstract away LLM-driven optimization."""

from __future__ import annotations

import os
import random
import time
from typing import Any

import litellm
from gepa.optimize_anything import EngineConfig
from gepa.optimize_anything import GEPAConfig
from gepa.optimize_anything import optimize_anything
from gepa.optimize_anything import ReflectionConfig
from gepa.strategies.candidate_selector import TopKParetoCandidateSelector

from llm_optimizer.optimizers.base_optimizer import Optimizer
from llm_optimizer.optimizers.opro import SolutionBank
from llm_optimizer.tasks.base_task import Task
from llm_optimizer.utils import _extract_token_usage


class GEPAOptimizer(Optimizer):
    """LLM optimizer.

    # https://arxiv.org/abs/2605.19633
    # https://github.com/gepa-ai/gepa
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

        self.experiment_dir = f'{kwargs["experiment_dir"]}/'
        print(f'{self.experiment_dir=}')
        os.makedirs(self.experiment_dir, exist_ok=True)

        # NOTE(MS): kept as a fallback for any gepa/litellm-internal code
        # that still reads these -- _metered_reflection_lm below passes
        # api_key/api_base explicitly on every call instead of relying on
        # this process-global state.
        os.environ['OPENAI_API_BASE'] = kwargs['base_url']
        os.environ['OPENAI_API_KEY'] = kwargs['api_key']
        self.LLM_MODEL = f'openai/{kwargs["model_name"]}'
        self._api_key = kwargs['api_key']
        self._base_url = kwargs['base_url']

        # NOTE(MS): variables to manage in-context examples/rewards
        self.n = num_past_sol
        self.noise = noise
        self.num_parallel_search = num_parallel_search
        self.max_population_size = kwargs['max_population_size']

        # for tracking token/time consumption per solution
        self.solution_bank = SolutionBank(
            seed=kwargs['seed'],
            max_population_size=self.max_population_size,
            pruning_strategy=kwargs['pruning_strategy'],
            failed_score=getattr(task, 'failed_score', None),
        )
        self.solution_bank.read_from_checkpoint(self.experiment_dir)

        # NOTE(MS): holds the most recent reflection_lm call's token/
        # timing usage, overwritten (not accumulated) on each successful
        # call. Overwrite -- not accumulate
        # each successful call already is the complete total for
        # whatever candidate follows it
        self._last_generation_metadata: dict[str, Any] | None = None

    def _metered_reflection_lm(
        self,
        prompt: str | list[dict[str, Any]],
    ) -> str:
        """Call litellm directly, recording token/timing usage.

        Passed to gepa as a callable (not a bare model-name string) so
        optimize_anything's own make_litellm_lm() conversion is
        enables token/time tracking
        """
        messages = (
            [{'role': 'user', 'content': prompt}]
            if isinstance(prompt, str)
            else prompt
        )
        start = time.perf_counter()
        response = litellm.completion(
            model=self.LLM_MODEL,
            messages=messages,
            api_key=self._api_key,
            api_base=self._base_url,
        )
        elapsed = time.perf_counter() - start

        usage = _extract_token_usage(response)
        # NOTE(MS): overwrite, not accumulate -- see __init__'s NOTE.
        self._last_generation_metadata = {
            **usage,
            'wallclock_seconds': elapsed,
        }
        return response.choices[0].message.content or ''

    def _pop_generation_metadata(self) -> dict[str, Any] | None:
        """Read-and-reset the pending reflection-call metadata.

        Returns None if no reflection call has completed since the last
        read -- true for the seed candidate, and for any re-check of an
        already-known candidate (see _gepa_evaluator's novelty gate,
        which never even calls this for those).
        """
        metadata = self._last_generation_metadata
        self._last_generation_metadata = None
        return metadata

    def _gepa_evaluator(self, candidate: str) -> tuple[float, dict[str, Any]]:
        """Adapt Task.evaluate's 3-tuple return to GEPA's (score, side_info).

        Task.evaluate returns (score, extra_info, val_score), but GEPA's
        evaluator protocol expects either a bare score or a
        (score, side_info) pair. Unpacking the raw 3-tuple as a 2-tuple
        inside gepa's EvaluatorWrapper raises a ValueError, so we adapt
        here instead of handing task.evaluate to gepa directly.

        NOTE(MS): val_score is intentionally NOT included in side_info.
        It's the held-out/test score, not training feedback -- exposing
        it to the reflection LM would let optimization "see" the test
        signal it's meant to generalize to, i.e. test-set hacking.

        here is where we track the token/time consumption
        """
        is_known = any(
            entry['solution'] == candidate
            for entry in self.solution_bank.never_prune_bank.values()
        )
        # NOTE(MS): only pop pending reflection-call metadata for a
        # genuinely new candidate -- a re-check's text is always already
        # known, so gating here means it can never accidentally inherit
        # metadata left over from an unrelated (e.g. since-failed)
        # proposal. This is a no-op on the normal path (a re-check would
        # have popped None anyway, since no reflection call precedes it)
        # and closes the one case that matters: a failed proposal that
        # already burned real tokens getting misattributed to a later,
        # unrelated candidate's cumulative total.
        generation_metadata = (
            None if is_known else self._pop_generation_metadata()
        )

        score, extra_info, val_score = self.task.evaluate(candidate)

        self.solution_bank.upsert_solution_score_pair(
            candidate,
            score,
            extra_info,
            val_score,
            generation_metadata,
        )
        self.solution_bank.prune_population()
        self.solution_bank.save_to_json(self.experiment_dir)

        return score, dict(extra_info)

    def optimize(self, num_iter: int = 5) -> None:
        """Optimization loop for task."""
        # TODO(MS): impl convergence criteria

        # rng = np.random.default_rng(seed=42)
        rng = random.Random(42)

        self.config = GEPAConfig(
            engine=EngineConfig(
                # NOTE(MS): setting run_dir gives us checkpoint/resume for
                # free -- gepa auto-loads it on the next call if it's
                # already present.
                run_dir=self.experiment_dir,
                # NOTE(MS): max_metric_calls counts individual evaluator
                # calls (parent re-eval + new-candidate eval each iteration,
                # plus a full-valset re-eval on every accepted proposal) --
                # 2-3 per proposed candidate, not 1. max_candidate_proposals
                # counts loop iterations (state.i, one per propose() call)
                # regardless of how many evaluator calls that iteration
                # burns, so num_iter here means "# of candidates GEPA gets
                # to propose", matching OPRO/OpenEvolve's num_iter semantics.
                max_candidate_proposals=num_iter,
                # NOTE(MS): parallel needs to be false to ensure
                # correct resource tracking
                parallel=False,
                # max_workers=64,
                # cache_evaluation=True,
                # track_best_outputs=True,
                # NOTE(K) defaults to 5...need to change this?
                # https://github.com/gepa-ai/gepa/releases
                # https://github.com/gepa-ai/gepa/pull/246
                candidate_selection_strategy=TopKParetoCandidateSelector(
                    k=self.max_population_size,
                    rng=rng,
                ),
            ),
            reflection=ReflectionConfig(
                # NOTE(MS): passing our own callable for resource tracking
                reflection_lm=self._metered_reflection_lm,
            ),
        )

        task_prompt = (
            f'{self.task.task_description} '
            f'Your goal is to {self.task.direction} {self.task.metric}. '
            f'Output only the bare minimum text to reach the objective goal.'
        )

        # NOTE (THIS RETURNS A RESULT)
        optimize_anything(
            # TODO(MS): give a seed candidate to the task definition!!
            seed_candidate=self.task.seed_candidate,
            evaluator=self._gepa_evaluator,
            objective=task_prompt,
            config=self.config,
        )
