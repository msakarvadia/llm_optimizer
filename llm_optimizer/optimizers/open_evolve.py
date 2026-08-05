"""Logic to abstract away LLM-driven optimization."""

from __future__ import annotations

from typing import Any

import openevolve.utils.code_utils as _code_utils
from openevolve import run_evolution
from openevolve.config import Config
from openevolve.config import LLMModelConfig

from llm_optimizer.optimizers.base_optimizer import Optimizer
from llm_optimizer.tasks.base_task import Task

# NOTE(MS): openevolve's evaluator callback only ever receives a file path
# (it runs in a forked worker process, so it can't receive a live Python
# object directly). A single module-level reference is enough to bridge
# that gap: each OpenEvolveOptimizer instance drives exactly one task per
# optimize() call, set right before run_evolution() spins up its (forked)
# worker pool, so workers inherit the correct reference automatically.
_ACTIVE_TASK: Task | None = None


def evaluator(solution_path: str) -> dict[str, float]:
    """Wrapper for task specific evaluator.

    Reads the full rewritten file back as the solution text -- no marker
    parsing or task lookup required, since _ACTIVE_TASK already points at
    the (only) task this optimizer instance is driving.

    NOTE(MS): openevolve executes this function in an isolated context (it
    dumps the evaluator to a file and reloads it), so it can't see this
    module's other globals/imports -- everything used here has to be
    (re)imported inline, same reason the old registry lookup did this too.
    """
    import re  # noqa: PLC0415 -- must stay local, see docstring above

    from llm_optimizer.optimizers.open_evolve import (  # noqa: PLC0415
        _ACTIVE_TASK,
    )

    with open(solution_path, encoding='utf-8') as f:
        solution = f.read()
    print('SOLUTION PATHvv' * 40)
    print(solution_path)
    print('SOLUTION PATH ^^' * 40)

    # openevolve auto-wraps initial_program in these markers if they're not
    # already present (see openevolve.api.run_evolution). Prefer extracting
    # just the delimited region -- it discards anything the LLM wrapped
    # around the actual solution (stray comments, variable assignments,
    # etc.) as long as it kept the markers. But nothing guarantees a
    # full-file rewrite echoes them back, so fall back to the whole file
    # (not a failure) when they're missing rather than scoring 0.
    match = re.search(
        r'#\s*EVOLVE-BLOCK-START\s*\n(.*?)\n\s*#\s*EVOLVE-BLOCK-END',
        solution,
        re.DOTALL,
    )
    solution = match.group(1).strip() if match else solution.strip()

    print('OPEN EVOLVE SOLUTION vv' * 40)
    print(solution)
    print('OPEN EVOLVE SOLUTION ^^' * 40)

    if _ACTIVE_TASK is None:
        return {'combined_score': 0.0}

    # NOTE(MS): val_score deliberately discarded -- it's held-out/test
    # signal and must not leak into the evolutionary feedback loop
    result, extra_info, _val_score = _ACTIVE_TASK.evaluate(solution)
    return {'combined_score': result} | extra_info


class OpenEvolveOptimizer(Optimizer):
    """LLM optimizer.

    # https://github.com/algorithmicsuperintelligence/openevolve/tree/main
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

        self.LLM_MODEL = kwargs['model_name']
        api_key = kwargs['api_key']
        if api_key is None:
            raise ValueError('API key not found.')
        self.config = Config()
        # NOTE(MS): openevolve's worker processes rebuild LLMConfig from
        # scratch (see ProcessParallelController._serialize_config /
        # _worker_init), which re-triggers LLMConfig.__post_init__ and
        # backfills any unset per-model field (incl. max_tokens) from this
        # top-level default. Leaving it at openevolve's own default (4096)
        # silently truncates reasoning-heavy responses mid-generation before
        # they ever reach a real code block -- setting it here (not just on
        # the per-model config below) is what actually reaches the LLM call.
        self.config.llm.max_tokens = None
        # NOTE(MS): openevolve's default per-request timeout (60s) is too
        # short once max_tokens is uncapped -- reasoning-heavy responses can
        # legitimately take longer than that to finish, and hitting the
        # timeout just burns through all retries and stalls instead of
        # truncating. Raise it accordingly.
        self.config.llm.timeout = 300
        self.config.llm.models = [
            LLMModelConfig(
                name=self.LLM_MODEL,
                api_key=api_key,
                api_base=kwargs['base_url'],
            ),
        ]

        # NOTE(MS): variables to manage in-context examples/rewards
        self.n = num_past_sol
        self.noise = noise
        self.num_parallel_search = num_parallel_search
        print(self.config)
        self.config.database.archive_size = num_past_sol

        # This is the default ratio that comes pre-defined w/ open-evolve
        self.config.database.population_size = 3.5 * num_past_sol

    def optimize(self, num_iter: int = 5) -> None:
        """Optimization loop for task."""
        # TODO(MS): impl convergence criteria

        global _ACTIVE_TASK  # noqa: PLW0603 -- see module docstring above
        _ACTIVE_TASK = self.task

        # NOTE(MS): in full-rewrite mode, openevolve's own worker code does
        # `from openevolve.utils.code_utils import parse_full_rewrite` right
        # before calling it (process_parallel.py), which does its own
        # ```{language}...``` markdown-fence stripping on the raw LLM
        # response before we ever see it -- and is what caused the stray
        # "python" leading-tag bug. We want ALL markdown/fence parsing to
        # happen in exactly one place (evaluator(), which already looks for
        # EVOLVE-BLOCK markers), so patch it to a no-op that hands back the
        # raw response untouched. Patching the module attribute (imported
        # at module scope above, not process_parallel's copy) works because
        # that import is resolved at call time, and workers are forked
        # after this runs, so they inherit the patched version too.
        _code_utils.parse_full_rewrite = lambda llm_response, _language=None: (
            llm_response
        )

        task_prompt = (
            f'{self.task.task_description} '
            f'Your goal is to {self.task.direction} {self.task.metric}. '
            f'Output only the bare minimum text to reach the objective goal.'
        )
        self.config.prompt.system_message = task_prompt
        # Force OpenEvolve to handle full-file string rewrites
        # Needed to unify interface b/w GEPA and openevolve
        self.config.diff_based_evolution = False

        # NOTE(MS): this returns an object
        run_evolution(
            initial_program=self.task.seed_candidate,
            evaluator=evaluator,
            iterations=num_iter,
            config=self.config,
        )

        # TODO(MS): temporarily save solution bank
        # experiment_dir = 'temp_results'
        # self.solution_bank.save_to_json(experiment_dir)
