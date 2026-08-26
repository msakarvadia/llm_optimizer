"""Logic to abstract away LLM-driven optimization."""

from __future__ import annotations

import os
import time
from typing import Any

import openevolve.controller as _oe_controller
import openevolve.utils.code_utils as _code_utils
from openai.resources.chat.completions.completions import Completions
from openevolve import run_evolution
from openevolve.config import Config
from openevolve.config import LLMModelConfig

from llm_optimizer.optimizers.base_optimizer import Optimizer
from llm_optimizer.optimizers.opro import SolutionBank
from llm_optimizer.tasks.base_task import Task
from llm_optimizer.utils import _extract_token_usage

# NOTE(MS): captured once, at import time, before _resumable_run below can
# ever patch OpenEvolve.run -- gives it something un-patched to delegate to.
_original_oe_run = _oe_controller.OpenEvolve.run

# NOTE(MS): same idea, for the token/timing patch below.
_original_completions_create = Completions.create

# NOTE(MS): openevolve's evaluator callback only ever receives a file path
# (it runs in a forked worker process, so it can't receive a live Python
# object directly). A single module-level reference is enough to bridge
# that gap: each OpenEvolveOptimizer instance drives exactly one task per
# optimize() call, set right before run_evolution() spins up its (forked)
# worker pool, so workers inherit the correct reference automatically.
_ACTIVE_TASK: Task | None = None
_ACTIVE_SOLUTION_BANK: SolutionBank | None = None
_ACTIVE_EXPERIMENT_DIR: str | None = None

# NOTE(MS): accumulates token usage/wall time across every
# Completions.create call since the last _pop_generation_metadata() read.
# None means nothing has been accumulated since the last read (used to
# tell "seed candidate, no generation call preceded it" apart from "a real
# generation call that happened to report 0 tokens"). See
# experiments/OPENEVOLVE_GENERATION_METADATA_PLAN.md for why this lives at
# the SDK layer (not openevolve's own private _call_api), why
# accumulate-not-overwrite is correct under retries and content-rejected
# generations (their tokens roll forward onto the next successful
# candidate), and why reading it at the top of evaluator() correlates it
# to the right candidate without needing per-candidate IDs or locking --
# each forked worker process has its own independent copy of this global.
_GENERATION_METADATA_ACCUMULATOR: dict[str, float] | None = None


def _metered_completions_create(
    self: Completions,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Patched Completions.create: accumulate token usage + wall time.

    Delegates fully to the original implementation and only observes the
    return value as a side effect -- see module docstring above for why
    this (not openevolve's OpenAILLM._call_api) is the patch point.
    """
    global _GENERATION_METADATA_ACCUMULATOR  # noqa: PLW0603

    start = time.perf_counter()
    response = _original_completions_create(self, *args, **kwargs)
    elapsed = time.perf_counter() - start

    usage = _extract_token_usage(response)
    if _GENERATION_METADATA_ACCUMULATOR is None:
        _GENERATION_METADATA_ACCUMULATOR = {
            'input_tokens': 0,
            'output_tokens': 0,
            'reasoning_tokens': 0,
            'total_tokens': 0,
            'wallclock_seconds': 0.0,
        }
    for key, value in usage.items():
        _GENERATION_METADATA_ACCUMULATOR[key] += value
    _GENERATION_METADATA_ACCUMULATOR['wallclock_seconds'] += elapsed

    return response


def _pop_generation_metadata() -> dict[str, float] | None:
    """Read-and-reset the token/timing accumulator.

    Called as the first thing inside evaluator(), before task.evaluate()
    can make any LLM calls of its own that would otherwise land in the
    same accumulator. Returns None if nothing was accumulated since the
    last read (the seed candidate, or -- in principle -- two evaluator()
    calls with no generation call in between), matching OPRO's convention
    of leaving generation_metadata unset for the seed.
    """
    global _GENERATION_METADATA_ACCUMULATOR  # noqa: PLW0603

    metadata = _GENERATION_METADATA_ACCUMULATOR
    _GENERATION_METADATA_ACCUMULATOR = None
    return metadata


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
        _ACTIVE_EXPERIMENT_DIR,
    )
    from llm_optimizer.optimizers.open_evolve import (  # noqa: PLC0415
        _ACTIVE_SOLUTION_BANK,
    )
    from llm_optimizer.optimizers.open_evolve import (  # noqa: PLC0415
        _ACTIVE_TASK,
    )
    from llm_optimizer.optimizers.open_evolve import (  # noqa: PLC0415
        _pop_generation_metadata,
    )

    # NOTE(MS): read-and-clear FIRST, before task.evaluate() gets a chance
    # to make any LLM calls of its own -- see _pop_generation_metadata's
    # docstring for why ordering (not scoping/IDs) is what makes this
    # correlate to the right candidate.
    generation_metadata = _pop_generation_metadata()

    with open(solution_path, encoding='utf-8') as f:
        solution = f.read()

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

    if _ACTIVE_TASK is None:
        return {'combined_score': 0.0}

    # NOTE(MS): val_score is used below only for our own SolutionBank
    # reporting -- it's still deliberately excluded from the returned
    # dict, so it never leaks into openevolve's own evolutionary feedback
    # loop (that dict becomes Program.metrics, which feeds future prompts).
    result, extra_info, val_score = _ACTIVE_TASK.evaluate(solution)

    if (
        _ACTIVE_SOLUTION_BANK is not None
        and _ACTIVE_EXPERIMENT_DIR is not None
    ):
        _ACTIVE_SOLUTION_BANK.add_solution_score_pair(
            solution,
            result,
            extra_info,
            val_score,
            generation_metadata,
        )
        _ACTIVE_SOLUTION_BANK.prune_population()
        _ACTIVE_SOLUTION_BANK.save_to_json(_ACTIVE_EXPERIMENT_DIR)

    return {'combined_score': result} | extra_info


def _find_latest_checkpoint(experiment_dir: str) -> str | None:
    """Find the highest-iteration checkpoint dir to resume from, if any.

    openevolve has no auto-resume -- even its own CLI requires the user
    to pass an explicit --checkpoint (see openevolve.cli), and just
    prints the latest one at the end of a run for the user to copy/paste
    next time. Same discovery logic as cli.py: sort
    checkpoints/checkpoint_<N> dirs by trailing numeric suffix.
    """
    checkpoint_root = os.path.join(experiment_dir, 'checkpoints')
    if not os.path.isdir(checkpoint_root):
        return None
    checkpoints = [
        os.path.join(checkpoint_root, name)
        for name in os.listdir(checkpoint_root)
        if os.path.isdir(os.path.join(checkpoint_root, name))
    ]
    if not checkpoints:
        return None
    return sorted(
        checkpoints,
        key=lambda p: int(p.rsplit('_', 1)[-1]) if '_' in p else 0,
    )[-1]


async def _resumable_run(
    self: _oe_controller.OpenEvolve,
    iterations: int | None = None,
    target_score: float | None = None,
    checkpoint_path: str | None = None,
) -> Any:
    """Patched OpenEvolve.run: auto-resume from self.output_dir if possible.

    run_evolution() (the high-level API optimize() calls below) never
    forwards a checkpoint_path to controller.run(), even though the
    controller itself supports resuming from one -- and openevolve has no
    auto-resume of its own. Patching run() itself (instead of
    reimplementing run_evolution()'s setup -- program/evaluator file prep,
    controller construction, asyncio.run -- by hand) means we only add the
    one missing lookup and let run_evolution() keep doing everything else.

    NOTE(MS): also caps `iterations` at the original target total on
    resume. optimize() calls this with the same `num_iter` on every
    invocation (fresh launch or resume alike), but controller.run() always
    treats `iterations` as "run this many MORE steps from wherever the
    checkpoint resumes" -- it never subtracts what the checkpoint already
    completed. Left unpatched, every job restart (crash/requeue/manual
    relaunch) tacks a full extra `num_iter` batch on top of the last one
    instead of topping up to it. Subtracting the checkpoint's
    `last_iteration + 1` (matching controller.run()'s own start_iteration
    formula) fixes that; when that's already >= the target we skip calling
    _original_oe_run entirely rather than pass iterations=0 through --
    controller.run() does `iterations or self.config.max_iterations`, and
    0 is falsy in Python, so 0 would silently fall back to
    config.max_iterations (10000) instead of running zero more steps.
    """
    if checkpoint_path is None:
        checkpoint_path = _find_latest_checkpoint(self.output_dir)

    if checkpoint_path is not None and iterations is not None:
        self._load_checkpoint(checkpoint_path)
        iterations = iterations - (self.database.last_iteration + 1)
        if iterations <= 0:
            return None

    return await _original_oe_run(
        self,
        iterations=iterations,
        target_score=target_score,
        checkpoint_path=checkpoint_path,
    )


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

        # build experiment path (also used as openevolve's checkpoint dir
        # -- see optimize(): run_evolution(output_dir=...) plus the
        # _resumable_run patch above, which looks here for a checkpoint
        # to resume from).
        self.experiment_dir = f'{kwargs["experiment_dir"]}/'
        print(f'{self.experiment_dir=}')
        os.makedirs(self.experiment_dir, exist_ok=True)

        # NOTE(MS): purely a reporting/observability side-channel -- does
        # NOT drive candidate generation (unlike OPRO, where this IS the
        # optimizer's state). openevolve's own checkpoints/ dir remains the
        # actual live optimization state, resumed separately via
        # _resumable_run. This just gives token-vs-score history in the
        # same long_running_solution_bank.json / current_solution_bank.json
        # format OPRO already writes, so the same notebooks can read either.
        self.solution_bank = SolutionBank(
            seed=kwargs['seed'],
            max_population_size=kwargs['max_population_size'],
            pruning_strategy=kwargs['pruning_strategy'],
            failed_score=getattr(task, 'failed_score', None),
        )
        self.solution_bank.read_from_checkpoint(self.experiment_dir)

        self.LLM_MODEL = kwargs['model_name']
        api_key = kwargs['api_key']
        if api_key is None:
            raise ValueError('API key not found.')
        self.config = Config()

        assert self.config.evaluator.parallel_evaluations == 1, (
            'OpenEvolveOptimizer does not support '
            'parallel_evaluations > 1 -- SolutionBank checkpointing is '
            'only safe with a single worker process.'
        )

        # ckpt every solution
        self.config.checkpoint_interval = 1
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
        # NOTE: evaluator.timeout also defaults to 300s, shorter than
        # circle_packing.py's 600s subprocess timeout -- raise it so the
        # outer timeout doesn't abandon (and orphan) a still-running eval.
        self.config.evaluator.timeout = 630
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

        global _ACTIVE_TASK, _ACTIVE_SOLUTION_BANK, _ACTIVE_EXPERIMENT_DIR  # noqa: PLW0603 -- see module docstring above
        _ACTIVE_TASK = self.task
        _ACTIVE_SOLUTION_BANK = self.solution_bank
        _ACTIVE_EXPERIMENT_DIR = self.experiment_dir

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

        # NOTE(MS): same reasoning as the parse_full_rewrite patch above --
        # patching OpenEvolve.run (imported at module scope, not a copy)
        # here means run_evolution()'s internal `OpenEvolve(...)` instance
        # picks up the patched, auto-resuming version. See _resumable_run's
        # docstring for why this is patched instead of exposing/duplicating
        # the run loop ourselves.
        _oe_controller.OpenEvolve.run = _resumable_run

        # NOTE(MS): same reasoning again -- patching the SDK class here
        # (before run_evolution()'s forked worker pool starts) means every
        # worker inherits the metered version. See
        # _metered_completions_create's docstring / plan doc for why this
        # is the patch point instead of openevolve's own _call_api.
        Completions.create = _metered_completions_create  # type: ignore[method-assign]

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
            output_dir=self.experiment_dir,
        )
