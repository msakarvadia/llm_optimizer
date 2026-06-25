"""Logic to abstract away LLM-driven optimization."""

from __future__ import annotations

from typing import Any

from openevolve import run_evolution
from openevolve.config import Config
from openevolve.config import LLMModelConfig

from llm_optimizer.optimizers.base_optimizer import Optimizer
from llm_optimizer.tasks.base_task import Task

# --- GLOBAL REGISTRY FOR OPENEVOLVE ---
# This dictionary lives in the memory space of your application process.
# Even when openevolve runs your evaluator in a new context, it will import
# this module and pull the live object from this dictionary.
_ACTIVE_TASK_REGISTRY: dict[str, Any] = {}


def evaluator(solution_path: str) -> dict[str, float]:
    """Wrapper for task specific evaluator.

    hacky workaround bc openevolve save evaluatory to a file
    need to load task specific evaluator from application process memory
    use process id to make sure no interference between tasks
    """
    import re  # noqa
    from llm_optimizer.optimizers.open_evolve import _ACTIVE_TASK_REGISTRY

    with open(solution_path, encoding='utf-8') as f:
        solution_file_text = f.read()
    print('SOLUTION PATHvv' * 40)
    print(solution_path)
    print('SOLUTION PATH ^^' * 40)

    print('OPEN EVOLVE SOLUTION vv' * 40)
    print(solution_file_text)
    print('OPEN EVOLVE SOLUTION ^^' * 40)

    # Extract out the registry ID injected right inside the file comment
    # Look for a comment line structured like: # REGISTRY_ID: 140401824103120
    id_match = re.search(r'# REGISTRY_ID:\s*(\d+)', solution_file_text)

    # mypy check guard
    if id_match is None:
        return {'combined_score': 0.0}

    task_id = int(id_match.group(1))
    current_task = _ACTIVE_TASK_REGISTRY.get(str(task_id))

    # mypy check guard
    if current_task is None:
        return {'combined_score': 0.0}

    # Extract the content safely
    match = re.search(
        r'# EVOLVE-BLOCK-START\s*(.*?)\s*# EVOLVE-BLOCK-END',
        solution_file_text,
        re.DOTALL,
    )

    # mypy check guard
    if match is None:
        return {'combined_score': 0.0}
    solution = match.group(1)

    result, extra_info = current_task.evaluate(solution)
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

        # register the instances unique system pointer address
        instance_id = id(self)
        _ACTIVE_TASK_REGISTRY[str(instance_id)] = self.task

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
            initial_program=f"""
            # REGISTRY_ID: {instance_id}
            # EVOLVE-BLOCK-START
        {self.task.seed_candidate}
        # EVOLVE-BLOCK-END
        """,
            evaluator=evaluator,
            iterations=num_iter,
            config=self.config,
        )

        # TODO(MS): temporarily save solution bank
        # experiment_dir = 'temp_results'
        # self.solution_bank.save_to_json(experiment_dir)
