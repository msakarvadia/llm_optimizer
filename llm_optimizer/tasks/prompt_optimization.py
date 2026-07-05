"""Optimize prompts for lm-eval harness evaluations."""

from __future__ import annotations

import os
import random
from typing import Any

import lm_eval
from lm_eval.tasks import TaskManager

from llm_optimizer.tasks.base_task import Task


class PromptOptimization(Task):
    """Prompt optimization 'task'."""

    def __init__(
        self,
        metric: str = 'f1',
        direction: str = 'maximize',
        **kwargs: Any,
    ) -> None:
        """Initialize task."""
        # for coding benchmarks
        os.environ['HF_ALLOW_CODE_EVAL'] = '1'
        # task_metric_dict = {'drop': {"metric_name":'f1',"filter_name":"none"}
        #'longbench_hotpotqa':{"metric_name":'qa_f1_score',"filter_name":""}
        #'gsm8k':{"metric_name":"exact_match","filter_name":'flexible-extract'},
        #'mbpp':'pass_at_1',
        #'humaneval':'pass@1'}
        task_metric_dict = {
            'drop': {
                'metric_name': 'f1',
                'filter_name': 'none',
            },
            'longbench_hotpotqa': {
                'metric_name': 'qa_f1_score',
                'filter_name': 'none',
            },
            'gsm8k': {
                'metric_name': 'exact_match',
                'filter_name': 'flexible-extract',
            },
            'mbpp': {
                'metric_name': 'pass_at_1',
                'filter_name': 'none',
            },
            'humaneval': {
                'metric_name': 'pass@1',
                'filter_name': 'create_test',
            },
        }
        self.benchmark = kwargs['benchmark']
        if self.benchmark not in task_metric_dict:
            raise KeyError(
                f'{self.benchmark} not supported yet. '
                'consider adding it to the task',
            )
        self.metric = task_metric_dict[self.benchmark]['metric_name']
        self.filter = task_metric_dict[self.benchmark]['filter_name']
        self.model_name = kwargs['model_name']
        self.task_description = (
            f'Craft a prompt prefix to boost a downstream '
            f"{self.model_name}'s score on the {self.benchmark} benchmark."
        )

        self.solution_description = 'prompt prefix'
        self.direction = direction
        self.seed_candidate = 'placeholder prompt prefix'
        self.eval_model_gpu_id = kwargs['eval_model_gpu_id']
        self.seed = kwargs['seed']

        self.device = 'cpu'
        if self.eval_model_gpu_id != 'cpu':
            self.device = f'cuda:{self.eval_model_gpu_id}'

        self.set_train_test_splits()

    def set_train_test_splits(self) -> None:
        """Initializes indices for train/test splits."""
        task_manager = TaskManager()
        task_dict = task_manager.load_task_or_group([self.benchmark])
        task_obj = task_dict[self.benchmark]
        task_obj.download()
        if task_obj.has_test_docs():
            total_instances = len(list(task_obj.test_docs()))
        elif task_obj.has_validation_docs():
            total_instances = len(list(task_obj.validation_docs()))
        elif hasattr(task_obj, 'config') and 'num_samples' in task_obj.config:
            total_instances = task_obj.config['num_samples']
        else:
            raise ValueError(
                f'Could not determine the dataset split'
                f' size for: {self.benchmark}',
            )
        random.seed(self.seed)

        # 10% test indices
        sample_count_10pct = max(1, int(total_instances * 0.10))
        test_indices = random.sample(
            range(total_instances),
            sample_count_10pct,
        )

        all_indices_set = set(range(total_instances))
        self.test_indices_set = set(test_indices)

        train_indices_set = all_indices_set - self.test_indices_set
        self.train_indices = list(train_indices_set)

    def evaluate(self, solution: str) -> tuple[float, dict[str, Any]]:
        """Evaluate LLM optimized solution."""
        # Pass the native 'drop' task directly
        chosen_indices = self.train_indices
        raw_results = lm_eval.simple_evaluate(
            model='hf',
            model_args=(
                f'pretrained={self.model_name},'
                f'trust_remote_code=True,'
                f'attn_implementation=sdpa'
            ),
            tasks=[
                self.benchmark,
            ],  # Use the official registered dataset string
            num_fewshot=0,  # 0-shot optimizes speed significantly
            batch_size='auto',
            device=self.device,
            # limit=50,  # Artificially limit prompts for quick testing
            system_instruction=solution,  # Forces prefix before generation
            apply_chat_template=True,  # Properly wraps the prompt
            confirm_run_unsafe_code=True,  # for CODING evaluation
            samples={self.benchmark: chosen_indices},
        )

        print(lm_eval.utils.make_table(raw_results))

        metrics = raw_results['results'][self.benchmark]
        score = metrics[f'{self.metric},{self.filter}']

        print(f'{self.metric} Score:        {score:.4f}')

        # NOTE(MS): this won't have meta-data
        return score, {}
