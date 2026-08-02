"""Optimize cloud scheduling strategies for multi-cloud data transfer.

Code reference:
https://github.com/gepa-ai/gepa/tree/main/examples/adrs/cloudcast
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from llm_optimizer.tasks.base_task import Task
from llm_optimizer.tasks.cloudcast_utils.dataset import load_config_dataset
from llm_optimizer.tasks.cloudcast_utils.simulation import (
    evaluation_failure_info,
)
from llm_optimizer.tasks.cloudcast_utils.simulation import (
    evaluation_success_info,
)
from llm_optimizer.tasks.cloudcast_utils.simulation import FAILED_SCORE
from llm_optimizer.tasks.cloudcast_utils.simulation import get_program_path
from llm_optimizer.tasks.cloudcast_utils.simulation import run_evaluation
from llm_optimizer.tasks.cloudcast_utils.simulation import syntax_failure_info
from llm_optimizer.tasks.cloudcast_utils.simulation import syntax_is_valid


class CloudCast(Task):
    """Could scheduling algo 'task'."""

    def __init__(
        self,
        metric: str = 'value_cost_ratio',
        direction: str = 'maximize',
        **kwargs: Any,
    ) -> None:
        """Initialize task."""
        self.task_description = f"""{OPTIMIZATION_OBJECTIVE}\n
        {OPTIMIZATION_BACKGROUND}"""

        self.failed_score = FAILED_SCORE
        self.seed = kwargs['seed']
        random.seed(self.seed)
        self.solution_description = 'broadcast routing algorithm'
        self.metric = metric
        self.direction = direction
        self.seed_candidate = INITIAL_PROGRAM

        # TODO(MS): load dataset
        dataset_root = (
            Path(__file__).resolve().parent
            / 'cloudcast_utils'
            / 'cloudcast'
            / 'config'
        )
        dataset = load_config_dataset(config_dir=dataset_root)

        print(
            f'Loaded {len(dataset)} configuration samples (used for train, val, and test)',
        )
        # TODO(MS): split dataset into train/val
        self.train_set = dataset
        # only 5 datapoints, grab 1 for validation set
        random_index = random.randrange(len(self.train_set))
        self.test_set = [self.train_set.pop(random_index)]
        print(self.train_set)
        print(self.test_set)

    def evaluate_set(
        self,
        program_path: str,
        eval_set: list[dict[str, Any]],
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate algorithm."""
        score = 0.0
        cost = 0.0
        transfer_time = 0.0
        example = eval_set[0]
        details = {'placeholder': 0}
        for example in eval_set:
            if not syntax_is_valid(program_path):
                return FAILED_SCORE, syntax_failure_info(example)

            success, cost, transfer_time, error, details = run_evaluation(
                program_path,
                example['config_file'],
                example['num_vms'],
            )

            if not success:
                return FAILED_SCORE, evaluation_failure_info(error, example)

            score += 1.0 / (1.0 + cost)
        return score / len(self.train_set), evaluation_success_info(
            score,
            cost,
            transfer_time,
            example,
            details,
        )

    def evaluate(self, solution: str) -> tuple[float, dict[str, Any], float]:
        """Evaluate algorithm."""
        # print(solution)
        program_path = get_program_path(solution)
        score, meta_data = self.evaluate_set(program_path, self.train_set)
        val_score, _ = self.evaluate_set(program_path, self.test_set)

        return score, meta_data, val_score


INITIAL_PROGRAM = """import networkx as nx
import pandas as pd
import os
from typing import Dict, List


class SingleDstPath(Dict):
    partition: int
    edges: List[List]  # [[src, dst, edge data]]


class BroadCastTopology:
    def __init__(self, src: str, dsts: List[str], num_partitions: int = 4, paths: Dict[str, 'SingleDstPath'] = None):
        self.src = src
        self.dsts = dsts
        self.num_partitions = num_partitions
        if paths is not None:
            self.paths = paths
        else:
            self.paths = {dst: {str(i): None for i in range(num_partitions)} for dst in dsts}

    def get_paths(self):
        return self.paths

    def set_num_partitions(self, num_partitions: int):
        self.num_partitions = num_partitions

    def set_dst_partition_paths(self, dst: str, partition: int, paths: List[List]):
        partition = str(partition)
        self.paths[dst][partition] = paths

    def append_dst_partition_path(self, dst: str, partition: int, path: List):
        partition = str(partition)
        if self.paths[dst][partition] is None:
            self.paths[dst][partition] = []
        self.paths[dst][partition].append(path)


def search_algorithm(src, dsts, G, num_partitions):
    \"\"\"
    Find broadcast paths from source to all destinations.

    Uses Dijkstra's shortest path algorithm based on cost as the edge weight.

    Args:
        src: Source node identifier (e.g., "aws:ap-northeast-1")
        dsts: List of destination node identifiers
        G: NetworkX DiGraph with cost and throughput edge attributes
        num_partitions: Number of data partitions

    Returns:
        BroadCastTopology object with paths for all destinations and partitions
    \"\"\"
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    for dst in dsts:
        path = nx.dijkstra_path(h, src, dst, weight="cost")
        for i in range(0, len(path) - 1):
            s, t = path[i], path[i + 1]
            for j in range(bc_topology.num_partitions):
                bc_topology.append_dst_partition_path(dst, j, [s, t, G[s][t]])

    return bc_topology
"""

OPTIMIZATION_OBJECTIVE = """Optimize a broadcast routing algorithm for multi-cloud data transfer.

The algorithm decides how to route data from a single source to multiple destinations
across cloud providers (AWS, GCP, Azure). The goal is to minimize total cost
(egress fees + instance costs) while maintaining good transfer times.
Output the executable Python code to accomplish the task inside a markdown code block."""

OPTIMIZATION_BACKGROUND = """Key information about the problem domain:

- The network is represented as a directed graph where:
  - Nodes are cloud regions (e.g., "aws:us-east-1", "gcp:europe-west1-a", "azure:eastus")
  - Edges have 'cost' ($/GB for egress) and 'throughput' (Gbps bandwidth) attributes

- Data is partitioned into num_partitions chunks that can be routed independently
- Each partition can take a different path to reach each destination
- Total cost = egress costs (data_vol × edge_cost) + instance costs (runtime × cost_per_hour)

- The algorithm must return a BroadCastTopology object containing:
  - paths[dst][partition] = list of edges [[src, dst, edge_data], ...]
  - Each destination must have at least one valid path for each partition

Evaluation feedback format:
- Cost: Total transfer cost in dollars
- Transfer time: Maximum time for all destinations to receive data (seconds)

Optimization targets:
1. Reduce total cost (egress + instance costs)
2. Find paths that balance cost and throughput
3. Consider multipath routing for better bandwidth utilization
4. Exploit cloud provider pricing differences (e.g., intra-provider is cheaper)"""

# if __name__ == '__main__':
#    optimization_task = CloudCastOptimization()
#    score, info = optimization_task.evaluate(optimization_task.seed_candidate)
#    print(score)
#    print(info)
