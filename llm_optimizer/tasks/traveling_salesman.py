"""Define Traveling Salesman Task."""

from __future__ import annotations

from typing import Any

import numpy as np
from ortools.sat.python import cp_model

from llm_optimizer.tasks.base_task import Task


class TravelingSalesman(Task):
    """Optimize TSP route."""

    def __init__(
        self,
        metric: str = 'negative length',
        direction: str = 'maximize',
        **kwargs: Any,
    ) -> None:
        """Initialize task."""
        # Finite sentinel (matches cloudcast/cant_be_late's FAILED_SCORE
        # convention) instead of -math.inf: the 'wheel' sampling strategy
        # in opro.py shifts scores by -min_score, and -inf + inf == NaN
        # crashes np.random.choice once an unparseable trace enters the
        # population. -1_000_000 (not -100_000) since TSP distances can
        # legitimately sum into the tens of thousands, so -100_000 isn't
        # safely out of range of a genuine score.
        self.failed_score = -1_000_000.0
        rng = np.random.default_rng(seed=0)

        self.num_points = kwargs['num_points']
        self.point_list = range(self.num_points)
        self.num_decimals = kwargs['num_decimals']
        x = rng.uniform(low=-100, high=100, size=self.num_points)
        y = rng.uniform(low=-100, high=100, size=self.num_points)
        self.x = [
            np.round(xi, self.num_decimals)
            if self.num_decimals > 0
            else int(xi)
            for xi in x
        ]
        self.y = [
            np.round(yi, self.num_decimals)
            if self.num_decimals > 0
            else int(yi)
            for yi in y
        ]

        self.task_description = (
            'You are given a list of points with coordinates below:\n'
        )
        for i, (xi, yi) in enumerate(zip(self.x, self.y, strict=False)):
            if i:
                self.task_description += ', '
            self.task_description += f'({i}): ({xi}, {yi})'
        self.task_description += (
            'Below are some previous traces and '
            'their scores. Score is the negative of the trace length, so a '
            'higher (less negative) score means a shorter, better trace. '
            'The trace should traverse all points exactly once. '
            'The trace should start with <trace> and end with </trace>.'
        )

        self.solution_description = 'trace'
        self.metric = metric
        self.direction = direction
        self.gt_sol, _min_dis = solve_tsp(
            self.x,
            self.y,
            self.num_points,
            self.num_decimals,
            'ortools_exact',
        )
        print(f'Minimum distance of solution: {_min_dis}')
        self.gt_sol_str = ','.join([str(i) for i in self.gt_sol])

        init_sol = None
        while not init_sol:
            sol = rng.permutation(self.point_list)
            sol_str = ','.join([str(i) for i in sol])
            if self.gt_sol_str != sol_str:
                init_sol = sol_str

        self.seed_candidate = f'<trace>{init_sol}</trace>'

    def evaluate(self, solution: str) -> tuple[float, dict[str, Any]]:
        """Evaluate the generated trace."""
        try:
            parsed_output = extract_string(solution)
            validate_trace(parsed_output, self.num_points)
            distance = evaluate_distance(
                self.x,
                self.y,
                parsed_output,
                self.num_decimals,
            )
            # Score is -distance so that shorter (better) tours rank higher,
            # matching the optimizer's always-maximize selection/pruning logic.
            score = -distance
            error_dict = {}
        except Exception as error:
            print('THERE IS AN ERROR PARSING RESPONSE')
            score = self.failed_score
            error_dict = {'error': str(error)}

        return score, error_dict, None


def evaluate_distance(
    x: list[float],
    y: list[float],
    trace: list[int],
    num_decimals: int,
) -> float:  # pylint: disable=invalid-name
    """Evaluate distance of path.

    #https://github.com/google-deepmind/opro/blob/main/opro/optimization/optimize_tsp.py#L172 # noqa
    """
    dis = 0.0
    for i in range(len(trace) - 1):
        id0 = trace[i]
        id1 = trace[i + 1]
        dis += np.sqrt((x[id0] - x[id1]) ** 2 + (y[id0] - y[id1]) ** 2)
    id0 = trace[-1]
    id1 = trace[0]
    dis += np.sqrt((x[id0] - x[id1]) ** 2 + (y[id0] - y[id1]) ** 2)
    dis = np.round(dis, num_decimals) if num_decimals > 0 else int(dis)
    return dis


def validate_trace(trace: list[int], num_points: int) -> None:
    """Validate that trace visits every city exactly once.

    Without this check, a trace that omits cities (fewer edges) can score
    *better* than a correct full tour, since evaluate_distance only sums
    consecutive-point distances over whatever list it is given -- silently
    rewarding degenerate/incomplete solutions over valid ones.
    """
    if len(trace) != num_points:
        raise ValueError(
            f'Trace visits {len(trace)} cities, expected {num_points} '
            '(missing and/or duplicate cities).',
        )
    seen = set(trace)
    if len(seen) != num_points:
        raise ValueError('Trace contains duplicate cities.')
    if seen != set(range(num_points)):
        raise ValueError('Trace contains out-of-range city indices.')


def extract_string(input_string: str) -> list[int]:
    """Extract produced soluiton.

    # https://github.com/google-deepmind/opro/blob/main/opro/optimization/optimize_tsp.py#L172C1-L185C15 # noqa
    """
    start_string = '<trace>'
    end_string = '</trace>'
    if start_string not in input_string:
        raise ValueError('<trace> not at beginning of input.')
    input_string = input_string[
        input_string.index(start_string) + len(start_string) :
    ]
    if end_string not in input_string:
        raise ValueError('</trace> not at ending of input.')
    input_string = input_string[: input_string.index(end_string)]
    parsed_list = []
    for p in input_string.split(','):
        p_str = p.strip()
        try:
            p_int: int = int(p_str)
        except ValueError:
            continue
        parsed_list.append(p_int)
    return parsed_list


def solve_tsp(
    x: list[float],
    y: list[float],
    num_points: int,
    num_decimals: int,
    starting_algorithm: str,
) -> tuple[list[int], float]:
    """3 different tsp solving algos.

    # https://github.com/google-deepmind/opro/blob/main/opro/optimization/optimize_tsp.py#L187C3-L252C27 # noqa
    """
    if starting_algorithm == 'nearest_neighbor':
        min_dis = 0.0
        gt_sol: list[int] = [0]
        remaining_points = list(range(1, num_points))
        while len(remaining_points) > 0:
            min_p = -1
            min_cur_dis = -1.0
            for p in remaining_points:
                cur_dis = np.sqrt(
                    (x[p] - x[gt_sol[-1]]) ** 2 + (y[p] - y[gt_sol[-1]]) ** 2,
                )
                if min_p == -1 or cur_dis < min_cur_dis:
                    min_p = p
                    min_cur_dis = cur_dis
            gt_sol.append(min_p)
            min_dis += min_cur_dis
            remaining_points.remove(min_p)
        min_dis += np.sqrt(
            (x[0] - x[gt_sol[-1]]) ** 2 + (y[0] - y[gt_sol[-1]]) ** 2,
        )
        min_dis = (
            np.round(min_dis, num_decimals)
            if num_decimals > 0
            else int(min_dis)
        )
        return gt_sol, min_dis
    elif starting_algorithm == 'farthest_insertion':
        gt_sol = [0]
        remaining_points = list(range(1, num_points))
        while len(remaining_points) > 0:
            max_p = -1
            max_cur_dis = -1.0
            max_cur_index = -1
            for p in remaining_points:
                min_cur_dis = -1.0
                min_cur_index = -1
                for index in range(1, len(gt_sol) + 1):
                    new_sol = gt_sol[:index] + [p] + gt_sol[index:]
                    cur_dis = evaluate_distance(x, y, new_sol, num_decimals)
                    if min_cur_dis == -1 or cur_dis < min_cur_dis:
                        min_cur_dis = cur_dis
                        min_cur_index = index
                if max_cur_dis == -1 or min_cur_dis > max_cur_dis:
                    max_p = p
                    max_cur_dis = min_cur_dis
                    max_cur_index = min_cur_index
            gt_sol = gt_sol[:max_cur_index] + [max_p] + gt_sol[max_cur_index:]
            remaining_points.remove(max_p)
        min_dis = evaluate_distance(x, y, gt_sol, num_decimals)
        return gt_sol, min_dis
    elif starting_algorithm == 'ortools_exact':
        scale = 10**num_decimals if num_decimals > 0 else 1
        dist = [
            [
                round(
                    scale
                    * float(
                        np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2),
                    ),
                )
                for j in range(num_points)
            ]
            for i in range(num_points)
        ]

        model = cp_model.CpModel()
        lits = {}
        arcs = []
        for i in range(num_points):
            for j in range(num_points):
                if i == j:
                    continue
                lit = model.NewBoolVar(f'x_{i}_{j}')
                lits[i, j] = lit
                arcs.append((i, j, lit))
        model.AddCircuit(arcs)
        model.Minimize(sum(dist[i][j] * lits[i, j] for i, j in lits))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 120
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(
                'ortools failed to find a TSP solution: '
                f'status={solver.StatusName(status)}',
            )

        gt_sol = [0]
        while len(gt_sol) < num_points:
            cur = gt_sol[-1]
            nxt = next(
                j
                for j in range(num_points)
                if j != cur and solver.Value(lits[cur, j])
            )
            gt_sol.append(nxt)
        min_dis = evaluate_distance(x, y, gt_sol, num_decimals)
        return gt_sol, min_dis

    f = {(0, 1): (0, [0])}
    q = [(0, 1)]
    min_dis = -1
    gt_sol = list(range(num_points))
    while len(q) > 0:
        p, status = q[0]
        q = q[1:]
        for i in range(num_points):
            if 2 << i >> 1 & status == 0:
                new_status = status + (2 << i >> 1)
                new_dis = f[(p, status)][0] + np.sqrt(
                    (x[i] - x[p]) ** 2 + (y[i] - y[p]) ** 2,
                )
                if (i, new_status) not in f or new_dis < f[(i, new_status)][0]:
                    f[(i, new_status)] = (new_dis, f[(p, status)][1] + [i])
                    if new_status == (2 << num_points >> 1) - 1:
                        new_dis += np.sqrt(
                            (x[i] - x[0]) ** 2 + (y[i] - y[0]) ** 2,
                        )
                        if min_dis == -1 or new_dis < min_dis:
                            min_dis = new_dis
                            gt_sol = f[(i, new_status)][1][:]
                    elif (i, new_status) not in q:
                        q.append((i, new_status))
    min_dis = (
        np.round(min_dis, num_decimals) if num_decimals > 0 else int(min_dis)
    )
    return gt_sol, min_dis
