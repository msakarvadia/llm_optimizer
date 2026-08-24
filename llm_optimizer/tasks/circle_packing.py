"""Abstraction to define a task (for which a solution will be produced)."""

# code adapted from:
# https://github.com/algorithmicsuperintelligence/openevolve/tree/main/examples/circle_packing
# https://github.com/gepa-ai/gepa/blob/main/examples/circle_packing/utils.py

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any

import numpy as np

from llm_optimizer.tasks.base_task import Task
from llm_optimizer.utils import extract_python_code


class CirclePacking(Task):
    """Circle Packing 'task'."""

    def __init__(
        self,
        metric: str = 'sum of all circle radii',
        direction: str = 'maximize',
        **kwargs: Any,
    ) -> None:
        """Initialize task."""
        self.task_description = """The circle packing problem involves placing n non-overlapping circles inside a container (in this case, a unit square) to optimize a specific metric. For this example:

We pack exactly 26 circles
Each circle must lie entirely within the unit square
No circles may overlap
We aim to maximize the sum of all circle radii.

Construct a specific arrangement of 26 circles in a unit square that attempts to maximize the sum of their radii.

Write a python function called 'run_packing' which acchomplishes this goal and     Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii

Output the executable Python code to accomplish the task inside a markdown code block."""
        self.solution_description = '26 circle packing configuration'
        self.metric = metric
        self.direction = direction
        self.seed_candidate = SEED_CANDIDATE

    def evaluate(self, solution: str) -> tuple[float, dict[str, Any], None]:
        """Evaluate the program by running it once and checking the sum of radii.

        Args:
            solution: Candidate program text to evaluate.

        Returns:
            Dictionary of metrics
        """
        # https://github.com/algorithmicsuperintelligence/openevolve/blob/main/examples/circle_packing/evaluator.py
        # LLMs are inconsistent about wrapping code in ```python ... ```
        # fences even when asked to (see task_description above); strip
        # them here rather than writing the raw completion (fences and
        # all) straight to a .py file, which would just SyntaxError in
        # the subprocess below.
        code = extract_python_code(solution)
        if code is None:
            print('Could not extract valid Python code from solution')
            return (
                0.0,
                {
                    'sum_radii': 0.0,
                    'target_ratio': 0.0,
                    'validity': 0.0,
                    'eval_time': 0.0,
                    'combined_score': 0.0,
                },
                None,
            )

        # Use a unique tempfile (not a fixed cwd-relative path) so concurrent
        # evaluations can't clobber each other's solution file.
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
            encoding='utf-8',
        ) as file:
            file.write(code)
            program_path = file.name

        # Target value from the paper
        TARGET_VALUE = 2.635  # AlphaEvolve result for n=26

        try:
            # For constructor-based approaches, a single evaluation is sufficient
            # since the result is deterministic
            start_time = time.time()

            # Use subprocess to run with timeout
            centers, radii, reported_sum = run_with_timeout(
                program_path,
                timeout_seconds=600,  # Single timeout
            )

            end_time = time.time()
            eval_time = end_time - start_time

            # Ensure centers and radii are numpy arrays
            if not isinstance(centers, np.ndarray):
                centers = np.array(centers)
            if not isinstance(radii, np.ndarray):
                radii = np.array(radii)

            # Check for NaN values before validation
            if np.isnan(centers).any() or np.isnan(radii).any():
                print('NaN values detected in solution')
                return (
                    0.0,
                    {
                        'sum_radii': 0.0,
                        'target_ratio': 0.0,
                        'validity': 0.0,
                        'eval_time': float(time.time() - start_time),
                        'combined_score': 0.0,
                    },
                    None,
                )

            # Validate solution
            valid = validate_packing(centers, radii)

            # Check shape and size
            shape_valid = centers.shape == (26, 2) and radii.shape == (26,)
            if not shape_valid:
                print(
                    f'Invalid shapes: centers={centers.shape}, radii={radii.shape}, expected (26, 2) and (26,)',
                )
                valid = False

            # Calculate sum
            sum_radii = np.sum(radii) if valid else 0.0

            # Make sure reported_sum matches the calculated sum
            if abs(sum_radii - reported_sum) > 1e-6:
                print(
                    f"Warning: Reported sum {reported_sum} doesn't match calculated sum {sum_radii}",
                )

            # Target ratio (how close we are to the target)
            target_ratio = sum_radii / TARGET_VALUE if valid else 0.0

            # Validity score
            validity = 1.0 if valid else 0.0

            # Combined score - higher is better
            combined_score = target_ratio * validity

            print(
                f'Evaluation: valid={valid}, sum_radii={sum_radii:.6f}, target={TARGET_VALUE}, ratio={target_ratio:.6f}, time={eval_time:.2f}s',
            )

            # Primary score matches self.metric ('sum of all circle radii').
            # `combined_score` (== target_ratio here, since validity is
            # already baked into sum_radii being forced to 0.0 above) is
            # reported in the metrics dict for visibility only and is not
            # itself returned as the score.
            return (
                float(sum_radii),
                {
                    'sum_radii': float(sum_radii),
                    'target_ratio': float(target_ratio),
                    'validity': float(validity),
                    'eval_time': float(eval_time),
                    'combined_score': float(combined_score),
                },
                None,
            )

        except Exception as e:
            print(f'Evaluation failed completely: {e!s}')
            traceback.print_exc()
            return (
                0.0,
                {
                    'sum_radii': 0.0,
                    'target_ratio': 0.0,
                    'validity': 0.0,
                    'eval_time': 0.0,
                    'combined_score': 0.0,
                },
                None,
            )

        finally:
            # Clean up the tempfile written above, regardless of outcome.
            if os.path.exists(program_path):
                os.unlink(program_path)


SEED_CANDIDATE = '''import numpy as np


def run_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    # Initialize arrays for 26 circles
    n = 26
    centers = np.zeros((n, 2))

    # Place circles in a structured pattern
    # This is a simple pattern - evolution will improve this

    # First, place a large circle in the center
    centers[0] = [0.5, 0.5]

    # Place 8 circles around it in a ring
    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers[i + 1] = [0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)]

    # Place 16 more circles in an outer ring
    for i in range(16):
        angle = 2 * np.pi * i / 16
        centers[i + 9] = [0.5 + 0.7 * np.cos(angle), 0.5 + 0.7 * np.sin(angle)]

    # Additional positioning adjustment to make sure all circles
    # are inside the square and don't overlap
    # Clip to ensure everything is inside the unit square
    centers = np.clip(centers, 0.01, 0.99)

    # Compute maximum valid radii for this configuration
    radii = compute_max_radii(centers)

    # Calculate the sum of radii
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.ones(n)

    # First, limit by distance to square borders
    for i in range(n):
        x, y = centers[i]
        # Distance to borders
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Then, limit by distance to other circles
    # Each pair of circles with centers at distance d can have
    # sum of radii at most d to avoid overlap
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))

            # If current radii would cause overlap
            if radii[i] + radii[j] > dist:
                # Scale both radii proportionally
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return radii
'''


class TimeoutError(Exception):
    """Raised when a candidate program's subprocess exceeds its timeout."""


def timeout_handler(signum: int, frame: Any) -> None:
    """Handle timeout signal."""
    raise TimeoutError('Function execution timed out')


def validate_packing(
    centers: np.ndarray[Any, Any],
    radii: np.ndarray[Any, Any],
) -> bool:
    """Validate that circles don't overlap and are inside the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle

    Returns:
        True if valid, False otherwise
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        print('NaN values detected in circle centers')
        return False

    if np.isnan(radii).any():
        print('NaN values detected in circle radii')
        return False

    # Check if radii are nonnegative and not nan
    for i in range(n):
        if radii[i] < 0:
            print(f'Circle {i} has negative radius {radii[i]}')
            return False
        elif np.isnan(radii[i]):
            print(f'Circle {i} has nan radius')
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if (
            x - r < -1e-6
            or x + r > 1 + 1e-6
            or y - r < -1e-6
            or y + r > 1 + 1e-6
        ):
            print(
                f'Circle {i} at ({x}, {y}) with radius {r} is outside the unit square',
            )
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if (
                dist < radii[i] + radii[j] - 1e-6
            ):  # Allow for tiny numerical errors
                print(
                    f'Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i] + radii[j]}',
                )
                return False

    return True


def run_with_timeout(
    program_path: str,
    timeout_seconds: int = 20,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], float]:
    """Run the program in a separate process with timeout.

    Uses a simple subprocess approach.

    Args:
        program_path: Path to the program file
        timeout_seconds: Maximum execution time in seconds

    Returns:
        centers, radii, sum_radii tuple from the program
    """
    # Create a temporary file to execute
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
        # Write a script that executes the program and saves results
        script = f"""
import sys
import numpy as np
import os
import pickle
import traceback

# Add the directory to sys.path
sys.path.insert(0, os.path.dirname('{program_path}'))

# Debugging info
print(f"Running in subprocess, Python version: {{sys.version}}")
print(f"Program path: {program_path}")

try:
    # Import the program
    spec = __import__('importlib.util').util.spec_from_file_location("program", '{program_path}')
    program = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(program)

    # Run the packing function
    print("Calling run_packing()...")
    centers, radii, sum_radii = program.run_packing()
    print(f"run_packing() returned successfully: sum_radii = {{sum_radii}}")

    # Save results to a file
    results = {{
        'centers': centers,
        'radii': radii,
        'sum_radii': sum_radii
    }}

    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {temp_file.name}.results")

except Exception as e:
    # If an error occurs, save the error instead
    print(f"Error in subprocess: {{str(e)}}")
    traceback.print_exc()
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{'error': str(e)}}, f)
    print(f"Error saved to {temp_file.name}.results")
"""
        temp_file.write(script.encode())
        temp_file_path = temp_file.name

    results_path = f'{temp_file_path}.results'

    try:
        # Run the script with timeout
        process = subprocess.Popen(
            [sys.executable, temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode

            # Always print output for debugging purposes
            print(f'Subprocess stdout: {stdout.decode()}')
            if stderr:
                print(f'Subprocess stderr: {stderr.decode()}')

            # Still raise an error for non-zero exit codes, but only after printing the output
            if exit_code != 0:
                raise RuntimeError(f'Process exited with code {exit_code}')

            # Load the results
            if os.path.exists(results_path):
                with open(results_path, 'rb') as f:
                    results = pickle.load(f)

                # Check if an error was returned
                if 'error' in results:
                    raise RuntimeError(
                        f'Program execution failed: {results["error"]}',
                    )

                return (
                    results['centers'],
                    results['radii'],
                    results['sum_radii'],
                )
            else:
                raise RuntimeError('Results file not found')

        except subprocess.TimeoutExpired:
            # Kill the process if it times out
            process.kill()
            process.wait()
            raise TimeoutError(
                f'Process timed out after {timeout_seconds} seconds',
            )

    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(results_path):
            os.unlink(results_path)
