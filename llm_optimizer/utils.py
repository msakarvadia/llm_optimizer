"""General purpose utility functions for experiments."""

from __future__ import annotations

import ast
import atexit
import os
import re
import signal
import subprocess
import time
from typing import Any

import numpy as np
import requests
from openai import APIConnectionError
from openai import APIStatusError
from openai import OpenAI
from sentence_transformers import SentenceTransformer


def extract_python_code(output_string: str | None) -> str | None:
    """Extract a single executable Python program from raw LLM output.

    Prefers the first ```...``` fenced block (stripping a leading language
    tag like "python"), falling back to the raw output if no fence is
    present -- LLMs are inconsistent about wrapping code in markdown even
    when asked to, so callers shouldn't assume either convention. Either
    path must parse as valid Python or this returns None; a fenced block is
    not on its own proof that a model put Python inside it.
    """
    if output_string is None:
        return None

    trimmed = output_string.strip()

    code_match = re.search(r'```(.*?)```', trimmed, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
        if code.startswith('python'):
            code = code[len('python') :].strip()
        try:
            ast.parse(code)
            return code
        except SyntaxError:
            pass  # fenced block wasn't valid Python; fall through to raw

    try:
        ast.parse(trimmed)
        return trimmed
    except SyntaxError:
        return None


def semantic_similarity(strings: list[str]) -> dict[str, float]:
    """Report group statistics about semantic similarity.

    return: dict of mean/std_dev/min/max

    usage:

    list_of_strs = ["I am a cat", "I am a dog", "I am a rhino"]
    semantic_similarity(list_of_strs)
    """
    # Handle edge case for fewer than 2
    if len(strings) < 2:  # noqa
        return {'mean': 0.0, 'std_dev': 0.0, 'min': 0.0, 'max': 0.0}

    # embed strs
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(strings, convert_to_numpy=True)

    # normalize embeddings
    # (Note: sentence-transformers models typically output normalized vectors,
    # but explicit normalization ensures safety for any model output)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / np.where(norms == 0, 1, norms)

    # compute pairwise semantic sim and exclude diagonal
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

    # Create a boolean mask to filter out the diagonal (self-similarity)
    mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
    pairwise_similarities = similarity_matrix[mask]

    # compute group stats: mean/std dev/min/max
    semantic_sim_stats: dict[str, float] = {
        'mean': float(np.mean(pairwise_similarities)),
        'std_dev': float(np.std(pairwise_similarities)),
        'min': float(np.min(pairwise_similarities)),
        'max': float(np.max(pairwise_similarities)),
    }

    return semantic_sim_stats


def _extract_token_usage(response: Any) -> dict[str, int]:
    """Pull token counts off a chat completion response.

    Not every OpenAI-compatible backend populates every field: local vLLM
    and some proxied backends may omit `usage` entirely. Missing values
    default to 0 rather than None so callers can sum them unconditionally.

    reasoning_tokens is handled in two ways depending on the backend:
    - OpenAI o-series / gpt-oss: reasoning_tokens is reported directly via
      `completion_tokens_details.reasoning_tokens`. There it's a labeled
      *subset* of completion_tokens, and total_tokens == prompt_tokens +
      completion_tokens exactly
    - Gemini (via its OpenAI-compat endpoint): `completion_tokens_details`
      is always null, even though Gemini models think/reason internally
      by default and bill those tokens into total_tokens. reasoning tokens =
      total_tokens - minus prompt_tokens minus completion_tokens.
    """
    usage = getattr(response, 'usage', None)
    if usage is None:
        print('prompt_lm: response had no `usage` field, defaulting to 0s.')
        return {
            'input_tokens': 0,
            'output_tokens': 0,
            'reasoning_tokens': 0,
            'total_tokens': 0,
        }

    input_tokens = getattr(usage, 'prompt_tokens', 0) or 0
    output_tokens = getattr(usage, 'completion_tokens', 0) or 0
    total_tokens = getattr(usage, 'total_tokens', 0) or 0

    details = getattr(usage, 'completion_tokens_details', None)
    explicit_reasoning = (
        getattr(details, 'reasoning_tokens', None) if details else None
    )
    if explicit_reasoning is not None:
        reasoning_tokens = explicit_reasoning
    else:
        # Best-effort recovery of hidden thinking tokens (e.g. Gemini);
        # 0 if the backend genuinely doesn't do token-metered reasoning.
        reasoning_tokens = max(0, total_tokens - input_tokens - output_tokens)

    return {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'reasoning_tokens': reasoning_tokens,
        'total_tokens': total_tokens,
    }


def prompt_lm(
    client: OpenAI,
    prompt: str,
    model_name: str = 'gemini-3.5-flash',
    max_tokens: int | None = None,
    max_retries: int = 3,
) -> tuple[str, dict[str, int]]:
    """Standard LLM api inference call.

    Retries transient server/connection errors (e.g. 502s from a shared
    proxy under concurrent load) with exponential backoff before giving up.

    Returns (text, usage_dict), where usage_dict holds token counts
    (input_tokens, output_tokens, reasoning_tokens, total_tokens) pulled
    from the response. Usage is always extracted (it's already-present
    response metadata, not an extra call) -- callers that don't need it
    just discard the second element.
    """
    backoff_seconds = 2.0
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a helpful assistant.',
                    },
                    {
                        'role': 'user',
                        'content': prompt,
                    },
                ],
                max_tokens=max_tokens,
            )
        except (APIStatusError, APIConnectionError):
            if attempt == max_retries:
                raise
            sleep_time = backoff_seconds * (2**attempt)
            print(
                f'prompt_lm transient error on attempt {attempt + 1}/'
                f'{max_retries + 1}, retrying in {sleep_time:.1f}s...',
            )
            time.sleep(sleep_time)
            continue

        raw_output = response.choices[0].message.content
        if raw_output is not None:
            return raw_output, _extract_token_usage(response)
        raise ValueError(
            'LM api call returned None instead of a valid string.',
        )

    raise RuntimeError('prompt_lm: exhausted retries without returning')


def resolve_visible_device(gpu_id: int) -> str:
    """Map a local GPU index to its physical CUDA_VISIBLE_DEVICES value.

    gpu_id is a LOCAL index into the devices already visible to this
    process (e.g. 0/1), not an absolute physical GPU id. Re-slice the
    parent's own CUDA_VISIBLE_DEVICES instead of returning gpu_id
    unchanged, otherwise callers can get pinned to a physical GPU
    outside this job's actual allocation whenever CUDA_VISIBLE_DEVICES
    isn't "0,1".
    """
    parent_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if parent_visible:
        return parent_visible.split(',')[gpu_id]
    return str(gpu_id)


def start_vllm_server(
    model_name: str,
    port: int = 8000,
    gpu_id: int = 0,
) -> subprocess.Popen[str] | None:
    """Start a vllm server for HF model."""
    cmd = [
        'python',
        '-m',
        'vllm.entrypoints.openai.api_server',
        '--model',
        model_name,
        '--host',
        '0.0.0.0',
        '--port',
        str(port),
        '--enable-prefix-caching',
        '--gpu-memory-utilization',
        '0.80',
        # if you want all INFO prints, disable flag:
        '--uvicorn-log-level',
        'warning',
        # scilences throughput/cache metrics
        '--disable-log-stats',
    ]

    print('Launching vLLM Server...')

    # Capture environment variables from your active shell session
    env_context = os.environ.copy()

    env_context['CUDA_VISIBLE_DEVICES'] = resolve_visible_device(gpu_id)

    # Tell Python HTTP engines to trust standard system root files
    # instead of local virtualenv variations that might be broken
    env_context['CURL_CA_BUNDLE'] = ''
    env_context['REQUESTS_CA_BUNDLE'] = ''

    process = subprocess.Popen(
        cmd,
        stdout=None,
        stderr=None,
        text=True,
        env=env_context,  # <-- Injects environmental data to background
    )

    print('Waiting for model to finish loading into GPU VRAM...')
    while True:
        try:
            response = requests.get(
                f'http://localhost:{port}/health',
                timeout=2,
            )
            if response.status_code == 200:  # noqa: PLR2004
                print('vLLM Server is up, healthy, and ready for queries!')
                break
        except requests.exceptions.ConnectionError:
            pass

        if process.poll() is not None:
            print(
                'vLLM process exited unexpectedly. '
                'Read the logs above for the error.',
            )
            return None

        time.sleep(3)

    def cleanup() -> None:
        """Forcefully kills the entire process group if it's still alive."""
        try:
            if process.poll() is None:
                print('\nCleaning up process tree...')
                # Kill the entire group ID matching the process PID
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass  # Process was already closed cleanly

    # This will catch regular exits, uncaught exceptions,
    # AND KeyboardInterrupts (Ctrl+C)
    atexit.register(cleanup)

    return process
