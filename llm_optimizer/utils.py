"""General purpose utility functions for experiments."""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import time

import numpy as np
import requests
from openai import OpenAI
from sentence_transformers import SentenceTransformer


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


def prompt_lm(
    client: OpenAI,
    prompt: str,
    model_name: str = 'gemini-3.5-flash',
) -> str:
    """Standard LLM api inference call."""
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
    )
    raw_output = response.choices[0].message.content
    if raw_output is not None:
        return raw_output
    else:
        raise ValueError(
            'LM api call returned None instead of a valid string.',
        )


def start_vllm_server(model_name: str) -> subprocess.Popen[str] | None:
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
        '8000',
        '--enable-prefix-caching',
    ]

    print('Launching vLLM Server...')

    # Capture environment variables from your active shell session
    env_context = os.environ.copy()
    print(f"{env_context['HF_TOKEN']=}")

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
            response = requests.get('http://localhost:8000/health', timeout=2)
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
