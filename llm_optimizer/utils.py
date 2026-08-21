"""General purpose utility functions for experiments."""

from __future__ import annotations

import ast
import atexit
import ipaddress
import os
import re
import signal
import subprocess
import time
from typing import Any
from urllib.parse import urlparse

import httpx
import numpy as np
import ray
import requests
import torch
from openai import APIConnectionError
from openai import APIStatusError
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


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


def is_local_url(base_url: str) -> bool:
    """Decide whether base_url points at a local/cluster-internal endpoint.

    Single source of truth for the local-vs-remote distinction used to
    pick proxy behavior (build_openai_client) and request concurrency
    (tasks/harm_bench.py). A vLLM server started on a Ray worker is
    reachable at the node's real cluster IP (e.g. 10.1.1.164), not
    literally 'localhost', so we resolve the hostname and check it
    against loopback/private-network ranges (RFC 1918) instead of
    string-matching a fixed set of hostnames.
    """
    host = urlparse(base_url).hostname or ''
    if not host:
        return False
    if host.endswith('alcf.anl.gov'):
        return True
    try:
        return ipaddress.ip_address(host).is_private
    except ValueError:
        # Not an IP literal (e.g. a public hostname like Google's API) --
        # 'localhost' resolves to loopback below; anything else is remote.
        return host == 'localhost'


def build_openai_client(api_key: str, base_url: str) -> OpenAI:
    """Build an OpenAI-compatible client for a given base_url.

    Shared by every LLM caller in this repo (mutators, llm_judge, ...) so
    the proxy-bypass behavior below stays in one place.

    NOTE(MS): work around global proxies specifically for compute nodes --
    local endpoints (vllm servers) need the cluster's system proxy env
    vars ignored entirely to be reachable, while external hosts (Google/
    ANL) need that same proxy respected to be reachable. is_local decides
    which.
    """
    if api_key is None:
        raise ValueError(
            'API key not found. Set the appropriate API key environment '
            'variable (see key_env_name in config.yaml for this model).',
        )

    is_local = is_local_url(base_url)
    print(f'DEBUG: base_url={base_url} | is_local={is_local}')

    if is_local:
        # FORCE bypass: Tell httpx to ignore ALL system
        # proxy variables entirely
        custom_http_client = httpx.Client(trust_env=False)
        print(
            '--> Local routing: '
            'Cluster environment proxy bypassed successfully.',
        )
    else:
        # FORCE use: Tell httpx to respect the system
        # proxy so it can reach ANL / Google
        custom_http_client = httpx.Client(trust_env=True)
        print(
            '--> Remote routing: '
            'Utilizing global cluster proxy for external connection.',
        )

    # TODO(MS): make generalizable to other base_urls
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=custom_http_client,  # WORK AROUND FOR GLOBAL PROXIES
    )


def prompt_lm(  # noqa: PLR0913
    client: OpenAI,
    prompt: str,
    model_name: str = 'gemini-3.5-flash',
    max_tokens: int | None = None,
    max_retries: int = 3,
    system_msg: str = 'You are a helpful assistant.',
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
                        'content': system_msg,
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


def get_shared_vllm_requirements(
    exp: dict[str, Any],
    config: dict[str, Any],
) -> list[tuple[str, str]]:
    """Return the (model_name, override_key) pairs `exp` needs a server for.

    Mirrors the vllm-routing gates already inline in main.py, so
    experiments.py can decide which models to share across a batch without
    re-deriving -- and risking drifting from -- those same gating rules. A
    model can appear more than once if its name is reused across roles
    (e.g. optimizer_llm == inference_model_name).
    """
    needed: list[tuple[str, str]] = []
    task_name = exp.get('task_name')

    # Role A: optimizer model -- always checked, every task_name resolves
    # one.
    optimizer_llm = exp.get('optimizer_llm', 'gemini-3.5-flash')
    if config[optimizer_llm]['key_env_name'] == 'vllm':
        needed.append((optimizer_llm, 'optimizer_base_url_override'))

    # Role B: inference/target model -- gating depends on task_name.
    inference_model_name = exp.get(
        'inference_model_name',
        'google/gemma-4-E4B-it',
    )
    if task_name == 'harmbench':
        # Role C: classifier model -- harmbench-only, fixed model name that
        # isn't surfaced as a CLI arg anywhere, so it's hardcoded here to
        # match HarmBench.__init__'s default.
        needed.append(
            ('cais/HarmBench-Llama-2-13b-cls', 'classifier_base_url_override'),
        )
        if config[inference_model_name]['key_env_name'] == 'vllm':
            needed.append(
                (inference_model_name, 'inference_base_url_override'),
            )
    elif task_name == 'prompt':
        # PromptOptimization always starts a local/shared server for
        # this model, regardless of config.yaml routing (pre-existing
        # behavior, not changed here).
        needed.append((inference_model_name, 'inference_base_url_override'))

    return needed


def resolve_vllm_endpoint(
    model_name: str,
    override_base_url: str | None,
    port: int,
    gpu_id: int = 0,
) -> str:
    """Return a base_url for model_name, sharing a server if one exists.

    If override_base_url is set, a server for this model is already
    running elsewhere (e.g. a shared Ray actor) -- just point at it
    instead of starting a redundant local one. Otherwise, fall back to
    spinning up a local vllm server exactly as before, so callers keep
    working standalone with no override supplied.
    """
    if override_base_url:
        return override_base_url
    start_vllm_server(model_name=model_name, port=port, gpu_id=gpu_id)
    return f'http://localhost:{port}/v1'


@ray.remote(num_gpus=1)
class VLLMServerActor:
    """Long-lived actor hosting one shared vllm server for a model.

    Ray schedules this actor onto a node with a free GPU and keeps it
    alive for as long as the actor handle is held, so its server can
    be reused by many experiment subprocesses instead of each one
    starting (and reloading weights for) its own.
    """

    def __init__(self, model_name: str, port: int, gpu_id: int = 0) -> None:
        """Start the shared vllm server."""
        process = start_vllm_server(
            model_name=model_name,
            port=port,
            gpu_id=gpu_id,
        )
        if process is None:
            raise RuntimeError(f'vLLM server failed to start for {model_name}')
        # Narrowed to non-Optional here via the local var + early raise
        # above, so self.process's inferred type is Popen[str] (not
        # Popen[str] | None) in every method, not just __init__ --
        # mypy doesn't carry a None-check's narrowing across methods for
        # an attribute assigned directly from an Optional-returning call.
        self.process: subprocess.Popen[str] = process
        self.node_ip = ray.util.get_node_ip_address()
        self.port = port

    def get_base_url(self) -> str:
        """Return the reachable base_url for this shared server."""
        return f'http://{self.node_ip}:{self.port}/v1'

    def shutdown(self) -> None:
        """Explicitly kill the underlying vllm process."""
        if self.process.poll() is None:
            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)


def get_similarity_percentiles(
    texts: list[str],
    model_name: str = 'all-MiniLM-L6-v2',
    percentiles: tuple[float, ...] = (75.0, 80.0, 95.0),
    device: str = 'cpu',
) -> dict[str, float]:
    """Embeds texts, computes pairwise cosine similarities.

    returns the requested percentiles of the similarity scores.
    """
    if len(texts) < 2:  # noqa: PLR2004
        raise ValueError("""At least two texts are required
        to compute pairwise similarities.""")

    # fp16 + capped seq length + small batches bound peak memory
    model_kwargs = {'torch_dtype': torch.float16} if device == 'cuda' else {}
    model = SentenceTransformer(
        model_name,
        device=device,
        model_kwargs=model_kwargs,
    )
    model.max_seq_length = min(model.max_seq_length, 4096)
    embeddings = model.encode(texts, batch_size=8)

    similarity_matrix = cosine_similarity(embeddings)

    # Extract upper triangle indices, excluding the diagonal (self-similarity)
    triu_indices = np.triu_indices_from(similarity_matrix, k=1)
    pairwise_scores = similarity_matrix[triu_indices]

    calculated_values = np.percentile(pairwise_scores, percentiles)

    # Map input percentiles to their calculated scores
    return {
        f'{p}th_percentile': float(val)
        for p, val in zip(percentiles, calculated_values, strict=True)
    }
