"""Prompts for the OPRO diversity check's LLM-as-a-judge step.

See SolutionBank.llm_judge_sim in opro.py: eval_cosine_sim flags a new
candidate as too similar to an existing active-pool solution, then these
prompts ask an LLM for a second opinion before rejecting it.
"""

from __future__ import annotations

# NOTE(MS): code-task (cloudcast, cantbelate, kernelbench) templates below
# are used verbatim, unmodified, from ShinkaEvolve's novelty judge:
# https://github.com/SakanaAI/ShinkaEvolve/blob/main/shinka/prompts/prompts_novelty.py
CODE_TASK_NAMES = frozenset({'cloudcast', 'cantbelate', 'kernelbench'})

NOVELTY_SYSTEM_MSG = """You are an expert code reviewer tasked with determining if two code snippets are meaningfully different from each other.

Your job is to analyze both programs and determine if the proposed code introduces meaningful changes compared to the existing code. Consider:

1. **Algorithmic differences**: Different approaches, logic, or strategies
2. **Structural changes**: Different data structures, control flow, or organization
3. **Functional improvements**: New features, optimizations, or capabilities
4. **Implementation variations**: Different ways of achieving the same goal that could lead to different performance characteristics
5. **Hyperparameter changes**: Different hyperparameters that could lead to different performance characteristics

Ignore trivial differences like:
- Variable name changes
- Minor formatting or style changes
- Comments or documentation changes
- Insignificant refactoring that doesn't change the core logic

Respond with:
- **NOVEL**: If the codes are meaningfully different
- **NOT_NOVEL**: If the codes are essentially the same with only trivial differences

After your decision, provide a brief explanation of your reasoning."""


NOVELTY_USER_MSG = """Please analyze these two code snippets:

**EXISTING CODE:**
```{language}
{existing_code}
```

**PROPOSED CODE:**
```{language}
{proposed_code}
```

Are these codes meaningfully different? Respond with NOVEL or NOT_NOVEL followed by your explanation."""


# NOTE(MS): non-code-task template -- same structure/vocabulary as
# ShinkaEvolve's novelty judge above (system/user split, numbered
# criteria list, NOVEL/NOT_NOVEL verdict), but grounded in this task's
# own description/goal/metric instead of being code-review-specific.
JUDGE_SYSTEM_MSG = """You are an expert evaluator tasked with determining if two candidate {solution_description} are meaningfully different from each other.

Task context: {task_description} The goal is to {direction} {metric}.

Your job is to analyze both candidates and determine if the proposed candidate introduces meaningful changes compared to the existing candidate, with respect to that goal. Consider:

1. **Strategic differences**: Different approaches, arguments, or strategies for pursuing the goal
2. **Structural differences**: Different organization, framing, or content structure
3. **Substantive additions**: New ideas, techniques, or angles not present in the existing candidate
4. **Implementation variations**: Different ways of pursuing the same goal that could plausibly lead to different {metric} outcomes

Ignore trivial differences like:
- Minor wording, phrasing, or formatting changes
- Reordering that doesn't change the underlying content
- Insignificant edits that don't change the core approach

Respond with:
- **NOVEL**: If the candidates are meaningfully different
- **NOT_NOVEL**: If the candidates are essentially the same with only trivial differences

After your decision, provide a brief explanation of your reasoning."""


JUDGE_USER_MSG = """Please analyze these two candidate solutions:

**EXISTING CANDIDATE:**
{existing_solution}

**PROPOSED CANDIDATE:**
{proposed_solution}

Are these candidates meaningfully different? Respond with NOVEL or NOT_NOVEL followed by your explanation."""
