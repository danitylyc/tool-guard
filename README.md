# ToolGuard — framework-agnostic tool-call governance for LLM agents

ToolGuard is a drop-in governance layer for the **tool-call step** of LLM agents.
It solves a real, recurring production problem: tool calls are the main source
of uncontrolled cost and unreliability in agent deployments, and mainstream
agent frameworks ship no built-in control layer.

> This is not a reproduction or a micro-optimization of an existing project.
> It is an original system designed from a concrete problem statement, with a
> deterministic test suite and measured benchmarks.

## Problem statement

Four failure modes observed in real agent tool usage:

| # | failure mode | consequence |
|---|---|---|
| 1 | **Duplicate executions** — the agent re-issues the same call (uncertain retry, plan re-check, redundant re-query) | re-pays compute/API/latency; the repeated result re-enters the model context and inflates token spend |
| 2 | **Oversized outputs** — a tool returns a huge payload (full table, large file, page dump) | the whole result is fed to the model, burning context budget or blowing the window |
| 3 | **No budget awareness** — no notion of "how many tokens did my tool I/O just cost" | one wild call can exhaust the context window mid-task |
| 4 | **No observability** — calls, hits, failures, truncations are not recorded | cost cannot be accounted for; bugs cannot be traced |

## Design: four cooperating layers

```
        agent / caller
             │
   L4 Telemetry ── every decision recorded (JSONL) + summary stats
             │
   L3 Policy   ── rate limiting (sliding window) · retry w/ backoff · circuit breaker
             │
   L2 Budget   ── output token estimation + truncation (head+tail | JSON field keep)
             │
   L1 Cache    ── persistent content-addressed result cache (SQLite, TTL, LRU cap)
             │
        underlying tool callable
```

- **L1 Cache** — key = sha1 of canonical (args, sorted kwargs), volatile args
  excludable; persisted in SQLite so hits survive across sessions/processes;
  TTL + LRU size cap.
- **L2 Budget** — transparent token estimate (`len/4` chars per token, reported
  as an *estimate*); oversized results are truncated (head+tail, or whitelisted
  JSON fields) before reaching the model context.
- **L3 Policy** — sliding-window rate limit (fail fast with a clear signal),
  retry with exponential backoff on retryable errors, circuit breaker so a
  failing tool cannot cascade.
- **L4 Telemetry** — one structured record per call (tool, args hash, cache
  status, status, duration, in/out token estimates, truncated, retries, error);
  written to JSONL and summarized (`summary()`).

Tool semantics are preserved: the guarded callable returns the same result,
same schema. Truncation is the only opt-in behavior change.

## Framework-agnostic + smolagents adapter

```python
from tool_guard import ToolGuard

# any callable (OpenAI tool loop, LangChain, LangGraph, plain function…)
search = ToolGuard(my_db_search, name="search", cache_dir=".cache",
                   max_output_tokens=300, telemetry_path="calls.jsonl")
result = search("SELECT * FROM orders WHERE id=42")
search.telemetry.summary()
```

smolagents tools are wrapped into a **real `smolagents.Tool` subclass** — the
agent sees an identical tool (same name / description / inputs / output_type):

```python
from tool_guard import wrap_smol_tool
from smolagents import CodeAgent, Model

agent = CodeAgent(tools=[wrap_smol_tool(my_tool, cache_dir=".cache")], model=model, max_steps=10)
```

## Measured results (all numbers measured on this machine, no LLM involved)

### Unit tests — `python -m pytest test_tool_guard.py -q`

**19 passed in 1.51s** covering: cache hit/miss, kwargs-order invariance,
volatile-arg ignoring, TTL expiry, cross-instance persistence, LRU eviction,
head+tail truncation, JSON-field truncation, rate limiting, retry recovery,
non-retryable error propagation, circuit breaker open, telemetry records +
JSONL, smolagents adapter schema preservation.

### Benchmark — `python bench_tool_guard.py` (deterministic scenarios)

| scenario | naive | guarded | effect |
|---|---|---|---|
| S1 same query ×3 | 3 tool executions, ~3036 out-tokens (3 calls) | **1 execution**, 2 cache hits, 324 out-tokens | −66% executions, −89% context tokens |
| S2 100,007-char output | ~25,001 tokens est. into context | **832 chars / ~208 tokens est** (truncated) | **−99.2% output tokens** |
| S3 flaky tool (2× `ConnectionError` then success) | crash | **recovered after 2 retries**, status ok | reliability |

### End-to-end demo — `python demo_smol_agent.py` (real smolagents `CodeAgent`, scripted model)

Trajectory replays the classic "uncertain retry": the agent calls the same
expensive tool twice. Guarded arm: **2 → 1 tool executions (cache hit), wall
0.428 s → 0.232 s, identical output**.

Raw artifacts: `bench_out/bench_results.json`, `bench_out/bench_telemetry.jsonl`.

## Honest limitations

- **Token estimate is a heuristic** (`len/4`), not a real tokenizer; reported as
  an estimate. Real tokenizers vary by model.
- **No real-LLM end-to-end run** in this repo (no API key used): the agent demo
  uses a scripted model for determinism. The governance layer itself is fully
  validated deterministically.
- **Truncation is opt-in**; with `max_output_tokens=None` the guarded callable
  is behavior-identical to the raw tool.
- Cache correctness assumes the tool is **deterministic for equal canonical
  args**; stateful or time-dependent tools should use `ignore_args` (e.g.
  timestamps) or disable caching.

## Layout

```
tool_guard.py        # the library: layers L1–L4, wrap_callable, wrap_smol_tool
test_tool_guard.py   # 19 deterministic tests (pytest)
bench_tool_guard.py  # 3-scenario measured benchmark (no LLM)
demo_smol_agent.py   # smolagents CodeAgent end-to-end demo (scripted model)
bench_out/           # measured artifacts (results.json + telemetry.jsonl)
```

MIT license.
