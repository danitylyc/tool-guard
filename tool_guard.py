"""ToolGuard: a framework-agnostic governance layer for LLM-agent tool calls.

Problem statement
-----------------
In real LLM-agent deployments, the tool-call layer is the main source of
uncontrolled cost and unreliability:

  L1-cost  duplicate execution   -- the agent re-issues the same call (uncertain
                                    retry, plan re-check), paying compute / API /
                                    latency again; the repeated result re-enters
                                    the model context and inflates token spend;
  L2-cost  oversized outputs     -- a tool returns a huge result (full table,
                                    large file, page dump) and the whole thing is
                                    fed to the model, burning context budget;
  L3-cost  no budget awareness   -- the agent has no notion of "how many tokens
                                    did my tool I/O just cost"; one wild call can
                                    exhaust the context window;
  L4-cost  no observability      -- calls, hits, failures and truncations are not
                                    recorded anywhere, so cost cannot be
                                    accounted for and bugs cannot be traced.

ToolGuard wraps any callable with four cooperating layers:

  L1 Cache     -- persistent, content-addressed result cache (SQLite + TTL +
                  LRU cap). Identical calls are replayed without re-execution.
  L2 Budget    -- output token estimation and truncation policy (head+tail, or
                  JSON field extraction), keeping oversized results bounded.
  L3 Policy    -- rate limiting (sliding window), retry with backoff, and a
                  circuit breaker so a failing tool cannot cascade.
  L4 Telemetry -- one structured record per call (JSONL) plus summary stats.

It is framework-agnostic: `wrap_callable` wraps any Python callable (usable with
OpenAI-style tool loops, LangChain, LangGraph, ...). A smolagents `Tool` adapter
(`wrap_smol_tool`) preserves the tool's name / description / inputs so the agent
sees an identical tool. Tool semantics are preserved; truncation is opt-in.

Design notes
------------
- Cache key = sha1 over a canonical JSON of (positional args, sorted kwargs),
  optionally ignoring volatile args (timestamps, request ids).
- Token estimate is a transparent heuristic (len/4, ~4 chars per token) and is
  reported as an *estimate*; it is not a real tokenizer.
- Every decision is recorded and can be inspected per call via `last_meta`.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# ----------------------------------------------------------------------------
# L0: canonical call key
# ----------------------------------------------------------------------------

def canonical_key(args: tuple, kwargs: dict, ignore_args: set[str] = frozenset()) -> str:
    """Stable, order-independent hash of a call signature (args + sorted kwargs)."""
    kw = {k: v for k, v in kwargs.items() if k not in ignore_args}
    payload = json.dumps([list(args), kw], sort_keys=True, default=lambda o: repr(o), ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------
# L1: persistent disk cache (SQLite, TTL, size cap)
# ----------------------------------------------------------------------------

class DiskCache:
    """Content-addressed result cache persisted in SQLite.

    Values are stored as JSON. `get(key, ttl)` returns None when absent or
    expired. A size cap (max_entries) evicts least-recently-used entries.
    Thread-safe via a lock + per-connection usage.
    """

    def __init__(self, path: str, max_entries: int = 10_000):
        self.path = path
        self.max_entries = max_entries
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL, created REAL NOT NULL,"
            "last_access REAL NOT NULL)"
        )
        self._conn.commit()

    def get(self, key: str, ttl: Optional[float] = None) -> Optional[Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT value, created FROM cache WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            value, created = row
            if ttl is not None and time.time() - created > ttl:
                self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                self._conn.commit()
                return None
            self._conn.execute(
                "UPDATE cache SET last_access = ? WHERE key = ?", (time.time(), key)
            )
            self._conn.commit()
            return json.loads(value)

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        with self._lock:
            now = time.time()
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, created, last_access) VALUES (?,?,?,?)",
                (key, json.dumps(value, default=repr), now, now),
            )
            # Enforce size cap (LRU): remove oldest last_access entries.
            total = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            if total > self.max_entries:
                excess = total - self.max_entries
                self._conn.execute(
                    "DELETE FROM cache WHERE key IN ("
                    "SELECT key FROM cache ORDER BY last_access ASC LIMIT ?)",
                    (excess,),
                )
            self._conn.commit()

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM cache")
            self._conn.commit()

    @property
    def size(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]


# ----------------------------------------------------------------------------
# L2: output token budget + truncation
# ----------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Transparent heuristic: ~4 chars per token. Reported as an estimate."""
    return max(1, len(text) // 4)


class OutputBudget:
    """Bounds how many tokens of a tool result are passed through.

    `truncate(text, max_tokens, mode)`:
      - "head_tail" : keep the head and the tail, drop the middle (for prose/logs).
      - "json_fields": keep only whitelisted top-level fields of a JSON object
                      (for structured payloads).
    Returns (bounded_text, truncated: bool).
    """

    def __init__(self, max_tokens: Optional[int] = None, mode: str = "head_tail",
                 keep_fields: Optional[list[str]] = None):
        self.max_tokens = max_tokens
        self.mode = mode
        self.keep_fields = keep_fields or []

    def truncate(self, text: str, max_tokens: Optional[int] = None) -> tuple[str, bool]:
        cap = max_tokens or self.max_tokens
        if cap is None or estimate_tokens(text) <= cap:
            return text, False
        if self.mode == "json_fields" and self.keep_fields:
            try:
                obj = json.loads(text)
                kept = {k: obj[k] for k in self.keep_fields if k in obj}
                out = json.dumps(kept, ensure_ascii=False)
                if estimate_tokens(out) <= cap:
                    return out, True
            except (ValueError, TypeError):
                pass  # fall through to head_tail
        # head_tail: keep first 60% and last 40% of the budget in characters.
        budget_chars = cap * 4
        head_chars = int(budget_chars * 0.6)
        tail_chars = budget_chars - head_chars
        if len(text) > head_chars + tail_chars:
            return text[:head_chars] + "\n...[truncated by ToolGuard]...\n" + text[-tail_chars:], True
        return text, False


# ----------------------------------------------------------------------------
# L3: call policy — rate limit, retry, circuit breaker
# ----------------------------------------------------------------------------

@dataclass
class CallPolicy:
    max_calls_per_minute: Optional[int] = None
    max_retries: int = 0
    backoff_base: float = 0.5
    circuit_max_failures: int = 5
    circuit_open_seconds: float = 30.0
    is_retryable: Optional[Callable[[Exception], bool]] = None

    def __post_init__(self):
        self._hits: deque[float] = deque()
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    # ---- rate limiting (sliding window) --------------------------------
    def allow(self) -> bool:
        if self.max_calls_per_minute is None:
            return True
        with self._lock:
            now = time.time()
            while self._hits and now - self._hits[0] > 60.0:
                self._hits.popleft()
            if len(self._hits) >= self.max_calls_per_minute:
                return False
            self._hits.append(now)
            return True

    # ---- circuit breaker ------------------------------------------------
    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._circuit_open_until = 0.0

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.circuit_max_failures:
                self._circuit_open_until = time.time() + self.circuit_open_seconds

    def circuit_open(self) -> bool:
        with self._lock:
            return time.time() < self._circuit_open_until

    def is_retryable_error(self, exc: Exception) -> bool:
        if self.is_retryable is not None:
            return self.is_retryable(exc)
        return isinstance(exc, (ConnectionError, TimeoutError, OSError))


# ----------------------------------------------------------------------------
# L4: telemetry — JSONL records + summary
# ----------------------------------------------------------------------------

@dataclass
class CallMeta:
    tool: str
    args_hash: str
    cache: str            # "hit" | "miss" | "disabled"
    status: str           # "ok" | "error" | "rate_limited" | "circuit_open"
    duration_ms: float
    in_tokens: int
    out_tokens: int
    truncated: bool
    retries: int
    error: Optional[str] = None

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["ts"] = time.time()
        return d


class Telemetry:
    """Collects one structured record per guarded call and writes JSONL.

    Records are appended in memory and optionally dumped to a JSONL file, so
    cost can be accounted for after a run (`summary()`).
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._records: list[dict] = []
        self._lock = threading.Lock()
        self._file = None
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._file = open(path, "a", encoding="utf-8")

    def record(self, meta: CallMeta) -> None:
        d = meta.as_dict()
        with self._lock:
            self._records.append(d)
            if self._file:
                self._file.write(json.dumps(d, ensure_ascii=False) + "\n")
                self._file.flush()

    def summary(self) -> dict:
        with self._lock:
            n = len(self._records)
            if n == 0:
                return {"calls": 0}
            hits = sum(1 for r in self._records if r["cache"] == "hit")
            truncated = sum(1 for r in self._records if r["truncated"])
            errors = sum(1 for r in self._records if r["status"] == "error")
            rate_limited = sum(1 for r in self._records if r["status"] == "rate_limited")
            out_tokens = sum(r["out_tokens"] for r in self._records)
            return {
                "calls": n,
                "cache_hits": hits,
                "cache_hit_rate": hits / n if n else 0.0,
                "truncated": truncated,
                "errors": errors,
                "rate_limited": rate_limited,
                "total_out_tokens": out_tokens,
            }

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None


# ----------------------------------------------------------------------------
# The guarded callable
# ----------------------------------------------------------------------------

class ToolGuard:
    """Wraps a callable so every call passes cache -> budget -> policy -> telemetry.

    Usage (framework-agnostic):
        guarded = ToolGuard(my_fn, name="search", cache_dir=".cache", max_output_tokens=500)
        result = guarded("query")
        meta = guarded.last_meta
        guarded.telemetry.summary()

    Or as a smolagents Tool (same instance is callable by the agent):
        from smolagents import Tool
        smol_tool = ToolGuard.wrap_smol_tool(original_smol_tool, ...)
    """

    def __init__(
        self,
        fn: Callable,
        *,
        name: str = "tool",
        cache_dir: Optional[str] = ".toolguard_cache",
        cache_ttl: Optional[float] = None,
        cache_max_entries: Optional[int] = None,
        ignore_args: Optional[set[str]] = None,
        max_output_tokens: Optional[int] = None,
        truncate_mode: str = "head_tail",
        keep_fields: Optional[list[str]] = None,
        policy: Optional[CallPolicy] = None,
        telemetry_path: Optional[str] = None,
        cache_enabled: bool = True,
    ):
        self._fn = fn
        self.name = name
        self.ignore_args = set(ignore_args or ())
        self.cache_ttl = cache_ttl
        self.cache_enabled = cache_enabled
        self.budget = OutputBudget(max_output_tokens, truncate_mode, keep_fields)
        self.policy = policy or CallPolicy()
        self.telemetry = Telemetry(telemetry_path)
        self.cache = DiskCache(os.path.join(cache_dir, "cache.db"),
                               max_entries=cache_max_entries if cache_max_entries is not None else 10_000) if (cache_dir and cache_enabled) else None
        self.last_meta: Optional[CallMeta] = None
        self._lock = threading.Lock()

    # ---- the guarded call -------------------------------------------------
    def __call__(self, *args, **kwargs) -> Any:
        result, meta = self.call_with_meta(*args, **kwargs)
        return result

    def call_with_meta(self, *args, **kwargs) -> tuple[Any, CallMeta]:
        t0 = time.time()
        key = canonical_key(args, kwargs, self.ignore_args)
        in_tokens = estimate_tokens(json.dumps([list(args), kwargs], default=repr))

        # circuit breaker: fail fast when open
        if self.policy.circuit_open():
            meta = CallMeta(self.name, key, "disabled", "circuit_open", (time.time() - t0) * 1000,
                            in_tokens, 0, False, 0, "circuit breaker open")
            self.last_meta = meta
            self.telemetry.record(meta)
            raise RuntimeError(f"ToolGuard: circuit breaker open for tool '{self.name}'")

        # rate limit
        if not self.policy.allow():
            meta = CallMeta(self.name, key, "disabled", "rate_limited", (time.time() - t0) * 1000,
                            in_tokens, 0, False, 0, "rate limit exceeded")
            self.last_meta = meta
            self.telemetry.record(meta)
            raise TimeoutError(f"ToolGuard: rate limit exceeded for tool '{self.name}'")

        # cache lookup
        cache_status = "disabled"
        if self.cache is not None:
            cached = self.cache.get(key, self.cache_ttl)
            if cached is not None:
                cache_status = "hit"
                self.policy.record_success()
                text = cached if isinstance(cached, str) else json.dumps(cached, ensure_ascii=False)
                out_tokens = estimate_tokens(text)
                meta = CallMeta(self.name, key, "hit", "ok", (time.time() - t0) * 1000,
                                in_tokens, out_tokens, False, 0)
                self.last_meta = meta
                self.telemetry.record(meta)
                return cached, meta
            cache_status = "miss"

        # execute with retry
        retries = 0
        while True:
            try:
                with self._lock:  # serialize actual tool execution
                    result = self._fn(*args, **kwargs)
                break
            except Exception as exc:  # noqa: BLE001 - policy decides
                self.policy.record_failure()
                if retries < self.policy.max_retries and self.policy.is_retryable_error(exc):
                    retries += 1
                    time.sleep(min(self.policy.backoff_base * (2 ** (retries - 1)), 5.0))
                    continue
                text_out = f"error: {exc!r}"[:400]
                meta = CallMeta(self.name, key, cache_status, "error", (time.time() - t0) * 1000,
                                in_tokens, estimate_tokens(text_out), False, retries, repr(exc))
                self.last_meta = meta
                self.telemetry.record(meta)
                raise

        self.policy.record_success()
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        bounded, truncated = self.budget.truncate(text)
        if self.cache is not None:
            # cache the bounded form if truncation applied, else the raw result
            self.cache.set(key, bounded if truncated else result, self.cache_ttl)
        out_tokens = estimate_tokens(bounded)
        meta = CallMeta(self.name, key, cache_status, "ok", (time.time() - t0) * 1000,
                        in_tokens, out_tokens, truncated, retries)
        self.last_meta = meta
        self.telemetry.record(meta)
        return (bounded if truncated else result), meta

    def clear_cache(self) -> None:
        if self.cache:
            self.cache.clear()

    def close(self) -> None:
        self.telemetry.close()


# ----------------------------------------------------------------------------
# smolagents adapter
# ----------------------------------------------------------------------------

def wrap_smol_tool(tool, **guard_kwargs):
    """Wrap a smolagents `Tool` so the agent sees an identical tool (same
    name / description / inputs) but every call is governed.

    Returns a real `smolagents.Tool` subclass whose `forward` runs through a
    ToolGuard, so existing agent code (CodeAgent, ToolCallingAgent) treats it
    as a plain tool. `guard_kwargs` are passed to ToolGuard. Requires
    smolagents installed (optional import).
    """
    try:
        from smolagents.tools import Tool  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("smolagents not installed; install it to use wrap_smol_tool") from exc

    if not isinstance(tool, Tool):
        raise TypeError(f"wrap_smol_tool expects a smolagents.Tool, got {type(tool).__name__}")

    guard_kwargs.setdefault("name", tool.name)
    guard = ToolGuard(tool.forward, **guard_kwargs)

    class _GuardedTool(Tool):
        skip_forward_signature_validation = True  # generic **kwargs signature is intentional

        def __init__(self):
            super().__init__()
            self.name = tool.name
            self.description = tool.description
            self.inputs = tool.inputs
            self.output_type = getattr(tool, "output_type", "any")
            self.output_schema = getattr(tool, "output_schema", None)

        def forward(self, *args, **kwargs):
            return guard(*args, **kwargs)

        @property
        def guard(self) -> ToolGuard:
            return guard

    return _GuardedTool()
