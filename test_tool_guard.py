"""Deterministic tests for ToolGuard (no LLM required).

Run:  python -m pytest test_tool_guard.py -q
"""
import json
import os
import time

import pytest

from tool_guard import (
    CallPolicy,
    DiskCache,
    ToolGuard,
    canonical_key,
    estimate_tokens,
    wrap_smol_tool,
)


# ---------------------------------------------------------------------------
# L1: cache layer
# ---------------------------------------------------------------------------

def test_same_args_are_cached(tmp_path):
    calls = []
    def fn(x):
        calls.append(x)
        return f"value-{x}"
    g = ToolGuard(fn, name="t", cache_dir=str(tmp_path / "c"))
    assert g(1) == "value-1"
    assert g(1) == "value-1"
    assert len(calls) == 1  # second call replayed from cache
    assert g.last_meta.cache == "hit"
    assert g.last_meta.status == "ok"


def test_different_args_miss(tmp_path):
    calls = []
    def fn(x):
        calls.append(x)
        return x * 2
    g = ToolGuard(fn, name="t", cache_dir=str(tmp_path / "c"))
    g(1); g(2)
    assert len(calls) == 2


def test_kwargs_order_invariant(tmp_path):
    def fn(a, b):
        return a + b
    g = ToolGuard(fn, name="t", cache_dir=str(tmp_path / "c"))
    g(a=1, b=2)
    g(b=2, a=1)
    assert g.last_meta.cache == "hit"


def test_ignore_args_exclude_volatile(tmp_path):
    calls = []
    def fn(q, ts):
        calls.append(q)
        return q.upper()
    g = ToolGuard(fn, name="t", cache_dir=str(tmp_path / "c"), ignore_args={"ts"})
    g("hi", ts=111); g("hi", ts=222)
    assert len(calls) == 1
    assert g.last_meta.cache == "hit"


def test_ttl_expiry(tmp_path):
    def fn(x):
        return x
    g = ToolGuard(fn, name="t", cache_dir=str(tmp_path / "c"), cache_ttl=0.1)
    g(1)
    time.sleep(0.15)
    g(1)
    assert g.last_meta.cache == "miss"


def test_cache_persists_across_instances(tmp_path):
    path = str(tmp_path / "c")
    calls = []
    def fn(x):
        calls.append(x)
        return f"v{x}"
    g1 = ToolGuard(fn, name="t", cache_dir=path)
    g1(5)
    g2 = ToolGuard(fn, name="t", cache_dir=path)
    g2(5)
    assert len(calls) == 1  # second instance hit the disk cache


def test_cache_lru_eviction(tmp_path):
    def fn(x):
        return x
    g = ToolGuard(fn, name="t", cache_dir=str(tmp_path / "c"), cache_max_entries=0)
    g(1); g(2); g(1)
    # cap 0 -> every store evicts; the second call of (1) is a miss again
    assert g.last_meta.cache == "miss"


def test_disk_cache_lru_cap(tmp_path):
    c = DiskCache(str(tmp_path / "c"), max_entries=2)
    c.set("a", 1); c.set("b", 2); c.set("c", 3)
    assert c.get("a") is None  # evicted
    assert c.get("b") == 2
    assert c.get("c") == 3


# ---------------------------------------------------------------------------
# L2: budget / truncation layer
# ---------------------------------------------------------------------------

def test_estimate_tokens_heuristic():
    assert estimate_tokens("x" * 100) == 25  # ~4 chars per token
    assert estimate_tokens("") == 1


def test_no_truncation_under_budget(tmp_path):
    def fn():
        return "small"
    g = ToolGuard(fn, name="t", cache_dir=str(tmp_path / "c"), max_output_tokens=1000)
    assert g() == "small"
    assert g.last_meta.truncated is False


def test_head_tail_truncation(tmp_path):
    big = "A" * 2000 + "|BODY|" + "Z" * 2000
    def fn():
        return big
    g = ToolGuard(fn, name="t", cache_dir=str(tmp_path / "c"), max_output_tokens=100)
    out = g()
    assert g.last_meta.truncated is True
    assert out.startswith("A")
    assert out.endswith("Z")
    assert "[truncated by ToolGuard]" in out
    assert estimate_tokens(out) <= 110


def test_json_field_truncation(tmp_path):
    payload = json.dumps({"id": 1, "big_field": "x" * 5000, "keep_me": "ok"})
    def fn():
        return payload
    g = ToolGuard(fn, name="t", cache_dir=str(tmp_path / "c"),
                  max_output_tokens=200, truncate_mode="json_fields", keep_fields=["id", "keep_me"])
    out = g()
    assert g.last_meta.truncated is True
    parsed = json.loads(out)
    assert set(parsed.keys()) == {"id", "keep_me"}


# ---------------------------------------------------------------------------
# L3: policy layer — rate limit, retry, circuit breaker
# ---------------------------------------------------------------------------

def test_rate_limit(tmp_path):
    def fn():
        return 1
    g = ToolGuard(fn, name="t", cache_dir=str(tmp_path / "c"),
                  policy=CallPolicy(max_calls_per_minute=2))
    g(); g()
    with pytest.raises(TimeoutError):
        g()
    assert g.last_meta.status == "rate_limited"


def test_retry_recovers(tmp_path):
    state = {"n": 0}
    def fn():
        state["n"] += 1
        if state["n"] < 3:
            raise ConnectionError("flaky")
        return "ok"
    g = ToolGuard(fn, name="t", cache_dir=str(tmp_path / "c"),
                  policy=CallPolicy(max_retries=3, backoff_base=0.01))
    assert g() == "ok"
    assert g.last_meta.retries == 2
    assert g.last_meta.status == "ok"


def test_non_retryable_error_propagates(tmp_path):
    def fn():
        raise ValueError("bad input")
    g = ToolGuard(fn, name="t", cache_dir=str(tmp_path / "c"),
                  policy=CallPolicy(max_retries=3, backoff_base=0.01))
    with pytest.raises(ValueError):
        g()
    assert g.last_meta.status == "error"
    assert g.last_meta.retries == 0


def test_circuit_breaker_opens(tmp_path):
    def fn():
        raise ConnectionError("down")
    g = ToolGuard(fn, name="t", cache_dir=str(tmp_path / "c"),
                  policy=CallPolicy(max_retries=0, circuit_max_failures=2,
                                    circuit_open_seconds=60))
    for _ in range(2):
        with pytest.raises(ConnectionError):
            g()
    assert g.policy.circuit_open() is True
    with pytest.raises(RuntimeError, match="circuit breaker"):
        g()  # fail fast without executing
    assert g.last_meta.status == "circuit_open"


# ---------------------------------------------------------------------------
# L4: telemetry
# ---------------------------------------------------------------------------

def test_telemetry_records_and_summary(tmp_path):
    def fn(x):
        return "r" * 100
    g = ToolGuard(fn, name="t", cache_dir=str(tmp_path / "c"),
                  max_output_tokens=10, telemetry_path=str(tmp_path / "calls.jsonl"))
    g(1); g(1); g(2)
    s = g.telemetry.summary()
    assert s["calls"] == 3
    assert s["cache_hits"] == 1
    assert s["cache_hit_rate"] == pytest.approx(1 / 3)
    assert s["truncated"] == 2  # truncation decisions on the execution path only
    g.close()
    lines = open(str(tmp_path / "calls.jsonl"), encoding="utf-8").read().strip().splitlines()
    assert len(lines) == 3
    rec = json.loads(lines[0])
    for k in ("tool", "args_hash", "cache", "status", "duration_ms", "in_tokens",
              "out_tokens", "truncated", "retries"):
        assert k in rec


# ---------------------------------------------------------------------------
# Framework adapter: smolagents
# ---------------------------------------------------------------------------

def test_wrap_smol_tool_preserves_schema_and_caches(tmp_path):
    from smolagents import Tool

    class Doubler(Tool):
        name = "doubler"
        description = "Doubles an integer."
        inputs = {"n": {"type": "integer", "description": "input"}}
        output_type = "integer"
        def forward(self, n: int) -> int:
            return n * 2

    tool = Doubler()
    wrapped = wrap_smol_tool(tool, cache_dir=str(tmp_path / "c"))
    assert wrapped.name == "doubler"
    assert wrapped.description == tool.description
    assert wrapped.inputs == tool.inputs
    assert wrapped.output_type == tool.output_type
    assert wrapped(21) == 42
    assert wrapped(21) == 42
    assert wrapped.guard.last_meta.cache == "hit"
    assert wrapped.guard.telemetry.summary()["cache_hits"] == 1


def test_canonical_key():
    a = canonical_key((), {"x": 1, "y": [1, 2]})
    b = canonical_key((), {"y": [1, 2], "x": 1})
    assert a == b
    c = canonical_key((), {"x": 1, "y": [1, 2]}, ignore_args={"y"})
    d = canonical_key((), {"x": 1, "y": [9, 9]}, ignore_args={"y"})
    assert c == d
