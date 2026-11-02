"""Deterministic benchmark for ToolGuard — no LLM required.

Three cost/unreliability scenarios, each measured with and without ToolGuard:

  S1 duplicate executions : the agent re-queries the same key 3 times
                            (uncertain retry / plan re-check pattern).
                            Metric: tool executions 3 -> ?, cache hits.
  S2 oversized output     : a tool returns ~100k chars; the guarded arm bounds
                            the result. Metric: output tokens before/after.
  S3 flaky tool           : the tool fails with ConnectionError twice then
                            succeeds. Metric: retries, final status.

All numbers are measured on this machine and written to JSONL + a text table.
"""
import json
import os
import time

from tool_guard import CallPolicy, ToolGuard, estimate_tokens

OUT_DIR = os.path.join(os.path.dirname(__file__), "bench_out")
os.makedirs(OUT_DIR, exist_ok=True)
TELEMETRY = os.path.join(OUT_DIR, "bench_telemetry.jsonl")


def scenario1_duplicate_calls():
    """Same query issued 3 times (uncached DB lookup returning a big row)."""

    def query_db(q):
        return "ORDER#42|user=lyc|amount=$123.45|items=42|payload=" + "x" * 4000

    def run(mode):
        guard = None
        fn = query_db
        if mode == "guarded":
            guard = ToolGuard(fn, name="query_db", cache_dir=os.path.join(OUT_DIR, "cache_s1"),
                              max_output_tokens=100)
            fn = guard
        t0 = time.time()
        for _ in range(3):
            fn("SELECT * FROM orders WHERE id=42")
        wall = time.time() - t0
        if guard:
            s = guard.telemetry.summary()
            executions = s["calls"] - s["cache_hits"]  # real tool executions
            hits = s["cache_hits"]
            out_tokens = s["total_out_tokens"]
            guard.close()
        else:
            executions = 3
            hits = 0
            out_tokens = 3 * estimate_tokens(query_db("SELECT * FROM orders WHERE id=42"))
        return {
            "mode": mode,
            "tool_executions": executions,
            "cache_hits": hits,
            "wall_ms": round(wall * 1000, 1),
            "out_tokens_total": out_tokens,
        }

    naive = run("naive")
    guarded = run("guarded")
    return naive, guarded


def scenario2_oversized_output():
    """A tool returns 100k chars; guarded arm truncates to a token budget."""
    big = "RECORD|" + "abcdefgh" * 12500  # ~100k chars

    def dump_all():
        return big

    guard = ToolGuard(dump_all, name="dump_all", cache_dir=os.path.join(OUT_DIR, "cache_s2"),
                      max_output_tokens=200)
    raw_tokens = estimate_tokens(big)
    t0 = time.time()
    out = guard()
    wall = time.time() - t0
    bounded_tokens = estimate_tokens(out)
    return {
        "raw_chars": len(big),
        "raw_tokens_est": raw_tokens,
        "bounded_chars": len(out),
        "bounded_tokens_est": bounded_tokens,
        "reduction_pct": round((1 - bounded_tokens / raw_tokens) * 100, 1),
        "truncated": guard.last_meta.truncated,
        "wall_ms": round(wall * 1000, 1),
    }


def scenario3_flaky_tool():
    """Tool fails twice (ConnectionError) then succeeds; retry policy recovers."""
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] <= 2:
            raise ConnectionError("upstream timeout")
        return "SUCCESS"

    guard = ToolGuard(flaky, name="flaky", cache_dir=os.path.join(OUT_DIR, "cache_s3"),
                      policy=CallPolicy(max_retries=3, backoff_base=0.01))
    t0 = time.time()
    result = guard()
    wall = time.time() - t0
    return {
        "result": result,
        "retries": guard.last_meta.retries,
        "status": guard.last_meta.status,
        "wall_ms": round(wall * 1000, 1),
    }


def main():
    s1n, s1g = scenario1_duplicate_calls()
    s2 = scenario2_oversized_output()
    s3 = scenario3_flaky_tool()

    print("== S1 duplicate executions (same query x3) ==")
    print(f"  {'metric':<20}{'naive':>14}{'guarded':>14}")
    print(f"  {'tool executions':<20}{s1n['tool_executions']:>14}{s1g['tool_executions']:>14}")
    print(f"  {'cache hits':<20}{'-':>14}{s1g['cache_hits']:>14}")
    print(f"  {'wall (ms)':<20}{s1n['wall_ms']:>14}{s1g['wall_ms']:>14}")
    print(f"  {'out tokens (3 calls)':<20}{s1n['out_tokens_total']:>14}{s1g['out_tokens_total']:>14}")

    print("== S2 oversized output (~100k chars) ==")
    print(f"  raw: {s2['raw_chars']} chars, ~{s2['raw_tokens_est']} tokens est")
    print(f"  guarded: {s2['bounded_chars']} chars, ~{s2['bounded_tokens_est']} tokens est, "
          f"reduction {s2['reduction_pct']}%, truncated={s2['truncated']}")

    print("== S3 flaky tool ==")
    print(f"  result={s3['result']} retries={s3['retries']} status={s3['status']} "
          f"wall={s3['wall_ms']}ms")

    report = {
        "S1_duplicate_calls": {"naive": s1n, "guarded": s1g},
        "S2_oversized_output": s2,
        "S3_flaky_tool": s3,
    }
    with open(os.path.join(OUT_DIR, "bench_results.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nresults -> {os.path.join(OUT_DIR, 'bench_results.json')}")
    print(f"telemetry -> {TELEMETRY}")


if __name__ == "__main__":
    main()
