"""End-to-end demo: ToolGuard wrapping a tool inside a real smolagents CodeAgent.

No LLM API is used: a scripted model replays a fixed trajectory (the classic
"uncertain retry" pattern — the agent calls the same expensive tool twice).
With ToolGuard the second call hits the persistent cache instead of
re-executing, and the telemetry shows it.

Run:  python demo_smol_agent.py
"""
import time

from smolagents import ChatMessage, CodeAgent, MessageRole, Model, Tool, TokenUsage

from tool_guard import wrap_smol_tool


class SlowSearch(Tool):
    name = "slow_search"
    description = "Searches the local product index (simulated, intentionally slow)."
    inputs = {"q": {"type": "string", "description": "search query"}}
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.executions = 0

    def forward(self, q: str) -> str:
        self.executions += 1
        time.sleep(0.2)  # simulated latency: real DB / HTTP tool
        return f"results for '{q}': 3 matches [A, B, C] | rank-1={q}-best"


class ScriptedLLM(Model):
    """Replays a fixed script so the demo is deterministic and LLM-free."""

    def __init__(self, steps):
        super().__init__()
        self._steps = list(steps)
        self.calls = 0

    def generate(self, prompt, **kwargs):
        self.calls += 1
        content = self._steps.pop(0)
        return ChatMessage(role=MessageRole.ASSISTANT, content=content,
                           token_usage=TokenUsage(10, 10))


SCRIPT = [
    "<code>\nslow_search('gpu servers')\n</code>",   # first call: miss, executes
    "<code>\nslow_search('gpu servers')\n</code>",   # retry/plan re-check: cache hit
    "<code>\nfinal_answer('done')\n</code>",
]


def main():
    import tempfile, os
    # Fresh cache dir per run: shows in-trajectory dedup (1st call executes,
    # 2nd hits). Persistent caching across runs is demonstrated by the
    # benchmark (S1 uses a shared cache dir) and by the unit test
    # test_cache_persists_across_instances.
    cache_dir = tempfile.mkdtemp(prefix="toolguard_demo_")

    # --- naive arm: raw tool ------------------------------------------------
    t0 = time.perf_counter()
    naive_tool = SlowSearch()
    model = ScriptedLLM(list(SCRIPT))
    naive_agent = CodeAgent(tools=[naive_tool], model=model, max_steps=6, verbosity_level=0)
    naive_out = naive_agent.run("Search the product index twice and report the result.")
    naive_wall = time.perf_counter() - t0

    # --- guarded arm: identical tool wrapped by ToolGuard --------------------
    t0 = time.perf_counter()
    raw = SlowSearch()
    guarded = wrap_smol_tool(raw, cache_dir=cache_dir, max_output_tokens=80)
    model2 = ScriptedLLM(list(SCRIPT))
    agent = CodeAgent(tools=[guarded], model=model2, max_steps=6, verbosity_level=0)
    guarded_out = agent.run("Search the product index twice and report the result.")
    guarded_wall = time.perf_counter() - t0

    s = guarded.guard.telemetry.summary()

    print(f"{'metric':<18}{'naive':>14}{'guarded':>14}")
    print(f"{'tool executions':<18}{naive_tool.executions:>14}{raw.executions - s['cache_hits']:>14}")
    print(f"{'cache hits':<18}{'-':>14}{s['cache_hits']:>14}")
    print(f"{'wall (s)':<18}{naive_wall:>14.3f}{guarded_wall:>14.3f}")
    print(f"{'output':<18}{str(naive_out):>14}{str(guarded_out):>14}")
    print(f"telemetry: {s}")

    assert naive_out == guarded_out, "outputs must match"
    assert raw.executions - s["cache_hits"] < raw.executions, "caching must cut executions"
    print("\nOK: identical output, fewer tool executions, telemetry recorded.")


if __name__ == "__main__":
    main()
