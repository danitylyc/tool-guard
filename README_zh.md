# ToolGuard — LLM Agent 工具调用治理中间件（框架无关）

> English version: [README.md](./README.md)

ToolGuard 是为 LLM Agent 的**工具调用步骤**提供的即插即用治理层。它解决一个真实且反复出现的问题：在 agent 部署中，工具调用是失控成本与不可靠性的主要来源，而主流 agent 框架都没有内置治理层。

> 本项目不是对现有项目的复现或微优化。它是一个**从具体问题出发的原创系统**，带有确定性测试套件与实测基准。

## 问题陈述

真实 agent 工具使用中观察到的四种失效模式：

| # | 失效模式 | 后果 |
|---|---|---|
| 1 | **重复执行** — agent 反复发出同一调用（不确定重试、计划复查、冗余再查） | 重复付出计算/API/延迟；重复结果再次进入模型上下文，token 成本上升 |
| 2 | **输出膨胀** — 工具返回超大负载（整表、大文件、整页抓取） | 全部结果被喂给模型，烧掉上下文预算或撑爆窗口 |
| 3 | **无预算意识** — 系统不知道"本次工具 I/O 花了多少 token" | 一次失控调用即可在任务中途耗尽上下文窗口 |
| 4 | **无观测** — 调用、命中、失败、截断均无记录 | 成本无法核算，故障无法追溯 |

## 设计：四层协作架构

```
        agent / 调用方
             │
   L4 观测   每次决策均记录（JSONL）+ 汇总统计
             │
   L3 策略   限流（滑动窗口）· 退避重试 · 熔断器
             │
   L2 预算   输出 token 估算 + 截断（头尾保留 | JSON 字段白名单）
             │
   L1 缓存   持久化内容寻址结果缓存（SQLite，TTL，LRU 上限）
             │
        底层工具 callable
```

- **L1 缓存** — key = 规范化 (args, 排序 kwargs) 的 sha1；易变参数可排除；SQLite 持久化使命中可跨会话/跨进程；TTL 过期 + LRU 容量上限。
- **L2 预算** — 透明 token 估算（每 4 字符 ≈ 1 token，标注为*估算*）；超长结果在进入模型上下文前被截断（头尾保留，或白名单 JSON 字段）。
- **L3 策略** — 滑动窗口限流（快速失败并给出明确信号）；对可重试错误指数退避重试；熔断器防止坏工具级联拖垮链路。
- **L4 观测** — 每次调用一条结构化记录（工具、参数哈希、缓存状态、状态、耗时、输入/输出 token 估算、是否截断、重试次数、错误）；写入 JSONL 并可 `summary()` 汇总。

工具语义保持不变：被治理的 callable 返回相同结果、相同 schema。截断是唯一可选的显式行为变更。

## 框架无关 + smolagents 适配器

```python
from tool_guard import ToolGuard

# 任意 callable（OpenAI tool loop、LangChain、LangGraph、普通函数……）
search = ToolGuard(my_db_search, name="search", cache_dir=".cache",
                   max_output_tokens=300, telemetry_path="calls.jsonl")
result = search("SELECT * FROM orders WHERE id=42")
search.telemetry.summary()
```

smolagents 工具被包装成**真正的 `smolagents.Tool` 子类**——agent 看到的仍是同一个工具（name / description / inputs / output_type 完全一致）：

```python
from tool_guard import wrap_smol_tool
from smolagents import CodeAgent, Model

agent = CodeAgent(tools=[wrap_smol_tool(my_tool, cache_dir=".cache")], model=model, max_steps=10)
```

## 实测结果（全部在本机测得，不涉及 LLM）

### 单元测试 — `python -m pytest test_tool_guard.py -q`

**19 项全部通过（1.51s）**，覆盖：缓存命中/失效、kwargs 顺序无关、易变参数忽略、TTL 过期、跨实例持久化、LRU 淘汰、头尾截断、JSON 字段截断、限流、重试恢复、不可重试错误透传、熔断开放、telemetry 记录与 JSONL、smolagents 适配器 schema 保持。

### Benchmark — `python bench_tool_guard.py`（确定性场景）

| 场景 | 裸工具 | ToolGuard | 效果 |
|---|---|---|---|
| S1 同一查询 ×3 | 3 次工具执行，约 3036 输出 token（3 次调用） | **1 次执行**，2 次缓存命中，324 输出 token | 执行 −66%，上下文 token −89% |
| S2 100,007 字符输出 | 约 25,001 token 估算进入上下文 | **832 字符 / 约 208 token 估算**（截断） | 输出 token **−99.2%** |
| S3 flaky 工具（2 次 `ConnectionError` 后成功） | 直接崩溃 | **2 次重试后恢复**，status ok | 可靠性 |

### 端到端 demo — `python demo_smol_agent.py`（真实 smolagents `CodeAgent`，脚本模型）

轨迹复现经典"不确定重试"模式：agent 两次调用同一个昂贵工具。治理后：**2 → 1 次工具执行（缓存命中），耗时 0.428s → 0.232s，输出一致**。

原始产物：`bench_out/bench_results.json`、`bench_out/bench_telemetry.jsonl`。

## 诚实的局限

- **token 估算是启发式**（`len/4`），不是真实 tokenizer；以"估算"名义报告。真实 tokenizer 因模型而异。
- **本仓库未做真实 LLM 端到端运行**（未使用 API key）：agent demo 使用脚本模型保证确定性。治理层本身已完全确定性验证。
- **截断是显式 opt-in**；`max_output_tokens=None` 时被治理的 callable 与裸工具行为完全一致。
- 缓存正确性假设工具对相同规范化参数是**确定性的**；有状态或依赖时间的工具应使用 `ignore_args`（如时间戳）或关闭缓存。

## 文件结构

```
tool_guard.py        # 库本体：L1–L4 各层、wrap_callable、wrap_smol_tool
test_tool_guard.py   # 19 项确定性测试（pytest）
bench_tool_guard.py  # 三场景实测 benchmark（无需 LLM）
demo_smol_agent.py   # smolagents CodeAgent 端到端 demo（脚本模型）
bench_out/           # 实测产物（results.json + telemetry.jsonl）
```

MIT License。
