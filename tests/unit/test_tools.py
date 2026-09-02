"""工具插件系统测试（doc/08）。

覆盖：
- 参数校验（validate_args）
- 注册表原子快照隔离 / 热更新回滚 / 影子灰度（doc/08 §3）
- entry_points 加载不崩溃（doc/08 §2）
- LLM function-calling 二级路由（解析 <tool_call> + 派发 + 危险工具门禁 + 去重）
- recall_memory 语义召回（经引擎 call_tool 与自动触发）
"""

import os
import time
from pathlib import Path

import pytest

from dla.config.loader import get_keyword_lib
from dla.config.settings import Settings
from dla.llm.openai_compat import make_llm_client
from dla.orchestration.engine import DialogueEngine
from dla.storage.migrator import migrate
from dla.storage.repositories import SQLiteRepo
from dla.storage.sqlite import get_connection
from dla.tools import build_builtin_registry
from dla.tools.loader import discover_entry_points
from dla.tools.watcher import HotReloadWatcher
from dla.tools.executor import run_tool
from dla.tools.protocol import Tool, ToolContext, ToolResult, validate_args
from dla.tools.registry import ToolRegistry
from dla.tools.plugins.recall_memory import TOOL as RECALL_TOOL


def _engine(tmp_path, reg=None):
    s = Settings()
    s.memory_retrieve_trigger = "always"
    s.memory_retrieve_sim_threshold = 0.0
    lib = get_keyword_lib()
    llm = make_llm_client("", "https://api.openai.com/v1", "gpt-4o-mini")
    db = os.path.join(str(tmp_path), "t.db")
    conn = get_connection(db, check_same_thread=False)
    migrate(conn, "migrations")
    repo = SQLiteRepo(conn)
    e = DialogueEngine(s, lib, llm, repo, tool_registry=reg or build_builtin_registry())
    return e


def _engine_no_db(reg=None):
    s = Settings()
    s.tools_enabled = True
    s.memory_retrieve_trigger = "always"
    lib = get_keyword_lib()
    llm = make_llm_client("", "https://api.openai.com/v1", "gpt-4o-mini")
    return DialogueEngine(s, lib, llm, repo=None, tool_registry=reg)


# ---- 参数校验 ----
def test_validate_args_missing_required():
    t = Tool(
        name="x", description="",
        parameters={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
        run=lambda a, c: ToolResult(ok=True),
    )
    assert validate_args(t, {}) is not None
    assert validate_args(t, {"q": "hi"}) is None
    assert validate_args(t, {"q": 1}) is not None  # 类型不符


# ---- 注册表原子快照 / 热更新隔离（doc/08 §3.1）----
def test_registry_snapshot_isolation():
    reg = ToolRegistry(tools=[RECALL_TOOL])
    snap1 = reg.snapshot()
    assert "recall_memory" in snap1
    # 外部对快照副本的修改不影响注册表内部
    snap1["recall_memory"] = "tampered"
    assert reg.get("recall_memory") is RECALL_TOOL
    # 新快照反映后续变更，旧快照不受影响
    reg._snapshot["echo"] = Tool(name="echo", description="echo tool",
                                 parameters={"type": "object", "properties": {}},
                                 run=lambda a, c: ToolResult(ok=True))
    assert "echo" in reg.snapshot()
    assert "echo" not in snap1


def test_reload_succeeds_and_keeps_tools():
    reg = ToolRegistry(tools=[RECALL_TOOL])
    res = reg.reload()
    assert res["ok"] is True
    assert "recall_memory" in reg.snapshot()


def test_reload_rolls_back_on_invalid(monkeypatch):
    reg = ToolRegistry(tools=[RECALL_TOOL])
    bad = Tool(name="bad", description="bad", parameters={}, run=None)  # type: ignore[arg-type]
    monkeypatch.setattr("dla.tools.registry.discover_all", lambda: [(bad, "bad", None)])
    res = reg.reload()
    assert res["ok"] is False
    assert res["error"]
    # 原快照未变（回滚到 last-good）
    assert reg.get("recall_memory") is not None


def test_reload_shadow_then_promote(monkeypatch):
    reg = ToolRegistry(tools=[RECALL_TOOL])
    good = Tool(name="extra", description="extra tool",
                parameters={"type": "object", "properties": {}},
                run=lambda a, c: ToolResult(ok=True))
    monkeypatch.setattr("dla.tools.registry.discover_all", lambda: [(good, "extra", None)])
    res = reg.reload(shadow=True)
    assert res["ok"] and res["shadow"]
    assert "extra" not in reg.snapshot()  # 影子未生效
    promote = reg.promote_shadow()
    assert promote["ok"]
    assert "extra" in reg.snapshot()


def test_set_enabled_toggles_tool():
    reg = ToolRegistry(tools=[RECALL_TOOL])
    assert reg.is_enabled("recall_memory")
    assert reg.set_enabled("recall_memory", False) is True
    assert not reg.is_enabled("recall_memory")
    assert reg.get("recall_memory") is None  # 禁用后 get 返回 None
    reg.set_enabled("recall_memory", True)
    assert reg.is_enabled("recall_memory")


def test_discover_entry_points_graceful_on_missing_group():
    # 不存在的 group 不应抛错，返回空列表（doc/08 §2 发行包可选）
    assert discover_entry_points("dla.tools.__nonexistent__") == []


# ---- 热更新监听器（doc/08 §3.2/§3.3）----
def test_hotreload_watcher_triggers_reload_on_change(tmp_path, monkeypatch):
    reg = ToolRegistry(tools=[RECALL_TOOL])
    calls = {"n": 0}

    def fake_discover():
        calls["n"] += 1
        return [(Tool(name="x", description="x", parameters={}, run=lambda a, c: ToolResult(ok=True)), "x", None)]

    monkeypatch.setattr("dla.tools.registry.discover_all", fake_discover)
    d = tmp_path / "plugins"
    d.mkdir()
    (d / "a.py").write_text("# initial", encoding="utf-8")

    watcher = HotReloadWatcher(reg, str(d), mode="watch")
    first = watcher.scan_once()
    assert first["changed"] is False  # 基线已建，无变化

    time.sleep(0.01)
    (d / "a.py").write_text("# changed content", encoding="utf-8")
    second = watcher.scan_once()
    assert second["changed"] is True
    assert calls["n"] >= 1  # 变更后触发发现并 reload


def test_hotreload_watcher_shadow_mode_does_not_activate(tmp_path, monkeypatch):
    reg = ToolRegistry(tools=[RECALL_TOOL])
    monkeypatch.setattr("dla.tools.registry.discover_all",
                        lambda: [(Tool(name="x", description="x", parameters={}, run=lambda a, c: ToolResult(ok=True)), "x", None)])
    d = tmp_path / "plugins"
    d.mkdir()
    (d / "a.py").write_text("# v1", encoding="utf-8")
    watcher = HotReloadWatcher(reg, str(d), mode="shadow")
    watcher.scan_once()  # 建基线
    (d / "a.py").write_text("# v2", encoding="utf-8")
    res = reg.reload(shadow=True)
    assert res["ok"] and res["shadow"]
    assert "x" not in reg.snapshot()  # 影子未生效
    assert reg.has_shadow() is True


# ---- LLM function-calling 二级路由（doc/08 §4.3）----
def test_parse_tool_calls_syntax():
    e = _engine_no_db(ToolRegistry(tools=[RECALL_TOOL]))
    text = '先看记忆<tool_call name="recall_memory" args=\'{"query":"母语"}\' />然后作答'
    calls = e._parse_tool_calls(text)
    assert calls == [("recall_memory", {"query": "母语"})]


def test_parse_tool_calls_invalid_json_skipped():
    e = _engine_no_db(ToolRegistry(tools=[RECALL_TOOL]))
    text = '<tool_call name="recall_memory" args=\'not-json\' />'
    # 非 JSON 也兼容为 query 字符串
    calls = e._parse_tool_calls(text)
    assert calls and calls[0][0] == "recall_memory"


def test_run_tool_calls_dispatches():
    called = {}

    def _run(args, ctx):
        called["args"] = args
        return ToolResult(ok=True, content="TOOL_RESULT")

    tool = Tool(name="calc", description="计算器",
                parameters={"type": "object", "properties": {"expr": {"type": "string"}}},
                run=_run)
    e = _engine_no_db(ToolRegistry(tools=[tool]))
    e.start_session(sid="A")
    sess = e._sessions["A"]
    results = e._run_tool_calls(sess, [("calc", {"expr": "1+1"})], set())
    assert results[0][1].ok and "TOOL_RESULT" in results[0][1].content
    assert called["args"] == {"expr": "1+1"}


def test_dangerous_tool_rejected_unless_enabled():
    def _run(args, ctx):
        return ToolResult(ok=True, content="x")

    tool = Tool(name="rm", description="删除", parameters={}, run=_run, is_readonly=False)
    e = _engine_no_db(ToolRegistry(tools=[tool]))
    e.start_session(sid="A")
    sess = e._sessions["A"]
    results = e._run_tool_calls(sess, [("rm", {})], set())
    assert not results[0][1].ok and "enable" in (results[0][1].error or "")
    e.tool_registry.set_enabled("rm", True)
    results = e._run_tool_calls(sess, [("rm", {})], set())
    assert results[0][1].ok


def test_run_tool_calls_dedup_with_auto_invoked():
    called = {"n": 0}

    def _run(args, ctx):
        called["n"] += 1
        return ToolResult(ok=True)

    tool = Tool(name="t", description="t", parameters={}, run=_run)
    e = _engine_no_db(ToolRegistry(tools=[tool]))
    e.start_session(sid="A")
    sess = e._sessions["A"]
    # 该工具已在本次轮次自动触发过 → 二级路由应跳过，避免重复调用
    e._run_tool_calls(sess, [("t", {})], {"t"})
    assert called["n"] == 0


# ---- recall_memory 语义召回（经引擎 call_tool）----
def test_recall_memory_semantic_via_engine(tmp_path):
    e = _engine(tmp_path)
    e.start_session(sid="A")
    e.send("我母语是粤语，平时在家都说粤语。")

    res = e.call_tool("recall_memory", {"query": "我的母语是什么"}, session_id="A")
    assert res.ok, res.error
    assert res.metadata.get("mode") == "semantic"
    assert "粤语" in res.content


def test_recall_memory_no_query_falls_back_to_summary(tmp_path):
    e = _engine(tmp_path)
    e.start_session(sid="B")
    e.send("今天练了过去式。")
    res = e.call_tool("recall_memory", {}, session_id="B")
    assert res.ok
    assert res.metadata.get("mode") == "summary"


def test_call_unknown_tool_errors(tmp_path):
    e = _engine(tmp_path)
    res = e.call_tool("no_such_tool", {}, session_id="X")
    assert not res.ok
    assert "未找到工具" in (res.error or "")


def test_tool_call_is_logged(tmp_path):
    e = _engine(tmp_path)
    e.start_session(sid="C")
    e.send("记一下：我喜欢用例子学。")
    e.call_tool("recall_memory", {"query": "例子"}, session_id="C")
    logs = e.repo.conn.execute("SELECT tool, ok FROM tool_log WHERE session_id='C'").fetchall()
    assert any(r["tool"] == "recall_memory" for r in logs)


def test_engine_auto_invokes_recall_memory_on_remember_question(tmp_path):
    e = _engine(tmp_path)
    e.start_session(sid="A")
    e.send("我母语是粤语，平时在家都说粤语。")

    captured = {}
    orig = e.assembler.assemble

    def spy(snapshot, **kw):
        captured.update(kw)
        return orig(snapshot, **kw)

    e.assembler.assemble = spy

    e.start_session(sid="B")
    e.send("你还记得我之前提过的母语吗？")

    detail = captured.get("detail_blocks") or []
    assert detail, "工具自动触发未注入 detail_blocks"
    assert any("粤语" in d for d in detail), f"recall_memory 未自动召回粤语相关记忆：{detail}"
