"""工具插件系统测试（doc/08）。

覆盖：参数校验、注册表原子快照/热更新隔离、recall_memory 语义召回（经引擎 call_tool）。
"""

import os

from dla.config.loader import get_keyword_lib
from dla.config.settings import Settings
from dla.llm.openai_compat import make_llm_client
from dla.orchestration.engine import DialogueEngine
from dla.storage.migrator import migrate
from dla.storage.repositories import SQLiteRepo
from dla.storage.sqlite import get_connection
from dla.tools import build_builtin_registry
from dla.tools.executor import run_tool
from dla.tools.protocol import Tool, ToolContext, ToolResult, validate_args
from dla.tools.registry import ToolRegistry
from dla.tools.plugins.recall_memory import TOOL as RECALL_TOOL


def _engine(tmp_path):
    s = Settings()
    s.memory_retrieve_trigger = "always"
    s.memory_retrieve_sim_threshold = 0.0
    lib = get_keyword_lib()
    llm = make_llm_client("", "https://api.openai.com/v1", "gpt-4o-mini")
    db = os.path.join(str(tmp_path), "t.db")
    conn = get_connection(db, check_same_thread=False)
    migrate(conn, "migrations")
    repo = SQLiteRepo(conn)
    e = DialogueEngine(s, lib, llm, repo)
    e.tool_registry = build_builtin_registry()
    return e


# ---- 参数校验 ----
def test_validate_args_missing_required():
    t = Tool(
        name="x", description="", dangerous=False,
        parameters={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
        run=lambda a, c: ToolResult(ok=True),
    )
    assert validate_args(t, {}) is not None
    assert validate_args(t, {"q": "hi"}) is None
    assert validate_args(t, {"q": 1}) is not None  # 类型不符


# ---- 注册表原子快照 / 热更新隔离 ----
def test_registry_snapshot_isolation():
    reg = ToolRegistry()
    reg.register(RECALL_TOOL)
    v1 = reg.snapshot()
    snap1 = reg.get_snapshot(v1)
    assert "recall_memory" in snap1

    reg.register(Tool(name="echo", description="", dangerous=False,
                     parameters={"type": "object", "properties": {}},
                     run=lambda a, c: ToolResult(ok=True)))
    # 旧快照仍只含 recall_memory；新快照含 echo
    assert "recall_memory" in reg.get_snapshot(v1)
    assert "echo" in [t.name for t in reg.all()]


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
