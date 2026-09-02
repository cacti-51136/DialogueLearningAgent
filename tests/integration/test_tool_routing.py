"""LLM function-calling 二级路由端到端（doc/08 §4.3）。

用会「先发 <tool_call> 再发终稿」的 FakeLLM，验证：引擎在 send 中解析 tool_call → 经注册表派发执行
recall_memory → 把结果回灌后重生成，且 TOOL_MAX_LOOPS 不失控。
"""

import os

from dla.config.loader import get_keyword_lib
from dla.config.settings import Settings
from dla.llm.openai_compat import FakeLLMClient, LlmResult
from dla.orchestration.engine import DialogueEngine
from dla.storage.migrator import migrate
from dla.storage.repositories import SQLiteRepo
from dla.storage.sqlite import get_connection
from dla.tools import build_builtin_registry


class ToolCallFakeLLM(FakeLLMClient):
    """每次 send 的「首次生成」返回带 tool_call 的回复，重生成返回终稿。

    用「奇数次 complete = 本轮首次生成」来区分：seed send 与待测 send 各自都会先发 tool_call。
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.calls = 0

    def complete(self, messages, **kw):
        self.calls += 1
        if self.calls % 2 == 1:
            return LlmResult(
                content='让我查一下记忆。<tool_call name="recall_memory" args=\'{"query":"母语"}\' />'
            )
        return LlmResult(content="你母语是粤语，我记得的。<turn_summary>已回忆母语</turn_summary>")


def _engine(tmp_path):
    s = Settings()
    s.memory_retrieve_trigger = "always"
    s.memory_retrieve_sim_threshold = 0.0
    s.tools_enabled = True
    s.analysis_cold_start_turns = 0     # 关闭分析，避免 analyzer 占用 LLM 调用干扰路由计数
    s.analysis_period = 10 ** 9
    lib = get_keyword_lib()
    llm = ToolCallFakeLLM(model=s.llm_model)
    db = os.path.join(str(tmp_path), "t.db")
    conn = get_connection(db, check_same_thread=False)
    migrate(conn, "migrations")
    repo = SQLiteRepo(conn)
    e = DialogueEngine(s, lib, llm, repo, tool_registry=build_builtin_registry())
    return e, llm


def test_llm_second_level_routing_dispatches_tool(tmp_path):
    e, llm = _engine(tmp_path)
    e.start_session(sid="A")
    e.send("我母语是粤语，平时在家都说粤语。")  # 落冷记忆

    reply, meta = e.send("请帮我回忆一下我的母语相关信息。")
    # 二级路由应已派发 recall_memory
    frames = dict(meta["debug_chain"])
    assert "tool_calls" in frames, "未触发二级路由 tool_calls 帧"
    executed = frames["tool_calls"]["executed"]
    assert ("recall_memory", True) in executed, f"recall_memory 未成功执行：{executed}"
    # 重生成后回复非空，且本轮 send 不超过 1（首次）+ TOOL_MAX_LOOPS(默认2) 次重生成
    assert reply
    assert llm.calls <= 4, f"调用次数异常：calls={llm.calls}"


def test_cli_entrypoint_connection_tolerates_threaded_tools(tmp_path, monkeypatch):
    """回归护栏：CLI 入口创建的连接必须允许跨线程工具调用。

    工具执行器（executor.run_tool）**恒在独立线程**运行工具，而工具（recall_memory）
    会经 ctx.repo/ctx.memory 访问同一个 sqlite 连接。若入口用默认的
    ``check_same_thread=True`` 建连，每次工具调用都会抛
    "SQLite objects created in a thread can only be used in that same thread"，
    并被 "仅 ok 才追加" 的逻辑静默吞掉（表现为 debug 链 invoked=[] 且无任何报错）。

    本测试直接复用 CLI 的 ``_build_engine``，确保该入口与其他入口（api/ui）一致。
    """
    # 让 CLI 使用临时库，避免污染开发库
    monkeypatch.setenv("DLA_DB__PATH", os.path.join(str(tmp_path), "cli.db"))
    from dla.config.settings import get_settings

    get_settings(reload=True)
    try:
        from types import SimpleNamespace

        import apps.cli.main as cli

        _s, _lib, _llm, repo, engine, _reg = cli._build_engine(
            SimpleNamespace(no_db=False), with_db=True, force_fake=True
        )
        engine.start_session(sid="CLI")
        engine.send("我母语是粤语，平时在家都说粤语。")  # 落冷记忆

        sess = engine._active_session()
        _details, _schema, invoked = engine._run_tool_step(sess, "你还记得我之前说过什么吗？")
        assert invoked == ["recall_memory"], f"CLI 路径下工具未被触发（连接线程亲和性问题？）：{invoked}"

        res = engine.call_tool("recall_memory", {"query": "母语"})
        assert res.ok, f"CLI 路径下工具执行失败：{res.error}"
    finally:
        get_settings(reload=True)


def test_llm_second_level_routing_respects_max_loops(tmp_path):
    # 终稿也含 tool_call 时，必须被 TOOL_MAX_LOOPS 截断，不无限循环
    class LoopFakeLLM(ToolCallFakeLLM):
        def complete(self, messages, **kw):
            self.calls += 1
            return LlmResult(content='<tool_call name="recall_memory" args=\'{"query":"x"}\' />')

    s = Settings()
    s.memory_retrieve_trigger = "always"
    s.tools_enabled = True
    s.analysis_cold_start_turns = 0     # 关闭分析
    s.analysis_period = 10 ** 9
    lib = get_keyword_lib()
    llm = LoopFakeLLM(model=s.llm_model)
    db = os.path.join(str(tmp_path), "t.db")
    conn = get_connection(db, check_same_thread=False)
    migrate(conn, "migrations")
    repo = SQLiteRepo(conn)
    e = DialogueEngine(s, lib, llm, repo, tool_registry=build_builtin_registry())
    e.start_session(sid="A")
    e.send("我母语是粤语。")  # 首次 send 也会触发路由，不计入本轮度量
    llm.calls = 0
    e.send("再查一次。")
    # 单次 send 不应超过 1（首次生成）+ TOOL_MAX_LOOPS(默认2) 次重生成
    assert llm.calls <= 1 + s.memory_tool_max_loops, f"循环未被截断：calls={llm.calls}"
