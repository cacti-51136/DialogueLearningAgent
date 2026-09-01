"""记忆子系统引擎集成测试（doc/07 §9 集成）。

验证：引擎在开启记忆后，能跨会话召回上一次对话中的关键事实，并将其作为「背景参考」
注入到下一轮 Prompt（经 PromptAssembler 的 cold_memory 参数）。使用零依赖确定性 embedding，
不联网。
"""

import os
import tempfile

from dla.config.loader import get_keyword_lib
from dla.config.settings import Settings
from dla.llm.openai_compat import make_llm_client
from dla.orchestration.engine import DialogueEngine
from dla.storage.migrator import migrate
from dla.storage.repositories import SQLiteRepo
from dla.storage.sqlite import get_connection


def _make_engine(tmp_path) -> DialogueEngine:
    s = Settings()
    # 强制每轮检索 + 阈值 0，保证跨会话召回路径被触发（与具体相似度无关）
    s.memory_retrieve_trigger = "always"
    s.memory_retrieve_sim_threshold = 0.0
    lib = get_keyword_lib()
    llm = make_llm_client("", "https://api.openai.com/v1", "gpt-4o-mini")
    db = os.path.join(str(tmp_path), "mem.db")
    conn = get_connection(db)
    migrate(conn, "migrations")
    repo = SQLiteRepo(conn)
    return DialogueEngine(s, lib, llm, repo)


def test_engine_builds_memory_store(tmp_path):
    eng = _make_engine(tmp_path)
    assert eng.memory is not None  # 有 repo 即应建起冷库


def test_engine_memory_none_when_repo_absent(tmp_path):
    # 无 repo（内存模式）→ 记忆关闭，主流程不受影响
    s = Settings()
    lib = get_keyword_lib()
    llm = make_llm_client("", "https://api.openai.com/v1", "gpt-4o-mini")
    eng = DialogueEngine(s, lib, llm, None)
    assert eng.memory is None
    reply, _ = eng.send("你好")
    assert reply


def test_cross_session_recall_injects_cold_memory(tmp_path):
    eng = _make_engine(tmp_path)

    # 会话 A：陈述一个稳定事实
    eng.start_session(sid="A")
    eng.send("我母语是粤语，平时在家都说粤语。")

    # 会话 B：新会话下问起，应能从冷库召回「粤语」相关记忆并注入 Prompt
    eng.start_session(sid="B")
    captured = {}

    orig = eng.assembler.assemble

    def spy(snapshot, **kw):
        captured.update(kw)
        return orig(snapshot, **kw)

    eng.assembler.assemble = spy

    eng.send("你还记得我之前提过的母语吗？")

    cold = captured.get("cold_memory") or []
    assert cold, "跨会话检索未产出冷记忆注入"
    assert any("粤语" in c for c in cold), f"冷记忆未召回粤语相关事实：{cold}"

    # 反向校验：会话 B 检索应命中会话 A 写入的记忆
    hits = eng.memory.search("你还记得我之前提过的母语吗？", scope="all")
    assert hits, "冷库未跨会话召回"
    assert any("粤语" in (h.summary or h.text) for h in hits)
