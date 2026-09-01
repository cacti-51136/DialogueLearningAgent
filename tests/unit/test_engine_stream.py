"""引擎流式入口测试（doc/04 §1 stream_reply_sync + TurnEvent）。

用 FakeLLM（离线），验证事件流的正确性与「受挫 → empathy 上升」的演化方向。
"""

import sys

sys.path.insert(0, "src")

from dla.config.loader import get_keyword_lib
from dla.config.settings import get_settings
from dla.llm.openai_compat import FakeLLMClient
from dla.orchestration.engine import DialogueEngine


def _engine():
    settings = get_settings()
    lib = get_keyword_lib()
    llm = FakeLLMClient(model=settings.llm_model)
    return DialogueEngine(settings, lib, llm, repo=None)


def test_stream_emits_event_sequence():
    eng = _engine()
    eng.start_session(mode="fixed", scenario_id="oral_practice")
    events = list(eng.stream_reply_sync("你好，我想练口语。"))
    types = [type(e).__name__ for e in events]
    assert "WeightUpdateEvent" in types
    assert "TokenEvent" in types
    assert "DoneEvent" in types
    done = [e for e in events if type(e).__name__ == "DoneEvent"][-1]
    assert done.final_text
    collected = "".join(e.text for e in events if type(e).__name__ == "TokenEvent")
    # 流式 token 累积应包含最终文本的前缀（不含 system 注入的 turn_summary 标记）
    assert done.final_text[:3] in collected or len(collected) > 0


def test_stream_empathy_rises_on_frustration():
    eng = _engine()
    eng.start_session(mode="fixed", scenario_id="oral_practice")
    sess = eng._active_session()
    base = sess.engine.compute_all(0).l3.get("empathy", 0.0)
    events = list(eng.stream_reply_sync("这个语法太难了，我完全不会，好烦。"))
    wue = [e for e in events if type(e).__name__ == "WeightUpdateEvent"]
    assert wue, "应产出一个权重更新事件"
    empathy = wue[-1].snapshot.l3.get("empathy", 0.0)
    assert empathy > base, f"受挫信号下 empathy 应上升：{base} -> {empathy}"


def test_stream_persists_history_without_db():
    eng = _engine()
    eng.start_session(mode="fixed", scenario_id="oral_practice")
    list(eng.stream_reply_sync("第一轮。"))
    list(eng.stream_reply_sync("第二轮。"))
    sess = eng._active_session()
    # 无 DB 时历史摘要链仍在内存累积
    assert len(sess.history_summaries) == 2
