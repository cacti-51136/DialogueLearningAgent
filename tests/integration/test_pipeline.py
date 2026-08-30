"""集成测试：FakeLLM 驱动完整对话闭环（分析→权重→落库→kw_agent_map 涌现）。"""
import os

from dla.config.settings import get_settings
from dla.llm.openai_compat import FakeLLMClient
from dla.orchestration.engine import DialogueEngine
from dla.storage.migrator import migrate
from dla.storage.repositories import SQLiteRepo
from dla.storage.sqlite import get_connection


def test_empathy_rises_and_kwmap_emerges(lib, scenario_dir, tmp_path):
    settings = get_settings()
    # 测试内强制绝对路径，避免 cwd 依赖
    settings.scenario_dir = scenario_dir
    llm = FakeLLMClient(model=settings.llm_model)
    db = str(tmp_path / "session.db")
    conn = get_connection(db)
    migrate(conn, os.path.join(os.path.dirname(__file__), "..", "..", "migrations"))
    repo = SQLiteRepo(conn)

    eng = DialogueEngine(settings, lib, llm, repo)
    eng.start_session(mode="fixed", scenario_id="oral_practice")

    base = eng.send("你好，我想练口语。")[1]["snapshot"].l3["empathy"]
    emps = [base]
    for m in [
        "这个语法太难了，我完全不会。",
        "我有点受挫，感觉学不会。",
        "还是不太懂，能不能简单点。",
    ]:
        emps.append(eng.send(m)[1]["snapshot"].l3["empathy"])

    # 受挫信号使共情度上升
    assert max(emps) > base

    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM kw_agent_map WHERE src_keyword='user_mood.frustrated'"
    )
    assert cur.fetchone()[0] >= 1
    # 不应出现由 agent 回复「别急」误抽出的 impatient 映射
    cur.execute(
        "SELECT COUNT(*) FROM kw_agent_map WHERE src_keyword='user_temper.impatient'"
    )
    assert cur.fetchone()[0] == 0
    # 持久化基本表都有记录
    cur.execute("SELECT COUNT(*) FROM messages"); assert cur.fetchone()[0] >= 2
    cur.execute("SELECT COUNT(*) FROM turn_summaries"); assert cur.fetchone()[0] >= 1
