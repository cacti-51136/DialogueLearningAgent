"""集成测试：auto 模式 warmup 后场景锁定（doc/02 §3.5）+ LVM 在线学习接线（doc/06）。

全部离线 FakeLLM 驱动，不触网；numpy 缺失时整文件 skip（LVM 强依赖 numpy）。
覆盖：
- auto 模式 warmup 后 L1 锁定（l1_locked + src_conf 升到 mode_auto_lock_conf，不再漂移）
- LVM 训练后 p_agent 经 γ 注入并改变 L3 权重分布（零回归门控：未训练不注入）
- LVM 状态跨会话持久化（同一 db 重建引擎恢复 has_learned）
- dla lvm reset 经 repo 清空状态
"""
import json
import os

import pytest

np = pytest.importorskip("numpy")

from dla.config.settings import get_settings
from dla.core.models import Layer
from dla.core.ports import LlmResult
from dla.llm.openai_compat import FakeLLMClient
from dla.orchestration.engine import DialogueEngine
from dla.storage.migrator import migrate
from dla.storage.repositories import SQLiteRepo
from dla.storage.sqlite import get_connection

MIGRATIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "migrations")


class _OpsFakeLLM(FakeLLMClient):
    """每轮分析注入给定 scene_ops / agent_ops（其余走真实 Fake 逻辑）。"""

    def __init__(self, scene_ops=None, agent_ops=None, **kw):
        super().__init__(**kw)
        self._scene_ops = scene_ops or []
        self._agent_ops = agent_ops or []

    def _fake_analyze(self, text):
        res = super()._fake_analyze(text)
        data = json.loads(res.content)
        data["scene_ops"] = self._scene_ops
        data["agent_ops"] = self._agent_ops
        return LlmResult(content=json.dumps(data, ensure_ascii=False))


def _make_engine(lib, scenario_dir, tmp_path, llm, db_name="t.db", enable_lvm=False):
    settings = get_settings()
    settings.scenario_dir = scenario_dir
    settings.lvm_enabled = enable_lvm
    db = str(tmp_path / db_name)
    conn = get_connection(db)
    migrate(conn, MIGRATIONS_DIR)
    repo = SQLiteRepo(conn)
    eng = DialogueEngine(settings, lib, llm, repo)
    return eng, repo, conn


# ---------------------------------------------------------------------------
# auto 模式 warmup 后场景锁定（doc/02 §3.5）
# ---------------------------------------------------------------------------
def test_auto_mode_l1_locked_after_warmup(lib, scenario_dir, tmp_path):
    """warmup 后累积的 auto_added 场景词升到 mode_auto_lock_conf 并锁定，不再漂移。"""
    l1k = lib.lexicon.for_layer(Layer.L1)[0].key
    llm = _OpsFakeLLM(scene_ops=[{"op": "add", "key": l1k, "intensity": 0.9, "reason": "练口语"}])
    eng, _repo, _ = _make_engine(lib, scenario_dir, tmp_path, llm)

    eng.start_session(mode="auto", sid="AL1")
    # 默认 mode_auto_warmup_turns=3，连发 4 轮确保越过 warmup 且 analyze 已运行累积 L1
    for msg in ("你好。", "我想练口语。", "继续练。", "再加点口语强度。"):
        eng.send(msg)

    sess = eng._sessions["AL1"]
    assert sess.l1_locked is True
    assert sess.engine.src_conf(l1k) == pytest.approx(0.8)
    # 锁定后再 add 同类场景词，置信度应维持锁定值（不回落到 0.5 推断值）
    eng.send("口语多练。")
    assert sess.engine.src_conf(l1k) == pytest.approx(0.8)


def test_fixed_mode_never_locks_l1(lib, scenario_dir, tmp_path):
    """固定模式不应触发 auto 锁定逻辑。"""
    l1k = lib.lexicon.for_layer(Layer.L1)[0].key
    llm = _OpsFakeLLM(scene_ops=[{"op": "add", "key": l1k, "intensity": 0.9}])
    eng, _repo, _ = _make_engine(lib, scenario_dir, tmp_path, llm)
    eng.start_session(mode="fixed", scenario_id="oral_practice", sid="FX1")
    eng.send("你好。")
    eng.send("我想练口语。")
    assert eng._sessions["FX1"].l1_locked is False


# ---------------------------------------------------------------------------
# LVM 在线学习接线（doc/06）
# ---------------------------------------------------------------------------
def _run_lvm_session(lib, scenario_dir, tmp_path, agent_ops, db_name, enable_lvm):
    llm = _OpsFakeLLM(agent_ops=agent_ops)
    eng, repo, conn = _make_engine(lib, scenario_dir, tmp_path, llm, db_name=db_name, enable_lvm=enable_lvm)
    eng.start_session(mode="auto", sid="LVM")
    for _ in range(6):
        eng.send("我想被鼓励一下。")
    return eng


def test_lvm_p_agent_injected_and_changes_l3(lib, scenario_dir, tmp_path):
    """LVM 训练后：p_agent 经 γ 注入并改变 L3 分布；未训练不注入（零回归门控）。"""
    l3k = lib.lexicon.for_layer(Layer.L3)[0].key
    agent_ops = [{"op": "add", "key": l3k, "intensity": 1.0, "reason": "需更鼓励"}]

    eng_off = _run_lvm_session(lib, scenario_dir, tmp_path, agent_ops, "off.db", enable_lvm=False)
    eng_on = _run_lvm_session(lib, scenario_dir, tmp_path, agent_ops, "on.db", enable_lvm=True)

    # 训练已发生
    assert eng_on.lvm.has_learned() is True
    sess_on = eng_on._sessions["LVM"]

    # p_agent 被注入且非均匀（学习到的 M 使其结构化）
    pa = sess_on.engine._p_agent
    assert pa, "已训练后应注入 p_agent"
    vals = list(pa.values())
    assert max(vals) - min(vals) > 1e-4

    # L3 因 γ 融合而不同于无 LVM 情形
    l3_off = eng_off._sessions["LVM"].engine.compute_all(1).l3
    l3_on = sess_on.engine.compute_all(1).l3
    dist = sum(abs(l3_off[k] - l3_on[k]) for k in l3_off)
    assert dist > 1e-4, f"LVM 注入未改变 L3 分布，dist={dist}"

    # p_agent 呈结构化（非均匀）→ 说明本地模型确实从 agent_ops 中学到了偏好，
    # 而不是均匀先验；这才是 LVM 在线学习在引擎闭环中生效的核心证据。
    # 注：端到端 6 轮在 d=64 随机嵌入下未必把"目标 L3 词"顶到最高（单样本、维度高），
    # 故只断言先验被塑形（非均匀）+ 已改变 L3 分布，不做单点抬高断言。


def test_lvm_not_injected_before_training(lib, scenario_dir, tmp_path):
    """零回归门控：has_learned()=False 时绝不注入 p_agent，避免均匀先验扰动既有权重。"""
    settings = get_settings()
    settings.lvm_enabled = True
    llm = FakeLLMClient(model=settings.llm_model)  # 无任何 agent_ops → 不训练
    eng, _repo, _ = _make_engine(lib, scenario_dir, tmp_path, llm, db_name="nope.db", enable_lvm=True)
    eng.start_session(mode="auto", sid="NOPE")
    eng.send("你好。")  # 单轮：未训练也无 ops
    assert eng.lvm.has_learned() is False
    assert eng._sessions["NOPE"].engine._p_agent == {}


# ---------------------------------------------------------------------------
# 持久化 / 复位
# ---------------------------------------------------------------------------
def test_lvm_persists_across_sessions(lib, scenario_dir, tmp_path):
    """同一 db 重建引擎应恢复已训练状态（doc-06 §6.5 跨会话个性化）。"""
    settings = get_settings()
    settings.lvm_enabled = True
    db = str(tmp_path / "persist.db")
    l3k = lib.lexicon.for_layer(Layer.L3)[0].key
    agent_ops = [{"op": "add", "key": l3k, "intensity": 1.0}]
    llm = _OpsFakeLLM(agent_ops=agent_ops)

    conn = get_connection(db)
    migrate(conn, MIGRATIONS_DIR)
    repo = SQLiteRepo(conn)
    eng1 = DialogueEngine(settings, lib, llm, repo)
    eng1.start_session(mode="auto", sid="P1")
    for _ in range(4):
        eng1.send("鼓励我。")
    assert eng1.lvm.has_learned()
    step_after_train = eng1.lvm.step

    # 重建引擎（同一 db）
    conn2 = get_connection(db)
    repo2 = SQLiteRepo(conn2)
    eng2 = DialogueEngine(settings, lib, llm, repo2)
    assert eng2.lvm is not None
    assert eng2.lvm.has_learned() is True
    assert eng2.lvm.step == step_after_train


def test_lvm_reset_clears_state(lib, scenario_dir, tmp_path):
    """dla lvm reset：清空关系头/步数，has_learned 回到 False。"""
    settings = get_settings()
    settings.lvm_enabled = True
    l3k = lib.lexicon.for_layer(Layer.L3)[0].key
    agent_ops = [{"op": "add", "key": l3k, "intensity": 1.0}]
    llm = _OpsFakeLLM(agent_ops=agent_ops)
    eng, repo, _ = _make_engine(lib, scenario_dir, tmp_path, llm, db_name="reset.db", enable_lvm=True)
    eng.start_session(mode="auto", sid="R1")
    for _ in range(4):
        eng.send("鼓励我。")
    assert eng.lvm.has_learned()

    eng.learner.reset()
    assert eng.lvm.has_learned() is False
    assert eng.lvm.step == 0
    assert repo.load_lvm_heads() is None


def test_cli_lvm_reset_invokes_repo(lib, scenario_dir, tmp_path, monkeypatch):
    """apps/cli 的 `dla lvm reset` 应清空 LVM 表（冒烟验证分发路径）。"""
    import io
    from contextlib import redirect_stdout

    import apps.cli.main as cli_main
    from apps.cli.main import build_parser, cmd_lvm

    settings = get_settings()
    settings.lvm_enabled = True
    db = str(tmp_path / "clireset.db")
    l3k = lib.lexicon.for_layer(Layer.L3)[0].key
    agent_ops = [{"op": "add", "key": l3k, "intensity": 1.0}]
    llm = _OpsFakeLLM(agent_ops=agent_ops)
    eng, repo, _ = _make_engine(lib, scenario_dir, tmp_path, llm, db_name="clireset.db", enable_lvm=True)
    eng.start_session(mode="auto", sid="CLI1")
    for _ in range(3):
        eng.send("鼓励我。")
    assert eng.lvm.has_learned()

    # cmd_lvm 内部的 get_settings 来自 src.dla.config.settings（与测试侧的 dla.config.settings
    # 是两个独立单例），故直接打补丁让 CLI 的 get_settings 指向临时库，验证 `dla lvm reset` 分发。
    class _CliSettings:
        db_path = db

    monkeypatch.setattr(cli_main, "get_settings", lambda: _CliSettings())

    parser = build_parser()
    args = parser.parse_args(["lvm", "reset"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_lvm(args)
    # 用全新连接核验：临时库 LVM 状态已被清空
    fresh = SQLiteRepo(get_connection(db))
    assert fresh.load_lvm_heads() is None
