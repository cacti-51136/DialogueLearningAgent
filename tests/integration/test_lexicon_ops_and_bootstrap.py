"""集成测试：#36 free 模式 Bootstrap 端到端（纯 StubLLM，避开真实限流）+
#37 scene_ops/agent_ops 消费与 lexicon_ops 审计。

全部用离线 StubLLM / FakeLLM 驱动，不触网、不依赖真实 LLM；可独立复跑。
"""
import json
import os

from dla.config.settings import get_settings
from dla.core.models import Layer
from dla.core.ports import LlmResult
from dla.llm.openai_compat import FakeLLMClient
from dla.orchestration.engine import DialogueEngine
from dla.storage.migrator import migrate
from dla.storage.repositories import SQLiteRepo
from dla.storage.sqlite import get_connection

MIGRATIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "migrations")


def _make_engine(lib, scenario_dir, tmp_path, llm):
    settings = get_settings()
    settings.scenario_dir = scenario_dir
    db = str(tmp_path / "t.db")
    conn = get_connection(db)
    migrate(conn, MIGRATIONS_DIR)
    repo = SQLiteRepo(conn)
    eng = DialogueEngine(settings, lib, llm, repo)
    return eng, repo, conn


# ---------------------------------------------------------------------------
# #36 · free 模式 Bootstrap（纯 StubLLM）
# ---------------------------------------------------------------------------
class _FreeStubLLM:
    """离线 Stub：区分 Bootstrap 调用 / 开场称呼生成 / 普通对话。"""

    def __init__(self, describe: str, seeds: dict) -> None:
        self.describe = describe
        self.seeds = seeds

    def complete(self, messages, **kw):
        sys = messages[0].content if messages else ""
        if "关键词引导器" in sys:
            return LlmResult(content=json.dumps(self.seeds, ensure_ascii=False))
        if "开场称呼生成器" in sys:
            return LlmResult(content="你好，我们开始练习吧～")
        return LlmResult(content="收到，我们继续。\n\n<turn_summary>练习中</turn_summary>")


def test_free_mode_bootstrap_injects_three_layer_seeds(lib, scenario_dir, tmp_path):
    lex = lib.lexicon
    l1k = lex.for_layer(Layer.L1)[0].key
    l2k = lex.for_layer(Layer.L2)[0].key
    l3k = lex.for_layer(Layer.L3)[0].key
    seeds = {
        "l1": [{"key": l1k, "intensity": 0.9}],
        "l2": [{"key": l2k, "intensity": 0.7}],
        "l3": [{"key": l3k, "intensity": 0.8}],
    }
    llm = _FreeStubLLM(describe="我想练英语口语", seeds=seeds)
    eng, repo, _ = _make_engine(lib, scenario_dir, tmp_path, llm)

    greeting = eng.start_session(mode="free", describe="我想练英语口语", sid="F1")
    sess = eng._sessions["F1"]
    snap = sess.engine.compute_all(0)

    # 三层种子真注入引擎
    assert l1k in snap.l1 and snap.l1[l1k] > 0
    assert l2k in snap.l2 and snap.l2[l2k] > 0
    assert l3k in snap.l3 and snap.l3[l3k] > 0
    # L1/L2 种子带 bootstrap source 置信度 0.6（doc/02 §3.2）
    assert sess.engine.src_conf(l1k) == 0.6
    assert sess.engine.src_conf(l2k) == 0.6
    # 开场称呼由 LLM 生成且非空
    assert isinstance(greeting, str) and greeting.strip()
    # Bootstrap 种子被记录
    assert sess.bootstrap_seeds is not None
    assert sess.bootstrap_seeds.total >= 3


def test_free_mode_degrade_to_auto_without_desc(lib, scenario_dir, tmp_path):
    """缺描述 + REQUIRE_DESC=true → 静默降级 auto，不抛错（doc/02 §3.6）。"""
    settings = get_settings()
    settings.mode_free_require_desc = True
    llm = FakeLLMClient(model=settings.llm_model)
    eng, repo, _ = _make_engine(lib, scenario_dir, tmp_path, llm)
    greeting = eng.start_session(mode="free", describe="", sid="F2")
    sess = eng._sessions["F2"]
    assert sess.mode == "auto"
    assert "降级" in "".join(sess.bootstrap_notes)
    assert isinstance(greeting, str)


# ---------------------------------------------------------------------------
# #36 · bootstrap.py 护栏单测
# ---------------------------------------------------------------------------
class _BootstrapStubLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def complete(self, messages, **kw):
        return LlmResult(content=json.dumps(self.payload, ensure_ascii=False))


def _bootstrap_with(lib, payload: dict):
    llm = _BootstrapStubLLM(payload)
    return __import__("dla.analysis.bootstrap", fromlist=["bootstrap"]).bootstrap(
        llm, "测试场景描述", lib.lexicon
    )


def test_bootstrap_accepts_valid_three_layer(lib):
    lex = lib.lexicon
    payload = {
        "l1": [{"key": lex.for_layer(Layer.L1)[0].key, "intensity": 0.9}],
        "l2": [{"key": lex.for_layer(Layer.L2)[0].key, "intensity": 0.7}],
        "l3": [{"key": lex.for_layer(Layer.L3)[0].key, "intensity": 0.8}],
    }
    seeds = _bootstrap_with(lib, payload)
    assert seeds.total == 3
    assert not seeds.rejected


def test_bootstrap_rejects_forbidden_temper_mood(lib):
    """user_temper.* / user_mood.* 禁止由 Bootstrap 预设（doc/02 §11.9）。"""
    payload = {"l1": [], "l2": [{"key": "user_temper.impatient", "intensity": 0.5}], "l3": []}
    seeds = _bootstrap_with(lib, payload)
    assert "user_temper.impatient" not in seeds.l2
    assert any(r["reason"] == "forbidden_dimension" for r in seeds.rejected)


def test_bootstrap_rejects_layer_mismatch(lib):
    """层归属必须匹配：L1 的词混进 L2 应被丢弃。"""
    l1k = lib.lexicon.for_layer(Layer.L1)[0].key
    payload = {"l1": [], "l2": [{"key": l1k, "intensity": 0.5}], "l3": []}
    seeds = _bootstrap_with(lib, payload)
    assert l1k not in seeds.l2
    assert any(r["reason"] == "layer_mismatch" for r in seeds.rejected)


def test_bootstrap_rejects_out_of_lexicon(lib):
    """词表外关键词不收编，记入 raw_unknown 反哺词库。"""
    payload = {"l1": [], "l2": [], "l3": [{"key": "totally.unknown.word", "intensity": 0.5}]}
    seeds = _bootstrap_with(lib, payload)
    assert "totally.unknown.word" in seeds.raw_unknown
    assert any(r["reason"] == "not_in_lexicon" for r in seeds.rejected)


# ---------------------------------------------------------------------------
# #37 · scene_ops(L1) / agent_ops(L3) 消费 + lexicon_ops 审计
# ---------------------------------------------------------------------------
class _OpsFakeLLM(FakeLLMClient):
    """在每轮分析里注入给定 scene_ops / agent_ops（其余走真实 Fake 逻辑）。"""

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


def test_scene_ops_add_applied_and_audited(lib, scenario_dir, tmp_path):
    """auto 模式：scene_ops add 生效并落 lexicon_ops（applied=1, layer=L1）。"""
    l1k = lib.lexicon.for_layer(Layer.L1)[0].key
    llm = _OpsFakeLLM(scene_ops=[{"op": "add", "key": l1k, "intensity": 0.9, "reason": "用户提到练口语"}])
    eng, repo, _ = _make_engine(lib, scenario_dir, tmp_path, llm)

    eng.start_session(mode="auto", sid="S1")
    eng.send("你好，我想练口语。")

    assert l1k in eng._sessions["S1"].engine.active_l1()
    rows = repo.recent_lexicon_ops("S1")
    applied = [r for r in rows if r["applied"] == 1 and r["op_type"] == "add"]
    assert applied, "scene_ops add 应落一条 applied=1 审计"
    assert applied[0]["layer"] == Layer.L1.value
    assert applied[0]["target_key"] == l1k


def test_agent_ops_add_changes_l3_and_audited(lib, scenario_dir, tmp_path):
    """auto 模式：agent_ops add 生效（L3 工作集新增）并落 lexicon_ops（layer=L3）。"""
    l3k = lib.lexicon.for_layer(Layer.L3)[0].key
    llm = _OpsFakeLLM(agent_ops=[{"op": "add", "key": l3k, "intensity": 0.8, "reason": "需更鼓励"}])
    eng, repo, _ = _make_engine(lib, scenario_dir, tmp_path, llm)

    eng.start_session(mode="auto", sid="S2")
    _, meta = eng.send("我有点受挫。")
    snap_l3 = meta["snapshot"].l3
    assert l3k in snap_l3 and snap_l3[l3k] > 0

    rows = repo.recent_lexicon_ops("S2")
    applied = [r for r in rows if r["applied"] == 1 and r["layer"] == Layer.L3.value]
    assert applied, "agent_ops add 应落一条 layer=L3 审计"


def test_fixed_mode_skips_ops(lib, scenario_dir, tmp_path):
    """固定模式：场景/人格锁定，scene_ops/agent_ops 一律不生效、不审计。"""
    eng0, repo0, _ = _make_engine(lib, scenario_dir, tmp_path, FakeLLMClient(model=get_settings().llm_model))
    eng0.start_session(mode="fixed", scenario_id="oral_practice", sid="probe")
    probe = eng0._sessions["probe"]
    # 选不在固定模板里的 L1/L3 词，避免「本就在模板里」干扰断言
    l1k = [k.key for k in lib.lexicon.for_layer(Layer.L1) if k.key not in probe.l1_template_keys][0]
    l3k = [k.key for k in lib.lexicon.for_layer(Layer.L3) if k.key not in probe.l3_template_keys][0]

    llm = _OpsFakeLLM(
        scene_ops=[{"op": "add", "key": l1k, "intensity": 0.9}],
        agent_ops=[{"op": "add", "key": l3k, "intensity": 0.8}],
    )
    eng, repo, _ = _make_engine(lib, scenario_dir, tmp_path, llm)
    eng.start_session(mode="fixed", scenario_id="oral_practice", sid="S3")
    eng.send("你好。")
    assert l1k not in eng._sessions["S3"].engine.active_l1()
    assert not repo.recent_lexicon_ops("S3")


# ---------------------------------------------------------------------------
# #37 · _consume_ops 护栏（直接单测，覆盖边界分支）
# ---------------------------------------------------------------------------
def _auto_session(eng, sid):
    eng.start_session(mode="auto", sid=sid)
    return eng._sessions[sid]


def test_consume_ops_guardrails(lib, scenario_dir, tmp_path):
    l3_all = [k.key for k in lib.lexicon.for_layer(Layer.L3)]
    l1_all = [k.key for k in lib.lexicon.for_layer(Layer.L1)]
    llm = FakeLLMClient(model=get_settings().llm_model)
    eng, repo, _ = _make_engine(lib, scenario_dir, tmp_path, llm)

    # --- 固定模式：整体跳过 ---
    eng.start_session(mode="fixed", scenario_id="oral_practice", sid="G0")
    g0 = eng._sessions["G0"]
    l3_tmpl = list(g0.l3_template_keys)[0]
    assert eng._consume_ops(g0, [{"op": "delete", "key": l3_tmpl}], Layer.L3) == (0, [])
    assert not repo.recent_lexicon_ops("G0")

    # --- auto 模式：模板词为空，专注护栏 ---
    sess = _auto_session(eng, "G1")

    # 层归属不匹配：把 L1 词当 L3 操作 → 拒绝
    n, _ = eng._consume_ops(sess, [{"op": "add", "key": l1_all[0], "intensity": 0.5}], Layer.L3)
    assert n == 0
    rows = repo.recent_lexicon_ops("G1")
    assert any(r["applied"] == 0 and "layer_mismatch" in (r["llm_reason"] or "") for r in rows)

    # 词表外 → 走审核不自动收编
    eng._consume_ops(sess, [{"op": "add", "key": "ghost.unknown.x", "intensity": 0.5}], Layer.L3)
    rows = repo.recent_lexicon_ops("G1")
    assert any(r["target_key"] is None and "out_of_lexicon_review" in (r["llm_reason"] or "") for r in rows)

    # update delta 超限 → 拒绝
    new_l3 = [k for k in l3_all if k not in sess.l3_template_keys][0]
    eng._consume_ops(sess, [{"op": "add", "key": new_l3, "intensity": 0.6}], Layer.L3)
    rows = repo.recent_lexicon_ops("G1")
    assert any(r["target_key"] == new_l3 and r["applied"] == 1 for r in rows)
    # 现在 update 但 delta 超 0.4 → 拒绝
    eng._consume_ops(sess, [{"op": "update", "key": new_l3, "delta": 0.9}], Layer.L3)
    rows = repo.recent_lexicon_ops("G1")
    assert any(r["target_key"] == new_l3 and r["applied"] == 0 and "delta_exceed_limit" in (r["llm_reason"] or "") for r in rows)

    # delete 非 auto_added → 拒绝（用从未被 add 的 l3_all[1]）
    other_l3 = l3_all[1]
    assert other_l3 != new_l3
    eng._consume_ops(sess, [{"op": "delete", "key": other_l3}], Layer.L3)
    rows = repo.recent_lexicon_ops("G1")
    assert any(r["target_key"] == other_l3 and r["applied"] == 0 and "not_auto_added" in (r["llm_reason"] or "") for r in rows)

    # 合法 update（delta<=0.4）生效
    eng._consume_ops(sess, [{"op": "update", "key": new_l3, "delta": 0.2}], Layer.L3)
    rows = repo.recent_lexicon_ops("G1")
    assert any(r["target_key"] == new_l3 and r["applied"] == 1 and r["op_type"] == "update" for r in rows)


def test_consume_ops_protects_template_key(lib, scenario_dir, tmp_path):
    """固定模式模板词受保护：delete 模板 L3 词被拒绝。"""
    llm = FakeLLMClient(model=get_settings().llm_model)
    eng, repo, _ = _make_engine(lib, scenario_dir, tmp_path, llm)
    eng.start_session(mode="fixed", scenario_id="oral_practice", sid="G2")
    sess = eng._sessions["G2"]
    tmpl_l3 = list(sess.l3_template_keys)[0]
    # 固定模式整体跳过，这里用 L3 等价逻辑单独验 delete 保护：临时改 mode 但保留模板集
    sess.mode = "auto"
    eng._consume_ops(sess, [{"op": "delete", "key": tmpl_l3}], Layer.L3)
    rows = repo.recent_lexicon_ops("G2")
    assert any(r["target_key"] == tmpl_l3 and r["applied"] == 0 and "protected_template_key" in (r["llm_reason"] or "") for r in rows)
    # 模板词仍在工作集
    assert tmpl_l3 in sess.engine.l3_working_keys()
