"""权重引擎核心算法单测（resolver / confidence / decay / coupling / engine）。"""
from dla.config.loader import CouplingRule
from dla.core.models import Evidence, Keyword, KeywordType, Layer
from dla.keywords.lexicon import Lexicon
from dla.weighting.confidence import confidence, decayed_total, half_life_lambda
from dla.weighting.coupling import apply_rules
from dla.weighting.decay import decay_factor, hysteresis_passed, passed_cooldown
from dla.weighting.engine import WeightEngine, WeightEngineConfig
from dla.weighting.resolver import resolve_layer


class _Clock:
    """确定性时钟（测试用），避免真实 time.time() 导致的证据衰减。"""

    def __init__(self, t: float = 1000.0) -> None:
        self._t = t

    def now(self) -> float:
        return self._t


def _lex() -> Lexicon:
    lex = Lexicon()
    lex.add(Keyword("scene.oral_practice", Layer.L1, "scene", "口语练习", KeywordType.CATEGORICAL))
    lex.add(Keyword("role.tutor", Layer.L3, "role", "导师", KeywordType.CATEGORICAL))
    lex.add(Keyword("role.pet", Layer.L3, "role", "宠物", KeywordType.CATEGORICAL))
    lex.add(Keyword("empathy", Layer.L3, "empathy", "共情", KeywordType.SCALAR))
    lex.add(Keyword("tone.encouraging", Layer.L3, "tone", "鼓励", KeywordType.SCALAR))
    lex.add(Keyword("tone.gentle", Layer.L3, "tone", "温和", KeywordType.SCALAR))
    lex.add(Keyword("user_mood.frustrated", Layer.L2, "mood", "受挫", KeywordType.SCALAR))
    return lex


# ---------------- confidence / decay ----------------
def test_confidence_monotonic_and_bounded():
    assert confidence(0.0, 2.0) == 0.0
    assert 0.0 < confidence(1.0, 2.0) < confidence(2.0, 2.0) < confidence(10.0, 2.0) < 1.0
    assert confidence(1e12, 2.0) > 0.999


def test_decayed_total_no_decay_at_now_and_halflife():
    ev = [Evidence("x", 1.0, timestamp=0.0, source="h", turn=1)]
    assert abs(decayed_total(ev, 0.0, 6.0) - 1.0) < 1e-9
    # 经过一个半衰期（6h）后约为 0.5
    assert abs(decayed_total(ev, 6.0 * 3600.0, 6.0) - 0.5) < 1e-6


def test_decay_factor_and_cooldown():
    assert abs(decay_factor(0.0, 6.0) - 1.0) < 1e-9
    assert abs(decay_factor(6.0 * 3600.0, 6.0) - 0.5) < 1e-6
    assert passed_cooldown(2, 5, 3) is True
    assert passed_cooldown(2, 4, 3) is False
    assert hysteresis_passed(0.5, 0.4) is True
    assert hysteresis_passed(0.3, 0.4) is False


# ---------------- resolver（标量透传 / 类别 argmax） ----------------
def test_resolver_categorical_argmax():
    lex = _lex()
    out = resolve_layer({"role.tutor": 0.3, "role.pet": 0.8}, lex, Layer.L3)
    assert out == {"role.pet": 1.0}


def test_resolver_scalar_passthrough_absolute_not_pinned():
    lex = _lex()
    # 单成员标量维度：透传绝对权重，不再被归一成 1.0（修复前会恒为 1.0）
    out = resolve_layer({"empathy": 0.6}, lex, Layer.L3)
    assert out["empathy"] == 0.6
    # 多成员标量：各自透传 + clamp
    out2 = resolve_layer({"tone.encouraging": 1.7, "tone.gentle": 0.0}, lex, Layer.L3)
    assert out2["tone.encouraging"] == 1.0  # 超界被 clamp
    assert out2["tone.gentle"] == 0.0


# ---------------- coupling ----------------
def test_coupling_sets_boosts_and_maps():
    lex = _lex()
    rules = [
        CouplingRule(
            id="base", when_l1=["scene.oral_practice"], when_l2=[],
            set_cmds=[("role", "role.tutor")], boost_cmds=[("empathy", 0.3)],
        ),
        CouplingRule(
            id="frust", when_l1=[], when_l2=["user_mood.frustrated"],
            set_cmds=[], boost_cmds=[("empathy", 0.3), ("tone.gentle", 0.4)],
        ),
    ]
    eff = apply_rules(rules, {"scene.oral_practice"}, {"user_mood.frustrated": 0.5}, lex)
    assert eff.sets == {"role": "role.tutor"}
    assert abs(eff.boosts["empathy"] - 0.6) < 1e-9
    assert abs(eff.boosts["tone.gentle"] - 0.4) < 1e-9
    srcs = {(s, d) for s, d, _ in eff.maps}
    assert ("user_mood.frustrated", "empathy") in srcs
    assert ("user_mood.frustrated", "tone.gentle") in srcs
    # 无 L2 情绪/脾性触发时不应产出可学习映射
    eff2 = apply_rules(rules, set(), {}, lex)
    assert eff2.maps == []


# ---------------- engine：L3 随对话演化 ----------------
def test_weight_engine_l3_rises_on_frustration():
    lex = _lex()
    rules = [
        CouplingRule(
            id="frust", when_l1=[], when_l2=["user_mood.frustrated"],
            set_cmds=[], boost_cmds=[("empathy", 0.3)],
        ),
    ]
    lib = type("Lib", (), {"lexicon": lex, "rules": rules})()
    clock = _Clock(1000.0)
    eng = WeightEngine(lib, WeightEngineConfig(), clock)
    eng.set_l3_baseline({"empathy": 0.6})
    base = eng.compute_all(0).l3["empathy"]
    assert abs(base - 0.6) < 1e-9

    eng.add_evidence(Evidence("user_mood.frustrated", 0.8, timestamp=clock.now(), source="llm", turn=1))
    risen = eng.compute_all(1).l3["empathy"]
    assert risen > base
    assert abs(risen - 0.9) < 1e-9  # base 0.6 + boost 0.3
