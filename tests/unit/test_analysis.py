"""分析层单测：本地启发式、重复护栏、LLM 抽取（FakeLLM）、词表外候选发现。"""
from dla.analysis.heuristics import detect_repetition, extract_heuristics
from dla.analysis.llm_analyzer import analyze
from dla.core.models import Keyword, KeywordType, Layer
from dla.evolution import CandidateRegistry, discover_from_extractions
from dla.keywords.lexicon import Lexicon
from dla.llm.openai_compat import FakeLLMClient


def test_extract_heuristics_frustrated(lib):
    ev = extract_heuristics("这个语法太难了，我完全不会", 1, 0.0, lib.lexicon)
    keys = {e.key for e in ev}
    assert "user_mood.frustrated" in keys


def test_detect_repetition_similar_and_self_degenerate():
    # 与近 N 轮高度相似
    hit, _ = detect_repetition("你好世界你好世界", ["你好世界你好世界", "今天天气不错"])
    assert hit is True
    # 完全不同
    hit2, _ = detect_repetition("一种全新的表达内容", ["你好世界你好世界", "今天天气不错"])
    assert hit2 is False
    # 单条自重复退化报文
    hit3, _ = detect_repetition("啊啊啊啊啊啊啊啊啊啊", [])
    assert hit3 is True


def test_fake_llm_analyze_only_user_part(lib):
    """回归：分析器拼接了上一轮 agent 回复，但抽取必须只看用户当前发言。

    agent 回复含「急」不应导致误抽 user_temper.impatient。
    """
    llm = FakeLLMClient()
    res = analyze(
        llm,
        user_text="你好，我想练口语。",
        prev_agent_text="别急，我们一步步来，你已经做得很好了～",
        lexicon=lib.lexicon,
    )
    keys = {e["key"] for e in res.extractions}
    assert "user_temper.impatient" not in keys
    assert "user_mood.frustrated" not in keys


def test_fake_llm_analyze_extracts_user_frustration(lib):
    llm = FakeLLMClient()
    res = analyze(llm, "这个语法太难了，我完全不会", "", lib.lexicon)
    keys = {e["key"] for e in res.extractions}
    assert "user_mood.frustrated" in keys


def test_discover_from_extractions_splits_known_unknown():
    lex = Lexicon()
    lex.add(Keyword("user_mood.frustrated", Layer.L2, "mood", "受挫", KeywordType.SCALAR))
    reg = CandidateRegistry()
    extractions = [
        {"key": "user_mood.frustrated", "intensity": 0.8},
        {"key": "some.unknown.word", "intensity": 0.7},
    ]
    known, unknown = discover_from_extractions(extractions, lex, reg)
    assert [e["key"] for e in known] == ["user_mood.frustrated"]
    assert unknown == ["some.unknown.word"]
    # 候选入队
    assert "some.unknown.word" in reg._cands
