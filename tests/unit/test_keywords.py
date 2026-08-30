"""词表 / 加载器单测（白名单归一 / 按层枚举 / 词库加载）。"""
from dla.core.models import Keyword, KeywordType, Layer
from dla.keywords.lexicon import Lexicon


def test_lexicon_add_get_normalize():
    lex = Lexicon()
    lex.add(
        Keyword("user_mood.frustrated", Layer.L2, "mood", "受挫", KeywordType.SCALAR),
        synonyms=["沮丧", "挫败"],
    )
    assert lex.is_known("user_mood.frustrated")
    assert lex.get("user_mood.frustrated").name == "受挫"
    # 同义词归一
    assert lex.normalize("沮丧") == "user_mood.frustrated"
    assert lex.normalize("挫败") == "user_mood.frustrated"
    # 大小写/空白不敏感
    assert lex.normalize(" 沮丧 ") == "user_mood.frustrated"
    # 未知返回 None
    assert lex.normalize("不存在的词") is None


def test_lexicon_for_layer_and_dimensions():
    lex = Lexicon()
    lex.add(Keyword("a", Layer.L1, "scene", "A", KeywordType.CATEGORICAL))
    lex.add(Keyword("b", Layer.L1, "scene", "B", KeywordType.CATEGORICAL))
    lex.add(Keyword("c", Layer.L2, "mood", "C", KeywordType.SCALAR))
    assert {k.key for k in lex.for_layer(Layer.L1)} == {"a", "b"}
    assert lex.dimensions_of(Layer.L1) == ["scene"]
    assert lex.dimensions_of(Layer.L2) == ["mood"]


def test_load_keyword_lib_real(lib):
    # 真实词库：L1/L2/L3 都有词，且含情绪/脾性白名单候选
    assert lib.lexicon.is_known("scene.oral_practice")
    assert lib.lexicon.is_known("user_mood.frustrated")
    assert lib.lexicon.is_known("user_temper.impatient")
    assert lib.lexicon.is_known("empathy")
    # 耦合规则存在
    ids = {r.id for r in lib.rules}
    assert "mood_frustrated_boost_empathy" in ids
    assert "oral_practice_baseline" in ids
