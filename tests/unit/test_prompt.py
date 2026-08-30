"""Prompt 预算单测：确定性 token 估算、按权重降序裁剪。"""
from dla.prompt.budget import estimate_tokens, truncate_by_budget


def test_estimate_tokens_chinese_per_char():
    assert estimate_tokens("你好世界啊") == 5
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") == 1  # 单非中文 4 字/token -> ceil(1/4)=1


def test_estimate_tokens_mixed():
    # "hi你" -> cjk=1, 非中文 "hi"=2 -> ceil(2/4)=1 -> 合计 2
    assert estimate_tokens("hi你") == 2
    # 纯英文空格不计入
    assert estimate_tokens("hello world") == 3  # 10 非中文 -> ceil(10/4)=3


def test_truncate_by_budget_respects_weight_order():
    segs = [
        ("a", 0.1, "短"),          # 1 token
        ("b", 0.9, "重要长文本"),   # 5 tokens
        ("c", 0.5, "中等"),         # 2 tokens
    ]
    # 预算=6：先放最重要的「重要长文本」(5)，下一个「中等」(2) 累加 7>6 -> 停止
    out = truncate_by_budget(segs, 6)
    assert out == ["重要长文本"]
    # 预算充足时全部纳入，且保持权重降序
    out2 = truncate_by_budget(segs, 20)
    assert out2 == ["重要长文本", "中等", "短"]
