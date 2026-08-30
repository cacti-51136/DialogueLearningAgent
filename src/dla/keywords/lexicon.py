"""词表（Lexicon）：白名单校验 + 同义词归一（doc/01 D11 / doc/02 §4）。

核心约束：LLM 抽取/用户对话中出现的关键词**必须命中白名单**，否则一律丢弃并记 warn
（用于反哺词库）。本类提供受控查询、同义词归一、按层/维度枚举。

零依赖（仅标准库 + core.models）。
"""

from __future__ import annotations

from typing import Iterable, Optional

from ..core.models import Keyword, KeywordType, Layer


class Lexicon:
    """受控关键词白名单。"""

    def __init__(self) -> None:
        self._by_key: dict[str, Keyword] = {}
        self._syn: dict[str, str] = {}  # 同义词小写 -> 规范 key

    # ---- 构建 ----
    def add(self, kw: Keyword, synonyms: Optional[Iterable[str]] = None) -> None:
        if kw.key in self._by_key:
            raise ValueError(f"重复关键词: {kw.key}")
        self._by_key[kw.key] = kw
        for s in synonyms or []:
            self._syn[str(s).strip().lower()] = kw.key

    # ---- 查询 ----
    def get(self, key: str) -> Optional[Keyword]:
        return self._by_key.get(key)

    def is_known(self, key: str) -> bool:
        return key in self._by_key

    def normalize(self, raw: str) -> Optional[str]:
        """将原始词（可能带同义词/大小写差异）归一为规范 key；未知返回 None。"""
        if raw is None:
            return None
        r = raw.strip().lower()
        if not r:
            return None
        if r in self._by_key:
            return r
        return self._syn.get(r)

    def for_layer(self, layer: Layer) -> list[Keyword]:
        return [k for k in self._by_key.values() if k.layer == layer]

    def dimensions_of(self, layer: Layer) -> list[str]:
        seen: list[str] = []
        for k in self.for_layer(layer):
            if k.dimension not in seen:
                seen.append(k.dimension)
        return seen

    def keywords_in(self, layer: Layer, dimension: str) -> list[Keyword]:
        return [k for k in self.for_layer(layer) if k.dimension == dimension]

    def all_keys(self) -> list[str]:
        return list(self._by_key.keys())

    def __len__(self) -> int:
        return len(self._by_key)
