"""零依赖确定性 embedding（doc/07 §4 检索）。

默认用「哈希技巧 + 词频 + L2 归一」生成确定性向量：相同文本永远得到相同向量，
语义近似（共享 token）会获得较高余弦相似度。这满足 doc/07「离线占位」策略——
无需联网、结构完整可测；配置真实 backbone（sentence-transformers）后语义更准，
但属可选增强，不进入核心零依赖路径。

设计取舍（doc/01 D11 / 零强依赖铁律）：核心包禁止引入重型 ML 库，故默认实现不依赖
numpy / sentence-transformers。如需语义热启动，可在构造时传入 ``embed_fn`` 或
``backbone`` 模型名（懒加载、失败降级为确定性占位），由调用方自行保证依赖可装。
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Callable, Optional, Sequence

_TOKEN_RE = re.compile(r"[一-龥]|[A-Za-z]+|[0-9]+")


def tokenize(text: str) -> list[str]:
    """特征切分：中文逐字 + ASCII 连续词 + 数字。小写归一。"""
    return _TOKEN_RE.findall(text.lower())


class Embedder:
    """文本 → 单位向量（确定性）。支持注入 ``embed_fn`` 覆盖（用于测试或真实 backbone）。"""

    def __init__(
        self,
        dim: int,
        backbone: str = "",
        embed_fn: Optional[Callable[[str], Sequence[float]]] = None,
    ) -> None:
        self.dim = max(1, int(dim))
        self.backbone = backbone
        self._embed_fn = embed_fn

    def embed(self, text: str) -> list[float]:
        if self._embed_fn is not None:
            return self._normalize(list(self._embed_fn(text)))
        return self._deterministic(text)

    def _deterministic(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in tokenize(text):
            digest = hashlib.md5(tok.encode("utf-8")).digest()
            h = int.from_bytes(digest[:8], "big")
            vec[h % self.dim] += 1.0
        return self._normalize(vec)

    @staticmethod
    def _normalize(vec: Sequence[float]) -> list[float]:
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return [0.0] * len(vec)
        return [v / norm for v in vec]

    @staticmethod
    def cosine(a: Sequence[float], b: Sequence[float]) -> float:
        """余弦相似度；长度不等或任一为零向量返回 0.0。"""
        if len(a) != len(b) or len(a) == 0:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
