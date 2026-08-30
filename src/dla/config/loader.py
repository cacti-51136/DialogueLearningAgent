"""词库加载器（doc/01 §5 config/loader.py）：读取 YAML 词库 + 耦合规则，构建 Lexicon 与规则集。

设计：不依赖 ``settings``（可由调用方传入路径），便于测试与离线加载。
输出 ``KeywordLib``（lexicon + rules），是权重引擎与编排层的输入。
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Optional

import yaml

from ..core.errors import LexiconError
from ..core.models import Keyword, KeywordType, Layer
from ..keywords.lexicon import Lexicon


@dataclass
class CouplingRule:
    """一条 L1×L2 → L3 确定性耦合规则（doc/02 §5.2 阶段一）。"""

    id: str
    when_l1: list[str] = field(default_factory=list)
    when_l2: list[str] = field(default_factory=list)
    set_cmds: list[tuple[str, str]] = field(default_factory=list)  # (dimension, value_key)
    boost_cmds: list[tuple[str, float]] = field(default_factory=list)  # (l3_key, weight)


@dataclass
class KeywordLib:
    """词库全集：词表 + 耦合规则。"""

    lexicon: Lexicon
    rules: list[CouplingRule]


def _load_one(path: str, lexicon: Lexicon) -> None:
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        raise LexiconError(f"加载词库失败 {path}: {exc}") from exc

    if not isinstance(data, dict) or "layer" not in data or "keywords" not in data:
        raise LexiconError(f"词库格式非法 {path}: 缺少 layer/keywords")
    try:
        layer = Layer(str(data["layer"]))
    except ValueError as exc:
        raise LexiconError(f"词库 {path} layer 非法: {data.get('layer')}") from exc

    for item in data["keywords"]:
        try:
            kw = Keyword(
                key=str(item["key"]),
                layer=layer,
                dimension=str(item["dimension"]),
                name=str(item["name"]),
                ktype=KeywordType(str(item["ktype"])),
                description=str(item.get("description", "")),
                is_base=bool(item.get("is_base", False)),
            )
        except KeyError as exc:
            raise LexiconError(f"词库 {path} 关键词缺字段 {exc}: {item}") from exc
        lexicon.add(kw, item.get("synonyms"))


def _load_rules(path: str) -> list[CouplingRule]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        raise LexiconError(f"加载耦合规则失败 {path}: {exc}") from exc

    rules: list[CouplingRule] = []
    for r in data.get("rules", []):
        when = r.get("when", {}) or {}
        then = r.get("then", {}) or {}
        rules.append(
            CouplingRule(
                id=str(r["id"]),
                when_l1=list(when.get("l1", [])),
                when_l2=list(when.get("l2", [])),
                set_cmds=[(s["key"], s["value"]) for s in then.get("set", [])],
                boost_cmds=[(b["key"], float(b["weight"])) for b in then.get("boost", [])],
            )
        )
    return rules


def load_keyword_lib(
    keywords_dir: str = "config/keywords",
    coupling_file: str = "config/coupling_rules.yaml",
) -> KeywordLib:
    """加载关键词库与耦合规则，返回 KeywordLib。"""
    if not os.path.isdir(keywords_dir):
        raise LexiconError(f"词库目录不存在: {keywords_dir}")
    if not os.path.isfile(coupling_file):
        raise LexiconError(f"耦合规则文件不存在: {coupling_file}")

    lexicon = Lexicon()
    for path in sorted(glob.glob(os.path.join(keywords_dir, "*.yaml"))):
        _load_one(path, lexicon)
    rules = _load_rules(coupling_file)
    return KeywordLib(lexicon=lexicon, rules=rules)


# 进程内缓存（同一进程内复用，避免重复解析）
_CACHE: dict[tuple[str, str], KeywordLib] = {}


def get_keyword_lib(
    keywords_dir: str = "config/keywords",
    coupling_file: str = "config/coupling_rules.yaml",
) -> KeywordLib:
    """带缓存的加载入口。"""
    key = (os.path.abspath(keywords_dir), os.path.abspath(coupling_file))
    if key not in _CACHE:
        _CACHE[key] = load_keyword_lib(keywords_dir, coupling_file)
    return _CACHE[key]
