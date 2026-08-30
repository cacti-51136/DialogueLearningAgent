"""场景模板加载器（doc/09 ScenarioTemplate / doc/01 §5 config/scenarios）。

读取 ``config/scenarios/*.yaml`` 为 ``Scenario`` 结构。约定（doc/09 §约定）：
- ``l2_preset`` 仅含身份/目标类（user_profile.* / user_goal.*），**不得**写入情绪/脾性类
  （user_mood.* / user_temper.*，必须由对话涌现，doc/02 §11.9）。
- ``coupling_seeds`` 为草稿，加载时不生效，需人工 review 并入 ``coupling_rules.yaml``。
- 亲密类（virtual_gf）含 ``safe_mode`` 否决级硬约束（doc/09 §6），不可关闭。
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

from ..core.errors import LexiconError


@dataclass
class Scenario:
    id: str
    name: str
    greeting: str
    l1: Dict[str, float] = field(default_factory=dict)
    l2_preset: Dict[str, float] = field(default_factory=dict)
    l3_baseline: Dict[str, float] = field(default_factory=dict)
    safe_mode: Optional[dict] = None
    coupling_seeds: List[dict] = field(default_factory=list)
    modes: List[str] = field(default_factory=lambda: ["fixed", "free"])


def load_scenario(path: str) -> Scenario:
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        raise LexiconError(f"加载场景失败 {path}: {exc}") from exc
    if not isinstance(data, dict) or "id" not in data:
        raise LexiconError(f"场景格式非法 {path}: 缺少 id")
    return Scenario(
        id=str(data["id"]),
        name=str(data.get("name", data["id"])),
        greeting=str(data.get("greeting", "")),
        l1={k: float(v) for item in data.get("l1", []) for k, v in [ (item["key"], item["intensity"]) ]},
        l2_preset={k: float(v) for item in data.get("l2_preset", []) for k, v in [ (item["key"], item["intensity"]) ]},
        l3_baseline={k: float(v) for k, v in data.get("l3_baseline", {}).items()},
        safe_mode=data.get("safe_mode"),
        coupling_seeds=list(data.get("coupling_seeds", [])),
        modes=list(data.get("modes", ["fixed", "free"])),
    )


def list_scenarios(scenario_dir: str) -> List[Scenario]:
    out: List[Scenario] = []
    for path in sorted(glob.glob(os.path.join(scenario_dir, "*.yaml"))):
        try:
            out.append(load_scenario(path))
        except LexiconError:
            continue
    return out


def load_scenario_by_id(scenario_dir: str, scenario_id: str) -> Scenario:
    path = os.path.join(scenario_dir, f"{scenario_id}.yaml")
    if not os.path.isfile(path):
        # 容错：遍历匹配 id 字段
        for sc in list_scenarios(scenario_dir):
            if sc.id == scenario_id:
                return sc
        raise LexiconError(f"未找到场景: {scenario_id}")
    return load_scenario(path)
