"""Prompt 组装器（doc/01 §5 prompt/assembler.py / §7 步骤7）。

把三层权重快照渲染为 System Prompt：角色/场景段 + 用户肖像段 + 交流风格段，
并对程度型段做**分层 Token 预算裁剪**（doc/01 D5：最重要的词一定进 Prompt，长尾被裁）。
历史摘要 / 冷记忆 / detail 块 / 工具 schema 作为附加段，各自受独立子预算约束。

注：人格稳定性判定（doc/01 §7 步骤6 Δ 比较）由编排层 DialogueEngine 负责，本模块只负责组装。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..core.models import Layer, WeightSnapshot
from ..keywords.lexicon import Lexicon
from .budget import budget_for_layer, estimate_tokens, truncate_by_budget
from .renderer import render_segments


class PromptAssembler:
    def __init__(self, cfg, lexicon: Lexicon) -> None:
        self.cfg = cfg
        self.lexicon = lexicon

    def assemble(
        self,
        snapshot: WeightSnapshot,
        *,
        scene_name: str = "",
        greeting: str = "",
        history: Optional[List[str]] = None,
        cold_memory: Optional[List[str]] = None,
        detail_blocks: Optional[List[str]] = None,
        tool_schema: str = "",
    ) -> Dict[str, object]:
        history = history or []
        cold_memory = cold_memory or []
        detail_blocks = detail_blocks or []
        L1, L2, L3 = snapshot.l1, snapshot.l2, snapshot.l3
        total = max(1, int(self.cfg.prompt_total_token_budget))

        # —— 角色段（L3 role，类别型 argmax 已定，完整渲染）——
        role_weights = {k: v for k, v in L3.items() if k.startswith("role")}
        role_seg = self._block(role_weights, Layer.L3, "角色")

        # —— 用户肖像段（L2，程度型，按预算裁剪）——
        l2_segs = render_segments(L2, self.lexicon, Layer.L2)
        l2_text = "；".join(
            truncate_by_budget(l2_segs, budget_for_layer(total, self.cfg.prompt_l2_ratio))
        )
        l2_seg = f"【用户肖像】{l2_text}" if l2_text else ""

        # —— 交流风格段（L3 非 role，程度型，按预算裁剪）——
        l3_trait = {k: v for k, v in L3.items() if not k.startswith("role")}
        l3_segs = render_segments(l3_trait, self.lexicon, Layer.L3)
        l3_text = "；".join(
            truncate_by_budget(l3_segs, budget_for_layer(total, self.cfg.prompt_l3_ratio))
        )
        l3_seg = f"【交流风格】{l3_text}" if l3_text else ""

        # —— 头部 ——
        header = "你是 DialogueLearningAgent，一个由关键词权重驱动的对话代理。"
        if greeting:
            header += f" {greeting}"
        if scene_name:
            header += f"\n当前场景：{scene_name}。"

        core_parts = [header]
        for seg in (role_seg, l2_seg, l3_seg):
            if seg:
                core_parts.append(seg)
        core = "\n".join(core_parts)

        # —— 附加段（独立子预算）——
        extras: List[str] = []
        if history:
            extras.append("历史摘要：\n" + "\n".join(history))
        cold_budget = budget_for_layer(total, self.cfg.memory_prompt_ratio)
        if cold_memory:
            cold_text = "；".join(cold_memory)
            if estimate_tokens(cold_text) > cold_budget:
                cold_text = cold_text[: cold_budget * 3] + "…"
            extras.append("背景参考：\n" + cold_text)
        if detail_blocks:
            extras.append("补充细节：\n" + "\n".join(detail_blocks))
        if tool_schema:
            extras.append(tool_schema)

        system_prompt = core + ("\n\n" + "\n\n".join(extras) if extras else "")
        debug = {
            "core_tokens": estimate_tokens(core),
            "system_tokens": estimate_tokens(system_prompt),
            "l1": L1,
            "l2": L2,
            "l3": L3,
        }
        return {"system_prompt": system_prompt, "core": core, "extras": extras, "debug": debug}

    def format_extras(
        self,
        history: Optional[List[str]] = None,
        cold_memory: Optional[List[str]] = None,
        detail_blocks: Optional[List[str]] = None,
        tool_schema: str = "",
    ) -> List[str]:
        """把附加段（独立于核心段）渲染为区块列表（doc/11 压缩后重组装复用）。"""
        history = history or []
        cold_memory = cold_memory or []
        detail_blocks = detail_blocks or []
        total = max(1, int(self.cfg.prompt_total_token_budget))
        extras: List[str] = []
        if history:
            extras.append("历史摘要：\n" + "\n".join(history))
        cold_budget = budget_for_layer(total, self.cfg.memory_prompt_ratio)
        if cold_memory:
            cold_text = "；".join(cold_memory)
            if estimate_tokens(cold_text) > cold_budget:
                cold_text = cold_text[: cold_budget * 3] + "…"
            extras.append("背景参考：\n" + cold_text)
        if detail_blocks:
            extras.append("补充细节：\n" + "\n".join(detail_blocks))
        if tool_schema:
            extras.append(tool_schema)
        return extras

    def render_system(self, core: str, extras: List[str]) -> str:
        """核心段 + 附加段组装为最终 System Prompt（压缩后重组装用）。"""
        return core + ("\n\n" + "\n\n".join(extras) if extras else "")

    def _block(self, weights: Dict[str, float], layer: Layer, title: str) -> str:
        segs = render_segments(weights, self.lexicon, layer)
        if not segs:
            return ""
        body = "；".join(text for _k, _w, text in segs)
        return f"【{title}】{body}"
