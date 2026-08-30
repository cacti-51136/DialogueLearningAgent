"""编排引擎 DialogueEngine（doc/01 §7 一轮对话完整数据流）。

流程所有者：把分析层 / 权重引擎 / Prompt 组装 / LLM / 存储 / 上下文压缩 / 人格演进 / kw_agent_map
串成一轮闭环。提供 ``start_session`` 与 ``send``（同步入口；异步包装见 apps 层）。

设计原则：
- 分析失败静默降级（doc/01 D12）：沿用上一轮权重继续对话。
- 人格演进（doc/10）默认关闭，仅达到阶梯里程碑 + 证据门槛才触发，且受锚相似度/偏离预算约束。
- kw_agent_map（doc/03 §2.15）：由已验证耦合规则沉淀为「情绪/脾性 → Agent 特质」可学习映射，随对话涌现累积。
- safe_mode（doc/09 §6）：亲密类场景否决级硬约束，回复若越界则拒绝。
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from ..analysis.heuristics import detect_repetition, extract_heuristics
from ..analysis.llm_analyzer import AnalysisError, analyze
from ..analysis.trigger import AnalysisTrigger
from ..config.scenario_loader import load_scenario_by_id
from ..core.errors import LlmError
from ..core.models import Evidence, Layer, WeightSnapshot
from ..core.ports import ChatMessage
from ..evolution import CandidateRegistry, discover_from_extractions
from ..prompt.assembler import PromptAssembler
from ..prompt.context_compact import ContextWindow, compact, fill_ratio
from ..prompt.budget import estimate_tokens
from ..weighting.engine import WeightEngine, WeightEngineConfig
from ..weighting.coupling import apply_rules

# safe_mode 简易黑名单（doc/09 §6 硬约束兜底；生产应接更强审核）
_SAFE_BLOCKLIST = ["裸", "性爱", "色情", "做爱", "约炮"]

# 触发分析的显式反馈信号
_EVENT_WORDS = ["不对", "错了", "很好", "谢谢", "满意", "不满意", "喜欢这样", "别这样"]


class DialogueEngine:
    def __init__(self, settings, lib, llm_client, repo=None, clock=None) -> None:
        self.settings = settings
        self.lib = lib
        self.llm = llm_client
        self.repo = repo
        self.clock = clock

        self.engine = WeightEngine(
            lib,
            WeightEngineConfig(
                prior_strength=settings.weight_prior_strength,
                default_half_life_hours=settings.weight_default_half_life_hours,
                llm_fusion_beta=settings.weight_llm_fusion_beta,
                lvm_gamma=settings.lvm_l3_prior_gamma,
            ),
            clock,
        )
        self.assembler = PromptAssembler(settings, lib.lexicon)
        self.trigger = AnalysisTrigger(
            cold_start_turns=settings.analysis_cold_start_turns,
            period=settings.analysis_period,
            enable_heuristics=settings.analysis_enable_heuristics,
        )
        self.candidates = CandidateRegistry()

        self.session_id = f"session-{int(self._now())}"
        self.turn = 0
        self._greeting = ""
        self._scene_name = ""
        self._safe_mode: Optional[dict] = None
        self._history_summaries: List[str] = []
        self._last_l3: Dict[str, float] = {}
        self._last_agent_text = ""
        self._recent_replies: List[str] = []
        self._pending_summary = ""
        self._baseline_spec = ""
        self._persona_specs: List[str] = []
        self._persona_last_turn = -999
        self._persona_drift = 0.0
        self._mode = "fixed"

    # ---- 工具 ----
    def _now(self) -> float:
        if self.clock is not None:
            return self.clock.now()
        return time.time()

    def _detect_event(self, text: str) -> bool:
        return any(w in text for w in _EVENT_WORDS)

    @staticmethod
    def _l3_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
        keys = set(a) | set(b)
        if not keys:
            return 0.0
        s = sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)
        return s / len(keys)

    @staticmethod
    def _split_summary(text: str) -> Tuple[str, str]:
        import re

        m = re.search(r"<turn_summary>(.*?)</turn_summary>", text, flags=re.S)
        if m:
            clean = text[: m.start()] + text[m.end():]
            return clean.strip(), m.group(1).strip()
        return text.strip(), text.strip()[:100]

    def _apply_safe_mode(self, text: str) -> str:
        if not self._safe_mode:
            return text
        if any(w in text for w in _SAFE_BLOCKLIST):
            return "（内容已按安全约束拦截，请换个健康的话题～）"
        return text

    # ---- 会话生命周期 ----
    def start_session(self, mode: str = "fixed", scenario_id: Optional[str] = None, describe: Optional[str] = None) -> str:
        self._mode = mode
        sid = scenario_id or self.settings.scenario_default
        sc = load_scenario_by_id(self.settings.scenario_dir, sid)
        self._scene_name = sc.name
        self._safe_mode = sc.safe_mode
        self.engine.set_l1_scene(dict(sc.l1))
        for k, v in sc.l2_preset.items():
            if self.lib.lexicon.normalize(k):
                self.engine.set_preset(k, v)
        self.engine.set_l3_baseline(dict(sc.l3_baseline))
        self._greeting = sc.greeting
        # doc/10 初始人格锚；同时把 _last_l3 种子设为基线，避免首轮与空权重相比产生虚假 notify
        baseline_snap = self.engine.compute_all(0)
        self._last_l3 = dict(baseline_snap.l3)
        self._baseline_spec = self._derive_baseline_spec(baseline_snap)
        self._persona_specs = [self._baseline_spec]
        if self.repo is not None:
            self.repo.save_persona_spec(self.session_id, 0, self._baseline_spec, version=0, is_baseline=True)
        return self._greeting

    def _derive_baseline_spec(self, snap=None) -> str:
        if snap is None:
            snap = self.engine.compute_all(0)
        parts = []
        for k, v in snap.l3.items():
            kw = self.lib.lexicon.get(k)
            if kw:
                parts.append(f"{kw.name}{v:.1f}")
        spec = f"初始人格（{self._scene_name}）：" + "；".join(parts)
        return spec[: self.settings.persona_spec_max_chars]

    # ---- 主循环 ----
    def send(self, user_text: str) -> Tuple[str, dict]:
        self.turn += 1
        now = self._now()
        if self.repo is not None:
            self.repo.add_message(self.session_id, self.turn, "user", user_text, now)

        chain: List[tuple] = []

        # 步骤2 触发判定
        force = self._detect_event(user_text)
        do_analyze = self.trigger.should_analyze(self.turn, force_event=force)
        chain.append(("trigger", {"turn": self.turn, "do_analyze": do_analyze, "event": force}))

        new_candidates: List[str] = []
        if do_analyze:
            heur = extract_heuristics(user_text, self.turn, now, self.lib.lexicon)
            for e in heur:
                self.engine.add_evidence(e)
            try:
                res = analyze(self.llm, user_text, self._last_agent_text, self.lib.lexicon)
                for ex in res.extractions:
                    self.engine.add_evidence(
                        Evidence(key=ex["key"], intensity=ex["intensity"], timestamp=now, source="llm", turn=self.turn)
                    )
                known, unknown = discover_from_extractions(res.extractions, self.lib.lexicon, self.candidates)
                new_candidates = unknown
                self._pending_summary = res.turn_summary
                chain.append(("analyze", {"heuristics": len(heur), "llm_extractions": len(res.extractions), "unknown": unknown}))
            except AnalysisError as e:
                chain.append(("analyze_error", {"error": str(e)}))  # 降级：沿用旧权重

        # 步骤4/5 重算三层
        snapshot = self.engine.compute_all(self.turn)
        chain.append(("weights", {"l1": snapshot.l1, "l2": snapshot.l2, "l3": snapshot.l3}))

        # 步骤6 稳定性
        delta = self._l3_distance(self._last_l3, snapshot.l3)
        rebuild = delta >= self.settings.weight_rebuild_threshold
        notify = delta >= self.settings.weight_notify_threshold
        self._last_l3 = snapshot.l3
        chain.append(("stability", {"delta": round(delta, 3), "rebuild": rebuild, "notify": notify}))

        # 步骤7 组装 + 预算护栏
        history = list(self._history_summaries)
        assembled = self.assembler.assemble(
            snapshot, scene_name=self._scene_name, greeting=self._greeting, history=history
        )
        system_prompt = assembled["system_prompt"]
        chain.append(("assemble", assembled["debug"]))

        window = ContextWindow(
            system_prompt=system_prompt, history=history, cold_memory=[],
            detail_blocks=[], tool_schema="", current_user_msg=user_text,
        )
        ratio = fill_ratio(window, self.settings)
        compact_actions: List[str] = []
        if self.settings.ctx_auto_compact and ratio >= self.settings.ctx_compact_ratio:
            new_window, actions = compact(window, self.settings)
            compact_actions = actions
            system_prompt = new_window.system_prompt
            self._history_summaries = list(new_window.history)
            history = list(new_window.history)
            ratio_after = fill_ratio(new_window, self.settings)
            if self.repo is not None:
                self.repo.log_compact(self.session_id, self.turn, ratio, ratio_after, actions)
            chain.append(("budget_guard", {"ratio_before": round(ratio, 3), "actions": actions, "ratio_after": round(ratio_after, 3)}))
        else:
            chain.append(("budget_guard", {"ratio": round(ratio, 3), "triggered": False}))

        # 步骤8 LLM 生成 + 重复护栏
        messages = [ChatMessage(role="system", content=system_prompt), ChatMessage(role="user", content=user_text)]
        reply_text, turn_summary, rep_hit = self._generate_with_guard(messages, user_text)
        chain.append(("repeat_guard", {"hit": rep_hit}))

        reply_text = self._apply_safe_mode(reply_text)
        self._recent_replies.append(reply_text)
        self._recent_replies = self._recent_replies[-self.settings.repeat_recent_n:]

        # 步骤9 落库
        if self.repo is not None:
            self.repo.add_message(self.session_id, self.turn, "agent", reply_text, self._now())
            self.repo.save_snapshot(self.session_id, snapshot, self._now())
            summary_to_save = turn_summary or self._pending_summary or reply_text[:100]
            self.repo.save_summary(self.session_id, self.turn, summary_to_save, "turn")
        self._history_summaries.append(turn_summary or reply_text[:100])
        self._last_agent_text = reply_text
        self._pending_summary = ""

        # 步骤10/11 人格演进（doc/10，默认关）+ kw_agent_map（doc/03 §2.15）
        self._maybe_update_persona(snapshot, now)
        self._update_kwmap(snapshot)

        meta = {
            "debug_chain": chain,
            "snapshot": snapshot,
            "new_candidates": new_candidates,
            "notify": notify,
            "compact_actions": compact_actions,
            "turn_summary": turn_summary,
        }
        return reply_text, meta

    # ---- 生成与重复护栏 ----
    def _generate_with_guard(self, messages: List[ChatMessage], user_text: str):
        try:
            resp = self.llm.complete(
                messages,
                frequency_penalty=self.settings.repeat_freq_penalty,
                presence_penalty=self.settings.repeat_presence_penalty,
            )
            reply = resp.content
        except LlmError:
            return "抱歉，我这边出了点问题，稍后再聊？", "", False

        reply_text, turn_summary = self._split_summary(reply)
        hit, _metrics = detect_repetition(
            reply_text, self._recent_replies,
            self.settings.repeat_sim_threshold, self.settings.repeat_self_ngram_ratio,
        )
        if hit:
            retry = messages + [ChatMessage(role="system", content="请勿重复之前的话，换种表达方式，给出新的回应。")]
            try:
                resp2 = self.llm.complete(retry, frequency_penalty=self.settings.repeat_freq_penalty, presence_penalty=self.settings.repeat_presence_penalty)
                r2, ts2 = self._split_summary(resp2.content)
                hit2, _ = detect_repetition(r2, self._recent_replies, self.settings.repeat_sim_threshold, self.settings.repeat_self_ngram_ratio)
                if not hit2:
                    return r2, ts2, True
            except LlmError:
                pass
            return self.settings.repeat_degrade_msg, "（降级兜底）", True
        return reply_text, turn_summary, False

    # ---- 人格演进（doc/10，默认关）----
    def _maybe_update_persona(self, snapshot: WeightSnapshot, now: float) -> None:
        if not self.settings.persona_auto_update:
            return
        stages = self.settings.persona_stages
        if self.turn not in stages:
            return
        if (self.turn - self._persona_last_turn) < 8:
            return
        # 证据门槛（用户参与度，用 L2 累积证据近似）
        user_evidence = sum(
            self.engine._e_total(k) for k in snapshot.l2
        ) if snapshot.l2 else 0.0
        if user_evidence < self.settings.persona_min_evidence:
            return
        spec = self._derive_persona_spec(snapshot)
        if not spec:
            return
        # 锚相似度校验
        sim = _jaccard_chars(self._baseline_spec, spec)
        if sim < self.settings.persona_min_anchor_sim:
            return
        if self._persona_drift + (1.0 - sim) > self.settings.persona_max_drift:
            return
        self._persona_drift += 1.0 - sim
        self._persona_last_turn = self.turn
        self._persona_specs.append(spec)
        if self.repo is not None:
            self.repo.save_persona_spec(
                self.session_id, self.turn, spec, version=len(self._persona_specs) - 1
            )

    def _derive_persona_spec(self, snapshot: WeightSnapshot) -> str:
        """生成本轮人格规格书（≤200 字，doc/10）。

        本期用「快照描述」作为可解释载体；接入 LLM 推演时替换为 doc/10 §4 的推演指令。
        """
        parts = []
        for k, v in snapshot.l3.items():
            kw = self.lib.lexicon.get(k)
            if kw:
                parts.append(f"{kw.name}{v:.1f}")
        spec = f"人格（第{self.turn}轮）：" + "；".join(parts)
        return spec[: self.settings.persona_spec_max_chars]

    # ---- kw_agent_map 涌现（doc/03 §2.15）----
    def _update_kwmap(self, snapshot: WeightSnapshot) -> None:
        if self.repo is None:
            return
        # 从已验证耦合规则沉淀「情绪/脾性 → Agent 特质」映射，随对话累积
        eff = apply_rules(self.lib.rules, self.engine.active_l1(), snapshot.l2, self.lib.lexicon)
        for src, dst, w in eff.maps:
            # 仅当该情绪/脾性源词在 L2 达到置信门槛才沉淀（避免噪声）
            if snapshot.l2.get(src, 0.0) < 0.3:
                continue
            if dst.startswith("user_"):
                continue
            self.repo.kwmap_upsert(src, dst, "boost", w, learn_rate=self.settings.kwmap_lr)


def _jaccard_chars(a: str, b: str) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)
