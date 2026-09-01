"""编排引擎 DialogueEngine（doc/01 §7 一轮对话完整数据流）。

流程所有者：把分析层 / 权重引擎 / Prompt 组装 / LLM / 存储 / 上下文压缩 / 人格演进 / kw_agent_map
串成一轮闭环。提供：
- ``start_session`` / ``switch_session`` / ``list_sessions``：多会话生命周期（UI 侧会话列表）。
- ``send``：同步一次性返回（CLI / 测试用）。
- ``stream_reply_sync``：同步生成器，逐事件产出 ``TurnEvent``（PyQt / API-SSE 用）。

设计原则：
- 分析失败静默降级（doc/01 D12）：沿用上一轮权重继续对话。
- 人格演进（doc/10）默认关闭，仅达到阶梯里程碑 + 证据门槛才触发，且受锚相似度/偏离预算约束。
- kw_agent_map（doc/03 §2.15）：由已验证耦合规则沉淀为「情绪/脾性 → Agent 特质」可学习映射，随对话涌现累积。
- safe_mode（doc/09 §6）：亲密类场景否决级硬约束，回复若越界则拒绝。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

from ..analysis.heuristics import detect_repetition, extract_heuristics
from ..analysis.llm_analyzer import AnalysisError, analyze
from ..analysis.trigger import AnalysisTrigger
from ..config.scenario_loader import load_scenario_by_id
from ..core.errors import LlmError
from ..core.events import (
    ChainStepEvent,
    DoneEvent,
    ErrorEvent,
    PersonaChangeEvent,
    TokenEvent,
    TurnEvent,
    WeightUpdateEvent,
)
from ..core.models import Evidence, Layer, WeightSnapshot
from ..core.ports import ChatMessage
from ..evolution import CandidateRegistry, discover_from_extractions
from ..memory import build_memory
from ..memory.store import ColdMemoryItem
from ..tools.executor import run_tool
from ..tools.protocol import ToolContext, ToolResult, validate_args as _validate_tool_args
from ..prompt.assembler import PromptAssembler
from ..prompt.context_compact import ContextWindow, compact, fill_ratio
from ..prompt.budget import estimate_tokens
from ..weighting.coupling import apply_rules
from ..weighting.engine import WeightEngine, WeightEngineConfig

# safe_mode 简易黑名单（doc/09 §6 硬约束兜底；生产应接更强审核）
_SAFE_BLOCKLIST = ["裸", "性爱", "色情", "做爱", "约炮"]

# 触发分析的显式反馈信号
_EVENT_WORDS = ["不对", "错了", "很好", "谢谢", "满意", "不满意", "喜欢这样", "别这样"]


@dataclass
class _Session:
    """单个会话的运行时状态（与 WeightEngine 实例一一绑定）。"""

    sid: str
    engine: WeightEngine
    mode: str = "fixed"
    scene_name: str = ""
    safe_mode: Optional[dict] = None
    greeting: str = ""
    turn: int = 0
    history_summaries: List[str] = field(default_factory=list)
    last_l3: Dict[str, float] = field(default_factory=dict)
    last_agent_text: str = ""
    recent_replies: List[str] = field(default_factory=list)
    pending_summary: str = ""
    baseline_spec: str = ""
    persona_specs: List[str] = field(default_factory=list)
    persona_last_turn: int = -999
    persona_drift: float = 0.0
    candidates: CandidateRegistry = field(default_factory=CandidateRegistry)


class DialogueEngine:
    def __init__(self, settings, lib, llm_client, repo=None, clock=None, tool_registry=None) -> None:
        self.settings = settings
        self.lib = lib
        self.llm = llm_client
        self.repo = repo
        self.clock = clock

        self.assembler = PromptAssembler(settings, lib.lexicon)
        self.trigger = AnalysisTrigger(
            cold_start_turns=settings.analysis_cold_start_turns,
            period=settings.analysis_period,
            enable_heuristics=settings.analysis_enable_heuristics,
        )
        # 冷记忆库（doc/07）：复用引擎已有的 SQLite 连接；无 repo（内存模式）时置 None
        self.memory = build_memory(self.repo.conn, self.settings) if self.repo is not None else None
        # 工具注册表（doc/08）：可选；引擎只消费契约，不参与工具实现
        self.tool_registry = tool_registry
        # 多会话：sid -> 运行时状态
        self._sessions: Dict[str, _Session] = {}
        self._active: Optional[str] = None

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
        # 防御：仅有开标签无闭标签时，截断到开标签之前，避免标签泄露进回复
        m2 = re.search(r"<turn_summary>", text)
        if m2:
            return text[: m2.start()].strip(), text[: m2.start()].strip()[:100]
        return text.strip(), text.strip()[:100]

    def _apply_safe_mode(self, text: str) -> str:
        if not self._safe_mode_active:
            return text
        if any(w in text for w in _SAFE_BLOCKLIST):
            return "（内容已按安全约束拦截，请换个健康的话题～）"
        return text

    @property
    def _safe_mode_active(self) -> bool:
        if self._active is None:
            return False
        return bool(self._sessions[self._active].safe_mode)

    # ---- 冷记忆检索（doc/07 §4）----
    def _retrieve_cold(self, sess: _Session, user_text: str, snapshot: WeightSnapshot) -> List[ColdMemoryItem]:
        """按触发策略检索冷记忆，返回命中的 ColdMemoryItem 列表（doc/07 §4.2/§4.3）。"""
        if not self.settings.memory_enable:
            return []
        if not self._should_retrieve_cold(sess):
            return []
        query = self._build_cold_query(user_text, snapshot)
        try:
            return self.memory.search(
                query,
                top_k=self.settings.memory_cold_top_k,
                sim_threshold=self.settings.memory_retrieve_sim_threshold,
                scope="all",
            )
        except Exception:  # noqa: BLE001 - 检索失败绝不阻断对话主链路
            return []

    def _should_retrieve_cold(self, sess: _Session) -> bool:
        trigger = self.settings.memory_retrieve_trigger
        period = max(1, self.settings.memory_retrieve_period)
        if trigger == "always":
            return True
        if trigger == "coldstart":
            return sess.turn <= 2
        if trigger == "periodic":
            return sess.turn % period == 0
        if trigger == "similarity":
            return True  # 阈值在 search 内部过滤
        return False

    def _build_cold_query(self, user_text: str, snapshot: WeightSnapshot) -> str:
        """查询 = 最新发言 + 当前 L2 活跃关键词（加权语义锚点，doc/07 §4.2）。"""
        parts = [user_text]
        for k, v in snapshot.l2.items():
            if v >= 0.3:
                kw = self.lib.lexicon.get(k)
                if kw:
                    parts.append(kw.name)
        return " ".join(parts)

    # ---- 会话生命周期 ----
    def start_session(
        self,
        mode: str = "fixed",
        scenario_id: Optional[str] = None,
        describe: Optional[str] = None,
        sid: Optional[str] = None,
    ) -> str:
        sid = sid or f"session-{int(self._now())}"
        sc = load_scenario_by_id(self.settings.scenario_dir, scenario_id or self.settings.scenario_default)
        eng = WeightEngine(
            self.lib,
            WeightEngineConfig(
                prior_strength=self.settings.weight_prior_strength,
                default_half_life_hours=self.settings.weight_default_half_life_hours,
                llm_fusion_beta=self.settings.weight_llm_fusion_beta,
                lvm_gamma=self.settings.lvm_l3_prior_gamma,
            ),
            self.clock,
        )
        eng.set_l1_scene(dict(sc.l1))
        for k, v in sc.l2_preset.items():
            if self.lib.lexicon.normalize(k):
                eng.set_preset(k, v)
        eng.set_l3_baseline(dict(sc.l3_baseline))

        sess = _Session(
            sid=sid, engine=eng, mode=mode,
            scene_name=sc.name, safe_mode=sc.safe_mode, greeting=sc.greeting,
        )
        baseline_snap = eng.compute_all(0)
        sess.last_l3 = dict(baseline_snap.l3)
        sess.baseline_spec = self._derive_baseline_spec(baseline_snap, sess)
        sess.persona_specs = [sess.baseline_spec]

        # 恢复历史摘要（若存在），使切换回旧会话时仍能"记得"主线
        if self.repo is not None:
            try:
                hist = self.repo.recent_summaries(sid, limit=50)
                sess.history_summaries = [h.text for h in reversed(hist)]
            except Exception:  # noqa: BLE001 - 降级：无历史则空
                pass
            self.repo.save_persona_spec(sid, 0, sess.baseline_spec, version=0, is_baseline=True)

        self._sessions[sid] = sess
        self._active = sid
        return sess.greeting

    def switch_session(self, sid: str) -> None:
        """切换到已有会话（不重建，沿用内存缓存或仅恢复历史摘要）。"""
        if sid in self._sessions:
            self._active = sid
            return
        # 不在内存：以该 sid 新建会话上下文；历史摘要从仓库恢复，权重重置基线
        self.start_session(sid=sid)

    def list_sessions(self) -> List[str]:
        ids = list(self._sessions.keys())
        if self.repo is not None:
            try:
                ids = self.repo.list_session_ids() or ids
            except Exception:  # noqa: BLE001
                pass
        return ids

    def _active_session(self) -> _Session:
        if self._active is None:
            self.start_session()
        return self._sessions[self._active]

    def _derive_baseline_spec(self, snap: WeightSnapshot, sess: _Session) -> str:
        parts = []
        for k, v in snap.l3.items():
            kw = self.lib.lexicon.get(k)
            if kw:
                parts.append(f"{kw.name}{v:.1f}")
        spec = f"初始人格（{sess.scene_name}）：" + "；".join(parts)
        return spec[: self.settings.persona_spec_max_chars]

    # ---- 前置流程（send / stream 共用）----
    def _prepare_turn(self, sess: _Session, user_text: str, now: float) -> dict:
        sess.turn += 1
        if self.repo is not None:
            self.repo.add_message(sess.sid, sess.turn, "user", user_text, now)

        chain: List[tuple] = []

        # 触发判定
        force = self._detect_event(user_text)
        do_analyze = self.trigger.should_analyze(sess.turn, force_event=force)
        chain.append(("trigger", {"turn": sess.turn, "do_analyze": do_analyze, "event": force}))

        new_candidates: List[str] = []
        if do_analyze:
            heur = extract_heuristics(user_text, sess.turn, now, self.lib.lexicon)
            for e in heur:
                sess.engine.add_evidence(e)
            try:
                res = analyze(self.llm, user_text, sess.last_agent_text, self.lib.lexicon)
                for ex in res.extractions:
                    sess.engine.add_evidence(
                        Evidence(key=ex["key"], intensity=ex["intensity"], timestamp=now, source="llm", turn=sess.turn)
                    )
                known, unknown = discover_from_extractions(res.extractions, self.lib.lexicon, sess.candidates)
                new_candidates = unknown
                sess.pending_summary = res.turn_summary
                chain.append(("analyze", {"heuristics": len(heur), "llm_extractions": len(res.extractions), "unknown": unknown}))
            except AnalysisError as e:
                chain.append(("analyze_error", {"error": str(e)}))  # 降级：沿用旧权重

        # 重算三层
        snapshot = sess.engine.compute_all(sess.turn)
        chain.append(("weights", {"l1": snapshot.l1, "l2": snapshot.l2, "l3": snapshot.l3}))

        # 稳定性
        delta = self._l3_distance(sess.last_l3, snapshot.l3)
        rebuild = delta >= self.settings.weight_rebuild_threshold
        notify = delta >= self.settings.weight_notify_threshold
        sess.last_l3 = snapshot.l3
        chain.append(("stability", {"delta": round(delta, 3), "rebuild": rebuild, "notify": notify}))

        # 组装 + 预算护栏
        history = list(sess.history_summaries)
        cold_items = self._retrieve_cold(sess, user_text, snapshot) if self.memory is not None else []
        assembled = self.assembler.assemble(
            snapshot, scene_name=sess.scene_name, greeting=sess.greeting,
            history=history, cold_memory=[it.display for it in cold_items],
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
            sess.history_summaries = list(new_window.history)
            history = list(new_window.history)
            ratio_after = fill_ratio(new_window, self.settings)
            if self.repo is not None:
                self.repo.log_compact(sess.sid, sess.turn, ratio, ratio_after, actions)
            chain.append(("budget_guard", {"ratio_before": round(ratio, 3), "actions": actions, "ratio_after": round(ratio_after, 3)}))
        else:
            chain.append(("budget_guard", {"ratio": round(ratio, 3), "triggered": False}))

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_text),
        ]
        return {
            "snapshot": snapshot, "chain": chain, "messages": messages,
            "notify": notify, "rebuild": rebuild, "delta": delta,
            "compact_actions": compact_actions, "new_candidates": new_candidates, "now": now,
        }

    def _persist(self, sess: _Session, user_text: str, reply_text: str, turn_summary: str, snapshot: WeightSnapshot, now: float) -> None:
        if self.repo is not None:
            self.repo.add_message(sess.sid, sess.turn, "agent", reply_text, self._now())
            self.repo.save_snapshot(sess.sid, snapshot, self._now())
            summary_to_save = turn_summary or sess.pending_summary or reply_text[:100]
            self.repo.save_summary(sess.sid, sess.turn, summary_to_save, "turn")
        sess.history_summaries.append(turn_summary or reply_text[:100])
        sess.last_agent_text = reply_text
        sess.pending_summary = ""
        # 落冷记忆库（doc/07 §4.1 ingestion）：本轮写入，供后续（含跨会话）检索
        if self.memory is not None:
            try:
                importance = max(snapshot.l2.values(), default=0.0)
                self.memory.add_turn(
                    sess.sid, sess.turn, user_text, reply_text, turn_summary,
                    importance=float(importance),
                )
            except Exception:  # noqa: BLE001 - 记忆写入失败不影响主流程
                pass

    # ---- 同步一次性入口（CLI / 测试）----
    def send(self, user_text: str) -> Tuple[str, dict]:
        sess = self._active_session()
        now = self._now()
        prep = self._prepare_turn(sess, user_text, now)
        messages = prep["messages"]
        reply_text, turn_summary, rep_hit = self._generate_with_guard(sess, messages, user_text)
        reply_text = self._apply_safe_mode(reply_text)
        sess.recent_replies.append(reply_text)
        sess.recent_replies = sess.recent_replies[-self.settings.repeat_recent_n:]
        self._persist(sess, user_text, reply_text, turn_summary, prep["snapshot"], now)
        self._maybe_update_persona(sess, prep["snapshot"], now)
        self._update_kwmap(sess, prep["snapshot"])
        meta = {
            "debug_chain": prep["chain"],
            "snapshot": prep["snapshot"],
            "new_candidates": prep["new_candidates"],
            "notify": prep["notify"],
            "compact_actions": prep["compact_actions"],
            "turn_summary": turn_summary,
        }
        return reply_text, meta

    # ---- 同步流式入口（PyQt / API-SSE）----
    def stream_reply_sync(self, user_text: str) -> Iterator[TurnEvent]:
        sess = self._active_session()
        now = self._now()
        try:
            prep = self._prepare_turn(sess, user_text, now)
        except Exception as exc:  # noqa: BLE001 - 绝不向上抛到 UI 线程
            yield ErrorEvent(message=f"处理出错：{exc}")
            return

        # 前置流程已算好权重，立即可刷新右侧面板
        yield WeightUpdateEvent(snapshot=prep["snapshot"])
        yield ChainStepEvent(step="prepare", detail={"chain": prep["chain"]})

        messages = prep["messages"]
        collected: List[str] = []
        emitted_len = 0
        past_tag = False
        try:
            for tok in self.llm.stream(
                messages,
                frequency_penalty=self.settings.repeat_freq_penalty,
                presence_penalty=self.settings.repeat_presence_penalty,
            ):
                collected.append(tok)
                full = "".join(collected)
                if not past_tag:
                    tag_pos = full.find("<turn_summary>")
                    if tag_pos != -1:
                        # 回复主体到此为止：推送标签之前的内容，并停止向 UI 推送可见 token；
                        # 但仍继续收集剩余 token（summary 与闭标签），以便结尾正确切分
                        if tag_pos > emitted_len:
                            yield TokenEvent(text=full[emitted_len:tag_pos])
                        emitted_len = tag_pos
                        past_tag = True
                        continue
                    if len(full) > emitted_len:
                        yield TokenEvent(text=full[emitted_len:])
                        emitted_len = len(full)
        except LlmError:
            yield ErrorEvent(message="生成失败，请稍后再试～")
            return
        except Exception as exc:  # noqa: BLE001
            yield ErrorEvent(message=f"生成异常：{exc}")
            return

        full = "".join(collected)
        reply_text, turn_summary = self._split_summary(full)

        # 重复护栏（doc/01 §9）
        hit, _metrics = detect_repetition(
            reply_text, sess.recent_replies,
            self.settings.repeat_sim_threshold, self.settings.repeat_self_ngram_ratio,
        )
        final_text = reply_text
        if hit:
            retry = messages + [ChatMessage(role="system", content="请勿重复之前的话，换种表达方式，给出新的回应。")]
            try:
                r2, ts2 = self._split_summary(
                    self.llm.complete(retry, frequency_penalty=self.settings.repeat_freq_penalty,
                                      presence_penalty=self.settings.repeat_presence_penalty).content
                )
                final_text = r2
                turn_summary = ts2 or turn_summary
            except LlmError:
                final_text = self.settings.repeat_degrade_msg
        final_text = self._apply_safe_mode(final_text)
        sess.recent_replies.append(final_text)
        sess.recent_replies = sess.recent_replies[-self.settings.repeat_recent_n:]
        self._persist(sess, user_text, final_text, turn_summary, prep["snapshot"], now)
        self._maybe_update_persona(sess, prep["snapshot"], now)
        self._update_kwmap(sess, prep["snapshot"])

        if prep["notify"]:
            yield PersonaChangeEvent(delta=prep["delta"], action="rebuilt" if prep["rebuild"] else "shifted")
        yield DoneEvent(
            turn=sess.turn, final_text=final_text, rep_hit=hit, notify=prep["notify"],
            summary=turn_summary, candidate_count=len(prep["new_candidates"]),
            compact_actions=prep["compact_actions"],
        )

    # ---- 生成与重复护栏 ----
    def _generate_with_guard(self, sess: _Session, messages: List[ChatMessage], user_text: str):
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
            reply_text, sess.recent_replies,
            self.settings.repeat_sim_threshold, self.settings.repeat_self_ngram_ratio,
        )
        if hit:
            retry = messages + [ChatMessage(role="system", content="请勿重复之前的话，换种表达方式，给出新的回应。")]
            try:
                resp2 = self.llm.complete(retry, frequency_penalty=self.settings.repeat_freq_penalty, presence_penalty=self.settings.repeat_presence_penalty)
                r2, ts2 = self._split_summary(resp2.content)
                hit2, _ = detect_repetition(r2, sess.recent_replies, self.settings.repeat_sim_threshold, self.settings.repeat_self_ngram_ratio)
                if not hit2:
                    return r2, ts2, True
            except LlmError:
                pass
            return self.settings.repeat_degrade_msg, "（降级兜底）", True
        return reply_text, turn_summary, False

    # ---- 人格演进（doc/10，默认关）----
    def _maybe_update_persona(self, sess: _Session, snapshot: WeightSnapshot, now: float) -> None:
        if not self.settings.persona_auto_update:
            return
        stages = self.settings.persona_stages
        if sess.turn not in stages:
            return
        if (sess.turn - sess.persona_last_turn) < 8:
            return
        # 证据门槛（用户参与度，用 L2 累积证据近似）
        user_evidence = sum(sess.engine._e_total(k) for k in snapshot.l2) if snapshot.l2 else 0.0
        if user_evidence < self.settings.persona_min_evidence:
            return
        spec = self._derive_persona_spec(sess, snapshot)
        if not spec:
            return
        # 锚相似度校验
        sim = _jaccard_chars(sess.baseline_spec, spec)
        if sim < self.settings.persona_min_anchor_sim:
            return
        if sess.persona_drift + (1.0 - sim) > self.settings.persona_max_drift:
            return
        sess.persona_drift += 1.0 - sim
        sess.persona_last_turn = sess.turn
        sess.persona_specs.append(spec)
        if self.repo is not None:
            self.repo.save_persona_spec(sess.sid, sess.turn, spec, version=len(sess.persona_specs) - 1)

    def _derive_persona_spec(self, sess: _Session, snapshot: WeightSnapshot) -> str:
        """生成本轮人格规格书（≤200 字，doc/10）。"""
        parts = []
        for k, v in snapshot.l3.items():
            kw = self.lib.lexicon.get(k)
            if kw:
                parts.append(f"{kw.name}{v:.1f}")
        spec = f"人格（第{sess.turn}轮）：" + "；".join(parts)
        return spec[: self.settings.persona_spec_max_chars]

    # ---- kw_agent_map 涌现（doc/03 §2.15）----
    def _update_kwmap(self, sess: _Session, snapshot: WeightSnapshot) -> None:
        if self.repo is None:
            return
        eff = apply_rules(self.lib.rules, sess.engine.active_l1(), snapshot.l2, self.lib.lexicon)
        for src, dst, w in eff.maps:
            if snapshot.l2.get(src, 0.0) < 0.3:
                continue
            if dst.startswith("user_"):
                continue
            self.repo.kwmap_upsert(src, dst, "boost", w, learn_rate=self.settings.kwmap_lr)

    # ---- 工具插件（doc/08）----
    def list_tools(self) -> List[dict]:
        """列出已注册工具（供 API / CLI 展示）。"""
        if self.tool_registry is None:
            return []
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "dangerous": t.dangerous,
            }
            for t in self.tool_registry.all()
        ]

    def call_tool(self, name: str, args: dict, session_id: Optional[str] = None, timeout: float = 10.0) -> ToolResult:
        """按需调用工具（doc/08 §5）。

        进行中对话在轮次开始处 ``snapshot`` 捕获当前工具集引用，热更新不影响正在跑的对话；
        执行经 ``validate_args`` 校验与 ``run_tool`` 超时熔断；调用记入 ``tool_log``（doc/08 G6）。
        """
        if self.tool_registry is None:
            return ToolResult(ok=False, error="工具系统未启用")
        sid = session_id or (self._active if self._active else "session-0")
        ctx = ToolContext(
            session_id=sid,
            repo=self.repo,
            settings=self.settings,
            memory=self.memory,
            clock=self.clock,
            llm=self.llm,
        )
        # 原子快照：持旧引用，热更新不影响本次调用
        version = self.tool_registry.snapshot()
        tools = self.tool_registry.get_snapshot(version)
        tool = tools.get(name)
        if tool is None:
            return ToolResult(ok=False, error=f"未找到工具: {name}")
        err = _validate_tool_args(tool, args)
        if err:
            return ToolResult(ok=False, error=err)
        result = run_tool(tool, args, ctx, timeout=timeout)
        if self.repo is not None:  # 可观测：tool_log
            try:
                self.repo.log_tool_call(
                    sid, name,
                    json.dumps(args, ensure_ascii=False),
                    int(result.ok), result.error or "",
                )
            except Exception:  # noqa: BLE001 - 日志失败不影响主流程
                pass
        return result


def _jaccard_chars(a: str, b: str) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)
