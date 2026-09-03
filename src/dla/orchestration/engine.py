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
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Set, Tuple

from ..analysis.bootstrap import bootstrap, should_degrade_to_auto
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
from ..prompt.context_compact import (
    ContextWindow,
    compact,
    fill_ratio,
    trigger_level_of,
    window_parts,
    window_tokens,
)
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
    # free 模式 Bootstrap（doc/02 §3.6）
    bootstrap_seeds: Optional[object] = None   # BootstrapSeeds | None
    bootstrap_notes: List[str] = field(default_factory=list)
    # L1 场景锁定（doc/02 §3.5）：scene_ops 建立到稳定后不再随意漂移
    l1_locked: bool = False
    # 运行期词库操作工作集追踪（doc/06 §4.2/§4.4 护栏）：模板词受保护、仅可删 auto_added
    l1_template_keys: Set[str] = field(default_factory=set)
    l3_template_keys: Set[str] = field(default_factory=set)
    l1_auto_added: Set[str] = field(default_factory=set)
    l3_auto_added: Set[str] = field(default_factory=set)


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

    # ---- 工具自动触发（doc/08 §4/§9）----
    def _run_tool_step(self, sess: _Session, user_text: str):
        """确定性工具路由/调用：仅自动触发**只读**工具（有副作用的工具需显式调用，doc/08 §5）。

        返回 (detail_blocks, tool_schema, invoked_names)。结果作为「补充细节」注入当轮 Prompt；
        tool_schema 把可用工具暴露给真实 LLM 做 function-calling。绝不参与权重计算。
        """
        detail_blocks: List[str] = []
        tool_schema_parts: List[str] = []
        invoked: List[str] = []
        if not self.settings.tools_enabled or self.tool_registry is None:
            return detail_blocks, "", invoked
        ctx = ToolContext(
            session_id=sess.sid, repo=self.repo, settings=self.settings,
            memory=self.memory, clock=self.clock, llm=self.llm,
        )
        # 捕获当前快照引用：本轮后续即便发生热更新也用这一份（doc/08 §3.1）
        snap = self.tool_registry.snapshot()
        for t in snap.values():
            if not self.tool_registry.is_enabled(t.name):
                continue  # 已禁用（含危险工具未显式 enable）不暴露给 LLM
            tool_schema_parts.append(f"- {t.name}: {t.description}")
            if not t.is_readonly:
                continue  # 有副作用工具不自动触发
            try:
                score = t.can_handle(user_text, ctx)
            except Exception:  # noqa: BLE001
                score = 0.0
            if score < self.settings.tools_auto_threshold:
                continue
            args: dict = {}
            if "query" in (t.parameters.get("properties") or {}):
                args["query"] = user_text
            res = self.call_tool(t.name, args, session_id=sess.sid)
            if res.ok and res.content:
                detail_blocks.append(f"[{t.name}] {res.content}")
                invoked.append(t.name)
        tool_schema = "可用工具（如需调用请说明意图）：\n" + "\n".join(tool_schema_parts) if tool_schema_parts else ""
        return detail_blocks, tool_schema, invoked

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
        # ---- 模式解析（doc/02 §3.5 auto / §3.6 free）----
        # 历史缺陷：mode 此前只被存进 _Session，**全代码零读取**，三种模式行为完全一致。
        eff_mode = mode
        boot_notes: List[str] = []
        if should_degrade_to_auto(mode, describe, self.settings.mode_free_require_desc):
            eff_mode = "auto"
            boot_notes.append("free 模式缺少场景描述且 FREE_REQUIRE_DESC=true → 降级为 auto")

        seeds = None
        if eff_mode == "fixed":
            # 加载场景模板三层（既有行为）
            eng.set_l1_scene(dict(sc.l1))
            for k, v in sc.l2_preset.items():
                if self.lib.lexicon.normalize(k):
                    eng.set_preset(k, v)
            eng.set_l3_baseline(dict(sc.l3_baseline))
        elif eff_mode == "free":
            seeds = self._bootstrap_seeds(describe, eng, boot_notes)
            if seeds is None:
                eff_mode = "auto"  # Bootstrap 失败 → 静默降级为 auto（doc/02 §3.6）
        # auto：不加载模板三层，L1 留空由 scene_ops 在对话中建立

        # 模板工作集（受保护，agent_ops/scene_ops 的 delete 不可动，update 仅可微调）
        l1_tpl: Set[str] = set()
        l3_tpl: Set[str] = set()
        if eff_mode == "fixed":
            l1_tpl = set(sc.l1.keys())
            l3_tpl = set(sc.l3_baseline.keys())
        elif eff_mode == "free" and seeds is not None:
            l1_tpl = set(seeds.l1.keys())
            l3_tpl = set(seeds.l3.keys())
        # auto：模板三层为空，工作集由 scene_ops 在对话中逐步建立

        if eff_mode == "auto":
            scene_name = ""
            greeting = self._pick_fallback_greeting()
        elif eff_mode == "free" and seeds is not None:
            scene_name = self._bootstrap_scene_name(describe, sc)
            greeting = self._generate_greeting(describe, seeds)
            # L3 种子作 w3_base 一次性写入（doc/02 §3.6：等同加载一个隐式场景模板基线）
            eng.set_l3_baseline(dict(seeds.l3))
        else:
            scene_name = sc.name
            greeting = sc.greeting

        sess = _Session(
            sid=sid, engine=eng, mode=eff_mode,
            scene_name=scene_name,
            # safe_mode 是 doc/09 §6 否决级硬约束，**任何模式都继承**，不可因无预设模板而丢失
            safe_mode=sc.safe_mode,
            greeting=greeting,
            # 模板工作集（受保护：scene_ops/agent_ops 的 delete 不可动，update 仅可微调）
            l1_template_keys=l1_tpl,
            l3_template_keys=l3_tpl,
        )
        if boot_notes:
            sess.bootstrap_notes = boot_notes
        if seeds is not None:
            sess.bootstrap_seeds = seeds
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

    # ---- 自由模式 Bootstrap（doc/02 §3.6 / 决策 D26）----
    def _bootstrap_seeds(self, describe: Optional[str], eng, notes: List[str]):
        """调用 LLM 由场景描述生成三层种子并注入引擎。失败返回 None（由调用方降级）。

        ``describe`` 参数此前被 ``start_session`` **完全忽略**，free 模式因此从未生效。
        """
        try:
            seeds = bootstrap(
                self.llm, describe or "", self.lib.lexicon,
                model=self.settings.bootstrap_model or None,
            )
        except AnalysisError as e:
            notes.append(f"Bootstrap 失败，降级为 auto：{e}")
            return None
        except Exception as e:  # noqa: BLE001 - LLM 不可用时静默降级，绝不阻塞会话
            notes.append(f"Bootstrap 异常，降级为 auto：{e}")
            return None

        conf = float(self.settings.mode_free_bootstrap_conf)

        # L1 播种：进入活跃场景集，src_confidence = bootstrap（默认 0.60）
        if seeds.l1:
            eng.set_l1_scene(dict(seeds.l1))
            for k in seeds.l1:
                eng.set_src_conf(k, conf)
        # L2 播种：弱种子作冷启动锚（user_temper./user_mood. 已在护栏层丢弃）
        for k, v in seeds.l2.items():
            eng.set_preset(k, v)
            eng.set_src_conf(k, conf)
        # L3 由调用方写入 baseline（w3_base）；此处不缩放，等同隐式场景模板基线

        if seeds.rejected:
            notes.append(f"Bootstrap 丢弃 {len(seeds.rejected)} 项（白名单/层归属/禁词护栏）")
        if not seeds.total:
            notes.append("Bootstrap 未产出任何合法种子")
        return seeds

    def _bootstrap_scene_name(self, describe: Optional[str], sc) -> str:
        """free 模式无预设模板，场景名直接取自描述摘要（供 Prompt 展示用）。"""
        text = (describe or "").strip().replace("\n", " ")
        short = text[:24] + ("…" if len(text) > 24 else "")
        return f"自由场景·{short}" if short else sc.name

    def _pick_fallback_greeting(self) -> str:
        """兜底称呼（doc/02 §12）：auto 模式 L1 为空时使用。"""
        try:
            pool = self.settings.greeting_fallback_list
        except Exception:  # noqa: BLE001
            pool = ["你好"]
        return pool[0] if pool else "你好"

    def _generate_greeting(self, describe: Optional[str], seeds) -> str:
        """由场景描述 + Bootstrap 种子生成贴合场景的开场称呼（doc/02 §12）。

        失败静默降级到兜底池（D12），绝不阻塞对话。
        """
        if not self.settings.greeting_enable:
            return self._pick_fallback_greeting()
        keys = list(seeds.l1)[:8] + list(seeds.l3)[:5]
        hint = "、".join(keys) if keys else "（无）"
        messages = [
            ChatMessage(role="system", content=(
                "你是开场称呼生成器。根据场景描述与已播种的关键词，"
                "生成一句简洁、贴合场景的中文开场欢迎语。只输出这一句话，不要解释、不要加引号。"
            )),
            ChatMessage(role="user", content=f"场景描述：{describe}\n已播种关键词：{hint}"),
        ]
        try:
            resp = self.llm.complete(
                messages, temperature=0.7,
                **({"model": self.settings.bootstrap_model} if self.settings.bootstrap_model else {}),
            )
            text = (resp.content or "").strip().strip("\"'")
            if text:
                return text[:120]
        except Exception:  # noqa: BLE001 - 与 D12 一致：静默降级
            pass
        return self._pick_fallback_greeting()

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
                # 消费 scene_ops(L1) / agent_ops(L3)（#37：此前被丢弃、白白消耗 token）
                ops_applied = 0
                ops_applied += self._consume_ops(sess, res.scene_ops, Layer.L1)
                ops_applied += self._consume_ops(sess, res.agent_ops, Layer.L3)
                chain.append(("analyze", {"heuristics": len(heur), "llm_extractions": len(res.extractions), "unknown": unknown, "ops_applied": ops_applied}))
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
        cold_displays = [it.display for it in cold_items]
        detail_blocks, tool_schema, invoked_tools = self._run_tool_step(sess, user_text)
        assembled = self.assembler.assemble(
            snapshot, scene_name=sess.scene_name, greeting=sess.greeting,
            history=history, cold_memory=cold_displays,
            detail_blocks=detail_blocks, tool_schema=tool_schema,
        )
        core = assembled["core"]
        system_prompt = assembled["system_prompt"]
        chain.append(("tools", {"invoked": invoked_tools}))
        chain.append(("assemble", assembled["debug"]))

        # ⑦ budget_guard（doc-11 §3 step 7½）：核心段 + 各附加段独立计 token，避免双重计数
        window = ContextWindow(
            system_prompt=core, history=history, cold_memory=cold_displays,
            detail_blocks=detail_blocks, tool_schema=tool_schema,
            current_user_msg=user_text,
        )
        ratio = fill_ratio(window, self.settings)
        parts = window_parts(window, self.settings)  # 分项估算（doc-11 §9 调试帧）
        compact_actions: List[str] = []
        if self.settings.ctx_auto_compact and ratio >= self.settings.ctx_compact_ratio:
            new_window, actions = compact(window, self.settings)
            compact_actions = actions
            # 用压缩后的字段【重新组装】system_prompt（修复此前只改 window 不重组的空操作 bug）
            new_extras = self.assembler.format_extras(
                history=new_window.history, cold_memory=new_window.cold_memory,
                detail_blocks=new_window.detail_blocks, tool_schema=new_window.tool_schema,
            )
            system_prompt = self.assembler.render_system(core, new_extras)
            sess.history_summaries = list(new_window.history)
            history = list(new_window.history)
            ratio_after = fill_ratio(new_window, self.settings)
            if self.repo is not None:
                try:
                    self.repo.log_compact(
                        sess.sid, sess.turn, ratio, ratio_after, actions,
                        trigger_level=trigger_level_of(ratio, self.settings),
                        tokens_before=window_tokens(window),
                        tokens_after=window_tokens(new_window),
                    )
                except Exception:  # noqa: BLE001 - 日志失败不影响主流程
                    pass
            chain.append(("budget_guard", {
                "ratio_before": round(ratio, 3), "actions": actions,
                "ratio_after": round(ratio_after, 3), "parts_before": parts,
            }))
        else:
            chain.append(("budget_guard", {"ratio": round(ratio, 3), "triggered": False, "parts": parts}))

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_text),
        ]
        return {
            "snapshot": snapshot, "chain": chain, "messages": messages,
            "notify": notify, "rebuild": rebuild, "delta": delta,
            "compact_actions": compact_actions, "new_candidates": new_candidates, "now": now,
            "invoked_tools": invoked_tools,
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
        # ---- LLM function-calling 二级路由（doc/08 §4.3）----
        # 若 LLM 在回复中以 <tool_call> 语法显式指定调用，经 registry 派发执行，结果回灌后重生成；
        # TOOL_MAX_LOOPS 防死循环。与本轮自动触发的只读工具去重。
        already_invoked = set(prep.get("invoked_tools", []))
        executed: List[Tuple[str, bool]] = []
        if self.settings.tools_enabled and self.tool_registry is not None:
            calls = self._parse_tool_calls(reply_text)
            if calls:
                loop_messages = list(messages)
                loop_assistant = reply_text
                loops = 0
                while calls and loops < self.settings.memory_tool_max_loops:
                    results = self._run_tool_calls(sess, calls, already_invoked)
                    executed = [(n, r.ok) for n, r in results]
                    block = self._tool_results_block(results)
                    if not block:
                        break
                    loop_messages = loop_messages + [
                        ChatMessage(role="assistant", content=loop_assistant),
                        ChatMessage(role="user", content="（工具返回结果）\n" + block + "\n请据此给出最终回应。"),
                    ]
                    try:
                        regen = self.llm.complete(
                            loop_messages,
                            frequency_penalty=self.settings.repeat_freq_penalty,
                            presence_penalty=self.settings.repeat_presence_penalty,
                        ).content
                    except Exception:  # noqa: BLE001
                        break
                    loop_assistant, _ = self._split_summary(regen)
                    reply_text = loop_assistant
                    calls = self._parse_tool_calls(loop_assistant)
                    loops += 1
                prep["chain"].append(("tool_calls", {"executed": executed}))
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

    # ---- 运行期词库操作消费（doc/06 §4.2/§4.4 / #37）----
    def _consume_ops(self, sess: _Session, ops: list, target_layer: Layer) -> int:
        """消费分析器产出的 scene_ops(L1) 或 agent_ops(L3)，带护栏并审计落库。

        返回本次**成功生效**的操作数。固定模式跳过（场景/人格锁定）。

        护栏（任一违例 → applied=0，记录原因）：
        - op 类型非法 / 词表外（非新维度才自动收编，新维度走审核不自动收编）→ 拒绝；
        - 层归属不匹配 → 拒绝；
        - update：``|delta| ≤ 0.4``，且目标须在当前运行期工作集内；
        - delete：仅可删 auto_added（运行期新增）词，模板基础词受保护。
        """
        if not ops:
            return 0
        if sess.mode == "fixed":
            return 0  # 固定模式：场景与人格均锁定，任何运行期增删改都不生效
        eng = sess.engine
        lex = self.lib.lexicon
        applied = 0
        for op in ops:
            if not isinstance(op, dict):
                continue
            op_type = str(op.get("op") or "").lower()
            key = op.get("key")
            canon = lex.normalize(key) if key else None
            kw = lex.get(canon) if canon else None
            payload = json.dumps(op, ensure_ascii=False)

            if op_type not in ("add", "update", "delete"):
                self._log_op(sess, op_type or "unknown", target_layer.value, canon, payload,
                             op.get("reason", ""), applied=False, reject_reason="unknown_op")
                continue

            # 层归属校验
            if kw is not None and kw.layer != target_layer:
                self._log_op(sess, op_type, target_layer.value, canon, payload,
                             op.get("reason", ""), applied=False, reject_reason="layer_mismatch")
                continue

            # 词表外 → 走审核流程，不自动收编（doc/06 §4.2/§4.4：新维度不自动收编）
            if canon is None:
                self._log_op(sess, op_type, target_layer.value, None, payload,
                             op.get("reason", ""), applied=False, reject_reason="out_of_lexicon_review")
                continue

            if target_layer == Layer.L1:
                working_set = eng.active_l1()
                tpl = sess.l1_template_keys
                auto_added = sess.l1_auto_added
            else:
                working_set = eng.l3_working_keys()
                tpl = sess.l3_template_keys
                auto_added = sess.l3_auto_added

            if op_type == "add":
                intensity = max(0.0, min(1.0, float(op.get("intensity", 0.6))))
                if target_layer == Layer.L1:
                    eng.add_l1_key(canon, intensity)
                    auto_added.add(canon)
                else:
                    eng.add_l3_key(canon, intensity)
                    auto_added.add(canon)
                # ops 为 LLM 推断，映射 doc/02 §3.2 inferred=0.50
                eng.set_src_conf(canon, 0.5)
                self._log_op(sess, "add", target_layer.value, canon, payload,
                             op.get("reason", ""), applied=True)
                applied += 1

            elif op_type == "update":
                delta = float(op.get("delta", 0.0))
                if abs(delta) > 0.4:
                    self._log_op(sess, "update", target_layer.value, canon, payload,
                                 op.get("reason", ""), applied=False, reject_reason="delta_exceed_limit")
                    continue
                if canon not in working_set:
                    self._log_op(sess, "update", target_layer.value, canon, payload,
                                 op.get("reason", ""), applied=False, reject_reason="not_in_working_set")
                    continue
                ok = eng.update_l1_key(canon, delta) if target_layer == Layer.L1 else eng.update_l3_key(canon, delta)
                if ok:
                    self._log_op(sess, "update", target_layer.value, canon, payload,
                                 op.get("reason", ""), applied=True)
                    applied += 1
                else:
                    self._log_op(sess, "update", target_layer.value, canon, payload,
                                 op.get("reason", ""), applied=False, reject_reason="update_failed")

            else:  # delete
                if canon in tpl:
                    self._log_op(sess, "delete", target_layer.value, canon, payload,
                                 op.get("reason", ""), applied=False, reject_reason="protected_template_key")
                    continue
                if canon not in auto_added:
                    self._log_op(sess, "delete", target_layer.value, canon, payload,
                                 op.get("reason", ""), applied=False, reject_reason="not_auto_added")
                    continue
                if target_layer == Layer.L1:
                    eng.remove_l1_key(canon)
                    auto_added.discard(canon)
                else:
                    eng.remove_l3_key(canon)
                    auto_added.discard(canon)
                self._log_op(sess, "delete", target_layer.value, canon, payload,
                             op.get("reason", ""), applied=True)
                applied += 1

        return applied

    def _log_op(self, sess: _Session, op_type: str, layer: str, target_key, payload: str,
                llm_reason: str, *, applied: bool, reject_reason: str = "") -> None:
        """落一条词库操作审计（doc/03 §2.12）。审计失败不影响主流程。"""
        if self.repo is None:
            return
        reason = llm_reason or (f"rejected:{reject_reason}" if not applied else "")
        try:
            self.repo.log_lexicon_op(
                sess.sid, sess.turn, op_type, layer, target_key, payload, reason,
                1 if applied else 0,
            )
        except Exception:  # noqa: BLE001
            pass

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
        # 原子快照：持旧引用，热更新不影响本次调用（doc/08 §3.1）
        snap = self.tool_registry.snapshot()
        tool = snap.get(name)
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

    # ---- LLM function-calling 二级路由（doc/08 §4.3）----
    _TOOL_CALL_RE = re.compile(
        r"<tool_call\s+name=\"([^\"]+)\"\s+args=('|\")(.*?)\2\s*/?>", re.S
    )

    def _parse_tool_calls(self, text: str) -> List[Tuple[str, dict]]:
        """解析 LLM 回复中的 function-calling 语法 ``<tool_call name="x" args='{...}' />``。

        返回 ``[(name, args_dict), ...]``；args 解析失败则跳过该条（不阻断对话）。
        """
        out: List[Tuple[str, dict]] = []
        for m in self._TOOL_CALL_RE.finditer(text or ""):
            name = m.group(1)
            raw = m.group(3)
            try:
                args = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                # 兼容无 JSON 的裸参（如 args='关键词'）
                args = {"query": raw} if raw else {}
            if isinstance(args, dict):
                out.append((name, args))
        return out

    def _run_tool_calls(self, sess: _Session, calls: List[Tuple[str, dict]], already_invoked: set) -> List[Tuple[str, ToolResult]]:
        """派发 LLM 选定的工具调用（doc/08 §4.3）。

        - 只读工具或已显式 enable 的工具方可执行；危险/disabled 工具拒绝并记错误。
        - 与自动触发去重（``already_invoked`` 含本轮已自动触发的只读工具），避免重复调用。
        - 经 ``call_tool`` → validate_args + 超时熔断 + tool_log。
        """
        results: List[Tuple[str, ToolResult]] = []
        if self.tool_registry is None:
            return results
        for name, args in calls:
            if name in already_invoked:
                continue
            tool = self.tool_registry.get(name)
            if tool is None:
                if self.tool_registry is not None and self.tool_registry.is_registered(name):
                    results.append((name, ToolResult(ok=False, error=f"工具已禁用（危险工具需显式 enable）: {name}")))
                else:
                    results.append((name, ToolResult(ok=False, error=f"未找到工具: {name}")))
                continue
            if not tool.is_readonly and not self.tool_registry.is_enabled(name):
                results.append((name, ToolResult(ok=False, error=f"危险工具需显式 enable: {name}")))
                continue
            res = self.call_tool(name, args, session_id=sess.sid)
            results.append((name, res))
            already_invoked.add(name)
        return results

    @staticmethod
    def _tool_results_block(results: List[Tuple[str, ToolResult]]) -> str:
        parts = []
        for name, res in results:
            if res.ok and res.content:
                parts.append(f"[{name}] {res.content}")
        return "\n".join(parts)

    def force_compact(self, sid: Optional[str] = None) -> Optional[dict]:
        """手动触发一次上下文压缩（doc-11；CLI ``dla ctx compact --force`` 用）。

        返回 ``{"triggered", "ratio_before", "ratio_after", "actions"}``；未达阈值则 triggered=False。
        """
        sid = sid or self._active
        if sid is None or sid not in self._sessions:
            return None
        sess = self._sessions[sid]
        history = list(sess.history_summaries)
        snapshot = sess.engine.compute_all(sess.turn)
        assembled = self.assembler.assemble(
            snapshot, scene_name=sess.scene_name, greeting=sess.greeting,
            history=history, cold_memory=[], detail_blocks=[], tool_schema="",
        )
        core = assembled["core"]
        window = ContextWindow(
            system_prompt=core, history=history, cold_memory=[],
            detail_blocks=[], tool_schema="", current_user_msg="",
        )
        ratio_before = fill_ratio(window, self.settings)
        if ratio_before < self.settings.ctx_compact_ratio:
            return {"triggered": False, "ratio_before": ratio_before, "ratio_after": ratio_before, "actions": []}
        new_window, actions = compact(window, self.settings)
        # 手动压缩：更新会话摘要链（原文已落冷库无损），并记入日志
        sess.history_summaries = list(new_window.history)
        ratio_after = fill_ratio(new_window, self.settings)
        if self.repo is not None:
            try:
                self.repo.log_compact(
                    sess.sid, sess.turn, ratio_before, ratio_after, actions,
                    trigger_level="MANUAL",  # CLI `dla ctx compact --force` 显式触发
                    tokens_before=window_tokens(window),
                    tokens_after=window_tokens(new_window),
                )
            except Exception:  # noqa: BLE001
                pass
        return {"triggered": True, "ratio_before": ratio_before, "ratio_after": ratio_after, "actions": actions}


def _jaccard_chars(a: str, b: str) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)
