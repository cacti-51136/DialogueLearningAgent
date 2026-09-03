"""配置加载（doc/01 §8）。

设计取舍：文档指定用 ``pydantic-settings``，但为保证**离线零依赖可运行**，这里用标准库
自实现等价功能（``os.environ`` + 简单 ``.env`` 解析 + dataclass）。若后续安装了
``pydantic-settings``，可平滑替换而接口不变。

所有环境变量前缀 ``DLA_``，段间用双下划线分隔（如 ``DLA_WEIGHT__PRIOR_STRENGTH``）。
启动缺失必填项时由调用方（CLI）决定如何处理；本模块只做类型转换与默认值填充。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

T = TypeVar("T")


def _load_dotenv(path: str = ".env") -> None:
    """把 .env 中尚未在环境中出现的键值注入 os.environ（不覆盖已有）。"""
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip()
                if not key:
                    continue
                val = val.strip('"').strip("'")
                os.environ.setdefault(key, val)
    except FileNotFoundError:
        pass


def _as_bool(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on", "y")


def _as_list(v: str, sep: str = ",") -> list[str]:
    return [x.strip() for x in v.split(sep) if x.strip()]


@dataclass
class Settings:
    # ---- LLM ----
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_analyzer_model: str = ""  # 留空复用 llm_model
    llm_timeout_seconds: int = 60
    llm_max_retries: int = 2

    # ---- 存储 ----
    db_path: str = "./data/dla.db"
    db_auto_migrate: bool = True

    # ---- 权重引擎 ----
    weight_prior_strength: float = 2.0  # K
    weight_default_half_life_hours: float = 6.0
    weight_llm_fusion_beta: float = 0.3  # β
    weight_rebuild_threshold: float = 0.15  # τ_rebuild
    weight_notify_threshold: float = 0.40  # τ_notify
    weight_rebuild_cooldown_turns: int = 2

    # ---- Prompt 预算 ----
    prompt_total_token_budget: int = 900
    prompt_l1_ratio: float = 0.25
    prompt_l2_ratio: float = 0.20
    prompt_l3_ratio: float = 0.55

    # ---- 上下文自动压缩（doc/11）----
    ctx_max_tokens: int = 128000
    ctx_reserve: float = 0.20
    ctx_tokenizer: str = "deterministic"
    ctx_warn_ratio: float = 0.70
    ctx_compact_ratio: float = 0.85
    ctx_hard_ratio: float = 0.95
    ctx_auto_compact: bool = True
    ctx_summary_compact_after: int = 40
    ctx_epoch_merge_n: int = 10
    ctx_epoch_max_chars: int = 200
    ctx_cold_trim_top_m: int = 2

    # ---- 分析触发 ----
    analysis_cold_start_turns: int = 2
    analysis_period: int = 3
    analysis_enable_heuristics: bool = True

    # ---- 本地向量化与在线学习（doc/06）----
    lvm_dim: int = 64
    lvm_backbone: str = ""  # 留空=随机初始化
    lvm_learning_rate: float = 0.01
    lvm_lr_decay: float = 0.0
    lvm_momentum: float = 0.9
    lvm_grad_clip: float = 1.0
    lvm_weight_decay: float = 1e-4
    lvm_temp: float = 0.2
    lvm_l3_prior_gamma: float = 0.2  # γ
    lvm_top_t_identity: int = 6
    lvm_margin_lambda: float = 0.1
    lvm_replay_buffer: int = 256
    lvm_train_every_n_turns: int = 1

    # ---- 运行模式 ----
    mode_scenario: str = "fixed"  # fixed | auto | free
    mode_auto_warmup_turns: int = 3
    mode_free_bootstrap_conf: float = 0.6
    mode_free_require_desc: bool = True
    mode_free_bootstrap_model: str = ""  # 留空复用 analyzer_model（可用更便宜模型做 Bootstrap）

    # ---- 对话历史冷热记忆（doc/07）----
    memory_enable: bool = True
    memory_hot_window: int = 6
    memory_cold_top_k: int = 4
    memory_embed_dim: int = 384
    memory_embed_backbone: str = ""
    memory_retrieve_trigger: str = "periodic"
    memory_retrieve_period: int = 3
    memory_retrieve_sim_threshold: float = 0.3
    memory_prompt_ratio: float = 0.15
    memory_compact_every_n_sessions: int = 10
    memory_importance_decay_days: int = 30
    memory_checkpointer: str = "sqlite"
    memory_inject_summaries_only: bool = True
    memory_summary_max_chars: int = 100
    memory_tool_retrieve_top_k: int = 3
    memory_tool_max_loops: int = 2

    # ---- 预设场景与角色模板库（doc/09）----
    scenario_dir: str = "config/scenarios"
    scenario_default: str = "oral_practice"

    # ---- 人格演进·受控自动补充（doc/10）----
    persona_auto_update: bool = False
    persona_stage_schedule: str = "10,50,100,200,400,800,1500,3000"
    persona_min_evidence: float = 5.0
    persona_spec_max_chars: int = 200
    persona_min_confidence: float = 0.7
    persona_conflict_policy: str = "review"
    persona_min_anchor_sim: float = 0.6
    persona_max_drift: float = 0.4
    persona_max_delta_ratio: float = 0.3

    # ---- 开场称呼生成 ----
    greeting_enable: bool = True
    greeting_fallback_pool: str = "你好,hi,嗨"
    greeting_regen_on_scene_change: bool = True

    # ---- 工具插件与路由（doc/08）----
    tools_enabled: bool = True
    tools_plugin_dir: str = "tools/plugins"
    tools_router: str = "hybrid"
    tools_router_top_n: int = 6
    tools_max_concurrent: int = 4
    tools_fallback_all_on_miss: bool = True
    tools_auto_invoke: bool = True  # 只读工具达阈值时自动触发（doc/08 §9）
    tools_auto_threshold: float = 0.5  # can_handle 自动触发阈值
    tools_auto_reload: str = "off"  # off | watch(文件监听) | manual（doc/08 §3.2）
    tools_hotreload_shadow: bool = True  # 新版本先 shadow 观测再切换（doc/08 §3.3）

    # ---- 回复重复/循环护栏（doc/04 §2.3）----
    repeat_freq_penalty: float = 0.3
    repeat_presence_penalty: float = 0.1
    repeat_recent_n: int = 3
    repeat_sim_threshold: float = 0.75
    repeat_self_ngram_ratio: float = 0.5
    repeat_max_retries: int = 1
    repeat_degrade_msg: str = "换个角度聊聊？你刚才说的，我可能没接住，再展开说说？"

    # ---- 语音（doc/01 D23）----
    speech_enabled: bool = False

    # ---- 日志 ----
    log_level: str = "INFO"
    log_json: bool = True
    log_log_content: bool = False

    # ---- API 服务（doc/04 §3）----
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_cors_origins: str = ""  # 逗号分隔的显式来源白名单；留空=不开放跨域（生产禁用 *）

    # ---- kw_agent_map（doc/03 §2.15）----
    kwmap_delta: float = 0.15
    kwmap_lr: float = 0.05
    kwmap_dmax: float = 0.30

    # ---- 派生/辅助 ----
    @property
    def analyzer_model(self) -> str:
        return self.llm_analyzer_model or self.llm_model

    @property
    def bootstrap_model(self) -> str:
        """Bootstrap 专用模型（doc/01 §D26）：留空复用分析器模型。"""
        return self.mode_free_bootstrap_model or self.analyzer_model

    @property
    def greeting_fallback_list(self) -> list[str]:
        return _as_list(self.greeting_fallback_pool, ",")

    @property
    def persona_stages(self) -> list[int]:
        out: list[int] = []
        prev = -1
        for part in _as_list(self.persona_stage_schedule, ","):
            try:
                v = int(part)
            except ValueError:
                continue
            out.append(v)
            prev = v
        return out

    @classmethod
    def load(cls, dotenv_path: str = ".env") -> "Settings":
        """从环境变量（含 .env）加载并类型转换。"""
        _load_dotenv(dotenv_path)

        def env(name: str, default: T, cast: Callable[[str], T] = str) -> T:  # type: ignore[valid-type]
            raw = os.environ.get(name)
            if raw is None or raw == "":
                return default
            try:
                return cast(raw)
            except (ValueError, TypeError):
                return default

        return cls(
            llm_api_key=env("DLA_LLM__API_KEY", ""),
            llm_base_url=env("DLA_LLM__BASE_URL", "https://api.openai.com/v1"),
            llm_model=env("DLA_LLM__MODEL", "gpt-4o-mini"),
            llm_analyzer_model=env("DLA_LLM__ANALYZER_MODEL", ""),
            llm_timeout_seconds=env("DLA_LLM__TIMEOUT_SECONDS", 60, int),
            llm_max_retries=env("DLA_LLM__MAX_RETRIES", 2, int),
            db_path=env("DLA_DB__PATH", "./data/dla.db"),
            db_auto_migrate=env("DLA_DB__AUTO_MIGRATE", True, _as_bool),
            weight_prior_strength=env("DLA_WEIGHT__PRIOR_STRENGTH", 2.0, float),
            weight_default_half_life_hours=env("DLA_WEIGHT__DEFAULT_HALF_LIFE_HOURS", 6.0, float),
            weight_llm_fusion_beta=env("DLA_WEIGHT__LLM_FUSION_BETA", 0.3, float),
            weight_rebuild_threshold=env("DLA_WEIGHT__REBUILD_THRESHOLD", 0.15, float),
            weight_notify_threshold=env("DLA_WEIGHT__NOTIFY_THRESHOLD", 0.40, float),
            weight_rebuild_cooldown_turns=env("DLA_WEIGHT__REBUILD_COOLDOWN_TURNS", 2, int),
            prompt_total_token_budget=env("DLA_PROMPT__TOTAL_TOKEN_BUDGET", 900, int),
            prompt_l1_ratio=env("DLA_PROMPT__L1_RATIO", 0.25, float),
            prompt_l2_ratio=env("DLA_PROMPT__L2_RATIO", 0.20, float),
            prompt_l3_ratio=env("DLA_PROMPT__L3_RATIO", 0.55, float),
            ctx_max_tokens=env("DLA_CTX__MAX_TOKENS", 128000, int),
            ctx_reserve=env("DLA_CTX__RESERVE", 0.20, float),
            ctx_tokenizer=env("DLA_CTX__TOKENIZER", "deterministic"),
            ctx_warn_ratio=env("DLA_CTX__WARN_RATIO", 0.70, float),
            ctx_compact_ratio=env("DLA_CTX__COMPACT_RATIO", 0.85, float),
            ctx_hard_ratio=env("DLA_CTX__HARD_RATIO", 0.95, float),
            ctx_auto_compact=env("DLA_CTX__AUTO_COMPACT", True, _as_bool),
            ctx_summary_compact_after=env("DLA_CTX__SUMMARY_COMPACT_AFTER", 40, int),
            ctx_epoch_merge_n=env("DLA_CTX__EPOCH_MERGE_N", 10, int),
            ctx_epoch_max_chars=env("DLA_CTX__EPOCH_MAX_CHARS", 200, int),
            ctx_cold_trim_top_m=env("DLA_CTX__COLD_TRIM_TOP_M", 2, int),
            analysis_cold_start_turns=env("DLA_ANALYSIS__COLD_START_TURNS", 2, int),
            analysis_period=env("DLA_ANALYSIS__PERIOD", 3, int),
            analysis_enable_heuristics=env("DLA_ANALYSIS__ENABLE_HEURISTICS", True, _as_bool),
            lvm_dim=env("DLA_LVM__DIM", 64, int),
            lvm_backbone=env("DLA_LVM__BACKBONE", ""),
            lvm_learning_rate=env("DLA_LVM__LEARNING_RATE", 0.01, float),
            lvm_lr_decay=env("DLA_LVM__LR_DECAY", 0.0, float),
            lvm_momentum=env("DLA_LVM__MOMENTUM", 0.9, float),
            lvm_grad_clip=env("DLA_LVM__GRAD_CLIP", 1.0, float),
            lvm_weight_decay=env("DLA_LVM__WEIGHT_DECAY", 1e-4, float),
            lvm_temp=env("DLA_LVM__TEMP", 0.2, float),
            lvm_l3_prior_gamma=env("DLA_LVM__L3_PRIOR_GAMMA", 0.2, float),
            lvm_top_t_identity=env("DLA_LVM__TOP_T_IDENTITY", 6, int),
            lvm_margin_lambda=env("DLA_LVM__MARGIN_LAMBDA", 0.1, float),
            lvm_replay_buffer=env("DLA_LVM__REPLAY_BUFFER", 256, int),
            lvm_train_every_n_turns=env("DLA_LVM__TRAIN_EVERY_N_TURNS", 1, int),
            mode_scenario=env("DLA_MODE__SCENARIO", "fixed"),
            mode_auto_warmup_turns=env("DLA_MODE__AUTO_WARMUP_TURNS", 3, int),
            mode_free_bootstrap_conf=env("DLA_MODE__FREE_BOOTSTRAP_CONF", 0.6, float),
            mode_free_require_desc=env("DLA_MODE__FREE_REQUIRE_DESC", True, _as_bool),
            mode_free_bootstrap_model=env("DLA_MODE__FREE_BOOTSTRAP_MODEL", ""),
            memory_enable=env("DLA_MEMORY__ENABLE", True, _as_bool),
            memory_hot_window=env("DLA_MEMORY__HOT_WINDOW", 6, int),
            memory_cold_top_k=env("DLA_MEMORY__COLD_TOP_K", 4, int),
            memory_embed_dim=env("DLA_MEMORY__EMBED_DIM", 384, int),
            memory_embed_backbone=env("DLA_MEMORY__EMBED_BACKBONE", ""),
            memory_retrieve_trigger=env("DLA_MEMORY__RETRIEVE_TRIGGER", "periodic"),
            memory_retrieve_period=env("DLA_MEMORY__RETRIEVE_PERIOD", 3, int),
            memory_retrieve_sim_threshold=env("DLA_MEMORY__RETRIEVE_SIM_THRESHOLD", 0.3, float),
            memory_prompt_ratio=env("DLA_MEMORY__PROMPT_RATIO", 0.15, float),
            memory_compact_every_n_sessions=env("DLA_MEMORY__COMPACT_EVERY_N_SESSIONS", 10, int),
            memory_importance_decay_days=env("DLA_MEMORY__IMPORTANCE_DECAY_DAYS", 30, int),
            memory_checkpointer=env("DLA_MEMORY__CHECKPOINTER", "sqlite"),
            memory_inject_summaries_only=env("DLA_MEMORY__INJECT_SUMMARIES_ONLY", True, _as_bool),
            memory_summary_max_chars=env("DLA_MEMORY__SUMMARY_MAX_CHARS", 100, int),
            memory_tool_retrieve_top_k=env("DLA_MEMORY__TOOL_RETRIEVE_TOP_K", 3, int),
            memory_tool_max_loops=env("DLA_MEMORY__TOOL_MAX_LOOPS", 2, int),
            scenario_dir=env("DLA_SCENARIO__DIR", "config/scenarios"),
            scenario_default=env("DLA_SCENARIO__DEFAULT", "oral_practice"),
            persona_auto_update=env("DLA_PERSONA__AUTO_UPDATE", False, _as_bool),
            persona_stage_schedule=env("DLA_PERSONA__STAGE_SCHEDULE", "10,50,100,200,400,800,1500,3000"),
            persona_min_evidence=env("DLA_PERSONA__MIN_EVIDENCE", 5.0, float),
            persona_spec_max_chars=env("DLA_PERSONA__SPEC_MAX_CHARS", 200, int),
            persona_min_confidence=env("DLA_PERSONA__MIN_CONFIDENCE", 0.7, float),
            persona_conflict_policy=env("DLA_PERSONA__CONFLICT_POLICY", "review"),
            persona_min_anchor_sim=env("DLA_PERSONA__MIN_ANCHOR_SIM", 0.6, float),
            persona_max_drift=env("DLA_PERSONA__MAX_DRIFT", 0.4, float),
            persona_max_delta_ratio=env("DLA_PERSONA__MAX_DELTA_RATIO", 0.3, float),
            greeting_enable=env("DLA_GREETING__ENABLE", True, _as_bool),
            greeting_fallback_pool=env("DLA_GREETING__FALLBACK_POOL", "你好,hi,嗨"),
            greeting_regen_on_scene_change=env("DLA_GREETING__REGEN_ON_SCENE_CHANGE", True, _as_bool),
            tools_enabled=env("DLA_TOOLS__ENABLED", True, _as_bool),
            tools_plugin_dir=env("DLA_TOOLS__PLUGIN_DIR", "tools/plugins"),
            tools_router=env("DLA_TOOLS__ROUTER", "hybrid"),
            tools_router_top_n=env("DLA_TOOLS__ROUTER_TOP_N", 6, int),
            tools_max_concurrent=env("DLA_TOOLS__MAX_CONCURRENT", 4, int),
            tools_fallback_all_on_miss=env("DLA_TOOLS__FALLBACK_ALL_ON_MISS", True, _as_bool),
            tools_auto_invoke=env("DLA_TOOLS__AUTO_INVOKE", True, _as_bool),
            tools_auto_threshold=env("DLA_TOOLS__AUTO_THRESHOLD", 0.5, float),
            tools_auto_reload=env("DLA_TOOLS__AUTO_RELOAD", "off"),
            tools_hotreload_shadow=env("DLA_TOOLS__HOTRELOAD_SHADOW", True, _as_bool),
            repeat_freq_penalty=env("DLA_REPEAT__FREQ_PENALTY", 0.3, float),
            repeat_presence_penalty=env("DLA_REPEAT__PRESENCE_PENALTY", 0.1, float),
            repeat_recent_n=env("DLA_REPEAT__RECENT_N", 3, int),
            repeat_sim_threshold=env("DLA_REPEAT__SIM_THRESHOLD", 0.75, float),
            repeat_self_ngram_ratio=env("DLA_REPEAT__SELF_NGRAM_RATIO", 0.5, float),
            repeat_max_retries=env("DLA_REPEAT__MAX_RETRIES", 1, int),
            repeat_degrade_msg=env("DLA_REPEAT__DEGRADE_MSG", "换个角度聊聊？你刚才说的，我可能没接住，再展开说说？"),
            speech_enabled=env("DLA_SPEECH__ENABLED", False, _as_bool),
            log_level=env("DLA_LOG__LEVEL", "INFO"),
            log_json=env("DLA_LOG__JSON", True, _as_bool),
            log_log_content=env("DLA_LOG__LOG_CONTENT", False, _as_bool),
            api_host=env("DLA_API__HOST", "127.0.0.1"),
            api_port=env("DLA_API__PORT", 8000, int),
            api_cors_origins=env("DLA_API__CORS_ORIGINS", ""),
            kwmap_delta=env("DLA_KWMAP__DELTA", 0.15, float),
            kwmap_lr=env("DLA_KWMAP__LR", 0.05, float),
            kwmap_dmax=env("DLA_KWMAP__DMAX", 0.30, float),
        )


# 进程内单例
_SETTINGS: Optional[Settings] = None


def get_settings(reload: bool = False) -> Settings:
    """获取（惰性加载）全局 Settings 单例。"""
    global _SETTINGS
    if _SETTINGS is None or reload:
        _SETTINGS = Settings.load()
    return _SETTINGS
