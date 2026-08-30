# 07 · 对话历史冷热记忆（LangGraph + RAG）

> 本篇是 [01 · 架构设计总览](./01-架构设计总览.md) 的**补充子系统**，落实用户在评审中提出的诉求：
> **对话历史的向量检索 / RAG / 语义搜索**等功能，需要**以 LangGraph 额外实现一套冷热记忆（Hot/Cold Memory）方案**。
>
> 它与 [06 · 向量化词库与在线学习](./06-向量化词库与在线学习.md) **并行但独立**：doc/06 向量化的是"关键词↔关键词"的关联，本篇向量化的是"对话片段↔对话片段 / 事实↔事实"的检索。二者是**两个独立向量空间、两套存储、两种用途**，不可混淆。

## 0. 边界澄清（与 doc/06 衔接）

| | doc/06 · 词库向量化 | **doc/07 · 对话历史冷热记忆（本篇）** |
|---|---|---|
| 向量化对象 | **关键词** `e_k`（d 维） | **对话片段 / 抽取事实**（turn / episode） |
| 用途 | 建模跨层关键词关联，驱动 L3 人格推导 | 跨会话长期记忆，向 Prompt 注入"背景/上下文" |
| 检索 | 关联打分 `S_r(q,j)`（非文本检索） | 语义相似度 top-K（真正的 RAG 检索） |
| 是否进权重 | 是（γ 融合进 L3） | **否**（仅作 Prompt 注入，不参与权重计算） |
| 框架 | 纯 numpy，无编排框架 | **LangGraph**（状态图 + checkpoint） |
| 训练 | 关联矩阵 `M_r` 在线学习 | 嵌入通常冻结（backbone），可选微调在外部 |

一句话：**doc/06 是"关键词的图谱"，doc/07 是"对话的记忆体"；LangGraph 是承载后者的状态机。** 二者通过 doc/03 的同一 SQLite 共存，但逻辑互不直接耦合。

---

## 1. 设计目标

1. **跨会话长期记忆**：Agent 能"记起"之前聊过什么、用户曾说过哪些稳定事实（如"用户母语是粤语""上周在准备面试"），而非每开新会话从零开始。
2. **冷热分层**：
   - **热记忆（Hot）**：当前会话最近 K 轮 + 当前 L1/L2/L3 活跃状态，始终在上下文，零检索延迟。
   - **冷记忆（Cold）**：历史会话中被滑出热窗口的轮次、跨会话抽取的事实/片段，按需语义检索注入。
3. **LangGraph 状态机编排**：用有状态图把 `retrieve → reflect → generate → store` 串成可复现流程，且**带 checkpoint**，可断点续会话、可回放调试。
4. **预算受控**：冷记忆注入**复用 doc/02 的 Prompt 预算纪律**，占独立子预算，绝不挤爆上下文。
5. **与三层权重解耦**：检索结果只向 Prompt 注入"背景参考"片段，并标注来源与置信度；**不参与 L1/L2/L3 权重计算**，人格仍由 doc/02/06 决定。
6. **压缩摘要驱动的历史注入（本次新增）**：每轮 LLM 回复必须在 schema 回传一段 **≤100 字的 `turn_summary`**（本会话/本轮回溯摘要）；**下一轮对话注入的"历史"仅由历轮 `turn_summary` 拼接而成**，而非完整对话原文；如需回溯某段历史的细节，由**本地检索工具 `recall_memory` 按需拉取**原文片段再喂回 LLM。完整对话始终存于冷热记忆，但默认不进 Prompt。

---

## 2. 冷热记忆定义

### 2.1 热记忆 Hot Memory

| 项 | 内容 |
|---|---|
| 范围 | ① 当前会话最近 `K` 轮（滑动窗口，默认 `K=6`）的**完整原文**——保留原文是为了让 `recall_memory` 工具能在同一会话内即时取回细节，但**原文默认不进下一轮 Prompt**；② 当前 L1/L2/L3 活跃关键词状态（来自 doc/02/06）；③ 当前会话 meta（场景、模式 fixed/auto）；④ 历轮 `turn_summary` 摘要链（下一轮 Prompt 注入的"历史"即此链） |
| 存储 | LangGraph `StateGraph` 的 **in-graph state**，每轮经 `SqliteSaver` 自动 checkpoint 到 SQLite |
| 特性 | 零检索、低延迟、始终在上下文；是"当下"的工作记忆 |

### 2.2 冷记忆 Cold Memory

| 项 | 内容 |
|---|---|
| 范围 | ① 历史会话中被滑出热窗口的轮次（turn-level）② 从对话中抽取的**事实 / 片段（episodic facts）**③ 跨会话的用户画像演进轨迹（可由 doc/06 的 L2 状态持久化而来） |
| 存储 | 本地向量库：`chunk 文本 + embedding + metadata(session_id, turn, layer 标签, importance, timestamp, source)` |
| 检索 | 语义相似度 top-K（余弦），可选 rerank |
| 特性 | 按需召回、有预算上限、可压缩（consolidation）、可软删除（遗忘） |

> **`turn_summary` 是冷记忆的主索引**：每条 turn 入库时同时写入其 `turn_summary`（来自 doc/06 §4 的 LLM 回传）。下一轮默认注入给 LLM 的历史 = 历轮 `turn_summary` 拼接（而非原文）；`summary` 字段同时作为 `recall_memory` 工具的"目录"，让 LLM/用户先看到每轮一句话概览，再决定是否拉取某轮原文。

---

## 3. LangGraph 实现

### 3.1 State Schema

```python
class DialogueState(TypedDict):
    session_id: str
    messages: list[BaseMessage]            # 当前会话消息（含热窗口）
    hot_memory: HotMemory                  # 热窗口 + 活跃关键词状态
    cold_context: list[ColdMemoryItem]     # 本轮检索到的冷记忆
    turn_summaries: list[TurnSummary]      # 历轮 turn_summary 链（下一轮默认注入的"历史"）
    scene_meta: SceneMeta                  # 场景、模式（fixed/auto）
    analysis: AnalysisResult               # 来自 doc/02/06 的分析结果
    response: str
    pending_tool_calls: list[ToolCall]    # LLM 本轮发起的 recall_memory 调用
    detail_blocks: list[DetailBlock]       # 工具取回的原文片段，临时注入，不持久化为"历史"
```

### 3.2 图节点（Graph Nodes）

```
        ┌─────────────────────────────────────────────────────────┐
        │  LangGraph StateGraph（记忆增强编排层，包在 doc/01 引擎外） │
        │                                                           │
  入口 ─┤  ① retrieve_cold ──▶ ② prepare_prompt ──▶ ③ generate      │
        │        │(条件边)              │                    │       │
        │        │                      │                    ▼       │
        │        │                      │             ④ reflect_store│
        │        │                      │                    │       │
        │        └──────────────────────┴────────────────────┘       │
        │                   ⑤ maybe_compact（周期性）                │
        └─────────────────────────────────────────────────────────┘
                         │
                         ▼  checkpoint → SQLite（SqliteSaver，同一 DLA_DB__PATH）
```

- **① `retrieve_cold`**：以"用户最新发言 + 当前 L2 活跃关键词（加权）"拼接为查询向量 → 冷库 top-K 检索 → 写入 `cold_context`。**条件边**：冷启动强制 / 周期触发 / 相似度阈值触发时才检索，避免每轮空转（延续 doc/01 D7 的触发思路）。
- **② `prepare_prompt`**：组装 System Prompt（doc/02 产出）+ 热记忆（始终）+ 冷记忆（条件注入、预算裁剪）。
- **③ `generate`**：调用 LLM 流式生成，**复用 doc/01 `orchestration/engine.py` 的生成逻辑**（LangGraph 是外层编排，不重写生成）。本轮 LLM 输出除正文与 doc/06 的结构化字段外，**必须回传 `turn_summary`（≤100 字）**（见 [06 §4](./06-向量化词库与在线学习.md)）；若模型判定"需要更早的历史细节"而发起工具调用 `recall_memory`，则转入 ⑥。
- **④ `reflect_store`**：生成后——① 将本轮（user + agent 正文 + `turn_summary` + doc/06 分析事实）切块 embed 写入冷库，**`turn_summary` 作为该 turn 的索引字段**；② 滑出热窗口的老轮次迁入冷库（原文保留在 cold store，供 `recall_memory` 取回）；③ 可选：抽取跨会话事实（episode fact extraction）写入冷库高 importance 区。
- **⑥ `handle_tools`（按需分支）**：当 ③ 的 LLM 发起 `recall_memory(query/fields)` 工具调用时触发——从热窗口（同会话近期）或冷库（更早/跨会话）取回对应的**原文片段/指定字段**，作为 `detail_block` 临时注入 ② `prepare_prompt`，并让 LLM 基于补充细节重新生成；工具结果**只进当前轮上下文、不污染长期权重、不写回默认历史链**。无工具调用时跳过此节点，直接进 ④。
- **⑤ `maybe_compact`**：周期性把 N 条相关冷记忆压缩为一条 episode 摘要（重要事实保留、冗余丢弃），控制冷库规模。

### 3.3 Checkpointer

使用 LangGraph `SqliteSaver` 绑定到同一个 `DLA_DB__PATH`（doc/03）。这同时满足：
- **状态持久化**：会话中断后可从 checkpoint 精确恢复热记忆。
- **断点续会话**：`dla chat --session <id>` 直接 resume。
- **可回放**：调试模式下可重放任意一轮的完整图状态。

---

## 4. 检索与 RAG 细节

### 4.1 Ingestion（入库管线）
- 每条 user / agent 消息 → 切块（按 turn 或语义句） → embed → upsert 冷库；同时把本轮 LLM 回传的 `turn_summary`（≤100 字）作为该 turn 的 `summary` 字段一并写入，既作下一轮默认历史注入的载体，也作 `recall_memory` 的概览目录。
- doc/06 分析产出的 `user_predictions` / `satisfaction` 等结构化事实，作为带 `layer` metadata 的记忆单元一并入库，使冷记忆不止于原文、还含"系统已判断的结论"。

### 4.2 Query 构造
用 **"用户最新发言 + 当前 L2 活跃关键词（加权拼接）"** 作为查询，比只用最新一句召回更准（关键词提供"用户此刻状态"的语义锚点）。

### 4.3 Retrieval
- 余弦相似度 top-K（默认 `K=4`），可选再按 `importance` 加权 rerank。
- 返回项携带 `similarity` 分，供调试面板展示。

### 4.4 注入预算（复用 doc/02 预算纪律）
- 冷记忆注入占 Prompt 预算的一个**独立子项** `DLA_MEMORY__PROMPT_RATIO`（默认 `0.15`），与 L1/L2/L3 三者共用 doc/02 的预算总盘。
- 超出时按相似度降序裁剪；检索结果以"背景参考"区块注入，明确标注来源会话与轮次，避免与当轮对话混淆。

### 4.5 防污染
- 检索到的冷记忆**只作上下文参考**，不参与权重计算，也不以"指令"形式注入（避免被注入内容劫持人格）。
- 低相似度 / 低 importance 的记忆默认不注入。

### 4.6 压缩摘要（turn_summary）：历史注入的默认载体

**职责**：把"每轮一次完整的对话"压缩成一句 ≤100 字的中文摘要，作为下一轮对话的**唯一默认历史上下文**。

- **由谁产出**：LLM 在每次回复时，于结构化输出（doc/06 §4 的 schema）中回传 `turn_summary`。System Prompt 显式提醒：*"回复后，额外给出不超过 100 字的中文摘要，概括本轮用户意图、关键事件、你的应对与用户反应，供后续对话回看。"*
- **约束**：硬上限 `DLA_MEMORY__SUMMARY_MAX_CHARS=100`；超出时由模型自检压缩，解析层再对越界字符串做硬截断（截断后仍保证语义完整，必要时末尾补"…"）。
- **下一轮怎么用**：`prepare_prompt` 把 `turn_summaries` 链（历轮摘要按序）渲染为"对话回顾"区块注入，每条约 ≤100 字，**完整原文默认不出现在此区块**。这把历史占用的 token 从"N 轮全文"压到"N×100 字"，与 doc/02 预算纪律天然契合。
- **与冷记忆检索的关系**：`turn_summary` 是**确定性的、每轮必有**的轻量历史；doc/07 §4.2/§4.3 的语义检索是**额外的、按需触发**的背景补充。两者不冲突——前者保证"永远知道聊过什么主线"，后者在相关时补充细节。

### 4.7 按需检索工具（local recall_memory，doc/08 首个内置插件）

**职责**：当压缩摘要不足以支撑当前回复（如用户问"你刚才说的那个例子是什么"），由 LLM 主动发起工具调用，从本地冷热记忆取回**原文片段/指定字段**，再喂回自己。该工具在 doc/08 中作为**第一个内置插件**实现，遵循统一 `Tool` 契约、走两级路由，由 `ToolRegistry` 管理（与后续所有本地工具一致）。

```python
class RecallMemoryArgs(BaseModel):
    query: str                       # 想找的内容（如"之前举的关于时态的例子"）
    scope: Literal["current_session", "all_sessions"] = "current_session"
    top_k: int = DLA_MEMORY__TOOL_RETRIEVE_TOP_K   # 默认 3
    fields: list[str] = ["text"]     # 可取回原文 text / 结构化字段（agent_ops 等）

def recall_memory(args: RecallMemoryArgs) -> list[DetailBlock]:
    # 1. 以 query 向量检索热窗口(同会话) + 冷库(跨会话)
    # 2. 取 top_k 命中项的指定 fields
    # 3. 返回 DetailBlock{ source_session, turn, text, similarity }
```

- **生命周期**：工具结果进 `detail_blocks`，**临时注入 ② `prepare_prompt`**，仅对当前这一轮生效；它**不写回 `turn_summaries`**（否则历史链会被原文污染），也**不参与 L1/L2/L3 权重计算**。
- **与 LangGraph / doc/08 的对接**：本工具作为 `tools/plugins/recall_memory` 插件注册进 `ToolRegistry`，并 `bind_tools` 给 LLM。节点 ③ `generate` 若检测到 tool_calls，转入节点 ⑥ `handle_tools`，由 `ToolRegistry` 按 `name` 派发执行、取回后把 `detail_blocks` 并入上下文重生成；重生成时若再次发起调用则继续，设最大工具循环 `DLA_MEMORY__TOOL_MAX_LOOPS=2` 防死循环。
- **护栏**：工具只读、不修改记忆；`scope=all_sessions` 需显式开启且受 `IMPORTANCE_DECAY` 约束；返回内容仍走 §4.5 防污染（仅作参考，不指令化）。

---

## 5. 冷热迁移与压缩（Consolidation）

| 动作 | 触发 | 说明 |
|---|---|---|
| 热 → 冷 | 滑窗滑出 | 老轮次立即迁入冷库，热窗口保持精简 |
| 冷 → 更冷（压缩） | 每 `DLA_MEMORY__COMPACT_EVERY_N_SESSIONS` 会话 | 相关冷记忆合并为一条 episode 摘要，降噪降本 |
| 遗忘 / 裁剪 | 按 `recency × importance` 双层打分 | 长尾低重要记忆软删除（保留可恢复），控制规模与隐私面 |

---

## 6. 与既有系统的集成点

- **不改动 doc/02 权重引擎**：冷记忆只是 Prompt 注入，人格仍由 L1×L2 推导。
- **复用 doc/01 `orchestration/engine.py` 生成**：LangGraph 是"外层编排"，把 `retrieve` 包在前面、`store` 包在后面。
- **复用 doc/03 的 SQLite**：同一 DB，新增 `cold_memory` / `memory_index` 两张表（与 doc/06 的 `keyword_embeddings` 物理隔离）。
- **与 doc/06 的再闭环（谨慎）**：doc/06 的 L2 长期状态可作为冷记忆的"用户画像"源；冷记忆中抽取的稳定事实也可**回写** doc/06 L2，但须带护栏（人工审核或高置信才回写），避免噪声污染用户肖像。
- **调试可视化**：doc/04 思维链面板新增"冷记忆检索帧"——展示检索到的 K 条记忆、相似度、来源轮次；`--show-prompt` 时一并打印注入的冷记忆区块。

---

## 7. 配置与环境变量

```bash
# ---- 冷热记忆（doc/07, LangGraph + RAG）----
DLA_MEMORY__ENABLE=true                    # 是否启用对话历史记忆
DLA_MEMORY__HOT_WINDOW=6                   # 热记忆滑动窗口轮数
DLA_MEMORY__COLD_TOP_K=4                   # 冷记忆检索返回条数
DLA_MEMORY__EMBED_DIM=384                  # 记忆向量维度（sentence-transformers 默认 384）
DLA_MEMORY__EMBED_BACKBONE=                # 留空=确定性占位(检索退化但结构可测)；可选 sentence-transformers 模型名(语义热启动)
DLA_MEMORY__RETRIEVE_TRIGGER=periodic      # always | periodic | similarity | coldstart
DLA_MEMORY__RETRIEVE_PERIOD=3              # periodic 模式下的触发周期（轮）
DLA_MEMORY__RETRIEVE_SIM_THRESHOLD=0.3    # similarity 模式下的相似度阈值
DLA_MEMORY__PROMPT_RATIO=0.15             # 冷记忆占 Prompt 预算比例
DLA_MEMORY__COMPACT_EVERY_N_SESSIONS=10    # 跨会话压缩周期
DLA_MEMORY__IMPORTANCE_DECAY_DAYS=30       # 重要性时间衰减半衰期
DLA_MEMORY__CHECKPOINTER=sqlite            # 状态持久化后端（LangGraph）
DLA_MEMORY__INJECT_SUMMARIES_ONLY=true     # true=下一轮历史仅注入 turn_summary 链（不注入完整原文）
DLA_MEMORY__SUMMARY_MAX_CHARS=100          # turn_summary 硬上限（字符，中文按字计）
DLA_MEMORY__TOOL_RETRIEVE_TOP_K=3          # recall_memory 每次取回的原文片段数
DLA_MEMORY__TOOL_MAX_LOOPS=2               # 单轮内工具循环上限（防死循环）
```

> **离线策略**：`DLA_MEMORY__EMBED_BACKBONE` 留空时，使用**确定性占位 embedding**（按 id 哈希），检索结构完整可测但语义近乎随机——仅用于架构验证与单测；配置真实 backbone（如 `paraphrase-multilingual-L12-v2`，支持中英混合）后，首次需联网下载约 80MB，之后完全离线。

---

## 8. 风险与权衡

| 风险 | 影响 | 缓解 |
|---|---|---|
| LangGraph 是额外依赖 | 依赖体积增加 | 纯 Python、可离线安装；checkpointer 无运行时联网要求 |
| Embedding 离线与否 | 占位模式下冷记忆无效 | 默认占位仅为验证结构；真实语义需配置 backbone（首次下载后可离线） |
| 检索质量依赖 embedding | 召回噪声 | 相似度阈值 + importance rerank + 预算裁剪三重过滤 |
| 冷记忆挤爆上下文 | 成本上升 | 严格复用 doc/02 预算纪律，`PROMPT_RATIO` 硬上限 |
| 隐私（存用户对话事实） | 数据泄露 | 冷库随 `data/` 排除进 `.gitignore`，永不入仓库；软删除可恢复 |
| 冷记忆劫持人格 | 行为偏移 | 仅作"背景参考"注入，不参与权重；不以指令形式呈现 |
| 压缩摘要失真/超长 | 历史主线丢失或 token 浪费 | 硬上限 `SUMMARY_MAX_CHARS` + 解析层截断；调试模式打印每轮 summary 供人工复核；可 `DLA_MEMORY__INJECT_SUMMARIES_ONLY=false` 临时回退为注入原文 |
| 工具循环 / 越权 | 成本失控或泄漏其他会话 | `TOOL_MAX_LOOPS` 上限；`scope=all_sessions` 需显式开启且受 decay 约束；工具只读不改 |

---

## 9. 测试策略

| 层 | 范围 | 用什么 |
|---|---|---|
| 单测 | 图可跑通 `retrieve → generate → store`；checkpoint 可恢复 | 确定性占位 embedding + 假 LLM |
| 检索单测 | top-K 召回正确性、相似度排序 | 精确 numpy 嵌入（可构造已知相似度） |
| 集成 | 多会话后能从冷库召回上一次关键事实 | 假 LLM + 假 embedding + 真实 SQLite |
| 预算单测 | 冷记忆超 `PROMPT_RATIO` 时按相似度裁剪 | 构造超长记忆集断言裁剪结果 |

**核心断言**：不联网、`ENABLE=true`、占位 embedding 下，LangGraph 图仍完整跑通且 checkpoint 可 resume；配置真实 backbone 后，跨会话能召回 `episode fact`。

| 层 | 范围 | 用什么 |
|---|---|---|
| 摘要单测 | 每轮 `turn_summary` 必填且 ≤100 字；越界被截断；下一轮注入的历史 = summary 链、不含原文 | 注入假 LLM 固定 summary + 断言 `prepare_prompt` 输出 |
| 工具单测 | LLM 发起 `recall_memory` → 返回 top-K 原文片段 → 重生成拿到细节；超 `TOOL_MAX_LOOPS` 被强制停止 | 假 LLM 触发 tool_call + 假 store |
| 注入隔离单测 | `detail_blocks` 仅当前轮可见、不写回 `turn_summaries`、不参与权重 | 断言 state 在两个轮次间不串味 |

---

下一篇（收尾）：回到 [01 · 架构设计总览](./01-架构设计总览.md) 查看编排层如何接入本子系统。
