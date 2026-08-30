# DialogueLearningAgent（DLA）

一个**基于关键词分析与提示（prompt）推导**的对话 Agent。它最不一样的地方在于：驱动对话的 System Prompt **不是写死的**，而是由**三层带权重的关键词**实时推导，并随对话进程自我调整。

一句话理解：用户说话的同时，系统在后台回答三个问题——

1. **这是什么样的对话？**（功能场景层 L1）
2. **对面这个人现在是什么状态？**（用户肖像层 L2）
3. **那么我此刻应该扮演什么样的交流对象？**（Agent 肖像层 L3）

第三个问题的答案，就是当轮真正喂给大模型的 System Prompt。

---

## 特性

- **三层权重模型**：L1 场景 → L2 用户肖像 → L3 Agent 肖像。证据累积（指数衰减）+ 置信度 `conf = E/(E+K)` + 维度内归一化/冲突消解，确定性耦合规则优先。
- **情绪/脾性只靠涌现**：`user_mood.*` / `user_temper.*` 类关键词**禁止预设、禁止用户自陈指定**，只能由 LLM 从对话中实时抽取（doc/02 §11.9）。
- **开箱即跑（零强依赖）**：核心引擎只依赖 `stdlib + pyyaml`。无 API key 时自动退回内置 `FakeLLMClient`，离线即可跑通整条闭环。
- **预设场景模板库**：内置 5 套场景（口语练习 / 淘宝客服 / 桌面宠物 / 虚拟恋人 / 职场面试），统一对齐关键词白名单；亲密类场景强制 `safe_mode` 安全硬约束，不可关闭。
- **对话历史冷热记忆**（doc/07）：热窗口 + 冷库 RAG 检索 + SQLite checkpoint。
- **上下文自动压缩**（doc/11）：每轮估算整窗 `fill_ratio`，阶梯阈值触发紧凑协议，长会话不撑爆上下文。
- **回复重复护栏**（doc/04 §2.3）：频率/存在惩罚 + 自重复 n-gram 检测，自动降级重生成。
- **工具插件化**（doc/08）：统一 Tool 契约 + 原子快照注册表 + 两级路由；`recall_memory` 为首个内置插件。
- **人格演进受控**（doc/10）：默认关闭、阶梯式、只能渐进不得大幅偏离原始人格锚。
- **`kw_agent_map` 涌现**（doc/03 §2.15）：从对话中沉淀「情绪/脾性 → Agent 特质」映射，随交互累积。

---

## 架构速览

```
┌─────────────────────────────────────────────┐
│  L1 功能场景层  这场对话是关于什么的？         │ 相对稳定，多由场景模板注入，权重最高
└──────────────────────┬──────────────────────┘
                       │ 约束候选空间
                       ▼
┌─────────────────────────────────────────────┐
│  L2 用户肖像层  对面是谁？现在什么状态？        │ 预设项(身份/目标) + 观察项(情绪/卡点…)，动态演化带置信度
└──────────────────────┬──────────────────────┘
                       │ 耦合规则 + 可选 LLM 精炼
                       ▼
┌─────────────────────────────────────────────┐
│  L3 Agent 肖像层  我该是什么样的人跟他聊？      │ 由 L1 × L2 推导出的"应然"，不直接来自输入
└──────────────────────┬──────────────────────┘
                       │ 加权渲染 + 预算裁剪
                       ▼
                System Prompt（当轮生效）
```

### 目录结构

```
DialogueLearningAgent/
├── apps/cli/main.py          # CLI 入口（argparse；`dla` 命令）
├── config/
│   ├── keywords/             # 三层关键词白名单 YAML（L1/L2/L3）
│   ├── scenarios/            # 5 套预设场景模板 YAML
│   └── coupling_rules.yaml   # 层间确定性耦合规则
├── migrations/001_init.sql   # SQLite 建表 + 迁移 runner
├── src/dla/
│   ├── core/                 # 错误体系、数据模型、端口协议
│   ├── keywords/             # 词表 Lexicon、同义词归一
│   ├── weighting/            # 置信度/衰减/耦合/解析/权重引擎
│   ├── analysis/             # 触发、启发式、LLM 结构化抽取
│   ├── llm/                  # OpenAI 兼容客户端 + FakeLLM（离线）
│   ├── prompt/               # token 预算、渲染、组装、上下文压缩
│   ├── storage/              # SQLite 连接、迁移、仓储
│   ├── orchestration/        # DialogueEngine（主流程编排）
│   ├── tools/                # 工具契约/注册/路由/执行/插件发现
│   ├── memory/               # 对话历史冷热记忆子系统（doc/07）
│   ├── evolution.py          # 词表/人格候选发现
│   └── config/               # 配置加载、场景加载
├── tests/                    # 单元 + 集成测试
└── doc/                      # 11 篇设计文档（doc/00–doc/11）
```

---

## 快速开始

### 1. 零依赖直接跑（无需 API key）

```bash
# 仓库根目录下
python apps/cli/main.py bench
# 输出：empathy 随"受挫"信号 0.600 → 0.900 上升，退出码 0
```

无 `DLA_LLM__API_KEY` 时自动使用 `FakeLLMClient`，整条「分析 → 权重 → 组装 → 生成 → 落库」闭环离线可跑。

### 2. 接真实大模型

复制 `.env.example`（或直接设置环境变量）并填入凭据：

```bash
export DLA_LLM__API_KEY=sk-xxxx
export DLA_LLM__BASE_URL=https://api.openai.com/v1   # 兼容 OpenAI 协议的任意端点
export DLA_LLM__MODEL=gpt-4o-mini
```

### 3. （可选）安装为命令行工具

```bash
pip install -e .          # 仅核心依赖（pyyaml）
# 或带 CLI 外观/开发/LLM/图能力：
pip install -e ".[cli,dev]"
```

安装后可用 `dla` 命令替代 `python apps/cli/main.py`。

---

## 配置

所有配置项前缀 `DLA_`，段间用双下划线分隔，可由环境变量或 `.env` 文件提供（`.env` 不覆盖已存在的环境变量）。核心字段：

| 配置项 | 环境变量 | 默认 | 说明 |
|---|---|---|---|
| LLM Key | `DLA_LLM__API_KEY` | 空（→FakeLLM） | 留空即离线模式 |
| LLM 端点 | `DLA_LLM__BASE_URL` | `https://api.openai.com/v1` | OpenAI 兼容协议 |
| 模型 | `DLA_LLM__MODEL` | `gpt-4o-mini` | 对话模型；分析模型 `DLA_LLM__ANALYZER_MODEL` 留空则复用 |
| 超时/重试 | `DLA_LLM__TIMEOUT_SECONDS` / `DLA_LLM__MAX_RETRIES` | `60` / `2` | — |
| 数据库路径 | `DLA_DB__PATH` | `./data/dla.db` | 自动建表迁移（`data/` 已被 gitignore） |
| 权重先验强度 K | `DLA_WEIGHT__PRIOR_STRENGTH` | `2.0` | 置信度分母 |
| 默认半衰期(时) | `DLA_WEIGHT__DEFAULT_HALF_LIFE_HOURS` | `6.0` | 证据指数衰减 |
| LLM 融合系数 β | `DLA_WEIGHT__LLM_FUSION_BETA` | `0.3` | L3 = (1-β-γ)·rule + β·llm + γ·prior |
| Prompt 总预算 | `DLA_PROMPT__TOTAL_TOKEN_BUDGET` | `900` | 确定性 token 估算 |
| 上下文压缩阈值 | `DLA_CTX__COMPACT_RATIO` / `DLA_CTX__HARD_RATIO` | `0.85` / `0.95` | 阶梯触发紧凑协议 |
| 默认场景 | `DLA_SCENARIO__DEFAULT` | `oral_practice` | `fixed` 模式默认加载 |
| 人格演进开关 | `DLA_PERSONA__AUTO_UPDATE` | `False` | 默认关闭（doc/10） |
| 工具启用 | `DLA_TOOLS__ENABLED` | `True` | 插件化工具系统 |
| 重复护栏降级语 | `DLA_REPEAT__DEGRADE_MSG` | （见源码） | 检测到循环时回退话术 |

> 完整字段见 `src/dla/config/settings.py`（全部 `DLA_*` 及其默认值与类型转换）。

---

## CLI 用法

```text
dla chat      对话（--mode fixed|auto|free / --scenario / --message / --explain / --debug / --no-db）
dla scenario  list | show <id> | validate | export
dla keyword   map list | map reset      # kw_agent_map 涌现映射
dla ctx       status | compact          # 上下文压缩日志
dla bench                           # 离线剧本回归（强制 FakeLLM，断言权重演化）
```

示例：

```bash
# 单次对话 + 打印三层权重 + 打印思维链
python apps/cli/main.py chat --scenario oral_practice \
    --message "这个语法太难了，我完全不会，好烦。" --explain --debug

# 校验 5 套场景模板是否对齐关键词白名单、无情绪/脾性预设
python apps/cli/main.py scenario validate

# 查看「情绪/脾性 → Agent 特质」涌现映射
python apps/cli/main.py keyword map list

# 交互式对话
python apps/cli/main.py chat --scenario taobao_cs
```

> 关注点：`--explain` 会打印 L1/L2/L3 当轮权重，`--debug` 打印每帧思维链；二者均不依赖真实 LLM。

---

## 设计文档

详细的架构、算法与子系统设计见 [`doc/`](./doc) 目录（共 11 篇，建议按序阅读）：

| 文档 | 内容 |
|---|---|
| [00-索引](./doc/00-索引.md) | 项目总览、文档地图、三层模型速览 |
| [01-架构设计总览](./doc/01-架构设计总览.md) | 目标/非目标、技术选型、分层架构、数据流 |
| [02-三层关键词权重模型](./doc/02-三层关键词权重模型.md) | **核心算法**：权重公式、耦合、冲突消解、稳定性、组装 |
| [03-数据模型与存储](./doc/03-数据模型与存储.md) | SQLite 表结构、仓储、迁移、`kw_agent_map` |
| [04-交互层设计](./doc/04-交互层设计.md) | CLI / FastAPI / PyQt 交互设计 |
| [05-实施路线与验收](./doc/05-实施路线与验收.md) | 里程碑、测试策略、验收标准 |
| [06-向量化词库与在线学习](./doc/06-向量化词库与在线学习.md) | 关键词 embedding、跨层关联、在线学习 |
| [07-对话历史冷热记忆](./doc/07-对话历史冷热记忆(LangGraph+RAG).md) | 热/冷记忆分层、RAG 检索、checkpoint |
| [08-工具插件化与路由](./doc/08-工具插件化与路由.md) | Tool 契约、原子快照、两级路由 |
| [09-预设场景与角色模板库](./doc/09-预设场景与角色模板库.md) | 5 套场景模板、内容安全边界 |
| [10-人格演进（受控自动补充）](./doc/10-人格演进（受控自动补充）.md) | 默认关、阶梯式、不偏离人格锚 |
| [11-上下文自动压缩与预算管控](./doc/11-上下文自动压缩与预算管控.md) | 整窗 `fill_ratio` 监控 + 阶梯紧凑协议 |

---

## 测试

```bash
pip install -e ".[dev]"        # 提供 pytest
pytest -q
```

测试覆盖：权重引擎（置信度/衰减/耦合/解析）、关键词词表与归一化、分析层（启发式/重复护栏/LLM 抽取/词表外候选）、Prompt 预算与渲染、以及「分析→权重→稳定性→组装→生成→落库」端到端集成。

---

## 开发约定

- **零强依赖**：核心引擎只允许 `stdlib + pyyaml`，禁止在 `src/dla` 核心路径引入 `typer`/`qt`/重型 ML 库；需要这些能力时走可选 extras 或子包隔离。
- **离线可跑**：任何"开箱"路径都应在无 API key 下闭环通过（由 `bench` 守护）。
- **白名单优先**：LLM 抽取的关键词一律经过词表校验，词表外进入 `raw_unknown` 待审，不污染权重。
- **情绪/脾性不预设**：`user_mood.*` / `user_temper.*` 仅能从对话涌现，见 doc/02 §11.9。
- **安全硬约束不可关闭**：亲密类场景 `safe_mode`、重复护栏等均为否决级约束。

---

## 许可证

待定（内部项目）。
