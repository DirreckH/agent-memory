# Agent Memory API（FastAPI + LLM + FastEmbed + SQLite）

本地可运行、可 Docker 部署的文本记忆服务：SQLite 持久化 + ONNX 语义召回 + 可选 LLM 索引增强与查询扩展。以 V3 时间感知检索为基础，新增有来源的多跳补充，以及 D1/F1 事实版本与抽取式摘要。服务只返回记忆证据，不生成最终答案。

## 当前实现与默认状态

| 能力 | 实现内容 | 代码与示例配置默认值 |
| --- | --- | --- |
| V3 基础检索 | 用户隔离、增强文本与原文向量、查询扩展、双通道混合检索 | 保留原流程，`QUERY_DUAL_CHANNEL=true` |
| 时间检索 | 规则优先的时间抽取、确定性窗口、时间打分 | `TEMPORAL_MODE=soft`、参照 `replay`、抽取 `hybrid` |
| 多跳补充 | 分步目标召回、原文实体桥接、批次内邻接补充 | `MULTIHOP_ENABLED=true`；结构化规划与独立目标嵌入关闭 |
| D1/F1 治理 | 事实版本、来源校验、分层抽取式摘要、依赖失效 | `GOVERNANCE_MODE=off`；验证环境显式使用 `shadow` |

【已验证】两轮改动保留既有消息库、向量、增强与时间模块。治理不会自动切换到 `active`。
【存疑】真实抽取语义、最终回答质量及官网 B2/D1/F1 得分改善仍需验证。
两轮修改与 V3 影响见 [改动说明](V3_TWO_ROUND_REVIEW.md)，实施与验收见
[治理指南](GOVERNANCE.md)、[验证报告](GOVERNANCE_VALIDATION.md)。

## 接口协议

同一组路径同时支持本地简化协议与 Agent Memory Leaderboard 官网 Add/Search 协议（路径名可自定，请求/响应格式才是固定契约）：

| 路径 | 简化请求 | 官网兼容请求 |
| --- | --- | --- |
| `POST /set` | `{"memory_text":"..."}` | 官方 Add JSON（`request_id`/`messages`/`user_id`/`session_id`，响应回显三个 ID） |
| `POST /get` | `{"query":"..."}` | 官方 Search JSON（`query`/`user_id`/`top_k`[/`options`]，响应 `{"data":[...]}`） |
| `GET /health` | 无鉴权健康检查 | 官网正式任务需要 |

关键契约：`user_id` 是唯一检索隔离边界；Add 在持久化完成且可检索后才返回 200；Search 只返回证据不回答问题；默认数据保留期为 30 天，由保留策略自动清理。Swagger/OpenAPI 页面默认关闭，不额外暴露业务接口。仓库为 OpenAI 模式预设 `gpt-4o-mini`；正式提交所需模型、接口与保留要求，以官网最新清单为准。

请求与响应字段保持兼容。`active` 治理结果可以是带来源的组合证据，ID、分数与
`created_at` 的含义不同于基础检索，详见 §4.3；不能将字段兼容解释为返回行为完全相同。

官方资料：[Agent Memory 文档](https://agentmemories.ai/home) · [DeepSeek API](https://api-docs.deepseek.com/) · [FastEmbed](https://qdrant.github.io/fastembed/)

## 项目结构

```text
app/
├── config.py       # 环境变量配置
├── embeddings.py   # FastEmbed/ONNX 向量模型
├── governance/     # 事实抽取、版本解析、分层摘要、治理检索及派生表
├── llm.py          # DeepSeek/OpenAI 调用；查询扩展与时间抽取独立
├── main.py         # FastAPI 路由
├── multihop.py     # 分步目标、原文共享片段桥接、有界补充
├── prompts.py      # 写入增强 V2 与查询扩展 V3 提示词
├── schemas.py      # 两套接口契约
├── service.py      # 写入、混合/时间/多跳检索、治理分流与索引维护
├── storage.py      # SQLite、幂等租约、迁移、快照与 user_id 隔离
├── temporal.py     # 时间约束解析、窗口计算、衰减打分（纯函数）
└── temporal_extraction.py  # 常见表达规则、LLM 字段与原文证据校验
scripts/
├── generate_benchmark.py   # 用 DeepSeek 生成严格校验的合成数据
├── manage_governance.py    # 备份、迁移、回填、修复、重建与代次激活
├── run_benchmark.py        # 只通过 /set、/get 执行批量评测
├── run_governance_eval.py  # D1/F1 四组固定数据对照
├── run_multihop_regression.py # 多跳逐例回归与冻结扩展对照
└── run_temporal_ab.py      # 时间检索对照，支持治理影子模式
tests/
├── test_api.py
├── test_benchmark_tools.py
├── test_prompt_v2.py       # 提示词 V2/V3 数据流（模拟 SDK 响应）
├── test_temporal.py        # 时间感知检索单元与服务级测试
├── test_multihop_retrieval.py # 多跳补充、来源与原排序保护
├── test_governance*.py     # 版本/摘要、迁移与存储一致性
└── fixtures/
```

## 1. 本地启动

需要 Python 3.10+，命令适用于 Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

不填任何 LLM 密钥即可运行基础向量写入与检索；多跳规划与独立事实抽取需要可用的 LLM。
首次使用 FastEmbed 时会将 ONNX 模型（`paraphrase-multilingual-MiniLM-L12-v2`）下载到
`data/model_cache`，之后复用缓存；SQLite 默认位于 `data/agent_memory.db`。

## 2. LLM 配置（可选）

LLM 用于 Add 索引增强、Search 查询扩展、复杂时间约束抽取，以及 `shadow/active` 下
独立的事实抽取。查询扩展和时间抽取独立调用、独立控制；常见时间表达无需 LLM。
事实抽取只读取来源消息及同批有限上下文，不读取问题、选项或预测答案。
摘要由本地代码组织完整来源片段，不增加摘要生成调用；治理查询复用已有查询准备。
提示词禁止回答问题或判断选项，返回内容为原始消息证据或带标签、来源的抽取式组合。

DeepSeek（本地实验，默认 `deepseek-v4-flash`）：

```dotenv
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=你的密钥
```

OpenAI 兼容配置：

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=你的密钥
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
LLM_FAILURE_MODE=strict
```

使用其他 OpenAI 兼容服务时，按所选服务的实际配置设置 `OPENAI_BASE_URL` 和密钥；
客户端直接读取该配置。模型身份、数据处理规则、配额与是否满足评测要求需另行核验。

DashScope 向量配置示例（正式评测的模型要求以官网清单为准）：

```dotenv
EMBEDDING_PROVIDER=dashscope
EMBEDDING_MODEL=text-embedding-v4
DASHSCOPE_API_KEY=你的百炼密钥
```

切换向量模型时，**必须使用全新数据库或完整重建原有向量，不能混用模型或分数**。
DashScope 调用由服务内部按最多 10 条文本分批，兼容端点复用 OpenAI SDK。
治理回填只构建派生事实索引，不会重建消息向量。

`LLM_FAILURE_MODE=fallback` 时放弃失败的增强、扩展或复杂时间抽取，保留已确认的规则时间约束；`strict` 时返回可重试的 HTTP 503，便于压测时暴露上游问题。

## 3. 时间感知检索（V3）

针对长期多轮对话中的时间序列推理，V3 把时间处理拆成“规则优先、LLM 补充抽取 + 确定性解析”两层：

- 常见表达（“昨天、今天、本月、过去三天、last week”等）先由规则解析；复杂表达通过独立 LLM 入口抽取。只读取原查询，不读取候选项或扩展文本。关闭查询扩展、或设置 `LLM_PROVIDER=none`，均不影响常见规则；
- 绝对窗口由 `app/temporal.py` 纯函数计算。显式查询时间优先；默认 `replay` 模式取该 user 的最大有效 `source_timestamp`，没有源消息时间时放弃相对窗口；`realtime` 模式固定使用本次检索开始时刻（在 LLM/向量调用之前捕获）；
- 记录时间只取源消息时间，缺失或不可解析时保持未知，**不回退到 `created_at`**，因此补录的入库时间不会移动历史查询锚点；
- 软模式（默认）：`分数 = w_sem·语义 + w_lex·词法 + w_time·time_match`。窗口内得 1.0，窗口外按半衰期衰减，无时间戳记录的时间项固定为 0.5，仍受最终相关度阈值约束；
- 窗口端点：过去滚动窗口为 `[参考时刻−长度, 参考时刻]`，包含最新记录；未来滚动窗口为 `[参考时刻, 参考时刻+长度)`；日历窗口左闭右开；事件“之前/之后”均不含事件时刻。日历计算仍使用 UTC；
- 严格模式：通过现有分数门槛后，依次返回“确认在窗内 → 时间未知 → 窗外补充”；各组内部按分数或首末次排序，最后统一截取 `top_k`。未知时间不冒充窗内证据，增大 `top_k` 不会让补充记录越过窗内记录。`strict` 保留不足时补充的语义，并非窗外一律禁止返回；
- 事件锚定采用两阶段检索：先用事件短语定位最佳命中记录，以其源消息时间代理事件边界，源时间未知或置信度不足时放弃该事件约束；尚未抽取正文中的事件发生时间；
- 程序化校验：核验数量、单位、方向、范围和原文证据；非法方向不再补为 `past`。事件短语须是原文片段，且前后关系一致。多窗口、无法完整表达的组合或模糊数量保守回退，不只采用其中一个条件；
- `TEMPORAL_MODE=off` 时基础打分旁路时间处理；它不会关闭多跳补充或 active 下的事实版本判断。对照 V3 基础查询路径还需关闭这两个功能。

```dotenv
TEMPORAL_MODE=soft                  # off / soft / strict
TEMPORAL_EXTRACTION_MODE=hybrid     # off / rules / hybrid；独立于 LLM_SEARCH_EXPANSION
TEMPORAL_REFERENCE_MODE=replay      # replay / realtime；与打分模式独立
TEMPORAL_WEIGHT=0.20                # 时间项权重（与 0.8/0.2 三路归一化）
TEMPORAL_DECAY_HALF_LIFE_DAYS=30    # 窗口外距离衰减的半衰期
TEMPORAL_EVENT_ANCHOR_MIN_SCORE=0.30 # 事件锚点定位的最低置信分
```

内部调用可通过 `MemoryService.search(..., query_time_ms=毫秒时间戳)` 指定参考时间，优先于上述模式；HTTP 请求字段保持不变。参考时间来源记录为 `query_time` / `conversation_frontier` / `request_time` / `unavailable`，可通过调试日志核对。回放缺少源消息时间且未指定查询时间时，退回普通检索；全部记录时间未知时也不执行首末次排序。

`TEMPORAL_EXTRACTION_MODE=rules` 仅启用本地规则；默认 `hybrid` 仅在规则不能完整解析且 LLM 可用时调用复杂抽取。复杂查询同时开启扩展时可能有两次 LLM 调用；规则命中不会增加时间抽取调用。`off` 只关闭时间抽取；`TEMPORAL_MODE=off` 则旁路全部时间处理。

### 3.1 多跳检索

开启 `MULTIHOP_ENABLED=true` 后，默认保持 V3 查询扩展的提示词、输入与输出限额，
用本地规则识别间接指代线索，并将已有扩展中的分号短语拆为 2–4 个检索目标。
这是启发式检索规划，不是经过语义验证的依赖分析；目标的 `evidence` 标记原问题来源，
不能证明目标由问题蕴含。无分号目标或规则未命中时，不启用补充。

`MULTIHOP_STRUCTURED_PLANNER=true` 可实验性地生成明确的分步目标和问题逐字片段；
片段经过程序校验，但规划完整性仍需核验。该模式会增加模型输出耗时，默认关闭。
它更换同一次查询扩展调用的提示词，并将输出上限从 512 调到 1024 token，不新增独立规划调用。
默认复用基础查询向量，用目标词法、原文名称与既有记录向量补充，不新增向量或
逐跳 LLM 调用。`MULTIHOP_EMBED_GOALS=true` 才独立嵌入目标，与原查询共用一个批次。
没有有效计划、查询含已解析时间约束、
LLM 关闭或 `top_k < 6` 时，沿用基础检索。可选目标向量生成失败时，重新生成
基础查询向量并回退。

扩展分两步：分别召回目标相关记录；再从原文名称或共享片段查找后继。桥接仅在
同一 `user_id` 内进行，可以跨 session；英文片段核验单词边界，排除路径中已访问
记录。复用既有记录向量和目标向量打分，桥接原文不发送给外部模型。调试日志只
记录来源 ID、目标 ID 和步骤，不记录原文。

针对仅用代词连接的片段，再补充种子前后各 2 条原文。范围严格限定为同一用户、
同一 session、同一写入 request，以输入 `ordinal` 确定邻接；它不代表事件时间。
不跨写入批次推断代词。`MULTIHOP_CONTEXT_RADIUS=0` 可关闭此补充；存储只读取
原有批次与顺序字段，旧数据库无需重建。

合并时完整保留基础检索已返回的顺序与分数，只在剩余名额中补充，最多新增 12 条。
基础结果已填满 `top_k` 时不再补充。默认至多 2 轮、每轮 3 个来源、4 个桥接候选；
本地补充阶段的时间预算为 6 秒，不含查询规划和向量生成。参数见 `.env.example`；
`MULTIHOP_ENABLED=false` 可关闭该功能。

简化 `/get` 的 `{"query":"..."}` 固定取 5 条，达不到默认的 `MULTIHOP_MIN_TOP_K=6`；
完整 Search 协议使用请求中的 `top_k`。开启多跳开关不表示每次查询都会执行补充。
active 治理中的版本感知证据组织可以在 `top_k=1` 时返回，见 §4。

**边界：**逐字共现是检索关联依据，不能证明实体归属、因果或完整推理链。
仅靠代词连接、共享名称缺失或计划不完整时，仍可能漏召回。第一轮补充自身不判断
事实版本；第二轮治理负责检查有效性，见 §4。服务不生成最终答案。额外目标和补充证据会增加计算与上下文开销；关闭功能
与开启功能的真实模型效果、理想计划效果应分别报告。

现有合成数据的对照命令：

```powershell
python -m scripts.run_multihop_regression --llm oracle --report data/multihop_default_oracle_final.json --cache data/multihop_default_oracle_cache.json
python -m scripts.run_multihop_regression --llm real --report data/multihop_default_real_final.json --cache data/multihop_default_real_cache.json
```

`real` 使用配置的真实查询模型；`oracle` 使用用例自带无答案扩展，不能视为真实
模型效果。四个模式共享写入和向量：V3 旧提示词、同计划的基础检索、分步补充、
本地桥接。报告记录完整链 Recall@10/@20/@100、单跳证据退步、规划与检索耗时；
缓存命中时的规划耗时不代表在线调用耗时。出现逐用例召回退步时返回非零状态。

默认模式用同一份 V3 响应做新旧对照，以隔离模型随机性。需要复用此前真实 V3 响应时，
可加 `--reuse-legacy-cache 路径`；脚本核对数据集、提示词和模型，缺项时停止，不调用模型。
`--structured-planner` 用于单独评估结构化规划。合成用例的链证据在写入批次中相邻，
因此上下文补充的收益依赖该布局；关键词链召回不代表最终回答正确率或官网得分。

## 4. D1/F1 事实版本与抽取式摘要

### 4.1 两项能力及联合一致性

- **D1 新值覆盖与当前状态：**独立抽取有来源的事实，按
  `(user_id, subject_id, predicate_key, scope_key)` 建立槽位；保存变化、更正、撤回、
  支持与冲突关系。当前查询读取完整版本，不仅在向量 `top_k` 中选新旧值。
- **版本规则：**真实变化保留旧状态；更正不把错误说法作为真实历史；撤回后无新值
  则保持未知。计划、假设、多值补充和迟到历史不直接覆盖当前状态；未知属性不采用
  无条件的“最后写入获胜”。消息时间、入库时间与有原文依据的有效时间分开保存。
- **F1 长历史综合：**从来源和事实构建会话/阶段、主题与全局摘要，分为当前、历史、
  事件/决定、计划、冲突/纠正五类。保留完整片段、否定、条件、归因及未知时间标签。
- **摘要更新：**记录来源、事实、子摘要与覆盖范围。新写入、更正、撤回、修复和删除
  使摘要失效；从最新事实与必要原文重建，发布前检查代次和修订号。
- **版本感知多跳：**当前状态使用适用事实桥接，候选和邻接再次核验，避免重新带回
  已替代实体；混合消息保留仍有效的事实片段。必要概览和来源可合并为一条结果。

来源缺项、截断、非法输出或超预算保存为 `partial/pending/failed`，不当作 `no_fact`。
默认保留原文写入成功并标记治理覆盖不完整；摘要失败不撤销已提交原文。
本地校验与抽取式组织不等于已经证明模型的语义分类正确。

### 4.2 模式、预算与调用成本

| `GOVERNANCE_MODE` | 写入 | 查询 |
| --- | --- | --- |
| `off`（默认） | 原有增强与向量，保存原文、提交顺序与租约元数据 | 原路径，包含已开启且符合条件的多跳补充 |
| `shadow` | 原流程加独立事实抽取与摘要索引 | 内部计算治理对照，返回原路径的 ID、内容、分数和顺序 |
| `active` | 同 shadow；可选严格抽取失败策略 | 当前、历史、变化和综合需求使用治理；普通查询沿用原路径 |

```dotenv
GOVERNANCE_MODE=off                 # off / shadow / active；不会自动启用 active
GOVERNANCE_STRICT_WRITE=false       # 仅 active 有效；true 时不完整抽取阻断写入
GOVERNANCE_INPUT_TOKENS=3000         # 每批抽取输入工程预算
GOVERNANCE_OUTPUT_TOKENS=4096        # 抽取输出上限
GOVERNANCE_ADD_BUDGET_SECONDS=60     # 每次写入的额外抽取预算
GOVERNANCE_SUMMARY_BUDGET_MS=50      # 本地摘要更新预算
GOVERNANCE_QUERY_BUDGET_MS=50        # 治理查询新增处理预算
GOVERNANCE_CONTENT_TOKENS=8192       # 治理返回内容预算
```

独立抽取发生在写入、回填或修复阶段，长消息可能分批调用。治理查询共享已有扩展与
查询向量，不新增查询模型调用、事实向量或摘要生成模型。`shadow/active` 仍会增加
写入成本、派生存储和本地处理；来源标签也可能使短历史输出变长。
`tiktoken==0.13.0` 的固定 `cl100k_base` 编码仅计量工程预算，不代表所有上游的计费数。

### 4.3 V3 兼容性与返回语义

`/set`、`/get` 的请求与响应字段、用户隔离、原有向量和基础检索保留。
active 治理命中通常是一条组合结果：

- `id` 使用 `gov_` 派生 ID，不是原 `memories` 行 ID；内容中保留来源 ID。
- `score` 当前固定为 `1.0`，不能解释为 V3 相似度或事实置信度。
- `created_at` 使用用户快照中来源的最新实际持久化时间，不表示事实生效时间。
- 内容可以是来源片段与标签的组合；外部按消息 ID 查库、分数过滤或时间排序的逻辑需核验。

对照 V3 基础查询路径需要同时配置：

```dotenv
MULTIHOP_ENABLED=false
GOVERNANCE_MODE=off
```

这不会撤销新代码的数据库迁移、原文列、提交记录或租约机制。
`TEMPORAL_MODE=off` 也不会关闭 active 下的版本解析。

### 4.4 迁移、回填与回退

升级前停止旧写入进程，处理遗留 `processing` 请求并备份。即使治理为 `off`，
新代码初始化也会安装新增存储结构；首次治理迁移自动备份旧库，遇处理中请求拒绝迁移。
迁移可重复执行。以下为维护示例，请替换实际数据库路径和用户 ID：

```powershell
python -m scripts.manage_governance backup --database data/agent_memory.db --output data/pre_governance.db
python -m scripts.manage_governance migrate --database data/agent_memory.db
python -m scripts.manage_governance backfill --user-id example-user
python -m scripts.manage_governance status --user-id example-user
```

`repair`、`rebuild` 与 `activate` 的代次、修订号参数见 [治理指南](GOVERNANCE.md)。
回填只读取已存来源及元数据，不重放 Add、不生成向量、不重复消息。
未完整覆盖的新代次不能激活；代次激活不会把功能模式切换为 `active`。
回退使用新代码的 `GOVERNANCE_MODE=off`，保留原文和派生表，不执行破坏性数据库降级。

## 5. curl 验证

简化写入与查询：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/set" -H "Content-Type: application/json" -d '{"memory_text":"用户最喜欢的饮料是无糖拿铁。"}'
curl.exe -X POST "http://127.0.0.1:8000/get" -H "Content-Type: application/json" -d '{"query":"用户喜欢喝什么饮料？"}'
```

无关查询低于 `MIN_RELEVANCE_SCORE` 时返回 `{"memory_text":"","results":[]}`。

官网兼容格式：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/set" -H "Content-Type: application/json" -d '{"request_id":"eval:demo:chunk-0","messages":[{"role":"user","timestamp":1704067200000,"content":"Alice prefers Ethiopian coffee."}],"user_id":"eval:demo:user-1","session_id":"eval:demo:session-1"}'
curl.exe -X POST "http://127.0.0.1:8000/get" -H "Content-Type: application/json" -d '{"query":"What coffee does Alice prefer?","user_id":"eval:demo:user-1","top_k":100}'
```

## 6. API 鉴权

公网部署必须设置 `MEMORY_API_KEY`（`python -c "import secrets; print(secrets.token_urlsafe(32))"` 生成），支持 `Authorization: Bearer`、`Authorization: Token` 与 `X-Api-Key` 三种传法；`/health` 按契约始终免鉴权。不要把密钥放在 URL、代码或公开仓库中。

## 7. 测试与验证结果

```powershell
python -m pytest
```

全部测试使用确定性的离线假向量器与桩 LLM，不下载模型、不调用外部 API。用例覆盖：简化/官网双协议契约、`user_id` 严格隔离、Add 幂等与冲突检测、可选鉴权、提示词 V2/V3 数据流、时间窗口数学（日历/滚动/月末收敛）、软打分与严格模式、事件锚定、幻觉守卫，以及 `TEMPORAL_MODE=off` 与旧版分数逐位一致的回归断言。`tests/test_temporal_reference.py` 另覆盖回放/实时模式、显式查询时间优先、补录与未知时间回退、跨月调用及接口兼容；本机 HTTP 测试需要允许监听回环端口。

2026-10-03 修订增加 `test_temporal_policy.py`、`test_temporal_extraction.py`：覆盖端点、严格模式各组顺序与 `top_k` 前缀稳定性、常见规则、原文证据、独立开关和失败回退。验证记录见 [V3 方案 §6.2](docs/v3-temporal-retrieval.md#62-窗口与抽取修订验证2026-10-03)。

### 7.1 最新联合验收（2026-10-09）

【已验证】冻结基线 241 项及新增测试全部通过，最终为 280 项，
0 失败、错误或跳过；本次提交前再次运行完整离线回归。测试覆盖事实版本、摘要覆盖与失效、部分抽取失败、旧租约拒绝、
用户与代次隔离、迁移重入、删除与重建一致性、旧实体桥接保护，以及 off/影子返回兼容。

| 数据与检查 | 结果 | 解释边界 |
| --- | --- | --- |
| 第一轮 15 个合成多跳例，冻结已有真实模型响应 | 完整链 Recall@100 从 14/15 到 15/15；@10/@20 均为 12/15 | 缓存响应回放，关键词链召回不代表推理或最终回答正确 |
| 第二轮既有多跳与时间回归 | 15 个多跳、15 个时间例的 off/影子逐例排名与既有记录一致 | 仅说明指定数据中未发现退步 |
| 13 个 D1 合成例，原检索对联合方案 | 当前状态核验从 2/13 到 13/13 | 抽取 SDK 替身，独立人工金标复核待完成 |
| 3 个 F1 合成例，原检索对联合方案 | 必要完整原文覆盖从 20% 到 100% | 检索代理指标，不是语义事实准确率或官网得分 |
| 五轮交替耗时对照 | 普通查询 P95 增量 -0.54 ms，治理 +12.96 ms，通过既定门槛 | 本机固定数据、空扩展、排除冷启动，不包含外部查询模型耗时 |

治理四组对照固定来源、向量、参照时间、`top_k=1` 和 8192 token 返回预算：

```powershell
python -m scripts.run_governance_eval --extraction fixture --repeats 5 --report data/governance_eval.json
python -m scripts.run_temporal_ab --llm oracle --governance-mode shadow --report data/temporal_shadow.json
```

【存疑】真实事实抽取的小样本共 9 次请求均遇 `RateLimitError`，仅验证原文保留和失败
状态；真实抽取语义、最终回答质量与官网改善尚未验证，因此总体 active 启用条件未满足。
完整四组质量、压缩率、性能门槛和剩余核验见 [验证报告](GOVERNANCE_VALIDATION.md)。
合成用例与说明位于 [tests/fixtures](tests/fixtures/README.md)；运行报告保存在忽略的
`data/` 目录，复现命令会重新生成，不随源码提交。

## 8. 合成数据批量测试

生成器始终读取 `.env` 的 DeepSeek 配置，与服务当前的 `LLM_PROVIDER` 无关。正例答案必须逐字存在于记忆且不泄漏在 query 中，负例必须无答案，重复项自动丢弃补齐：

```powershell
python -m scripts.generate_benchmark --count 200 --negative-count 20 --output data/benchmark_200.jsonl
```

建议为批量测试使用独立数据库，并保持 `LLM_FAILURE_MODE=strict`，避免本地回退掩盖上游失败：

```powershell
$env:DATABASE_PATH="data/benchmark_run.db"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# 另一窗口：
python -m scripts.run_benchmark --dataset data/benchmark_200.jsonl --base-url http://127.0.0.1:8000 --top-k 5 --min-recall 0.90 --report data/benchmark_report.json
```

运行器只走 HTTP，输出 Set/Search 成功率、Recall@1/@k、MRR、负例误召回率与延迟 p50/p95/max；每次运行使用独立 `run_id` 隔离，不会检索到历史数据。不要用真实个人信息、隐藏样本或已知金标生成测试数据，合成结果也不能替代官方评测。

## 9. Docker 与公网调试

```powershell
docker build -t agent-memory-api .
docker run --rm -p 8000:8000 --env-file .env -v "${PWD}/data:/app/data" agent-memory-api
```

必须挂载 `/app/data`，否则容器删除后 SQLite 记忆和模型缓存都会丢失。`ngrok http 8000` 可用于临时联调（Add/Search/Health 分别指向 `/set`、`/get`、`/health`）；正式部署应依据评测清单配置稳定 HTTPS 地址、持久卷和进程守护。

## 原有实现边界与调参

- SQLite 使用 WAL、同步事务与带令牌的 `request_id` 幂等租约；Add 在原文与向量提交后才返回 200。治理启用时事实、版本关系、来源状态、修订号和失效标记一并原子提交；摘要在事务外更新。
- 接口输入日志默认开启：`/set`、`/get` 的请求正文写入 `INPUT_LOG_PATH`（JSONL，按天轮转）。它是评测数据的第二份副本，保留份数对齐 `DATA_RETENTION_DAYS`；只记请求体，永不记请求头。
- 检索只加载当前 `user_id`；无时间约束时为语义余弦 + Unicode 关键词重合度的二元混合（默认 0.8/0.2），有时间窗口时三路归一化。
- `MIN_RELEVANCE_SCORE` 越高误召回越少但漏召回越多；只能用公开材料调参，禁止接触或硬编码评测金标。
- 更换 `EMBEDDING_MODEL` 后旧向量维度可能不一致，应换新数据库或离线重建索引，不能混用分数。
- 数据按写入时间与配置保留期自动清理，默认 30 天；删除同时处理派生索引与摘要依赖。

## 依赖与数据披露

- API 封装、SQLite 幂等/隔离、混合检索与时间感知逻辑均为本实现原创，未复制现成记忆论文或仓库；向量运行时为 Qdrant 维护的 FastEmbed（默认模型 Apache-2.0），其余依赖见 `requirements.txt`。
- 开启 DeepSeek/OpenAI 后，原始消息和查询会发送给所选第三方 API；部署方需自行确认比赛数据规则与所在地区合规要求。不能发送评测数据时保持 `LLM_PROVIDER=none`，只运行本地向量检索。
- 本实现不含任何数据集硬编码；禁止用 benchmark 金标、泄漏样本、人工实时答题或提示注入调整本系统。
