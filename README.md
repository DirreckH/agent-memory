# Agent Memory API（FastAPI + LLM + FastEmbed + SQLite）

本地可运行、可 Docker 部署的文本记忆服务：SQLite 持久化 + ONNX 语义召回 + 可选 LLM 索引增强与查询扩展，V3 起支持时间感知检索。服务只返回记忆证据，不生成最终答案。

## 接口协议

同一组路径同时支持本地简化协议与 Agent Memory Leaderboard 官网 Add/Search 协议（路径名可自定，请求/响应格式才是固定契约）：

| 路径 | 简化请求 | 官网兼容请求 |
| --- | --- | --- |
| `POST /set` | `{"memory_text":"..."}` | 官方 Add JSON（`request_id`/`messages`/`user_id`/`session_id`，响应回显三个 ID） |
| `POST /get` | `{"query":"..."}` | 官方 Search JSON（`query`/`user_id`/`top_k`[/`options`]，响应 `{"data":[...]}`） |
| `GET /health` | 无鉴权健康检查 | 官网正式任务需要 |

关键契约：`user_id` 是唯一检索隔离边界；Add 在持久化完成且可检索后才返回 200；Search 只返回证据不回答问题；评测数据 30 天内删除（由数据保留策略自动清理）。Swagger/OpenAPI 页面默认关闭，不额外暴露业务接口。注意：官网 Full 清单当前要求 `gpt-4o-mini`，DeepSeek 仅用于本地实验与方法验证，不能把 DeepSeek 运行结果冒充为合规勾选。

官方资料：[Agent Memory 文档](https://agentmemories.ai/home) · [DeepSeek API](https://api-docs.deepseek.com/) · [FastEmbed](https://qdrant.github.io/fastembed/)

## 项目结构

```text
app/
├── config.py       # 环境变量配置
├── embeddings.py   # FastEmbed/ONNX 向量模型
├── llm.py          # DeepSeek/OpenAI 兼容调用与结构化查询扩展
├── main.py         # FastAPI 路由
├── prompts.py      # 写入增强 V2 与查询扩展 V3 提示词
├── schemas.py      # 两套接口契约
├── service.py      # 写入、混合检索、时间感知打分
├── storage.py      # SQLite、幂等、user_id 隔离
└── temporal.py     # 时间约束解析、窗口计算、衰减打分（纯函数）
scripts/
├── generate_benchmark.py   # 用 DeepSeek 生成严格校验的合成数据
└── run_benchmark.py        # 只通过 /set、/get 执行批量评测
tests/
├── test_api.py
├── test_benchmark_tools.py
├── test_prompt_v2.py       # 提示词 V2/V3 数据流（模拟 SDK 响应）
├── test_temporal.py        # 时间感知检索单元与服务级测试
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

不填任何 LLM 密钥即可运行完整的向量写入与检索。首次 `/set` 会下载约 220 MB 的 ONNX 模型（`paraphrase-multilingual-MiniLM-L12-v2`）到 `data/model_cache`，之后复用缓存；SQLite 位于 `data/agent_memory.db`。

## 2. LLM 配置（可选）

LLM 负责三件事：Add 时生成事实保真的索引增强文本；Search 时扩展同义表达；从查询中抽取结构化时间约束（V3）。提示词明确禁止回答问题或判断选项，返回的 `content` 始终是原始消息。

DeepSeek（本地实验，默认 `deepseek-v4-flash`）：

```dotenv
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=你的密钥
```

官网 Full 合规模式：

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=你的密钥
OPENAI_BASE_URL=https://api.chatanywhere.tech/v1
LLM_FAILURE_MODE=strict
```

国内服务器无法直连官方端点时，用 ChatAnywhere 中转调用 `gpt-4o-mini`（`api.chatanywhere.tech`，走 OpenAI 官方转发；免费线限 200 次/天/IP，正式评测需付费 key）。配置无需改代码——[llm.py](app/llm.py) 的 OpenAI 兼容客户端直接读取 `OPENAI_BASE_URL`。

学术榜（开源方法榜）向量配置——官网要求统一使用 `text-embedding-v4`（阿里百炼 DashScope，Qwen3-Embedding 系列）：

```dotenv
EMBEDDING_PROVIDER=dashscope
EMBEDDING_MODEL=text-embedding-v4
DASHSCOPE_API_KEY=你的百炼密钥
```

切换后向量维度从本地 MiniLM 的 384 变为 1024（默认，可选 64-2048），**必须使用全新数据库，不能混用分数**。`text-embedding-v4` 单次请求最多 10 条文本，由服务内部自动分批；兼容端点复用 OpenAI SDK，无需新增依赖。

`LLM_FAILURE_MODE=fallback` 时上游异常自动退回纯向量检索；`strict` 时返回可重试的 HTTP 503，便于压测时暴露上游问题。

## 3. 时间感知检索（V3）

针对长期多轮对话中的时间序列推理，V3 把时间处理拆成“LLM 抽取 + 确定性解析”两层：

- 查询扩展提示词 V3 输出结构化时间约束：滚动窗口（“最近三个月”）、日历窗口（“上个月”）、事件锚点（“搬家之前”）、首末次排序（“第一次/最后一次”）。模型不知道也不计算当前时间，不做任何日期换算；
- 绝对窗口由 `app/temporal.py` 纯函数计算。锚点优先取该 user 的对话前沿（记忆中最大事件时间），全库无时间信息才回退服务器时钟；
- 软模式（默认）：`分数 = w_sem·语义 + w_lex·词法 + w_time·time_match`。窗口内得 1.0，窗口外按半衰期衰减，**无时间戳记录取中性值 0.5，永不惩罚**；
- 严格模式：窗口内候选足够时截掉窗口外记录；时间未知的记录始终保留；
- 事件锚定采用两阶段检索：先用事件短语定位最佳命中记录，以其事件时间为窗口边界，置信度低于阈值则放弃约束、退回常规检索；
- 程序化守卫：LLM 声称的时间约束必须能在查询原文中回查到时间信号，否则视为幻觉丢弃；
- `TEMPORAL_MODE=off` 时打分路径与旧版逐位一致，可安全回归。

```dotenv
TEMPORAL_MODE=soft                  # off / soft / strict
TEMPORAL_WEIGHT=0.20                # 时间项权重（与 0.8/0.2 三路归一化）
TEMPORAL_DECAY_HALF_LIFE_DAYS=30    # 窗口外距离衰减的半衰期
TEMPORAL_EVENT_ANCHOR_MIN_SCORE=0.30 # 事件锚点定位的最低置信分
```

## 4. curl 验证

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

## 5. API 鉴权

公网部署必须设置 `MEMORY_API_KEY`（`python -c "import secrets; print(secrets.token_urlsafe(32))"` 生成），支持 `Authorization: Bearer`、`Authorization: Token` 与 `X-Api-Key` 三种传法；`/health` 按契约始终免鉴权。不要把密钥放在 URL、代码或公开仓库中。

## 6. 运行 pytest

```powershell
python -m pytest
```

全部测试使用确定性的离线假向量器与桩 LLM，不下载模型、不调用外部 API。71 个用例覆盖：简化/官网双协议契约、`user_id` 严格隔离、Add 幂等与冲突检测、可选鉴权、提示词 V2/V3 数据流、时间窗口数学（日历/滚动/月末收敛）、软打分与严格模式、事件锚定、幻觉守卫，以及 `TEMPORAL_MODE=off` 与旧版分数逐位一致的回归断言。

## 7. 合成数据批量测试

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

## 8. Docker 与公网调试

```powershell
docker build -t agent-memory-api .
docker run --rm -p 8000:8000 --env-file .env -v "${PWD}/data:/app/data" agent-memory-api
```

必须挂载 `/app/data`，否则容器删除后 SQLite 记忆和模型缓存都会丢失。`ngrok http 8000` 可用于临时联调（Add/Search/Health 分别指向 `/set`、`/get`、`/health`），但不满足“至少 30 天稳定公网可达”的正式要求，正式提交应使用带 HTTPS、持久卷、进程守护的云部署。

## 实现边界与调参

- SQLite 使用 WAL、同步事务与 `request_id` 幂等表；Add 在向量写入提交后才返回 200；每条消息独立存储。
- 接口输入日志默认开启：`/set`、`/get` 的请求正文写入 `INPUT_LOG_PATH`（JSONL，按天轮转）。它是评测数据的第二份副本，保留份数对齐 `DATA_RETENTION_DAYS`；只记请求体，永不记请求头。
- 检索只加载当前 `user_id`；无时间约束时为语义余弦 + Unicode 关键词重合度的二元混合（默认 0.8/0.2），有时间窗口时三路归一化。
- `MIN_RELEVANCE_SCORE` 越高误召回越少但漏召回越多；只能用公开材料调参，禁止接触或硬编码评测金标。
- 更换 `EMBEDDING_MODEL` 后旧向量维度可能不一致，应换新数据库或离线重建索引，不能混用分数。
- 数据按写入时间自动清理，是“评测数据 30 天内删除”的保守实现。

## 依赖与数据披露

- API 封装、SQLite 幂等/隔离、混合检索与时间感知逻辑均为本实现原创，未复制现成记忆论文或仓库；向量运行时为 Qdrant 维护的 FastEmbed（默认模型 Apache-2.0），其余依赖见 `requirements.txt`。
- 开启 DeepSeek/OpenAI 后，原始消息和查询会发送给所选第三方 API；部署方需自行确认比赛数据规则与所在地区合规要求。不能发送评测数据时保持 `LLM_PROVIDER=none`，只运行本地向量检索。
- 本实现不含任何数据集硬编码；禁止用 benchmark 金标、泄漏样本、人工实时答题或提示注入调整本系统。
