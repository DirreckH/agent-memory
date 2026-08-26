# Agent Memory API（FastAPI + DeepSeek + FastEmbed + SQLite）

这是一个可本地运行、可 Docker 部署的文本记忆服务。记忆内容持久化到 SQLite；
本地 ONNX 向量模型完成语义召回；DeepSeek 可选地生成索引增强文本和查询扩展。
服务返回的始终是记忆证据，不生成最终答案。

## 先说明：原始接口定义与官网现行协议不一致

截至 2026-08-26，Agent Memory Leaderboard 官网要求参赛方提供同步 **Add / Search**：

- Add 请求必须包含 `request_id`、`messages`、`user_id`、`session_id`，成功响应要原样回显三个 ID；
- Search 请求必须包含 `query`、`user_id`、`top_k`，可选 `options`；响应必须是
  `{"data": [...]}`，每项至少包含 `id` 和 `content`；
- `user_id` 是唯一检索隔离边界，不得跨用户召回；正式评测 `top_k=100`；
- Add 必须在持久化完成且可立即检索后才返回 HTTP 200；
- 正式任务需要一个无鉴权的 GET Health 地址，缺省检查 Add 同源 `/health`；
- Search 只能返回记忆证据，不能直接生成最终答案；评测数据应在任务完成后 30 天内删除。

因此，仅实现最初给出的 `{"memory_text": ...}` 与 `{"query": ...}` 无法通过官网 smoke。
本项目在同一组路径上同时支持两种协议：

| 路径 | 本地简化请求 | 官网兼容请求 |
| --- | --- | --- |
| `POST /set` | `{"memory_text":"..."}` | 官方 Add JSON |
| `POST /get` | `{"query":"..."}` | 官方 Search JSON |
| `GET /health` | 无鉴权健康检查 | 官网正式任务需要 |

路径名称本身不必叫 `/add`、`/search`；官网允许分别配置 Add URL 和 Search URL，
请求与响应格式才是固定契约。FastAPI 的 Swagger/OpenAPI 页面默认关闭，因此除了必要的健康检查，
不额外暴露业务接口。

还有一个不能忽略的规则冲突：官网当前 Full 提交清单明确要求 Add/Search 使用
`gpt-4o-mini`。因此，**DeepSeek 模式可以用于本地开发、方法实验或兼容性验证，但按当前规则不能
如实勾选 Full 的模型合规项**。本项目同时保留 `LLM_PROVIDER=openai` 的切换能力；正式提交前必须
再次阅读当期规则，不应把 DeepSeek 运行结果冒充为 `gpt-4o-mini` 结果。

官方资料：

- [Agent Memory Leaderboard 文档与 API Guide](https://agentmemories.ai/home)
- [DeepSeek API 首次调用](https://api-docs.deepseek.com/)
- [DeepSeek JSON Output](https://api-docs.deepseek.com/guides/json_mode/)
- [FastEmbed 官方文档](https://qdrant.github.io/fastembed/)

## 项目结构

```text
.
├── app/
│   ├── config.py       # 环境变量配置
│   ├── embeddings.py   # FastEmbed/ONNX 向量模型
│   ├── llm.py          # DeepSeek/OpenAI 兼容调用
│   ├── main.py         # FastAPI 路由
│   ├── schemas.py      # 两套接口契约
│   ├── service.py      # 写入、混合检索、阈值过滤
│   └── storage.py      # SQLite、幂等、user_id 隔离
├── tests/test_api.py
├── .env.example
├── Dockerfile
├── pytest.ini
└── requirements.txt
```

## 1. 本地安装与启动

需要 Python 3.10+。以下命令适用于 Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

不填任何 LLM 密钥也能运行完整的本地向量写入与检索：

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

第一次真实调用 `/set` 时会下载
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` 的 ONNX 文件（约 220 MB）到
`data/model_cache`；之后复用本地缓存。SQLite 文件位于 `data/agent_memory.db`。

健康检查：

```powershell
curl.exe http://127.0.0.1:8000/health
```

## 2. 填入 DeepSeek API Key

编辑 `.env`，只改本地文件，不要把它提交到 Git：

```dotenv
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=在这里填入你的真实密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

然后重启 Uvicorn。当前 DeepSeek 官方文档列出的通用模型是
`deepseek-v4-flash` / `deepseek-v4-pro`；旧的 `deepseek-chat` 已进入弃用流程，因此没有把旧名称写死。

DeepSeek 在本项目中执行两件事：

1. Add 时给每条原始消息生成“仅用于检索”的事实保真索引文本；
2. Search 时扩展同义表达，但提示词明确禁止回答问题或判断选项。

最终返回的 `content` 仍是原始消息证据。`LLM_FAILURE_MODE=fallback` 时，上游异常会自动退回纯向量检索；
若要求每次调用严格使用同一模型，可改为 `strict`，此时 LLM 异常返回可重试的 HTTP 503。

正式 Full 若仍执行官网当前的 `gpt-4o-mini` 规则，应使用：

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=在这里填入对应密钥
OPENAI_MODEL=gpt-4o-mini
LLM_FAILURE_MODE=strict
```

## 3. 可直接复制的 curl 验证

### 简化接口：写入

```powershell
curl.exe -X POST "http://127.0.0.1:8000/set" -H "Content-Type: application/json" -d '{"memory_text":"用户最喜欢的饮料是无糖拿铁。"}'
```

预期返回：

```json
{"success":true,"message":"记忆写入成功，现已可检索","memory_count":1}
```

### 简化接口：相关查询

```powershell
curl.exe -X POST "http://127.0.0.1:8000/get" -H "Content-Type: application/json" -d '{"query":"用户喜欢喝什么饮料？"}'
```

`memory_text` 应包含“无糖拿铁”，`results` 给出排序、分数和稳定记忆 ID。

### 简化接口：无关查询

```powershell
curl.exe -X POST "http://127.0.0.1:8000/get" -H "Content-Type: application/json" -d '{"query":"鲸鱼如何在水下交流？"}'
```

结果低于 `MIN_RELEVANCE_SCORE` 时返回：

```json
{"memory_text":"","results":[]}
```

### 官网兼容 Add

```powershell
curl.exe -X POST "http://127.0.0.1:8000/set" -H "Content-Type: application/json" -d '{"request_id":"eval:demo:chunk-0","messages":[{"role":"user","timestamp":1704067200000,"content":"Alice prefers Ethiopian coffee."}],"user_id":"eval:demo:user-1","session_id":"eval:demo:session-1"}'
```

### 官网兼容 Search

```powershell
curl.exe -X POST "http://127.0.0.1:8000/get" -H "Content-Type: application/json" -d '{"query":"What coffee does Alice prefer?","user_id":"eval:demo:user-1","top_k":100}'
```

响应是官网规定的对象：

```json
{"data":[{"id":"mem_...","content":"[user | 2024-01-01T00:00:00.000Z]\nAlice prefers Ethiopian coffee.","score":0.8,"created_at":"2024-01-01T00:00:00.000Z"}]}
```

实际分数由本地模型计算，不保证与示例数字完全相同。

## 4. API 鉴权

公网服务不要保持匿名。先生成强随机密钥并写入 `.env` 的 `MEMORY_API_KEY`：

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

服务兼容官网支持的三种传法：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/get" -H "Content-Type: application/json" -H "Authorization: Bearer 你的MEMORY_API_KEY" -d '{"query":"test"}'
curl.exe -X POST "http://127.0.0.1:8000/get" -H "Content-Type: application/json" -H "Authorization: Token 你的MEMORY_API_KEY" -d '{"query":"test"}'
curl.exe -X POST "http://127.0.0.1:8000/get" -H "Content-Type: application/json" -H "X-Api-Key: 你的MEMORY_API_KEY" -d '{"query":"test"}'
```

`/health` 按官网契约始终无需鉴权。不要把密钥放在 URL、代码、截图或公开仓库中。

## 5. 运行 pytest

```powershell
python -m pytest
```

测试使用确定性的离线假向量器，不下载模型、不调用 DeepSeek/OpenAI。覆盖：

- 简化 `/set` 写入与 `/get` 正向召回；
- 无关 query 返回空结果；
- 官网 Add/Search 请求与响应格式；
- `user_id` 严格隔离；
- Add 重试幂等与冲突检测；
- 可选 Bearer API Key 和无鉴权 Health。

## 6. Docker

```powershell
docker build -t agent-memory-api .
docker run --rm -p 8000:8000 --env-file .env -v "${PWD}/data:/app/data" agent-memory-api
```

必须挂载 `/app/data`，否则容器删除后 SQLite 记忆和模型缓存都会丢失。

## 7. ngrok 公网调试

服务启动后另开终端：

```powershell
ngrok http 8000
```

把 ngrok 给出的 HTTPS 地址配置为：

- Add URL：`https://你的域名/set`
- Search URL：`https://你的域名/get`
- Health URL：`https://你的域名/health`

ngrok 临时隧道适合本地联调和 smoke，不满足官网“托管接口至少 30 天稳定公网可达”的正式提交要求。
正式评测应使用带 HTTPS、持久卷、进程守护和监控的 VPS/云容器，并按实际压测结果申报并发、超时和限流能力。

## 实现边界与调参

- SQLite 使用 WAL、同步事务和 `request_id` 幂等表；Add 在向量写入提交后才返回 200。
- 每条消息单独存储，适合官网每次最多约 20 条消息/2000 词的分块方式。
- 检索只加载当前 `user_id`，语义余弦分数与 Unicode 关键词重合度混合排序。
- `MIN_RELEVANCE_SCORE` 越高，误召回更少但漏召回更多；正式打榜前应只用公开训练/验证材料调参，不能接触或硬编码评测金标。
- 自动清理按写入时间执行，是对“30 天内删除评测数据”的保守实现；如需精确按 Job 完成时间清理，应在获得平台回调/任务元数据后扩展内部管理流程，不能靠公开 Search 接口完成。
- 更换 `EMBEDDING_MODEL` 后，旧 SQLite 向量可能维度不一致。应使用新数据库文件或离线重建索引，不能混用分数。

## 原创性、依赖与数据出境披露

- 本仓库中的 API 封装、SQLite 幂等/隔离逻辑和混合检索逻辑是为本实现新编写的，未复制某个记忆论文或现成记忆仓库；
- 向量运行时使用 Qdrant 维护的 FastEmbed，默认模型为 Apache-2.0 许可的
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`；其余 Python 依赖及版本范围见
  `requirements.txt`，正式投稿时应按主办方表单要求列出对应作者和许可证；
- 开启 DeepSeek/OpenAI 后，原始消息和查询会被发送给所选第三方 API。部署方必须自行确认比赛数据规则、
  第三方数据保留政策、所在地区合规要求和主办方是否允许该处理方式；若不能发送评测数据，应保持
  `LLM_PROVIDER=none`，只运行本地向量检索；
- 禁止用 benchmark 金标、泄漏样本、人工实时答题或提示注入调整本系统。本实现不包含任何数据集硬编码。
