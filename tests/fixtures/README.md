# V2 提示词样例与验证边界

`prompt_v2_cases.json` 包含 8 组人工合成样例。`add_output` 和 `query_output`
是用于驱动回归测试的模拟 LLM 响应，不是真实调用记录，也不代表唯一正确的改写。

覆盖场景：无需增强的问候、时间变化、否定、不确定性、说话人归属、乱序及省略的
source_index、数值边界、英文选择题，以及包含指令文本的消息。

运行新测试：

```sh
.venv/bin/python -m pytest tests/test_prompt_v2.py
```

测试执行真实的 FastAPI 路由、LLM 响应解析和 SQLite 写入，替换 SDK 的网络调用和
向量模型。它检查实际发送的 V2 system prompt、输入 payload、增强文本进入向量器、
空增强保留原文、按 source_index 对齐、Search 返回原始证据、选项传递、重试幂等、
增强开关，以及无效 JSON/字段在 strict 和 fallback 模式下的处理。

2026-09-14 验证结果：新增 17 项测试通过；整个仓库 27 项通过，1 条现有
Starlette/httpx 弃用警告。全套测试中的本机 HTTP 服务需要允许监听回环端口。

## 不能从回归测试推导的结论

- 模拟响应中的否定或时间表述正确，不证明真实 LLM 也会输出正确表述。
- 消息中的指令以数据发送，不证明模型能够抵御提示注入。
- 记录向量器使用的文本，不证明真实向量检索的排序得到提升。

本次配置为 `LLM_PROVIDER=none`，OpenAI/DeepSeek 密钥均未配置，因此没有进行真实
LLM 调用，也没有进行 V1/V2 检索效果对照。

真实效果验证时，应将同一批 messages、query 和 options 分别输入 V1/V2，人工核对
主体、否定、时间、数值和未知事实是否保持，再使用相同向量模型、候选记忆和阈值
比较检索排名、漏检、误召回、延迟及 token 用量。不要把模拟输出当作模型实测结果。

## 多跳与时间合成数据（2026-10-09）

`multihop_cases.jsonl` 含 15 个多跳用例（中文 9 个、英文 6 个，其中 4 个为三跳），
评测时每例加入 180 条干扰记忆。`temporal_ab_cases.jsonl` 含 15 个时间用例。
两者均为仓库内的合成数据，不是官网隐藏评测日志。

多跳回归使用 `scripts/run_multihop_regression.py`，冻结查询扩展，并让四个模式
共享同一原文库和向量。`oracle` 是用例提供的理想目标，`real` 是模型响应或其缓存；
报告保留响应来源，检索耗时不含规划。指标仅核对每跳关键词和完整链是否被召回。
任一旧有链或单跳证据在指定截断位置退步时，脚本返回非零状态。

`tests/test_multihop_retrieval.py` 覆盖 17 项服务及 SDK 数据流用例：来源片段校验、
跨 session 桥接、用户隔离、英文词界、批次内上下文、预算耗尽、可选目标向量失败回退，
以及原有排序/分数保护和默认 V3 模型请求复用。SDK 与向量测试替身不代表真实模型效果。

```powershell
.venv/Scripts/python.exe -m pytest tests/test_multihop_retrieval.py
.venv/Scripts/python.exe -m scripts.run_multihop_regression --llm oracle
```

多跳用例的相关片段位于同一写入批次开头、彼此相邻。上下文窗口带来的收益不能
外推为跨批次代词消解能力；还需用经过授权、布局更分散的数据验证。

## D1/F1 合成数据

`governance_cases.jsonl` 包含 13 个 D1、3 个 F1 场景，分别声明抽取草稿与验收标签。
金标不是由待测版本解析器生成；独立人工复核仍待完成。`fixture` 仅替代外部事实抽取
SDK，实际执行来源校验、事务存储、版本解析、摘要和检索，不能代表真实模型抽取质量。

四组对照为原有检索、仅版本解析、仅摘要、版本解析加摘要；固定来源、向量、空扩展、
参照时间、`top_k=1` 与内容预算，交替运行五轮。指标是当前值与必要原文覆盖的检索
代理指标，尚未执行最终回答模型，也不能换算为官网 D1/F1 得分。

```powershell
.venv/Scripts/python.exe -m scripts.run_governance_eval --extraction fixture --repeats 5 --report data/governance_eval.json
```

`--extraction real` 使用配置的模型。`--audit-output` 可选输出本合成数据的已校验事实
片段供核验；常规服务监控仅记录数量、状态、失败类型与耗时。所有对照数据库均为
临时数据库，不读取隐藏评测日志，不激活正式环境。
