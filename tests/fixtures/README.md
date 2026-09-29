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
