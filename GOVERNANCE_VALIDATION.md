# D1/F1 实施验证报告

日期：2026-10-09。实现与维护说明见 [GOVERNANCE.md](GOVERNANCE.md)。

## 1. 已完成的修改

【已验证】新增独立事实抽取、受控槽位版本解析、来源校验与处理状态、抽取式分层摘要、
摘要依赖失效、版本感知多跳、写入租约、一致快照、索引回填/修复/代次管理。
公共请求与响应字段保持兼容。默认 off，验证写入显式 shadow，未启用正式 active。

补充保护包括：真实变化与纠正分开；定向纠正保留更早真实同值历史；别名合并要求明确
身份场景且唯一匹配；第三方引语不能归为用户第一人称；计划不能通过错误规则标签
绕过校验；旧显示前缀日期不能充当有效时间；模型抽出的已替代实体不能重新桥接。

## 2. 自动化与兼容回归

| 检查 | 结果 | 依据 |
|---|---|---|
| 冻结基线 | 241 项，0 失败/错误/跳过 | data/governance_baseline_tests.xml |
| 最终全套 | 280 项，0 失败/错误/跳过 | data/governance_tests_final.xml |
| 既有多跳 | 15/15 完整链，逐例排名不退步 | data/governance_multihop_regression.json |
| 多跳影子对照 | 四个检索模式逐例跳排名、完整链、候选数一致 | data/governance_multihop_shadow.json |
| 既有时间及影子对照 | 15 例逐例目标/干扰排名与原报告一致 | data/governance_temporal_regression.json、governance_temporal_shadow.json |
| 管理命令 | 9 项生命周期检查通过，包括拒绝不完整代次激活 | data/governance_cli_check.json |

服务集成测试核对影子返回的 ID、内容、分数和顺序与 off 一致，查询 SDK 调用仅执行原
有扩展；原有多跳/时间回归另核对逐例检索指标。源码编译与差异格式检查通过。上述
测试没有读取隐藏评测日志，没有迁移或写入正式数据库。

## 3. 四组固定数据对照

【已验证】16 个公开合成用例：13 个 D1、3 个 F1。抽取草稿与期望值独立声明于用例，
不是由解析器生成；独立人工复核待完成。使用本机同一 FastEmbed 模型、来源和向量，
固定空查询扩展、参照时间、top_k=1、8192 token 内容预算，五轮交替测量。

下列指标是检索代理指标；D1 核对所需当前值、应拒绝的过期值和冲突/未知标识，F1
核对必要完整原文是否出现。没有执行最终回答模型，没有测量官网得分或语义事实准确率。

| 指标 | 原方案 G0 | 仅版本 G1 | 仅摘要 G2 | 联合 G3 |
|---|---:|---:|---:|---:|
| D1 当前状态核验通过 | 2/13 | 13/13 | 2/13 | 13/13 |
| F1 必要原文覆盖率 | 20% | 40% | 100% | 100% |
| 当前分区出现禁用旧值的用例数 | 7 | 0 | 9 | 0 |
| 返回预算全部通过 | 是 | 是 | 是 | 是 |

联合方案在上述数据中没有逐例退步，五轮质量输出稳定；完整来源校验、覆盖与合成
改善门槛均通过。仅摘要能增加覆盖，但会把尚未治理的旧值带入当前分区；联合方案
通过版本解析生成当前与历史的不同摘要分区。

压缩效果需要单独解释。重复提及场景原文 619、返回 184 工程 token，输入/输出为
3.36；三个 F1 场景合计输入/输出为 0.9913，来源与限定标签使短历史输出扩张。
本实现没有证明所有历史都能压缩；低覆盖原检索的高压缩比也不能代表完整摘要。

## 4. 耗时与写入成本

【已验证】最新五轮对照，普通与治理查询各模式各 80 次测量，包含原有本地查询向量
推理，不含外部查询模型耗时（扩展固定为空），排除模型冷启动。

| P95 | 原方案 G0 | 联合 G3 | 增量 | 门槛 |
|---|---:|---:|---:|---:|
| 普通查询 | 36.02 ms | 35.48 ms | -0.54 ms | ≤10 ms |
| 治理查询 | 35.19 ms | 48.14 ms | 12.96 ms | ≤50 ms |

两项门槛通过。普通查询的小幅负增量属于本次测量差异，不作为提速结论。首轮曾有
超限，修复为普通查询按需求加载快照、重复同值支持关系保留最近观察连接；验收阈值
保持不变，所有事实和依赖保留。

合成影子写入共 136 条来源、40 次抽取接口调度，来源不完整比例为 0/136。
这是 SDK 替身测量，不能代表真实抽取成本。每例写入总耗时和抽取耗时记录在原始
报告 writes 字段。查询无新增模型调用或事实向量生成；维护命令不重放 Add 或生成向量。

## 5. 真实抽取与剩余门槛

【已验证】两项 D1、一项 F1 共 9 条虚构消息的真实模型小样本，9 次请求均返回
`RateLimitError`，不完整比例 9/9。原文成功写入、来源状态 failed，保留可修复索引。
没有得到可用于语义核验的事实抽取结果。

真实失败报告中的必要原文覆盖来自待确认原文回退，不能解读为真实摘要改善；空事实
上的来源校验也不能解读为真实抽取成功。本次只验证了限流降级路径。

【存疑】真实抽取的语义准确性、独立人工金标复核、最终回答质量及官网 D1/F1 改善
尚未验证。因此总体启用条件尚未满足，保持 off/影子验证，不自动开启 active。
恢复模型配额后，使用同一合成数据完成抽取核验，再补充授权数据和最终回答对照。

## 6. 复现与原始证据

```powershell
.venv/Scripts/python.exe -m pytest -q --junitxml=data/governance_tests_final.xml
.venv/Scripts/python.exe -m scripts.run_governance_eval --extraction fixture --repeats 5 --report data/governance_fixture_eval_final_verified.json
.venv/Scripts/python.exe -m scripts.run_temporal_ab --llm oracle --governance-mode shadow --report data/governance_temporal_shadow.json
```

多跳冻结模型响应的命令：

```powershell
.venv/Scripts/python.exe -m scripts.run_multihop_regression --llm real --governance-mode shadow --reuse-legacy-cache data/multihop_default_real_cache.json --cache data/governance_multihop_shadow_cache.json --report data/governance_multihop_shadow.json
```

真实小样本复现需要可用模型配额；数据由本合成集中的 d1_implicit、d1_correction、
f1_global 三例组成，未使用官网日志。

```powershell
.venv/Scripts/python.exe -m scripts.run_governance_eval --extraction real --dataset data/governance_real_probe_cases.jsonl --repeats 5 --report data/governance_real_probe.json --audit-output data/governance_real_audit.json
```

原始四组数据位于本地 `data/governance_fixture_eval_final_verified.json`，剩余
门槛与实现文件 SHA256 位于 `data/governance_acceptance.json`。管理命令、
迁移前备份、修复与回退步骤见实施说明。数据报告保存在本机忽略目录，未自动提交。
