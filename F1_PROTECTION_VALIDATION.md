# F1 事实与归因保护、受约束合并：实现与验证

更新日期：2026-10-18。主基线：`b005cefccca104b8292b23f4bcd5b566ecdace7f`，
`GOVERNANCE_MODE=active`、`F1_MODE=active`。本轮不以 V3 或 `0ec1465` 替代基线。

## 1. 结论与证据边界

【已验证】修改前工作区 372 项测试通过，完成后 390 项通过。修改前产品代码是 b005cef，
372 项中包含上一轮尚未提交的 11 项服务器只读核验测试；不是把 b005cef 的历史测试数改为 372。

【已验证】固定外部模型替身、来源、向量、扩展、参照时间、持久化时钟和内容预算后：

- 既有 33 题的返回 ID、正文、分数、顺序和查询模型调用与 b005cef 完全一致。
- 新增 8 道独立开发题，必要完整引文覆盖 3 题提高、5 题持平；没有丢失任何基线已覆盖的金标证据原子。
- 全部 41 题的原治理 `GovernanceResult` 与查询调用记录一致；24 道未进入 F1 的题返回完全一致。
- off、总治理 shadow、F1 shadow 的冻结路径逐字段对照全部通过。
- 五份已有日志共 240 次正式请求完成原文回放；20 次进入 F1，输出及原文范围均与 b005cef 一致。
  多轮包含重复问题，不能把 240 当作独立评测样本，也不能把触发次数当作官网 F1 覆盖率。

【存疑】本轮没有调用真实抽取或回答模型。日志没有实际召回、向量、抽取事实、查询扩展、
最终回答及官方逐题标签。以上结果不能证明官网 F1 的 16.67 分已经保住，或 A1/A2 已恢复。
尤其日志探针的 F1 返回没有变化，不能将本轮修复宣称为已解决官网 A1/A2 下降。
服务器实际索引核验继续使用 [SMOKE_INDEX_AUDIT.md](SMOKE_INDEX_AUDIT.md)。

## 2. 实现

### 2.1 任务保护需求

`summary_planning.py` 新增 `direct_requirements` 与 `attribution_requirements`。
仅在原 F1 路由已经适用时，从请求区识别明确的属性与归因要求；没有扩大 F1 触发范围。
已支持属性注册表中的名称、关系名称、编号/金额，以及提出、转述、执行、批准等明确归因表达。
每项保留原请求触发文本；材料、选项及引用不作为事实，互动 likes 不作为个人偏好。
未支持的表达不猜测实体或行动关系，继续使用既有 F1 路径。

### 2.2 版本与来源保护

`summary_retrieval.py` 只额外暴露已经由原 `StateResolver` 算出的当前、历史及冲突事实 ID；
不更改版本判定。`summary_guard.py` 在已有范围内选择完整证据单元：

- 已有结构化属性依据原 D1 选中版本；历史任务可保留真实历史。更正、撤回、同名身份及冲突标签沿用原结果。
- 无结构化事实的完整来源可作为候选；保护成功不等于将候选认证为真实事实。
- 同一单元的否定、条件、计划性质、更正依赖共同保留。
- 受保护片段旁显示消息角色，并声明归因以原文为准；不把 `user/assistant` 自动变成人物或行动者。
- 原治理事实引用与基础召回顺序用于范围内证据的优先选择；不解析 `gov_` 渲染文本来重建事实。

共享来源目录仍只记录一次完整来源 ID。完整段落、代码、表格、续句及限定条件保持不拆分。

### 2.3 受约束合并

每个适用的混合任务先产生原 b005cef 风格的 F1 包，再构建候选包：

1. 优先装入直接事实/归因的完整证据单元。
2. 为原包已有的显式综合需求保留支持证据，再执行原覆盖选择。
3. 校验受保护单元、来源哈希、原需求覆盖，以及同一序列化 JSON token 预算。
4. 检查通过才采用候选包；失败时返回原 F1 包，并记录保护失败。

该回退选择保留同一查询范围内的既有 F1 结果，不直接拼接未经范围和版本校验的基础原文。
原 F1 包为空且没有完整可用证据时仍返回空，不制造只有标题的结果，也不恢复范围外或未来来源。

`SummaryOutcome.protection` 记录 `passed`、`decision`、保护单元数、实际保留数、
`missing_requirements`、`omitted_units`、`coverage_regressions` 和无效来源数。
`passed` 仅表示所识别证据保护约束通过，不是事实真实性或最终回答正确性的判定。
运行日志只新增数量与状态；不新增原文日志。

覆盖检查保证原来已有证据的需求不会变成完全缺失，不能证明其所有隐含语义均已完整。
因此离线报告还逐个比较人工金标原子，分别记录 `lost_atoms` 和 `gained_atoms`，不允许总比例掩盖得失相抵。

## 3. 保护与综合覆盖冲突时的行为

新增数据特意保留一个 850 token 下无法同时装入两组完整材料的例子：

| 方法 | 编号完整证据 | 决定、原因、例外完整证据 |
| --- | --- | --- |
| b005cef 双 active | 缺失 | 保留 |
| 仅事实/归因优先 | 保留 | 丢失 |
| 完整受约束合并 | 缺失 | 保留 |

完整方案拒绝该次候选合并，保留原 F1 包，并记录 `passed=false`、
`decision=preserved_f1_baseline` 和缺失的 `direct:identifier`。
这道题的 A1 需求仍未解决；不能将回退算作全面修复。两组方法都可能有 1/2 的总引文覆盖，
但丢失的是不同证据，因此必须查看原子明细。

## 4. 兼容性与测试范围

V3 的 `service.py`、增强与查询扩展、向量、双通道、阈值、时间排序和基础多跳未修改。
D1 的 `planning.py`、`resolution.py`、抽取和存储未修改；只在原治理计算完成后将其只读证据
引用传入 F1。公开 API、默认配置、查询模型调用、数据库结构和持久化摘要机制不变。

新增回归包括：编号完整限定、人物/转述/否定、中文问题、当前与历史版本、混合消息更正依赖、
同名主体、未知及超大证据、无效哈希、外用户来源、未来知识截止、材料隔离、范围隔离、
共享目录、预算冲突回退，以及普通事实和 D1 返回一致性。

已有 33 题分别采用原预算：F1 7 题与治理 21 题为 8192 token，证据单元 5 题为 1400 token。
新增 8 题两侧均为 850 token。预算按完整可见 `memories` JSON 计量；没有通过扩大单条内容取得优势。
全量测试有一个已有 `python_multipart` 弃用提示，无失败或跳过。

## 5. 复现

本地替身对照，无外部模型调用：

```bash
python -m pytest -q
python scripts/run_governance_e2e.py selftest --f1-guard-comparison --dataset tests/fixtures/f1_e2e_cases.json --visible-tokens 8192 --output data/f1_guard_verified_f1
python scripts/run_governance_e2e.py selftest --f1-guard-comparison --dataset tests/fixtures/governance_e2e_cases.json --visible-tokens 8192 --output data/f1_guard_verified_governance
python scripts/run_governance_e2e.py selftest --f1-guard-comparison --dataset tests/fixtures/f1_evidence_units_cases.json --visible-tokens 1400 --output data/f1_guard_verified_units
python scripts/run_governance_e2e.py selftest --f1-guard-comparison --dataset tests/fixtures/f1_protection_cases.json --visible-tokens 850 --output data/f1_guard_verified_new_final
python scripts/compare_f1_log_evidence.py --baseline b005cef --logs logs/input.jsonl.2026-10-06 logs/input.jsonl logs/input.jsonl-2026-10-14.jsonl logs/input.jsonl-2026-10-16.jsonl logs/input.jsonl-2026-10-17.jsonl --visible-tokens 8192 --output data/f1_guard_verified_logs
```

输出目录中的 `manifest.json` 固定代码、数据、模型指纹及预算；改变输入时须换新目录。
`comparison.json` 保存逐题/逐原子差异，`runs/` 保存来源、抽取和治理中间结果，`REVIEW.md`
提供人工语义核验入口。报告不将替身当成真实模型；`final_answer_accuracy` 保持未知。

服务器可运行真实模型冻结对照，再核验最终答案：

```bash
python scripts/run_governance_e2e.py run --f1-guard-comparison --dataset tests/fixtures/f1_protection_cases.json --visible-tokens 850 --output data/f1_guard_server
python scripts/run_governance_e2e.py review --output data/f1_guard_server
```

`run` 会使用服务器配置的模型接口并产生调用；不连接生产服务或修改生产数据库。
先检查 `REVIEW.md` 并填写 `review.json`，再执行 `review` 汇总；未评项不能填成通过。
新增开发用例不代表保留集或官网题目，也不能由此声称泛化提升。

没有新增开关。默认仍 `GOVERNANCE_MODE=off`、`F1_MODE=legacy`；已配置双 active 的部署
更新代码后会采用本轮保护。`F1_MODE=shadow/legacy` 返回旧治理路径，不等于 b005cef 双 active；
精确对照使用 `--f1-guard-comparison`。本轮没有修改服务器配置或生产数据库；代码提交记录以 Git 历史为准。

## 6. 逐题对照

下表由固定运行结果整理；覆盖数值仅表示人工预设必要完整引文的保存比例。

| 数据集 / 场景 / 问题 | b005cef | 本轮 | 返回完全一致 | D1 结果一致 |
| --- | ---: | ---: | --- | --- |
| f1 / scoped_rule / q1 | 100.00% | 100.00% | 是 | 是 |
| f1 / indirect_stance / q1 | 100.00% | 100.00% | 是 | 是 |
| f1 / report_context / q1 | 100.00% | 100.00% | 是 | 是 |
| f1 / long_source / q1 | 100.00% | 100.00% | 是 | 是 |
| f1 / d1_control / q1 | 100.00% | 100.00% | 是 | 是 |
| f1 / d1_control / q2 | 100.00% | 100.00% | 是 | 是 |
| f1 / ordinary_control / q1 | 100.00% | 100.00% | 是 | 是 |
| governance / address / current | 100.00% | 100.00% | 是 | 是 |
| governance / address / past | 100.00% | 100.00% | 是 | 是 |
| governance / correction / historical_truth | 100.00% | 100.00% | 是 | 是 |
| governance / correction / known_then | 100.00% | 100.00% | 是 | 是 |
| governance / correction / real_change | 100.00% | 100.00% | 是 | 是 |
| governance / withdrawal / unknown | 0.00% | 0.00% | 是 | 是 |
| governance / relation / leader | 100.00% | 100.00% | 是 | 是 |
| governance / relation / leader_en | 100.00% | 100.00% | 是 | 是 |
| governance / missing_edge / missing | 33.33% | 33.33% | 是 | 是 |
| governance / project_history / overview | 100.00% | 100.00% | 是 | 是 |
| governance / project_history / small_overview | 100.00% | 100.00% | 是 | 是 |
| governance / field_trial / stages | 100.00% | 100.00% | 是 | 是 |
| governance / preferences / both | 0.00% | 0.00% | 是 | 是 |
| governance / preferences / fact | 100.00% | 100.00% | 是 | 是 |
| governance / policy / exception | 100.00% | 100.00% | 是 | 是 |
| governance / policy / normal | 100.00% | 100.00% | 是 | 是 |
| governance / experiments / constrained | 100.00% | 100.00% | 是 | 是 |
| governance / experiments / unobserved | 100.00% | 100.00% | 是 | 是 |
| governance / time_sequence / interval | 100.00% | 100.00% | 是 | 是 |
| governance / time_sequence / order | 100.00% | 100.00% | 是 | 是 |
| governance / identity / ambiguous | 100.00% | 100.00% | 是 | 是 |
| units / complete_structures / q1 | 100.00% | 100.00% | 是 | 是 |
| units / compact_provenance / q1 | 100.00% | 100.00% | 是 | 是 |
| units / material_task_scope / q1 | 100.00% | 100.00% | 是 | 是 |
| units / task_evidence_families / q1 | 100.00% | 100.00% | 是 | 是 |
| units / scoped_stance / q1 | 100.00% | 100.00% | 是 | 是 |
| new_final / direct_identifier / mixed | 0.00% | 100.00% | 否 | 是 |
| new_final / direct_identifier / atomic | 100.00% | 100.00% | 是 | 是 |
| new_final / actor_reporter / mixed | 0.00% | 100.00% | 否 | 是 |
| new_final / actor_reporter / atomic | 100.00% | 100.00% | 是 | 是 |
| new_final / protected_and_coverage / mixed | 33.33% | 100.00% | 否 | 是 |
| new_final / corrected_mixed_source / mixed | 100.00% | 100.00% | 否 | 是 |
| new_final / corrected_mixed_source / current | 100.00% | 100.00% | 是 | 是 |
| new_final / infeasible_joint_budget / mixed | 50.00% | 50.00% | 是 | 是 |

基线已有的 0% 或部分覆盖保持原值，没有以本轮任务为由隐藏。`infeasible_joint_budget` 的 50% 表示保留综合证据、仍缺直接编号上下文；该题不算保护成功。
