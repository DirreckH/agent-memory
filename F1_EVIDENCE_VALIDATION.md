# F1 证据单元、打包与任务覆盖：实施及回归

更新日期：2026-10-18。

## 1. 基线、结论与边界

**主基线是 `0ec14657f7aaf80b5ab873bb156d3b49b507d768`，两项开关均为 active。**
本轮没有以 V3 或 `fe846f7` 代替修改前版本。基线源码通过 Git archive 单独加载，
当前源码在另一进程执行；来源、增强输出、抽取输出、向量、查询扩展、参照时点、持久化
时钟及可见内容预算固定。测试使用确定性外部模型替身，真实回答阶段未运行。

【已验证】基线全量测试 351 项通过；完成后 361 项通过。已有两套数据共 28 题，
必要完整引文覆盖均未下降。21 道未触发 F1 的控制题，返回 ID、正文、分数、顺序和
查询调用记录完全一致。新增 5 道独立开发用例，4 道完整必要引文覆盖提高，1 道持平。

【已验证】四份已有日志共 192 次正式查询完成来源快照回放。此前进入 F1 的 8 次查询，
没有已输出原文字符范围的丢失；打包保留了更多完整内容。该回放仅测试 F1 规划与原文
打包，不能复现服务器实际检索。

【存疑】最终回答正确性、真实事实抽取效果和官网 F1 得分尚未验证。日志没有返回正文、
向量、抽取事实、逐题类别或答案。引文覆盖、触发数量和字符保留均不能代替这些指标。
新增用例为独立声明的开发样例，没有据此主张保留集或总体泛化能力。

## 2. 实现内容

| 部分 | 具体修改 | 保留的约束 |
| --- | --- | --- |
| 证据单元 | 新增 `summary_units.py`；以段落、代码围栏、完整表格、标题与正文组织原文；表格注释和续句随主句收纳 | 字符偏移及内容哈希对应原文；完整事实、规则例外、更正依赖共同竞争预算 |
| 任务规划 | 支持无冒号的任务/材料分区，隔离 transcript、artifacts、prior outputs；区分互动指标 likes 与个人偏好；识别明确的多类数据综合及有主题范围的对话轨迹 | V3 仍读取原查询；不把材料、选项或计划当作事实；歧义的新组合不接管原路径 |
| 需求绑定 | 显式列出的证据类别分别建立需求；规则与例外分列；时间轨迹补充首末有日期观察 | 每项需求保留任务中的触发文本及来源类别；未知关系不猜测 |
| 支持状态 | 新增 `summary_requirements.py`，区分直接原文支持、待核验候选、预算缺口、未找到、范围歧义 | 仅出现“原因”等关键词不能视为因果证据；原文支持不代表事实真实性或答案正确 |
| 打包 | 每个来源只写一次目录，用短编号与字符范围引用；共享同一更正片段；优先覆盖尚未支持的需求，再补候选和不同阶段 | 同值压缩保留全部支持来源；D1 逐片段版本、归因及有效日期标签仍在 |
| 预算 | 对完整 `memories` JSON 计量，包括 ID、分数、日期、转义和引用开销 | 大代码块/表格整体放弃并记录缺口，不返回半个条件、表头或语法片段；不提高预算 |
| 对照程序 | 增加 `--f1-units-comparison` 和 `compare_f1_log_evidence.py` | 固定当前双 active 基线；金标仅进入离线评分；失败不伪装为真实模型成功 |

`SummaryCoverage.status` 继续表示候选的装入情况；新增的 `support_status` 表示来源支持
状态。`source_supported` 是本地规则识别到对应陈述/条件/数据的标记，需要结合原文及
角色核验。跨片段隐含因果、复杂指代与最终综合结论不由该标记认证。

## 3. V3 和 D1 的兼容性

本轮产品代码修改局限于 F1 查询期模块。`app/service.py`、`app/llm.py`、时间排序、
向量生成、基础多跳、`planning.py`、`resolution.py`、事实抽取及存储代码均未修改。
F1 继续读取 D1 的 `knowledge_view`、`state_at`、`history_until` 与更正/撤回关系。

未修改公开 API、默认配置或服务器配置。没有新增模型调用、数据库迁移、Add 回放或
向量重建，也不依赖旧的持久化摘要缓存。

| 配置 | 本轮行为 |
| --- | --- |
| `GOVERNANCE_MODE=off` | 原路径 |
| `GOVERNANCE_MODE=shadow` | 返回原检索结果，内部对照 |
| `GOVERNANCE_MODE=active`、`F1_MODE=legacy` | 原治理路径，D1 生效 |
| `GOVERNANCE_MODE=active`、`F1_MODE=shadow` | 返回原治理路径，内部计算修改后的 F1 |
| 两项均为 `active` | 可识别且范围明确的综合请求采用本轮 F1；其他请求沿用现有路径 |

注意：`F1_MODE=shadow/legacy` 的返回目标一直是旧治理路径，并不是 `0ec1465` 的
双 active 实现。本轮没有改变这一语义。需要精确复现修改前双 active 时，使用冻结
版本评测程序或部署该提交；不能把切换 shadow 宣称为同一个基线。

33 题分别验证了以下逐字段一致性：`baseline_off → off`、
`baseline_shadow → shadow`、`baseline_f1_shadow → f1_shadow`。所有比较通过。
33 题的主基线与完整方案查询模型调用记录也全部一致。

## 4. 既有数据逐题比较

以下数值是人工预设必要原文的**完整逐字覆盖比例**，不是答案正确率。原有 28 题均使用
8192 token 的可见 `memories` 预算；分母保持不变。

| 数据/场景/问题 | 修改前双 active | 修改后 | 变化 |
| --- | ---: | ---: | --- |
| F1 / scoped_rule / q1 | 100% | 100% | 持平 |
| F1 / indirect_stance / q1 | 100% | 100% | 持平 |
| F1 / report_context / q1 | 100% | 100% | 持平 |
| F1 / long_source / q1 | 100% | 100% | 持平 |
| F1 / d1_control / q1 | 100% | 100% | 返回完全一致 |
| F1 / d1_control / q2 | 100% | 100% | 返回完全一致 |
| F1 / ordinary_control / q1 | 100% | 100% | 返回完全一致 |
| 治理 / address / current | 100% | 100% | 返回完全一致 |
| 治理 / address / past | 100% | 100% | 返回完全一致 |
| 治理 / correction / historical_truth | 100% | 100% | 返回完全一致 |
| 治理 / correction / known_then | 100% | 100% | 返回完全一致 |
| 治理 / correction / real_change | 100% | 100% | 返回完全一致 |
| 治理 / withdrawal / unknown | 0% | 0% | 返回完全一致 |
| 治理 / relation / leader | 100% | 100% | 返回完全一致 |
| 治理 / relation / leader_en | 100% | 100% | 返回完全一致 |
| 治理 / missing_edge / missing | 1/3 | 1/3 | 返回完全一致 |
| 治理 / project_history / overview | 100% | 100% | 持平 |
| 治理 / project_history / small_overview | 100% | 100% | 持平 |
| 治理 / field_trial / stages | 100% | 100% | 持平 |
| 治理 / preferences / both | 0% | 0% | 返回完全一致 |
| 治理 / preferences / fact | 100% | 100% | 返回完全一致 |
| 治理 / policy / exception | 100% | 100% | 返回完全一致 |
| 治理 / policy / normal | 100% | 100% | 返回完全一致 |
| 治理 / experiments / constrained | 100% | 100% | 返回完全一致 |
| 治理 / experiments / unobserved | 100% | 100% | 返回完全一致 |
| 治理 / time_sequence / interval | 100% | 100% | 返回完全一致 |
| 治理 / time_sequence / order | 100% | 100% | 返回完全一致 |
| 治理 / identity / ambiguous | 100% | 100% | 返回完全一致 |

保留的 0% 和 1/3 是既有代理指标结果，本轮没有解决或掩盖它们。尤其撤回后未知、
缺边等问题需要核验回答是否正确保留不确定性，不能仅依据引文比例判断成败。

## 5. 新增独立用例

五题固定 1400 token 可见预算，两组使用相同来源与模型替身输出。

| 场景 | 必要完整引文：修改前 → 修改后 | 实际可见 token：修改前 → 修改后 |
| --- | --- | --- |
| 完整段落、函数与带注释表格 | 0/3 → 3/3 | 1048 → 310 |
| 同一来源的 18 个不同检查项 | 13/18 → 18/18 | 1357 → 602 |
| 材料日期、互动指标与历史报告约定 | 1/1 → 1/1 | 94 → 230 |
| 两类测量及各自限定 | 0/2 → 2/2 | 1399 → 880 |
| 有主题范围的三阶段立场 | 1/3 → 3/3 | 163 → 408 |

第一行的 0/3 指旧输出没有连续保留必要的完整结构，不代表旧输出中一个字符也没有。
改善指标限定为可核验的完整证据保存。输出包含完整否定和限定，例如“未测试湿试样”
和“仅用于内部用途”；没有由本地代码推导最终答案。

新增行为回归还覆盖：超大代码块整体遗漏、来源目录只出现一次、原因标题不能认证因果、
模糊对话范围保留原路径、日志回放不得串用户/读取未来消息/接受失败写入或重复 Add。
原有更正、撤回、混合消息、来源校验、同名主体、未知日期、摘要失效和开关回归继续通过。

## 6. 已有请求日志的来源打包探针

输入为 `logs/input.jsonl.2026-10-06`、`logs/input.jsonl`、
`logs/input.jsonl-2026-10-14.jsonl`、`logs/input.jsonl-2026-10-16.jsonl`。
每份读取 48 次正式查询，排除部署自测。每次只读取当时已经成功写入的同一用户来源，
去除相同请求的幂等重复，不合并不同用户，也不读取后续写入。

边界固定为：原文来源、无事实替身、无向量或基础召回、以先前消息最大源时间为参照、
8192 token 包预算、两组使用相同的短日志来源 ID。缺失的服务器时间窗口不伪造。
这是规划/原文打包探针，不能称为端到端检索或官方得分回放。

四份日志中旧 F1 各触发 2 次，新 F1 各触发 4 次。此前触发的 8 次查询没有输出字符
范围丢失。10 月 16 日具体结果如下；单位是完整返回 JSON 的工程 token。

| 日志行 | 请求类型 | 旧候选/输出单元 | 新候选/输出单元 | 旧包 token | 新包 token |
| --- | --- | ---: | ---: | ---: | ---: |
| 76 | 明确要求结合两类数据 | 未触发 | 9/9 | 不适用 | 2192 |
| 82 | 附材料的报告任务 | 未触发 | 6/6 | 不适用 | 1198 |
| 88 | 文档概览 | 177/83 | 10/10 | 8125 | 2343 |
| 90 | 跨会话功能讨论顺序 | 234/77 | 22/22 | 8128 | 4442 |

行 88、90 中，旧输出的原文 quote token 分别为 371、566，新输出为 1671、2889。
长度不超过 10 字符的主片段由 24、50 个降为 0、0 个。减少的单元来自结构合并，
不是删掉这些字符。由短编号替代重复来源元数据，更多预算用于完整原文。

【存疑】行 82 仍报告结构/结果证据不足、变化关系待核验。这些需求可能依赖查询里的
材料和后续回答过程；本轮没有假定它们已解决。范围含糊的对话立场题保持原检索路径。
以上任一请求均不能据此认定为官网 F1 的实际计分题。

## 7. 复现与后续真实评测

本地离线检查（不读取生产数据库、不调用真实模型）：

```bash
python -m pytest -o addopts= -q
python scripts/run_governance_e2e.py selftest --f1-units-comparison --dataset tests/fixtures/f1_e2e_cases.json --visible-tokens 8192 --output data/f1_units_existing_f1
python scripts/run_governance_e2e.py selftest --f1-units-comparison --dataset tests/fixtures/governance_e2e_cases.json --visible-tokens 8192 --output data/f1_units_existing_governance
python scripts/run_governance_e2e.py selftest --f1-units-comparison --dataset tests/fixtures/f1_evidence_units_cases.json --visible-tokens 1400 --output data/f1_units_new
python scripts/compare_f1_log_evidence.py --logs logs/input.jsonl.2026-10-06 logs/input.jsonl logs/input.jsonl-2026-10-14.jsonl logs/input.jsonl-2026-10-16.jsonl --visible-tokens 8192 --output data/f1_units_logs_replay
```

源码或数据变化时使用新输出目录；程序拒绝混合不同的冻结清单。每套模型替身结果包含
`manifest.json`、逐模式 `runs/*.json`、`comparison.json`、`REVIEW.md` 和待填的 `review.json`。
清单保存基线完整提交号、当前代码哈希和输入数据哈希；报告均放在被忽略的 `data/`。

本轮最终产物位于 `data/f1_units_verified_existing_f1/`、
`data/f1_units_verified_existing_governance/`、`data/f1_units_verified_new/` 和
`data/f1_units_verified_logs/`；全量测试记录为 `data/f1_units_verified_tests.xml`。

服务器真实配对评测（会使用当前模型配置并产生调用费用）：

```bash
python scripts/run_governance_e2e.py run --f1-units-comparison --dataset tests/fixtures/f1_evidence_units_cases.json --visible-tokens 1400 --output data/f1_units_real
python scripts/run_governance_e2e.py review --output data/f1_units_real
```

先核验冻结抽取结果，再核验实际可见记忆、限定条件、错误状态污染和最终答案；填写
`review.json` 后重新执行 review。没有成功真实调用或人工语义核验的字段保持未验证。
本轮实现和评测没有自动开启 active 或修改服务器；代码提交与推送单独按用户授权执行。
