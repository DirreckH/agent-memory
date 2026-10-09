# D1/F1 治理索引实施说明

## 1. 状态与适用边界

【已验证】代码已实现事实版本与抽取式摘要，公共 API 字段保持不变。代码与示例配置
均缺省 `off`；本次验证显式使用 `shadow`。正式数据库与实际 `.env` 未修改，未启用
`active`，未执行破坏性数据库降级。

【推测】版本感知的检索与摘要能够改善官网 D1/F1；这一点仍需真实抽取、独立人工核验
与最终回答评测。合成检索指标不能换算为官网得分。

| 模式 | 写入 | 查询 |
|---|---|---|
| off | 原有增强和向量，加持久化提交顺序 | 原有返回结果 |
| shadow | 原有流程加独立事实抽取与摘要索引 | 内部治理对照，返回原有结果 |
| active | 同 shadow；可选严格抽取失败策略 | 当前、历史、变化、综合需求使用治理检索 |

普通查询复用原路径；治理查询共享一次原有查询扩展与向量生成。没有新增查询模型
调用、事实向量或摘要生成模型。抽取式摘要直接组织有来源的完整片段，不生成新原因。

## 2. 代码与数据

| 文件 | 职责 |
|---|---|
| app/governance/models.py | 事实、证据、处理状态、快照、版本解析和摘要模型 |
| app/governance/extraction.py | 独立模型调用、分块、字符范围/哈希及语义限定校验 |
| app/governance/registry.py | 受控属性别名、基数、动态性与默认场景 |
| app/governance/resolution.py | 提交时版本关系、当前与历史选择、冲突和未知状态 |
| app/governance/summaries.py | 会话/阶段、主题、全局摘要及来源依赖 |
| app/governance/retrieval.py | 本地需求规划、范围覆盖、版本感知多跳、片段预算 |
| app/governance/persistence.py | 派生表、复合外键、失效、修复和发布 |
| app/storage.py | 迁移备份、带令牌租约、原子提交、一致快照与索引代次 |
| app/service.py | 原流程接入、影子对照、摘要更新和维护接口 |
| scripts/manage_governance.py | 备份、迁移、状态、回填、修复、重建、代次激活 |

保留原 `memories` 与向量；新消息另存确切原文 `raw_content`。旧库原文列不存在可靠
原文时，使用已有存储内容原样回填，不猜测剥离显示前缀或把增强文本当来源。已知
显示前缀的日期不能用作事实有效日期或纠正目标日期。旧记录
没有可靠提交顺序与持久化时间时保持未知。`created_at` 不推断事实有效时间。

槽位为 `(user_id, subject_id, predicate_key, scope_key)`。派生关系同时约束用户和代次；
来源处理状态记录完整范围。`ready/no_fact` 必须覆盖完整消息；缺项、截断、非法字段、
限流与超预算保留 `partial/failed/pending`。监控不新增原文日志。

## 3. 版本与摘要规则

- 单值动态属性按明确更新或可信观察次序解析，不编造改变日期。未知属性不自动覆盖。
- 更正与真实变化分别保存；更正的错误旧说法不进入真实历史。重复同值的支持关系
  连接最近观察，所有事实、来源和摘要依赖仍保留。
- 定向纠正可携带原文支持的 `target_date`；多次同值历史且无法确定目标时保留冲突。
- 多值补充、计划、假设、建议和不确定内容不覆盖已确认现实状态。否定保留证据。
- 同名跨会话身份无依据时保持分离；有明确别名与一致身份场景、且匹配唯一时合并。
- 迟到历史按来源/有效时间处理。撤回、删除或重索引后，不恢复已失效旧值为当前状态。
- 摘要固定当前、历史、事件/决定、计划、冲突/纠正五个分区。未知时间明确标注；
  不能归类的来源保留目录，抽取不完整的范围以待确认原文补充。
- 新写入、更正、撤回、修复或删除保守地使用户摘要失效。重建读取最新事实与原文，
  发布校验代次与预期修订号；并发写入后旧摘要不能发布。
- 当前状态多跳只用适用事实的实体桥接，候选和邻接再次筛选。混合消息只投影有效
  片段，不删掉整条消息中其他有效属性。`top_k=1` 也能合并必要概览和来源。

本地规则和来源校验无法证明模型事实分类正确；时间解析首版仅接受有原文 ISO 日期
依据的有效日期。跨语言实体别名、隐含身份、复杂时间和否定仍需要人工与真实模型核验。

## 4. 配置与预算

验证环境设置 `GOVERNANCE_MODE=shadow`，正式启用必须另行显式改为 `active`。
`GOVERNANCE_STRICT_WRITE=true` 仅在 active 下使不完整抽取阻断本次写入；默认保留
原文并标记索引不完整。摘要失败不撤销已提交原文，查询取最新事实与待确认增量。

初始工程上限为每批输入 3000、输出 4096 token，独立抽取总预算 60 秒，本地摘要与
治理查询各 50 毫秒，返回内容 8192 token。配置项见 `.env.example`。固定
`tiktoken==0.13.0` 的 `cl100k_base` 只计量工程预算，不是所有上游模型的精确计费数。
网络调用在事务外；提交和失败标记均校验本次租约，过期执行者不能覆盖重试。

## 5. 迁移、回填、修复与回退

以下为维护命令示例。部署前先停止旧写入进程，处理遗留 `processing` 请求，并备份。
初始化会在首次治理迁移前自动备份旧库；遇到处理中请求拒绝迁移。迁移可重复执行。
请将示例用户 ID 替换为实际用户；代次与修订号以命令输出为准。

```powershell
.venv/Scripts/python.exe -m scripts.manage_governance backup --database data/agent_memory.db --output data/pre_governance.db
.venv/Scripts/python.exe -m scripts.manage_governance migrate --database data/agent_memory.db
.venv/Scripts/python.exe -m scripts.manage_governance status --user-id example-user
.venv/Scripts/python.exe -m scripts.manage_governance backfill --user-id example-user
# backfill 输出 generation；重复 repair 补齐 failed/partial/pending 来源
.venv/Scripts/python.exe -m scripts.manage_governance repair --user-id example-user --generation 2
.venv/Scripts/python.exe -m scripts.manage_governance rebuild --user-id example-user --generation 2
.venv/Scripts/python.exe -m scripts.manage_governance status --user-id example-user --generation 2
```

回填读取已存来源及元数据，不重放 Add、不调用向量器、不重复消息。新代次有未覆盖
或失败来源时拒绝激活；激活还要求修订号一致。修复期间仍可使用原索引。
代次激活只改变派生索引选择，不会将 `GOVERNANCE_MODE` 切换为 active。

```powershell
# 仅在覆盖核验与启用门槛满足后，使用 status 返回的实际 revision
.venv/Scripts/python.exe -m scripts.manage_governance activate --user-id example-user --generation 2 --expected-revision 7
```

回退使用新代码的 `GOVERNANCE_MODE=off`，保留原文与派生表。不要用旧代码继续写入
已迁移数据库，不执行删表或破坏性降级。

## 6. 验证与启用条件

原基线 241 项测试全部通过；最终测试结果见 [实施验证报告](GOVERNANCE_VALIDATION.md)。
新增治理测试覆盖版本策略、来源归因、低频事件、规则例外、抽取失败与修复、租约、
跨用户与批次约束、删除失效、重建修订校验、代次激活、top_k=1 与旧实体桥接保护。

```powershell
.venv/Scripts/python.exe -m pytest -q --junitxml=data/governance_tests_final.xml
.venv/Scripts/python.exe -m scripts.run_governance_eval --extraction fixture --repeats 5 --report data/governance_fixture_eval_final_verified.json
```

`tests/fixtures/governance_cases.jsonl` 为 13 个 D1、3 个 F1 的独立声明合成标签，人工
复核待完成。对照固定来源、原向量、空扩展、时间参照和 top_k=1；四组为原方案、仅
版本、仅摘要、联合方案。指标仅核对当前值、过期值污染、必要完整原文与预算。
来源片段校验不等于语义事实正确性，最终回答质量尚未测量。

原有 15 个多跳用例复用既有真实 V3 模型响应，逐例 hop_ranks/完整链与前次报告一致，
联合治理的 off 路径保持完整链 15/15。15 个时间用例 off/soft 的汇总与前次报告一致；
需将合成结果与真实模型或官网分数区分。

两个回归脚本均支持 `--governance-mode shadow`。本次另行核验影子模式的多跳逐例排名、
完整链、候选数量，以及时间逐例排名，均与 off/既有报告一致。

真实抽取小样本为两项 D1、一项 F1，共 9 条虚构消息；9 次请求均发生 `RateLimitError`，
不完整来源为 9/9，原文写入成功。记录见 `data/governance_real_probe.json`。这只能验证
限流降级，不能得出真实抽取或 F1 改善结论；没有自动重试放宽门槛或开启 active。

启用须同时满足来源/隔离/事务/失效检查、逐例无退步、合成改善及真实抽取/语义核验；
五轮普通查询 P95 增量不超过 `max(10 ms, 基线 P95 的 20%)`，治理查询增量不超过
50 ms。具体本轮质量与成本数值见实施验证报告，正式启用仍待真实与人工核验。
