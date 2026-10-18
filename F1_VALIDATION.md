# F1 查询期优化：实现与验证

本文件记录 2026-10-15 的历史轮次，基线为 `fe846f7`。2026-10-18 的证据单元与任务覆盖
改动以 `0ec1465` 双 active 为基线，见 [最新实施及回归报告](F1_EVIDENCE_VALIDATION.md)。

更新日期：2026-10-15。

## 1. 结论与验证范围

【已验证】本轮实现了独立 F1 综合计划、原文补证、版本感知的完整片段组织，以及输出后覆盖检查。
全量测试 **351 项通过**。两套冻结对照共 28 个问题，相对本轮修改前的 `fe846f7`，
3 个问题的必要引文覆盖提高，25 个保持一致，没有该指标的新增退步。

【存疑】真实事实抽取、最终回答正确率和官网 F1 得分改善尚未验证。本轮全部评测使用
确定性替身模型，未调用真实模型；官网没有提供逐题 F1 标签、返回正文或评分细则。
因此本报告不将任何合成题认定为官网实际 F1 题，也不把引文覆盖当作语义正确率。

代码默认保持 `GOVERNANCE_MODE=off`、`F1_MODE=legacy`；本轮没有修改服务器配置或生产数据库。

## 2. 改动及其目的

| 模块 | 实现 | 要解决的问题 |
| --- | --- | --- |
| `app/governance/summary_planning.py` | 根据原问题生成综合范围、时间用途、需求和展示顺序；隔离明确标识的材料、引用及选项 | 综合请求不一定含“总结”一词；材料中的日期或指令可能误导检索范围 |
| `app/governance/summary_retrieval.py` | 在同一用户快照中收集事实和相关原文，保留来源范围、归因、日期及版本标签 | 低频决定、规则例外、未抽取事实的原文可能不在基础 `top_k` 中 |
| `app/governance/summary_assembly.py` | 按需求覆盖选择完整证据单元，压缩同值重复支持，并检查实际输出 | 按相似度或原文长度直接截取会漏掉阶段和条件；候选中存在不代表最终可见 |
| `app/governance/retrieval.py`、`app/config.py` | 独立 F1 开关，保留原治理执行路径及未触发查询 | 将 F1 验证与 D1 启用状态分开，便于逐步验证和回退 |
| 配对评测程序、测试和合成用例 | 冻结旧代码与模型输出，新增规划、证据打包消融及逐题核验包 | 将程序正确性、引文可见性和最终回答质量分开验证 |

处理流程：

```text
原有 V3 召回及治理结果
  → 独立判断 F1 是否适用
  → 读取同一用户快照中的事实与相关原文
  → 复用 D1 解析版本，保留条件、更正及撤回依据
  → 按需求和阶段收集完整证据
  → 在统一内容预算内组织抽取式结果
  → 统计实际输出中的覆盖与缺口
  → 由 F1_MODE 决定返回旧结果或新结果
```

### 范围与证据规则

- 支持显式总结、跨讨论综合、报告需求，以及明确的项目、属性和字面主题范围。
  未识别或不支持的范围按“不适用”处理，保留原治理结果。
- 请求中的背景材料不写入记忆，也不作为事实来源；明确标识的材料日期不自动成为查询窗口。
- `no_fact`、部分抽取、未分类来源仍可提供原文，但不因此被认定为已确认当前状态。
- 按段落和句子边界保留完整语义。主句与例外、条件和多行证据一起处理；同批次、同角色、
  相邻消息中的明确续句可以绑定。不推断跨会话的隐含续接关系。
- 原文混合旧说法和其他有效信息时，保留必要片段，标注失效内容并附上更正或撤回依据。
- 已验证的同值重复观察可以压缩重复文字，但保留全部支持来源和日期。
  不合并不同事件、不同限定条件、不同身份或真实变化。
- 同名身份不明确时分别呈现并标注歧义。没有可靠日期时显示文档位置，不创造事件时间。

### 内容预算与覆盖

`GOVERNANCE_CONTENT_TOKENS` 对新 F1 计量完整返回 `memories` JSON，包括派生 ID、分数、
日期字段和换行转义；使用固定 `cl100k_base` 编码，不能视为所有上游模型的精确计费量。
此处不包括外层回答提示词或调用方额外包装。

新结果使用可追溯的 `f1_` ID；分数继承相关基础结果或基础排序值，不固定设为 `1.0`。
输出满足 `top_k`，完整证据可以合并到一条结果中。配对评测仍按相同的实际可见 JSON
预算检查各组，避免仅靠向单条结果塞入更多文字取得优势。

每个需求分别报告：已输出、部分输出、已找到但预算未覆盖、证据不足。
这是对候选证据的计数，不是对语义完整性的证明。必要片段不能安全装入时不截断否定或条件，
也不返回只有标题的摘要；已识别的范围内没有完整可用证据时可以返回空结果。

为避免提前停止扫描遗漏早期低频证据，新 F1 不使用旧的 `GOVERNANCE_QUERY_BUDGET_MS`
作为扫描截止时间。旧治理部分继续保留原时间预算。新 F1 在大量来源上的耗时和并发负载
尚未测量；本轮没有据此宣称性能改善。

## 3. 对 V3 和 D1 的影响

【已验证】以下文件相对本轮开始的 `fe846f7` 没有修改：

- `app/service.py`、`app/storage.py`。
- `app/llm.py`、`app/prompts.py`、`app/temporal.py`、`app/multihop.py`。
- `app/governance/resolution.py`、`app/governance/summaries.py`。

V3 写入增强、查询扩展、向量生成、双通道评分、阈值和时间排序逻辑保持原实现。
D1 的 `state_at`、历史观察、后来更正与“当时已知”的区分继续由原版本解析器决定；
F1 读取其结果后组织证据，不重写事实版本。

新 F1 不新增模型或向量调用，不改公开 API，不修改持久化摘要格式，不需要迁移数据库、
重新 Add 或重新生成原文向量。查询直接读取最新来源与事实，不依赖旧摘要缓存。
上述结论限于本轮 F1 改动，不能据此省略此前治理版本升级所要求的迁移或回填。

| `GOVERNANCE_MODE` | `F1_MODE` | 返回行为 |
| --- | --- | --- |
| `off` | 任意 | 基础路径 |
| `shadow` | 任意 | 基础路径；内部可计算治理和 F1 对照 |
| `active` | `legacy` | 本轮修改前的治理路径 |
| `active` | `shadow` | 返回本轮修改前的治理结果，内部计算新 F1；D1 继续生效 |
| `active` | `active` | 适用的综合任务使用新 F1；其余查询保持原治理路径 |

F1 影子验证配置：

```dotenv
GOVERNANCE_MODE=active
F1_MODE=shadow
```

明确启用新 F1 时将 `F1_MODE` 改为 `active`。回退新 F1 时改回 `legacy`，可继续保留 D1。
没有自动切换配置。

## 4. 自动化回归

全量测试由本轮开始的 317 项增加到 **351 项，全部通过**。新增 33 项 F1 行为测试和
1 项评测汇总测试。测试结果存于本地 `data/f1_tests_final.xml`；存在一项既有
`python_multipart` 弃用提示，无测试失败。

新增测试覆盖：

- 综合范围、材料日期隔离、隐式综合、普通问题和否定总结请求。
- 范围内 `no_fact` 补证、混合项目、未知属性主题与无关个人信息隔离。
- 长来源、早期低频决定、规则例外、多行证据、同批续句、重复支持和实际 JSON 预算。
- 更正依据、有效时间与消息时间、乱序入库、未知时间、同名身份及“当时已知”。
- 输出覆盖的部分、遗漏、缺失区分；预算不足时不返回伪摘要或越界来源。
- 普通及 D1 问题的 ID、正文、分数、顺序与调用一致性，两个开关的组合行为。

另在 `F1_MODE=active` 下运行既有治理测试中的非摘要部分，**51 项通过、7 项未选择**。
该检查针对 D1 与既有治理控制题；不能推导任意线上输入均无退步。

## 5. 冻结配对对照

### 方法

固定三份代码参照：V3 `9f096cf`、原 V4 `e90b153`、本轮开始版本 `fe846f7`（`legacy`）。
这些提交用于本地对照，不据此推断任何一次官网实际部署的版本。

同一套数据内冻结增强、抽取替身、查询扩展、向量、时间参照和可见内容预算。
程序提供相同回答模型及提示词的真实调用入口，但本轮 `selftest` 没有执行最终回答。
对照时旧治理本地处理预算统一设为 1,000 ms，以减少本机时间截断造成的混杂；
这一设置仅用于评测，不修改服务配置。

新增用例采用独立声明的合成原文、预期结论、必要来源及推导理由，不读取官网隐藏答案。
这组用例是本轮开发样本。既有数据保留原 `development/holdout` 标签，但已用于回归，
不能再将其宣称为本轮未见的独立泛化测试集。

下列数值是**必要引文覆盖**：某个预设必要事实的支持片段全部出现在预算内可见记忆中，
才计为该项可见。该指标检查字面完整性，不自动判定版本适用性、限定语义或答案正确性。

### 新增 7 个问题：统一可见预算 1,500 token

`f1_plan` 仅使用新综合计划，保留旧证据组织；`f1_pack` 保留旧触发与范围语义，
使用新证据收集和打包；`full` 同时使用两者。

| 场景/问题 | V3 | 原 V4 | 修改前 legacy | 仅计划 | 仅收集/打包 | 完整 F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scoped_rule/q1：项目状态与规则例外 | 1/2 | 1/2 | 1/2 | 1/2 | 1 | 1 |
| indirect_stance/q1：间接表达的跨讨论综合 | 1/3 | 0 | 1/3 | 1/3 | 1/3 | 1 |
| report_context/q1：带材料的报告请求 | 1 | 0 | 1 | 1 | 1 | 1 |
| long_source/q1：长原文中的决定、原因与规则 | 0 | 0 | 0 | 0 | 1 | 1 |
| d1_control/q1：当前值 | 0 | 1 | 1 | 1 | 1 | 1 |
| d1_control/q2：历史值 | 0 | 1 | 1 | 1 | 1 | 1 |
| ordinary_control/q1：普通偏好 | 1 | 1 | 1 | 1 | 1 | 1 |

【已验证】相对 `legacy`，3 项提高、4 项一致。新 F1 触发 4 题；其余 3 题的完整返回与
调用记录和 `legacy` 一致。长来源场景的旧结果在统一可见预算下不能完整容纳；新方案
选择其中完整的必要证据单元。该场景的替身抽取可能标记部分处理，用于检验原文补证，
不代表真实模型已成功抽取长来源。

### 既有 21 个问题：统一可见预算 8,192 token

| 场景/问题 | V3 | 修改前 legacy | 完整 F1 | 相对 legacy |
| --- | ---: | ---: | ---: | --- |
| address/current | 1 | 1 | 1 | 一致 |
| address/past | 1 | 1 | 1 | 一致 |
| correction/historical_truth | 1 | 1 | 1 | 一致 |
| correction/known_then | 1 | 1 | 1 | 一致 |
| correction/real_change | 1 | 1 | 1 | 一致 |
| withdrawal/unknown | 1 | 0 | 0 | 一致 |
| relation/leader | 0 | 1 | 1 | 一致 |
| relation/leader_en | 0 | 1 | 1 | 一致 |
| missing_edge/missing | 2/3 | 1/3 | 1/3 | 一致 |
| project_history/overview | 2/7 | 1 | 1 | 一致 |
| project_history/small_overview | 1/5 | 1 | 1 | 一致 |
| field_trial/stages | 3/4 | 1 | 1 | 一致 |
| preferences/both | 0 | 0 | 0 | 一致 |
| preferences/fact | 1 | 1 | 1 | 一致 |
| policy/exception | 1 | 1 | 1 | 一致 |
| policy/normal | 1 | 1 | 1 | 一致 |
| experiments/constrained | 1 | 1 | 1 | 一致 |
| experiments/unobserved | 1 | 1 | 1 | 一致 |
| time_sequence/interval | 1 | 1 | 1 | 一致 |
| time_sequence/order | 1 | 1 | 1 | 一致 |
| identity/ambiguous | 1 | 1 | 1 | 一致 |

【已验证】21 题均未出现本轮新增的引文覆盖退步；其中 3 题触发新 F1，其余 18 题的
完整返回与调用记录保持一致。三个基线未满覆盖的问题仍未满覆盖，本轮没有将其计为成功。
V3 与治理方案在撤回、缺边等题上存在原有差异，故不能概括为“全链路全面优于 V3”。

### 兼容检查

两套数据的下列检查全部通过：

| 检查项 | 比较内容 |
| --- | --- |
| `v3_vs_off` | 完整返回 ID、内容、分数、时间、顺序及调用记录 |
| `off_vs_shadow` | 同上，验证治理影子模式不改变返回 |
| `legacy_vs_f1_shadow` | 同上，验证仅对照新 F1 时保留现有 D1 和治理结果 |
| `capture_vs_replay_full` | 冻结记录与重放的完整返回 |
| `legacy_vs_full_untouched` | 未触发新 F1 的问题，完整返回及调用记录一致 |

两套报告均为 `answer_status=not_run`、`real_model_evaluation=false`、
`overall_quality_conclusion=unverified`；最终回答正确性、错误状态污染和限定语义的人工
核验结果仍为空，没有用自动关键词指标填充这些结果。

## 6. 复现与服务器核验

在仓库根目录、已安装项目依赖的 Python 环境中执行：

```bash
python -m pytest -o addopts= -q
python scripts/run_governance_e2e.py validate --dataset tests/fixtures/f1_e2e_cases.json
python scripts/run_governance_e2e.py selftest --f1-comparison --dataset tests/fixtures/f1_e2e_cases.json --visible-tokens 1500 --output data/f1_verified_new
python scripts/run_governance_e2e.py selftest --f1-comparison --visible-tokens 8192 --output data/f1_verified_existing
```

本次已完成结果在 `data/f1_verified_new/`、`data/f1_verified_existing/`：
`comparison.json` 保存逐题与兼容结果，`REVIEW.md` 和 `review.json` 用于人工核验。
这些本地运行产物不进入 Git；代码、合成用例和本报告保留复现依据。
新增数据集 SHA-256 为
`5ede3f2fad87f03285bb9c08c02c00eb60f7f5f43401a932058c6a83bfe1d56d`。

服务器真实链路运行命令如下。该运行使用服务器现有模型配置，会产生模型调用；输出目录
应与自检目录分开。它不修改生产数据库或服务配置。

```bash
python scripts/run_governance_e2e.py run --f1-comparison --dataset tests/fixtures/f1_e2e_cases.json --visible-tokens 1500 --expect-embedding-model text-embedding-v4 --output data/f1_server
python scripts/run_governance_e2e.py run --f1-comparison --visible-tokens 8192 --expect-embedding-model text-embedding-v4 --output data/f1_controls_server
```

完成后检查各目录的 `REVIEW.md`，逐题核对来源、版本、必要证据、限定条件和实际回答，
填写对应 `review.json`，再汇总：

```bash
python scripts/run_governance_e2e.py review --output data/f1_server
python scripts/run_governance_e2e.py review --output data/f1_controls_server
```

访问失败或尚未核验的项目应保持未验证。特别需要检查：真实抽取是否漏掉规则、否定与
更正；F1 范围是否误识别；输出预算是否遗漏必要证据；回答模型是否准确使用证据。
存在未解释退步时继续使用 F1 影子模式，不依据上述离线引文结果自动启用。

## 7. 尚未覆盖的边界

- 本地范围与需求识别依赖规则和目录，尚不支持任意语义表达；不适用问题继续走旧路径。
- 原始记忆缺失、角色归因错误或跨会话指代不明确时，F1 不能自动补造缺失事实。
- 未结构化原文会带不确定标签，其语义正确性仍需下游判断或人工核验。
- 已识别范围内证据不足或预算过小时可能返回空结果；调用方需保留“信息不足”的回答能力。
- 全量相关快照扫描的规模成本未做线上测量。本轮没有增加模型调用，但这不等于总耗时不变。
- 合成回归只能证明所列行为；真实最终回答和官网评分仍需独立验证。
