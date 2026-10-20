# A／C／F1 协同优化：实现与验证

日期：2026-10-20。

本轮从 `f930ac62b730b4cd6cd558cd412b0136d3c4437b` 建立并推送
`codex/joint-memory-evidence-optimization` 分支后修改源码。
范围是已复现的属性查询、转述归因、自然语言日期和证据打包问题。

## 实现与交互

| 模块 | 改动 | 对应能力 |
| --- | --- | --- |
| `app/governance/planning.py` | 明确属性、主体或范围的事实问句进入完整槽位解析；收紧泛化的 work／position／like；共享任务与材料分离；保留有依据的日期限制 | A1、C1，普通查询误触发控制 |
| `app/governance/attribution.py`、`extraction.py` | 检查目标值所在子句与引语的说话人；区分报告者与事实主体、项目标题与角色说话人 | A2，防止错误 self 事实污染 C3 |
| `app/governance/time_support.py`、`extraction.py` | 核验完整中英文日历日期及源消息 UTC 日锚定的相对日；保留原始时间引文；拒绝串用其他人物日期 | C1、C2，与属性历史查询联动 |
| `app/governance/summary_planning.py` | 扩展列表、比较、顺序任务；区分事件发生顺序与消息提及顺序；材料中的日期不替代实际任务约束 | F1、C2 |
| `app/governance/summary_retrieval.py` | 标题与正文、局部代词与前文、规则与例外、代码围栏和 Markdown 表格作为完整候选组；继承更正／撤回依赖 | A2、C3、F1 |
| `app/governance/summary_assembly.py` | 同一来源共享属性元数据，保留逐片段引用；实际 JSON 预算打包；序列化后检查已编码依赖，缺失则不返回该包 | A、C、F1 的共同证据完整性 |
| 新增回归与配对程序 | 独立虚构用例、固定模型草稿和向量、临时数据库、完整响应与调用轨迹比较 | 改善与退步分别记录 |

事实更新、更正、撤回与“当时已知”继续由 `StateResolver` 解释。
基础评分、查询扩展、向量生成、原时间检索实现、API、数据库结构及默认模式没有修改。
规划与打包不增加查询模型或向量调用；这不代表耗时不变。

## 验证口径

【已验证】修改前完整离线回归为 **351 passed**。修改后结果与配对结果在下表记录。

| 检查 | 冻结后的结果 | 解释 |
| --- | --- | --- |
| 完整回归 | **540 passed**，1 条既有 `python_multipart` 弃用提示 | 保留原 351 项，新增 189 项；无失败或跳过 |
| 新增联合配对，1500 tokens | 19 题：5 题必要检查改善，14 题无此指标变化，0 退步 | 共 30 个声明检查项，9 项改善、21 项不变；当前全部通过 |
| 紧预算归因，256 tokens | 1 题改善：旧包 215 tokens 丢失 Dana，保留 She；新包 216 tokens 同时保留两句 | 两组使用同一预算、同一输入、同一向量；不是增加预算获得改善 |
| 既有综合集合，1500 tokens | 7 题必要引文覆盖全部保持一致 | 不把引文覆盖当答案准确率 |
| 既有治理集合，8192 tokens | 21 题必要引文覆盖全部保持一致 | 包含历史更正、当时已知、真实变化、撤回等 |
| 模式与声明的兼容性 | 四组共 158 项比较全部通过 | 包含 144 项模式比较、9 项完整结果不变比较和5项普通路径比较 |
| 查询调用轨迹 | 四组所有配对均无查询模型／向量调用变化 | 比较输入与输出哈希，不仅比较调用数量 |
| 离线隔离 | 最终基线与当前进程均无 `.env` 读取、联网或临时目录外数据库打开尝试 | 同时检查没有导入 `app.main`；环境变量覆盖测试通过 |

5 道改善题分别是普通雇主属性查询、转述者隔离、英文完整日期、源消息相对日和事件日期排序。
新增 19 题与既有 28 题共 47 个不同问题；紧预算题是其中一题在另一预算下的再次比较。
检查项改善数不能解释为改善题数。四组报告的应用代码哈希均与最终测试源码一致。

C3 的真实变化当前／历史、更正当前／历史／当时已知共 **5 题**，完整返回及查询调用轨迹与基线一致。
普通查询的事实、话题和建议控制共 **4 题**同样完全一致；配置建议查询修正为基础返回，
并以独立 `expect_base_response` 检查验收，未将其正文变化误称为“逐字不变”。

全部验证使用确定性 SDK 和向量替身，无真实模型抽取或最终回答。
人工草稿仍经过待测事实校验，因此错误归因草稿会被拒绝，合法日期草稿可能被回归检测发现误拒绝。
新用例为开发回归，不包含官网隐藏答案或实际日志正文；既有集合已用于本轮诊断，不能作为新的未见泛化证据。

配对程序 `scripts/run_memory_joint_regression.py` 将基线 Git 源码导出到独立临时目录，
分别在隔离子进程和临时数据库中执行基线及工作区版本。固定来源、人工草稿、向量替身、
查询扩展输出、参照时间、持久化时钟、`top_k` 和调用方可见 JSON 预算。
它不读 `.env`、运行中的数据库或日志，不启动服务。

每题检查：

- 基线与当前完整方案的必要引文、禁止引文、已声明顺序与事实字段。
- 基线 `off` 与当前 `off`；当前 `off` 与 `shadow`；当前 `legacy` 与综合 `shadow`。
- 上述兼容性比较包括最终 ID、正文、分数、时间、顺序及查询模型／向量输入输出哈希。
- 显式声明的 C3／普通查询控制一旦完整响应变化即失败；应走基础路径的请求一旦触发替换返回也失败。
- 既有评测集沿用必要引文覆盖指标；该指标不验证任意语义、归因或最终答案。
- 新用例的当前必要检查失败、配对退步、兼容性失败或查询调用变化均使程序退出非零。

输出中的 `real_model_evaluation=false`，`answer_accuracy=null`，不将未执行回答记为正确。
报告同时保留基线、当前结果、逐题变化、数据集哈希和当前应用代码哈希。

## 复现

在项目虚拟环境中执行，基线提交须存在于本地 Git 对象库：

```powershell
.\.venv\Scripts\python.exe -m pytest -o addopts= -q --junitxml=data/joint_optimization/final_tests.xml

.\.venv\Scripts\python.exe scripts/run_memory_joint_regression.py --output data/joint_optimization/joint_paired

.\.venv\Scripts\python.exe scripts/run_memory_joint_regression.py --scenario pronoun_packet --budget 256 --output data/joint_optimization/packet_256

.\.venv\Scripts\python.exe scripts/run_memory_joint_regression.py --dataset tests/fixtures/f1_e2e_cases.json --budget 1500 --output data/joint_optimization/f1_paired

.\.venv\Scripts\python.exe scripts/run_memory_joint_regression.py --dataset tests/fixtures/governance_e2e_cases.json --budget 8192 --output data/joint_optimization/governance_paired
```

新增针对性测试：

- `tests/test_query_evidence_planning.py`：完整槽位、主体与建议／配置边界、任务材料隔离、日期范围和排序语义。
- `tests/test_fact_evidence_support.py`：整体／内部／嵌套转述、本人引语、无引号报告、标题、日期及更正链。
- `tests/test_evidence_packet_support.py`：多种紧预算、依赖篡改、最终 renderer 遗漏、结构保全、项目边界、日期排序与窗口。
- `tests/test_memory_joint_regression.py`：拒绝环境／数据库泄漏，验证普通查询响应改变会使配对程序失败。
- `tests/fixtures/memory_joint_cases.json`：11 个虚构场景、19 个配对查询。

运行结果保存在忽略的 `data/joint_optimization/` 下；生成报告不随源码提交。
本文件为 Markdown 验证记录，核对了源码路径、复现命令与报告数据；未进行 HTML 或 Office 页面视觉验收。

## 本轮捕获的退步与修正

- 收紧说话人归属时，项目标题一度被误识别为说话人，导致合法事件事实丢失；加入标题与角色标签的区分。
- 日期不确定校验一度将“之前写成……属于笔误”视作日期约束，过滤更正事实；改为约束紧邻日期的修饰语，并补完整更正链测试。
- “跨讨论总结事件顺序”一度被错误解释为消息顺序；将讨论背景与明确的提及顺序请求区分。
- 日期排序必须区分未知日期和已知超出窗口；引用形式的日期不能被任务材料隔离静默删除。
- 评测助手原先会间接导入应用入口并读取配置，且普通查询完整返回变化仅展示、不使评测失败；最终程序使用自有替身、配置源隔离与进程审计，并将声明的普通查询不变性纳入失败条件。前期诊断结果不替代修复后的重跑记录。

## 使用边界

【已验证】这是受控归因、日期核验及已编码依赖保全；未实现通用语义角色识别、完整共指或任意长历史的语义压缩证明。

- 仅规范化完整日历日及 yesterday／today／tomorrow（昨天／今天／明天）。年、月精度不扩成某日；复杂相对时间、引语内相对日、显式非 UTC 时区保守拒绝。
- `valid_to` 保持原排他终点。明确包含末日的 `through` 不强行映射为同日排他终点；`target_date` 仍沿用原 ISO 校验。
- 局部依赖规则不推断跨来源的人物身份或因果。保留原文的明确说法，不将原文存在等同于语义分类正确。
- 序列化支持审核检查已编码组内依赖；组间事件顺序另由独立 `ordered` 检查和服务级回归验证。
- 已识别范围但完整证据装不下时可以返回空结果；固定预算下不存在“始终完整回答”的保证。
- 不自动修改开关或修复存量索引。先前抽取的错误事实不会因更新 Python 文件自动消失；存量使用需要另行验证回填／重建及索引激活。

【存疑】真实抽取质量、平台最终回答正确率以及 A／C／F1 分数增益尚未验证。
本轮结论仅是所列离线用例中已复现问题修复，以及指定对照中是否出现退步。
