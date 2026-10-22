# G 执行证据修复与回归记录

日期：2026-10-22。

## 结论与证据等级

【已验证】按“复合任务识别 → 程序性材料补证 → 具体需求优先打包 → 最终输出保护”的顺序完成实现。
完整离线回归 **658 passed，0 failed，0 errors，0 skipped**。新增 118 项针对性检查。

【已验证】在独立编写的 11 个 G 开发问题上，相对本次修改前的冻结代码，完整来源证据检查
**5 改善、6 不变、0 退步**；相对更早代码为 **3 改善、8 不变、0 退步**。
另以 128 token 检查原文可放入、综合开销却可能使结果变空的场景，当前完整证据仍可返回。

【已验证】既有联合、综合与治理三套材料共 47 个问题的检查均无新增退步；关闭、影子、
普通查询等兼容性检查全部通过，查询模型与向量调用记录未变化。

【存疑】以上不是官网题目、最终答案正确率或线上 G 恢复的证据。真实回答验证仅发起首次
请求，上游返回 `PermissionDeniedError`，没有生成答案；后续调用停止。真实抽取质量、
真实回答效果以及官网 A/C/F1/G 得分均需后续验证。

## 冻结对照与输入

- 当前优势保护基线：`f97629b9a9e3b4712ba6a5850906a796f286c8a7`。
- 历史 G 对照基线：`f930ac62b730b4cd6cd558cd412b0136d3c4437b`。
- 最终应用源码指纹：`8e4ec071a4b3fb5dea28b6c78bcdc25c648d667a8ba94ae1454093bdd7150b3e`。
- G 材料：[g_execution_cases.json](tests/fixtures/g_execution_cases.json)，10 场景、11 问题；
  原文、抽取草稿、完整证据、来源、条件依赖及误导材料均预先声明，没有复制日志。
- 各对照固定来源、抽取结果、查询扩展、向量、参照时间、top_k 和实际调用方可见的序列化预算。
  四组 G 配对记录的 `.env` 读取、联网和隔离目录外数据库访问次数均为 0。

这些材料已经参与开发，不属于未见泛化集。完整引文与来源绑定检查可以证明已声明证据的
保留，不能证明任意语义蕴涵、人物归因或调用方回答正确。

## 实现及模块交互

| 模块 | 本次修改 | 保持的约束 |
| --- | --- | --- |
| `planning.py`、`task_requirements.py` | 分离召回与新建议子句；识别显式历史模板、规则、公式和 include/retain 需求；处理任务/材料边界 | 纯建议保持普通路径；不把引用、代码或角色发言当作新任务 |
| `summary_planning.py` | 编码完整目标词组和 mandatory 需求；保留项目或主题范围 | 无主题的程序请求不自动变成全历史扫描；指代限定词不成为匹配障碍 |
| `procedural.py`、`summary_retrieval.py` | 补充显式调用的程序性原文，绑定标题、公式、变量、条件、例外、示例及更新依据 | 保留角色与来源；已知其他项目不因同名规则被纳入；不将旧规则计为当前执行支持 |
| `summary_assembly.py` | 优先尝试完整的具体需求组，再选择其余证据；紧凑模式仍保留状态、来源与缺口 | 原子证据组不截半句，按实际 JSON 返回预算计量 |
| `summary_selection.py`、`retrieval.py` | 比较主包与经范围、版本过滤后的紧凑包；最后检查实际输出依赖 | 不直接拼接未经筛选的基础命中；不重复实现版本解析，不增加查询模型调用 |

完整目标规则槽交给既有 `StateResolver`，规则的更正、变化、撤回、冲突、时点和来源截止
均沿用其判断。多值槽中的其他规则不能仅因共享槽位而冒充目标规则的后继。
无法结构化确认的原文保留“适用性待核验”标签，不自动变成当前有效指令。

小预算原文保全仅在以下条件同时满足时适用：无结构化事实需要状态标签、无收集器警告、
完整原文的全部非空内容均已被合格证据覆盖、所有声明需求都有支持、完整返回仍满足预算。
角色和时间前缀必须对应同一快照中的已存储显示文本及来源元数据，不能通过任意前缀匹配绕过检查。

## 验证结果

| 检查 | 数据/预算 | 结果 |
| --- | --- | --- |
| 完整回归 | 全部自动化测试，离线替身 | 658 通过，33 条依赖弃用警告 |
| G 当前优势对照 | 11 问题，1500 token，基线 `f97629b` | 5 改善、6 不变、0 退步 |
| G 历史对照 | 同 11 问题和预算，基线 `f930ac62` | 3 改善、8 不变、0 退步 |
| 小预算完整原文 | 1 问题，128 token | 对 `f97629b` 改善；对 `f930ac62` 不变 |
| 既有联合保护 | 19 问题，1500 token | 30 项声明检查不变，兼容性检查通过 |
| 既有综合保护 | 7 问题，1500 token | 必要引文覆盖不变，无退步 |
| 既有治理保护 | 21 问题，1500 token | 必要引文覆盖不变，无退步 |
| 真实回答 | 4 个合成问题各旧/新一答，最多 8 次调用 | 首次请求遭拒，0 个答案；未验证 |

47 个既有问题中，45 个最终响应完全一致；两个报告类问题因新增具体需求或紧凑选择而改变
内容组织，但既有完整证据检查没有退步。不能将该结果写成“所有 active 响应完全不变”。
三套保护数据共 155 项兼容性断言全部通过，包括响应 ID、正文、分数、时间、顺序及调用记录。
实际结论限定在这些测试材料，不能证明全部普通问题或线上 C3 永不退步。

完整回归首次暴露本地依赖冲突：虚拟环境允许继承系统包，加载了全局 `httpx 0.28.1`，
与已有 `Starlette 0.36.3` 的测试客户端不兼容，产生 26 失败、6 错误。仅在项目 `.venv`
安装符合现有 requirements 范围的 `httpx 0.27.2` 后，受影响的 123 项检查和全量回归均通过。
没有修改全局包、requirements、服务代码来绕过接口测试或生产配置。

真实回答程序使用现有 `openai` 适配与配置的 `gpt-4o-mini`，温度 0；只允许发送问题及合成
记忆，不把金标、oracle 草稿或预期答案传给模型。失败只记录异常类型，未记录凭据或上游错误正文。
调用成功后仍需逐题语义审核，不能以程序解析成功或关键词命中直接判分。

## 复现

```powershell
.\.venv\Scripts\python.exe -m pytest -o addopts= -q --junitxml=data/g_execution/full_pytest.xml
.\.venv\Scripts\python.exe scripts/run_g_execution_regression.py --baseline f97629b --output data/g_execution/protection_final
.\.venv\Scripts\python.exe scripts/run_g_execution_regression.py --baseline f930ac62 --output data/g_execution/historical_final
.\.venv\Scripts\python.exe scripts/run_g_execution_regression.py --baseline f97629b --scenario tight_budget --budget 128 --output data/g_execution/protection_tight_final
.\.venv\Scripts\python.exe scripts/run_g_execution_regression.py --baseline f930ac62 --scenario tight_budget --budget 128 --output data/g_execution/historical_tight_final
.\.venv\Scripts\python.exe scripts/run_memory_joint_regression.py --baseline f97629b --output data/g_execution/joint_protection_final
.\.venv\Scripts\python.exe scripts/run_memory_joint_regression.py --baseline f97629b --dataset tests/fixtures/f1_e2e_cases.json --output data/g_execution/f1_protection_final
.\.venv\Scripts\python.exe scripts/run_memory_joint_regression.py --baseline f97629b --dataset tests/fixtures/governance_e2e_cases.json --output data/g_execution/governance_protection_final
.\.venv\Scripts\python.exe scripts/run_g_answer_probe.py --paired-output data/g_execution/protection_final
```

最后一行仅做离线输入验证。需真实回答时显式追加 `--run`；程序最多调用 8 次，首次失败
即停止，成功结果缓存以避免重复计费。本次失败记录保存在
`data/g_execution/protection_final/real_answers/answers.json`，答案与语义评分保持未验证。
各 `comparison.json`、`g_support_comparison.json` 与测试 XML 位于忽略的 `data/g_execution/`，不随源码提交。

## 适用范围与剩余问题

- 程序性补证和最终包选择的输出生效条件仍为 `GOVERNANCE_MODE=active`、`F1_MODE=active`。
  `legacy`、`shadow` 与关闭模式的含义不变。治理的复合任务规划也会用于 active 的 legacy 路径。
  本次没有切换模式、修改 `.env`、生产数据库或部署服务。
- 基础增强、查询扩展、双通道评分、时间模块、抽取器、事实版本解析和存储代码未修改。
  版本与时间是否正确仍依赖已抽取事实和来源依据，不能由打包校验补足。
- 无标记自然语言的材料/任务分界、目标词组与局部依赖识别仍为保守规则。
  未显式标记的跨消息变量、例外及复杂改写不保证识别；任务来源不能仅凭“Now”获得可信认证。
- `release rules` 与 `rule: set release checks...` 可能不直接匹配：配置召回样例虽已返回两条
  完整规则，需求覆盖头仍保守显示缺口。不得将此覆盖计数等同于语义有无支持。
- 单独请求“我教过你的规则”时，助手的同名原文可能作为带角色标签的主题材料返回，但不会
  被计为用户教授的规则；未承诺将全部不适用诊断材料从正文删除。
- 程序来源与任务事实仍共享时间窗口，可能遗漏窗口前教授的规则；本轮没有放宽时间边界。
  更精细的“任务事件窗口/执行规则适用时点”分离需要独立保护测试。

下一步证据应为：恢复模型接口访问后完成已准备的小样本答案审核，再在同配置、明确部署提交、
相同历史写入条件下进行平台配对，保留逐题 Search 返回、最终答案和评分类别。
只有这些证据才能判断 G 是否恢复，以及线上 A/C/C3、普通查询和流式能力是否退步。
