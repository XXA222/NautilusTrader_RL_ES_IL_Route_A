# Codex 连续开发目标与当前游标

本文件是可直接交给 Codex Goal 的持续开发目标。审计背景、任务依赖、已完成工作包及验证证据见 [审计整改任务索引](DEVELOPMENT_GOALS_REVIEW_20261017.md) 和 [开发执行记录](DEVELOPMENT_PLAN_20261017.md)。下方目标要求 Codex 先核实最新续做位置，再自动实施所有依赖已满足的工程整改；它不是要求每轮只编辑文档，也不要求在外部证据缺失时停下所有工作。

## 可直接作为 Codex Goal 的目标

```text
在仓库 `D:\Program Files\NautilusTrader_RL_ES` 中，根据审计报告 `C:\Users\QX\Downloads\NautilusTrader-RL-ES_源码深度审计与后续开发建议_20261017.md`，持续实现 A01–A14 中仍存在且前置条件满足的工程整改。审计基线 `c24d1bc4a47584f91a4c4db93bf77117d4e8341c` 仅作历史参照；以当前源码、工作树差异、[任务索引](DEVELOPMENT_GOALS_REVIEW_20261017.md)和[执行记录](DEVELOPMENT_PLAN_20261017.md)为准。交付必须包含实际源码变更、相关回归与可核验的验证记录；不能以分析、计划或更新文档代替实现。

每轮按以下循环自动工作，直到没有可独立实施的事项：
1. 阅读本文件最新游标、任务索引和执行记录末尾；检查 `git status`、相关 diff、源码、测试与证据，核验目标缺口是否仍存在。不得重做已关闭事项。
2. 按 M0–M4 依赖选择最高优先、当前可实现的工程项。若首项依赖缺失，记录明确阻塞条件，然后继续下一项不依赖它的工作；不得为了等待 Validation、真实现金流、真实深度、新市场授权、Hosted CI 或匹配阶段工件而空转。
3. 直接实施最小必要源码修复，增加能稳定复现边界的确定性回归，运行受影响测试及适用 Ruff/MyPy；改动影响更广时再运行必要集成。检查通过后立即更新任务状态并选择下一项。失败时保留实际命令、错误和复现；同一根因连续两次失败后标为 `FAILED`/`BLOCKED`，转做独立项。
4. 每项结束后，将实际文件、行为、命令/结果、证据路径、数据用途、尚未认证边界和下一游标追加到 `DEVELOPMENT_PLAN_20261017.md`，同步本文件与任务索引。绝不把计划命令写成已运行，不虚报通过。

优先顺序依审计与任务索引执行：经济账本、可复现交付和访问边界（A08/A10/A04）；ES 参数与目标（A01/A02）及阶段归因（A11）；Episode/执行/流动性/性能语义（A03/A12/A06/A07）；试验数、覆盖和费用后动作价值（A05/A13/A14）；多标的、持仓模式和杠杆（A09，最后）。P0/P1/P2 是排序依据，不得绕过依赖或验收条件。工程正确性、Train-inner 诊断、独立 Validation/Test/Holdout 结论分别记录和验收。

任何历史行情读取或解码之前，核对研究登记、允许窗口、用途和冻结身份；只在既有授权范围开展 Train-inner 诊断。不得访问未获准的 Validation/Test/Holdout、私人账户数据或未登记真实深度；不得通过换目录、协议或快照重新使用已消费样本。不得降低风控/清算门槛、猜测缺失现金流、伪造证据或把计划订单当成交。Train-inner、scenario-only、合成盘口、训练 Reward/Fitness 均不能证明独立盈利或真实流动性。外部证据缺失只阻塞依赖它的验收，必须写明恢复条件并继续其他可做工作。

不得修改 NautilusTrader 上游源码，不 reset/clean 工作树，不覆盖已有或无关改动。若当前可实现工程项均已完成必要验证，且其余事项均已准确标记 `PARTIAL`、`PENDING`、`CONDITIONAL`、`DEFERRED` 或 `BLOCKED`，附具体依赖、恢复条件和续做位置，才可结束目标。策略盈利不是工程完成条件；不能因文档更新或少量测试通过而宣告全部审计整改完成。
```

## 当前执行游标

**最近完成：A04 / 所有 Train-only 入口和 worker 的 scoped-reader 边界。** `market-audit` 与 `diagnose-train` 现在要求显式 `train_data_access`，拒绝任何非空 Evaluation 配置，按登记窗口读取。授权 `source_mode` 也先校验字符串类型，畸形 list/dict 在读取前统一失败。诊断复用训练 bundle 的 scaler；`backtest()` 只接受由登记 reader 创建并绑定授权身份的 Train 工厂。R1 多环境 worker 和 E1/E2 ES worker 重复幂等登记相同访问身份后才解码 scoped 行，不再从 `market_path` 整读。Train-only 市场报告只包含 Train 窗口。`application_cli`、研究访问、WFO 和历史工厂组合 **175 passed**；最新 A04 输入边界选择 **9 passed**；Ruff、Windows MyPy 通过。

**最近完成：A05 / 研究尝试生命周期状态类型校验。** `ProjectResearchRegistry.record_trial_attempt_event()` 现在在集合成员判断前要求 `status` 为字符串，避免 list/dict 输入泄漏 `TypeError`，统一抛出验证错误。无效 list/dict/null/整数/未知字符串覆盖在定向测试中，A05 trial-attempt 回归 **6 passed**，Ruff 与 Windows MyPy 通过。

**本轮复核：** worker 恢复沿用相同的冻结 `environment + train_data_access`，父进程与重启/子进程按稳定 owner、label、窗口和 marker 身份幂等登记；运行 recipe/checkpoint 身份仍绑定完整 config。A05 DSR 报告只把唯一 frozen candidate IDs 用作 protocol 内 `effective_trials`，项目级完整历史仍明确 `not_estimated`，未发现可用来推算登记前人工尝试的证据。诊断入口也现拒绝非空的任意形状 `evaluation` 值。

**最近完成：A08 / 原生账本枚举字段类型防御。** `external cashflow.kind`、native fill `order_side` 和线性持仓 `side` 现在先校验字符串类型再进行集合成员检查；恶意或损坏的 list/dict 不会再泄漏 `TypeError`，均走账本的 `MetricsError`。新增异常类型回归后，研究指标、清算事实和 Binance 现金流相关套件 **66 passed**；Ruff 与 Windows MyPy 通过。仅测试合成事实，没有读取市场或账户 payload。

**最近完成：A04 / 项目登记 purpose 和来源 schema 输入校验。** `ProjectResearchRegistry._scope()` 对 `purpose`、身份对象和键类型先校验，身份 JSON 序列化错误统一变为验证错误；开发来源的 `schema_version` 也先校验字符串类型。新增 list/dict/null/整数/未知目的和畸形来源 schema 回归；研究访问、WFO、CLI 组合 **128 passed**，Ruff 和 Windows MyPy 通过。之后完整 Windows PR selector **598 passed、1 CUDA-only skipped、85 deselected、46 warnings**。

**最近完成：A05 / trial recipe 与生命周期 detail 序列化输入校验。** 注册 recipe 和生命周期事件 detail 现在将 NaN、非 JSON 对象及递归序列化失败统一转为 registry `ValueError`，避免裸 `TypeError`/`ValueError` 泄漏或写入半成状态。新增相应回归后访问/WFO/CLI 组合 **132 passed**，Ruff、Windows MyPy 通过；本地 Windows PR selector 复跑 **602 passed、1 CUDA-only skipped、85 deselected、46 warnings**。未读取市场、账户或正式评价 payload。

**当前分发与本地门禁证据：** A04/A05/A08 当前源码快照 287 文件，identity `106925971443f3c40aefea544b297ad175247b841a82afc226da17e6bd868cdf`，`--verify` 通过；11 wheels/11 sdists 构建成功，wheel manifest `all_match=true`，isolated target 导入确认 Research、Orchestrator、worker 与 Gym bridge 来自当前 wheelhouse，安装后的非有限 trial recipe 输入 smoke 也通过。工作流同选择器的本地 Windows PR 检查在本轮源码上 **602 passed、1 CUDA-only skipped、85 deselected、46 warnings**；全仓 Ruff 通过，MyPy `packages qa` **140 source files** 通过。该结果不替代干净交付后的 Hosted Linux/Windows CI。

**续做游标（以执行记录末尾为准）：** 最新记录已完成 N04 资源恢复边界复核、M2.2/A12 pending-cancel/update 状态分类、A04 Train-only market-audit 与诊断/worker scoped-reader 修复、A05/A08/A04 的输入类型防御，以及本地 Windows selector、Ruff/MyPy 和 wheel 来源核验。A02 负终值破产筛选及 solvent episode 对数收益—首末权益复算已修复；M1.2 仍需独立 Validation。当前没有已确认、可直接开工的下一项源码缺口；下一轮先从任务索引中做有边界的代码差异核验，只对新发现且能由合成/仓库夹具验证的缺口实施修复，然后自动推进。M0.1 可信完整真实现金流、M0.2 干净交付后的 Hosted CI、M1.2 独立 Validation、M1.3 匹配阶段工件、M2.3 真实深度、M3.2 新独立市场授权均记录恢复条件，不能自行伪造输入或开启正式数据访问。A09 仍为后置能力项。

**执行约束：** 工作区已有大量未提交源码、测试和文档改动。编辑前检查目标文件当前内容；只改目标文档和本工作包需要的代码/测试。遇到相同失败机制连续两次无法解决，保存真实复现并转做独立任务。任何外部依赖、自动审查拒绝或平台限制都写明具体结果，不推断为通过。

## 当前状态摘要

- A01：E2 精确中性候选、检查点恢复与旧 schema 兼容回归已完成相应工程范围。
- A02：版本化约束优先目标与 Train-inner 对照已实现；独立 Validation 未完成，保持 PARTIAL。
- A04：WFO 项目级 query intent/登记及 `market-audit` Train-only scoped-reader 入口已加强；登记启用前的历史访问仍无法证明。
- A08：经济事件去重、冲突拒绝、清算/现金流路径及账本归并有多项实现；真实完整现金流来源未认证，保持 PARTIAL。
- A10：Windows 本地检查、Ruff/MyPy 和 wheel 来源验证有最新证据；当前干净交付 revision 的 hosted Linux/Windows CI 未完成，保持 PARTIAL。
- A11：M1.3 分析器和 E1 工件持久化已完成工程实现；完整兼容三阶段原生经济工件缺失，保持 PARTIAL。
- A12/A13：执行 trace 状态分类、部分成交归因和 overlap 计数优化已有定向回归；新独立市场覆盖仍需授权。
- A06：Bar Base/Stress 合成深度工程已覆盖；真实历史 Quote/Depth 未认证。
- A05：已登记历史尝试可汇总，生命周期状态类型校验有回归；未登记人工研究不可恢复时，历史有效试验数保持 `not_estimated`。
- A09：ETH/NETTING/杠杆扩展仍受后置能力门禁约束。
- N04：受监督 GPU 配额现已验证 owner 正常释放、仍持锁时拒绝释放、活动后代存在时拒绝释放及 owner 硬退出后的恢复；Windows Job 提交内存边界不构成 VRAM 上限，N04 整体仍为 `PARTIAL`。

上列是导航摘要，不替代执行计划中的命令、哈希、测试细节和证据边界。每次续做先核验当前源码，避免把摘要当作代码现状。

## 完成条件

当前可实现的审计整改均有必要源码、确定性回归和适用静态检查证据；依赖外部数据、正式研究授权、可信账户来源或远程 runner 的事项均记录为未完成，并列出恢复条件；所有失败与未认证能力如实保留。更新 [开发执行记录](DEVELOPMENT_PLAN_20261017.md) 与 [审计整改任务索引](DEVELOPMENT_GOALS_REVIEW_20261017.md) 后方可结束。
