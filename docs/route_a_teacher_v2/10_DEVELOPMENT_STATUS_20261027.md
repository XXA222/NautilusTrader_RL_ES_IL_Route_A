# Route A 1m 教师整改阶段报告 - 2026-10-27

基线：bd313c5241c31f1ccff96fa4bf76a7966be48767。所有原生历史结果均来自已消费 Train 开发数据，按 DEVELOPMENT_DIAGNOSTIC_ONLY / SCENARIO_ONLY 解释；不代表新样本外盈利、G1/G2 或项目经济验收。

## 任务一：EMA 多空反手

**PASS（状态机与工程验证）；FAIL（经济结果）。** EMA 保留旧 X1 EXIT_ONLY_V1 对照，新增平仓确认后下一根合法 1m bar 复核并反手。覆盖多空、六分钟待确认、同分钟重放、EMA 方向失效、未决订单和原生风险否决。生命周期测试 30 项通过；Ruff、聚焦 MyPy 通过。

原生旧版/反手版配对覆盖 2024 年 2、3、4 月 × Base/Stress 共 6 组；身份检查无失败，M2.2 无缺口，强平为零。反手版产生双向交易，但净收益和最大回撤在六组中都比旧版差。净收益（旧版 -> 反手版）：二月 Base -2.33% -> -5.12%、Stress -6.70% -> -12.91%；三月 Base -3.31% -> -5.83%、Stress -8.21% -> -14.43%；四月 Base -0.72% -> -1.56%、Stress -2.77% -> -5.16%。

证据：packages/route_a_imitation/src/rles_route_a/teachers_1m/base.py；packages/route_a_imitation/src/rles_route_a/teachers_1m/ema.py；packages/route_a_imitation/tests/test_teachers_1m_lifecycle.py；.artifacts/route-a-ema-reversal-task1/paired_base_stress_summary.json。

## 任务二：ROC 过度交易

**PASS（候选实现、配对工程验证）；FAIL（绝对盈利）。** 成本/ATR/RV 感知滞回候选具有双根确认、短期反向 HOLD/EXIT 区分，未设置统一持仓上限。ROC 专项测试 21 项通过；Ruff、聚焦 MyPy 通过。

同源、同账户、同成本配对覆盖三个已消费开发块的 Base/Stress 六组；身份检查无失败，M2.2 无缺口，强平为零。旧版合计 8,309 笔；候选完成 1,046 笔（Base）和 643 笔（Stress）。候选在两种情景的三个块中均改善账户净收益和回撤，但候选每个块仍为负收益；因此只确认相对减亏，不确认盈利资格。逐笔原生案例、费用归因和配对结果见 .artifacts/route-a-roc-hysteresis-task2/paired_native_results.json 及其 native/ 子目录。

## 任务三：动态退出与逐笔归因

**PASS（分析与候选机制）；FAIL（普遍经济改善）。** X1/X2 报告包含 EMA、ROC、Donchian、Z-score 四位教师 × 三个开发块 × Base/Stress 共 24 组，配对身份检查无失败；逐笔指标文件 53,968 行，汇总报告标出 8,667 笔 MFE 至少达到往返成本但最终净亏损的案例，并包含 MFE、MAE、持仓时间、浮盈回撤、利润捕获和同一入场的 X1/X2 现金流对照。X2 在 ROC 上改善各开发块结果但仍亏损；其他教师没有跨块一致改进。X3/X8 等 ATR 利润保护/交接候选有代码和回归覆盖；现有 X8 原生诊断未稳定胜过 X1。

证据：.artifacts/route-a-exit-economics-task3/x1_x2_exit_economics.json；.artifacts/route-a-exit-economics-task3/per_trade_exit_metrics.jsonl；.artifacts/route-a-roc-x2-handoff-x8-task4/paired_x1_x2_x6_x8_results.md；packages/route_a_imitation/src/rles_route_a/teachers_1m/exits.py；packages/route_a_imitation/src/rles_route_a/integration/exit_lifecycle.py。

### 新增逐笔亏损路径、方向经济与块级退出时机归因（2026-10-27）

对既有 S06 Train 衍生逐笔输出做只读分析（53,968 行）。分类按教师 × 退出 × 开发块 × 情景拆成 48 组；X1/X2 精确入场比较按教师 × 开发块 × 情景拆成 24 组。没有重跑原生撮合或读取市场源。

跨组汇总有 45,838 条亏损结果记录（同一入场可出现在多个退出版本/情景中，所以不是唯一交易数或单账户亏损额）。组内类别互斥；跨组亏损金额按 USDT 汇总为：逆向 MAE 主导 29,652 条、27,991.02；较大 MFE 后回吐/潜在退出过晚 5,063 条、3,713.46；负价格收益或弱边际 4,224 条、2,659.46；正价格收益被费用/资金费吞噬 6,899 条、1,509.84。分类仅描述路径，不能单独判定信号或退出的因果责任。

按教师 × 退出 × 块 × 情景 × LONG/SHORT 共 96 个方向单元中，90 个有普通完整交易，且这 90 个单元的合计净 PnL 全为负；另 6 个空单元全部属于原始 EMA X1：inner-screen-1/2 没有普通完整 SHORT，inner-screen-3 没有普通完整 LONG。单元级结果是交易案例归因，不替代账户级原生资格判断。现金流复核覆盖 53,944 条普通完整交易，realized_pnl = gross_cash_pnl_before_fees_and_funding - fees + funding 最大绝对误差为 1.78e-15 USDT，零条超过 1e-6；滑点保留为单独归因，不重复扣除。按方向单元汇总，Base 有 45 个有普通完整交易的单元，其中 11 个扣费用/资金费前的实际成交价 PnL 为正但净 PnL 为负，另 34 个扣除这些费用前已为负；Stress 的 45 个单元扣费用/资金费前均为负。此结果提示 Base 存在费用侵蚀，Stress 还存在更基础的价格/信号表现问题；这是 D1 描述性推断，不是因果识别。

X1/X2 同一入场的退出时机按块列在归因报告中。三块汇总仍显示教师方向不一：ROC 较早组多于较晚组；EMA、Donchian、Z-score 较晚组多于较早组。它们是实际现金流的描述性匹配，不是随机因果比较，也不足以推出统一提前/延迟规则。

证据：.artifacts/route-a-exit-loss-attribution-task3-20261027/exit_loss_attribution.json、.artifacts/route-a-exit-loss-attribution-task3-20261027/exit_loss_attribution.md、qa/analyze_route_a_exit_loss_attribution.py、tests/test_route_a_exit_loss_attribution.py。最新定向测试 4 项通过；Ruff、MyPy 通过；MyPy 使用 MYPYPATH=.。


## 任务四：原生回测与筛选器

**PASS（已完成的 Train 开发诊断）；BLOCKED（新窗口评分与经济验收）。** 现有报告覆盖 EMA 旧版/反手版、ROC 旧版/抗噪候选、四教师 X1/X2 和固定方向对照；配对报告含信号漏斗、LONG/SHORT、成本、收益、风险、持仓与交易级指标。筛选器测试 11 项通过；Ruff、MyPy 通过。固定方向对照也显示收益方向随窗口反转，不能作为稳定盈利策略。

当前冻结预检为 PREFLIGHT_PASS_STAGED_SOURCES_PARTIAL_METADATA_NOT_READ：授权源登记存在，预期 5 个导入中仅 1 个预热导入存在，窗口 2026-10-23 至 2026-10-30 的 import.json 缺失，market_payload_read=false。因此没有执行部分窗口评分；没有读取 Test/Holdout，也没有启动学生训练。不得声称 G1/G2 或样本外盈利通过。

证据：.artifacts/route-a-fixed-controls-task4/paired_fixed_side_vs_x1.json；.artifacts/route-a-ema-reversal-task1/paired_base_stress_summary.json；qa/validate_route_a_teacher_student_prereg_20261026.py；tests/test_route_a_teacher_train_screen_runner.py。

### 本轮评分器验收门槛复核（2026-10-27）

静态核验冻结的预注册、源勘误和原生评分器，教师资格规则与项目验收要求一致：四个不重叠窗口 × Base/Stress × 两个种子共 16 个配对槽；候选每槽都须有费用后净收益 > 0、严格高于 FLAT/FIXED_LONG_0.3/FIXED_SHORT_0.3、最大回撤不超过 0.30、零强平且全部决策 M2.2 匹配。缺任一槽或控制组即不能通过。学生训练由完整教师筛选及合格候选门控；冻结的学生屏幕使用后续独立窗口，并要求每个配对槽费用后正收益且严格超过同槽教师。当前未训练学生。无需为这项规则另加重复门禁。

窗口 0 需导入 `2026-10-23` 至 `2026-10-29` 日档，计分期为 `2026-10-25` 至 `2026-10-30`。截至 10 月 27 日，只有两个暖机日导入完成，窗口 0 导入目录尚不存在，所需 10 月 29 日日档还未发布。冻结源审计规定完整发布后导入到 `data/routea_teacher_train_window0_20261023_20261029`，再通过只读预检并执行本报告所列 `--window-index 0` 原生评分命令；不评分部分窗口，也不读 Test/Holdout。

## 总状态与下一步

工程候选和已授权开发诊断已完成；经济目标未通过。现有结果指向弱且不稳定的方向信号叠加交易成本侵蚀，部分退出规则还会增加亏损；不能据此宣称实际交易盈利。新窗口完整导入并通过冻结预检后，执行：

wsl.exe --distribution Ubuntu-24.04 --exec /bin/bash -lc 'cd "/mnt/d/Program Files/NautilusTrader_RL_ES_IL_Route_A" && .venv-ubuntu/bin/python qa/run_route_a_teacher_student_train_screen_20261026.py --window-index 0'

完成该新窗口的原生评分前，不做教师盈利、学生超越教师或项目经济验收结论。


## Frozen screen recipe-identity integrity repair — 2026-10-27

A source-independent audit found that resume and summary validation keyed native-run records by teacher family but did not compare the serialized teacher_recipe. A stale or misplaced run could therefore be attributed to the ROC X1/X7 or EMA reversal candidate named by its directory. The runner now checks the exact recipe on resume and in summaries, rejects non-registered scenario/seed slots, and derives active-screen completion from the exact expected key set. The original preregistered matrix remains reported separately.

The focused runner suite passes (13 tests); Ruff and focused runner MyPy pass. The preregistration preflight passes with the registered row still recorded, one of five staged imports present, and market_payload_read: false. Source-errata SHA-256 is 7db874aef4eef349b9eb19b27614454c06646a22e7f77b959fa3067b25bdd1b0.

This repair changes execution-evidence validation only. Teacher recipes, thresholds, source dates and roles, seeds, controls, scenario costs, and acceptance rules are unchanged. No market payload or economic outcome was read, no partial window was scored, and student training remains unstarted.


## ROC episode-reset latch correction - 2026-10-27

A lifecycle review found that the cost/ATR profit-lock rearm latch was intentionally preserved by on_discontinuity() across missing-bar gaps, but BaseTeacher1m.reset() invokes that same hook. A reused ROC teacher could therefore carry a prior episode's opposite-consensus requirement into a new episode and suppress valid entries. RocCostHysteresisTeacher.reset() now clears the episode-scoped latch, exit side, and opposite-run counter after the shared reset, while ordinary within-episode gaps continue to preserve the latch. The new regression covers both paths and confirms same-regime entry is available again after a new episode warms up.

The focused ROC suite passes (22 tests); Ruff and focused MyPy pass. The active ROC implementation is now bound in the existing runtime_code_sha256 map (module SHA-256 85c17057c0ab9e0dca98f18c7dfed021ffda1bdd135c5d8bd8ed9e8dc0d94f23). Read-only preflight passes with registry state recorded, one of five staged imports present, and market_payload_read false. Source errata SHA-256 is ea9a78ff5c2c5f43a2f59b13e5f0aecdcbfd2bbe58b5d98d0405b48957d0d900.

This is a lifecycle correctness repair; thresholds, candidates, exit recipe, window roles, source dates, seeds, costs, controls, and acceptance criteria are unchanged. No new market payload or scored outcome was read.

## 教师 source/fold 身份隔离修复（2026-10-27）

此前 BaseTeacher1m 只在 episode_id 改变时清空状态。同一 episode 更换 source_id 或 fold_id 时，EMA 可能把不同数据身份的快照误判为交叉；ROC 滚动收益与共识、回撤教师历史、待处理反手和 ATR 退出状态也可能跨边界沿用。现在 episode、source 或 fold 任一身份改变都会重置教师与退出状态，再从新身份的下一根 bar 建立状态。

回归覆盖 source 切换和同 episode 的 fold 切换：新身份的首根 bar 不会产生跨边界 EMA 入场，后续本身份快照仍能形成自己的合法交叉；ROC 回归确认来源切换清空滚动波动率和共识历史。生命周期与 ROC 定向 pytest 共 55 项通过，Ruff、实现和测试文件的聚焦 MyPy 均通过。base.py 新哈希已写入现有评分前源勘误，只读预检通过。本修复未读取行情载荷或生成原生经济结果，也未改变候选配方、阈值、窗口角色、成本、种子、对照或验收规则。

## X3/X8 部分成交后的 ATR basis 修复（2026-10-27）

X3/X8 在原生持仓均价因追加或部分成交变化时会重置退出状态。旧逻辑用该根已完成 bar 的 high/low 初始化新 basis，但极值可能发生在追加成交之前，导致 X3 把新持仓未经历过的盈利当作利润锁触发条件，或让 X8 提前交接给指标退出。现在均价变化当根从完成收盘价建立新极值，后续 bar 再按常规 high/low 跟踪；X5 继承 X3 的修复。

新增 X3 多空假触发回归和 X8 提前交接回归。生命周期与 ROC 定向套件共 59 项通过，Ruff 和聚焦 MyPy 通过。exits.py 新哈希已绑定到现有源勘误，只读预检通过。本轮没有读取新行情或生成原生经济结果，冻结配方阈值、成本、日期、角色、种子、对照和验收规则均未改变。

## X2 首次观察持仓 bar 的 ATR basis 修复（2026-10-27）

只读代码审查发现，X2 首次看到原生持仓时用整根已完成 bar 的 high/low 建立追踪极值；原生成交可能发生在极值之后，导致成交前的尖峰把追踪线推到随后价格之上/下并提前平仓。现在首次初始化改用该 bar 收盘价，后续 bar 继续按 high/low 跟踪。新增多空假退出回归，代码哈希已写入源勘误。修复未改变阈值、候选配方、成本、日期、窗口角色、种子、对照或验收规则。本轮未读取行情载荷或经济结果；既有 X2 结果仍为开发诊断，未重跑。

验证：生命周期与 ROC hysteresis 定向套件 71 项通过；exits.py 和生命周期测试的 Ruff 通过，聚焦 MyPy 对 2 个源文件未发现问题。预注册预检通过，registry 为 recorded，5 个暂存导入中只有 1 个存在，market_payload_read 为 false。冻结源导入齐备前，窗口 0 继续保持未评分。

## X3/X8 首次观察持仓 bar 的 ATR basis 修复（2026-10-27）

后续代码审查发现，X3/X8 首次看到原生持仓时仍使用整根 bar 的 high/low。X3 可能借用成交前极值提前激活成本利润锁；X8 可能借用该极值达到 4 倍成本交接点并关闭 ATR 止损。现两者首次建仓均从完成收盘价初始化，与 X2 及均价变化修复一致。新增多空回归，覆盖 X3 提前激活和 X8 提前交接。本轮未改配方、阈值、数据角色、成本、对照或验收规则，也未读取行情载荷或经济结果。

X2/X3/X8 首次观察持仓修复的最终验证：生命周期与 ROC hysteresis 定向套件 71 项通过，Ruff 通过，聚焦 MyPy 对 2 个源文件未发现问题。冻结预注册预检通过，5 个导入仅 1 个存在且 market_payload_read 为 false。本轮未评分任何原生窗口。

## X2/X8 首根持仓 bar 的不利 ATR 止损判定（2026-10-27）

首根只按收盘价初始化虽能排除成交前极值，但若该收盘已相对原生持仓均价不利超过 3×ATR，X2/X8 会把止损推迟到下一根 bar。X2/X6 共用 AtrTrailingExit：首次建线时多头采用 max(均价, 收盘)、空头采用 min(均价, 收盘)，不计入成交前 high/low，并立即检查收盘是否越过止损；X8 的交接路径同步采用该规则。新增 X2/X8 多空回归；没有读取行情载荷或经济结果。

## X3 首根持仓按收盘激活利润保护（2026-10-27）

此前 X3 首次看到持仓时虽排除了成交前 high/low，却直接返回，未处理已完成收盘相对原生均价达到既有激活阈值的情形。现在首次观察或均价变化时以原生均价和收盘价建立状态，仅该收盘参与当根激活；X4/X5 继承该路径。多空回归确认不借用成交前尖峰。阈值、成本与配方未变，本轮未读取行情载荷或经济结果。
## X2 动态退出输入完整性修复（2026-10-27）

只读代码审查发现，ATR 退出器在 OHLC/ATR 特征缺失时会返回无退出信号，而通用教师仍可能仅凭 EMA/ROC 等入场字段开仓。现在所有启用 ATR 退出的教师在账户空仓时遇到必需退出字段缺失会保持空仓并记录 ATR_EXIT_NOT_READY；已有持仓遇到这些字段缺失则显式抛出数据完整性错误，避免静默继续而遗漏冻结退出规则。X1 无 ATR 退出依赖，行为不变。

生命周期与 ROC hysteresis 定向套件共 76 项通过；Ruff 和聚焦 MyPy（base.py 与生命周期测试）通过；git diff --check 通过。现有源勘误已绑定 base.py 新哈希，source errata SHA-256 为 d5cc6c5642d2a00113b1060e234d50d21dfe1b542a68ac7d39ff67a8f61c8002。只读预检通过：注册仍为 recorded，4 个教师窗口、4 个学生窗口，5 个阶段导入仅 1 个存在，market_payload_read=false。未读取行情载荷或生成经济结果；窗口 0 未评分，配方、阈值、成本、窗口角色、种子与验收门槛均未改变。

## Dependency-following MyPy closeout — 2026-10-28

The dependency-following MyPy run had previously exposed callback annotations that were narrower than the NativeGymTeacherProvider Mapping contract, and a runner parameter typed only as TrendTeacher although the same path also passes WaitTeacher. The callback inputs now use Mapping[str, Any], the teacher parameter accepts both concrete teacher types, and the two installed rles_orchestrator imports are marked untyped because that package ships without stubs.

Validation in Ubuntu-24.04: MYPYPATH=packages/data/src:packages/route_a_imitation/src:. .venv-ubuntu/bin/python -m mypy qa/run_route_a_teacher_student_train_screen_20261026.py — passed; .venv-ubuntu/bin/ruff check qa/run_route_a_train_student_screen_20261020.py — passed. The source preflight passed with errata SHA-256 25c8052a480f6e6a156ccfc61df225daeb0d110ad93c9ca75e71c6c9ffa9af4a, while correctly reporting only 2/5 staged imports and market_payload_read=false. This was a typing-only correction; no market payload or economic outcome was read, and candidate recipes, source scope, costs, and acceptance rules are unchanged.
