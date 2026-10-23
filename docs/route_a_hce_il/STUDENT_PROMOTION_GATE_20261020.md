# 学生超越教师的 Train-inner 筛选门（2026-10-20）

`rles_route_a.evaluation.screen_student()` 为多窗学生筛选增加不可放宽的默认准入：

- 先要求同一市场源、同一教师和基准通过 `screen_teacher()` 的门，并逐窗绑定其完整教师配对结果与 `receipt_sha256`。
- 学生评估必须覆盖与教师门完全相同的窗口 × Base/Stress × seed 矩阵；默认至少 2 个不重叠窗口、2 个 seed。
- 每一对都要求教师与学生回报严格大于零，且学生回报严格高于同窗/场景/seed 的教师；任一不满足都拒绝。
- 任一学生清算、超出回撤门、缺少成交、M2.2 屏障未匹配、窗口非重叠未由宿主核验，均拒绝。
- 全矩阵使用同一学生 actor SHA-256、Train source、observation/action schema 和 execution recipe；每一对保留教师与学生原生回放证据 SHA-256。学生筛选收据同时哈希完整配对数据和阈值。

`TRAIN_INNER_SCREEN_PASS` 仅表示这个冻结 Train-inner 场景矩阵满足预先设定的筛选条件。若教师成本规则带有情景假设，结果的 `economic_status` 仍是 `SCENARIO_ONLY`；该门不替代 L6，不开启 Validation/Test/Holdout，不证明历史真实净收益或策略盈利。窗口是否独立只能由宿主登记及真实时间边界核验，调用方不得自行把窗口 ID 声称为不重叠证明。

**WFO 集成状态：** `make_route_a_wfo_train()` 已强制要求 PASS 收据，在写 actor/bundle 和调用 Validation 前重算教师与学生筛选、核对冻结权重及 schema/source/execution 身份，并在候选包里保存完整 teacher/student 配对矩阵。actor 序列化后会重新加载并再次比对权重哈希。缺失、拒绝或身份不匹配的收据会阻断候选。

**运行编排状态：** `evaluate_bc_on_registered_train_inner_rollout()` 仍只产生单个 selection window。新增的 `qa/run_route_a_train_student_screen_20261020.py` 在注册 Train 工厂上运行多窗 × Base/Stress × seed 矩阵、保留各方 M2.2/L6 证据，并把完整 `StudentScreen` 交给 WFO；当前它是研究/QA runner，尚未成为通用 WFO 收集器。WFO 对缺失或失败收据继续 fail closed。

实现与回归：`packages/route_a_imitation/src/rles_route_a/evaluation/teacher_gate.py`；固定输入测试覆盖通过、任一负收益/未胜过教师、未过教师门、矩阵缺项、actor/来源/收据混用和伪造教师收益。针对性回归为 **37 passed**；完整 `packages/route_a_imitation/tests` 回归通过；Ruff 和单文件 MyPy 均通过。该测试未读取行情。

## 2026-10-20 注册 Train-only 实测

- 只通过宿主注册的 whole-snapshot reader 读取历史源中 2024-01-02 至 2024-04-08 的 Train 区间；来源 marker 与每次 reader receipt 绑定。没有使用 Validation/Test/Holdout。旧快照质量审计从 DQ-D 修复为允许训练的 DQ-B；仍保留 `conservative_availability` 提示。
- 成本、规则和保证金仍为 `SCENARIO_ONLY`。最初的顺势教师在 2/3 月双窗矩阵净收益为负，拒绝晋级。
- 在已参与规则调参的 2/3 月窗口，冻结反向趋势、阈值 0.004、中性时持仓教师得到全配对正收益；这只作为调参发现，不能当作独立晋级证据。
- 独立 4/1 与 4/8 矩阵复核该教师。教师在 4/1 配对从 -0.0234% 到 +0.0001%，在 4/8 为 -0.3651% 到 -0.3478%；教师门拒绝。Jan fit 的 BC 学生在这 8 个配对均为负收益，因此未生成学生通过收据，也未导出 actor/bundle。
- 进一步以 2/3 月和 4/1/8 的 8 个 base/stress 配对探索弱趋势反转、强趋势跟随规则。4 个 regime 强度阈值最多只有 4/8 配对为正，没有任何阈值通过全矩阵；这组数据继续作为开发样本，不作为晋级证据。
- 完整证据：`.artifacts/route-a-train-student-screen-20261020-run13/screen-report.json`；阈值探索：`.artifacts/route-a-teacher-rule-probe-20261020-run11/probe-report.json`。经济状态均为 `SCENARIO_ONLY`。
- Regime 探索证据：`.artifacts/route-a-regime-teacher-probe-20261020-run15/regime-probe.json`；4 月直接/反向方向对照：`.artifacts/route-a-april-direction-probe-20261020-run14/direction-probe.json`。

## 长周期继续开发结果

- 完整 M2.2 审计快照现在逐 barrier 保留原生 reader 已观察到的累计订单、成交、资金费与强平事件，避免已关闭订单被 reader 清理后漏掉先前交付给 Gym 的 ID。mean-reversion 重跑的 32/32 episode 通过 M2.2 核验，但所有收益仍为负。资金费时点的 15 ms 容差来自全 Train 318 个事件的测量；导入声明的 1 ms availability delay 上限未放宽。
- GRU 使用 Feb 1–10 的 12,000 个注册 Train 决策拟合，在 Apr 13/14 完成两个日窗口的 Base/Stress × 两 seed 筛选。Apr 13 学生回报 -0.431% 至 -0.479%，低于教师 -0.361% 至 -0.397%；学生成交 60–64 笔，教师 6 笔。Apr 14 学生 +0.498% 至 +0.528%，高于教师 +0.472% 至 +0.487%，但学生成交 87 笔，教师 5 笔。教师门与学生门均拒绝，未产生 PASS 收据或 actor/bundle。证据：`.artifacts/route-a-long-horizon-student-screen-20261020-run22/screen-report.json`。
- 固定趋势方向在 12 个不重叠训练日、Base/Stress、seed 11/29 的滚动原生对照共 96 个 episode。顺势方向 20/48 配对为正、均值 +0.0039%、最差 -0.5205%；反向方向 28/48 配对为正、均值 -0.0392%、最差 -0.6074%。两方向在不同日期轮流胜出，没有一个方向通过全配对正收益筛选。最终报告及逐 episode M2.2 原生证据：`.artifacts/route-a-rolling-teacher-probe-20261020-run25/rolling-teacher-probe.json`；96 份证据哈希和报告 sidecar 均已核验，并与初跑的结果一致。
- 新日期均已作为开发数据使用，不能再作为独立晋级证据。眼下需解决教师跨行情稳定性和学生过度交易；所有上述收益仍为 `SCENARIO_ONLY`。

## 意图监督缺口与多 episode 实验

- 原训练将未成交 OPEN 命令的 execution weight 同时用于屏蔽意图和仓位损失。32 个平仓重置 Train episode 记录了 24 个 OPEN 决策，但都因未成交在意图头上被屏蔽，学生在 Apr13/14 对新空仓均只输出 FLAT。
- 已将 `LabeledStep` 的 `intent_supervision_weight` 与实际执行仓位的 `supervision_weight` 分离。核验过的 teacher request 可监督意图头；未成交执行仓位依旧不会被当作真实成交目标。意图感知推理对 HOLD、FLAT、EXIT 应用离散含义，对不一致的 ADD/REDUCE 输出采取不交易。各推理/损失/抽样模式写入 BC 配置及学生 recipe identity；旧 bundle 默认连续仓位推理。
- run34 通过 32 个六小时注册 Train episode 拟合 11,200 个决策，训练集 OPEN 召回 100%、总体意图准确率 89.6%。多 episode 使学生 Apr14 的 4 个配对都超过教师，净收益 +0.48% 至 +0.56%；Apr13 的 4 个配对全部为负且低于教师，净收益 -0.50% 至 -0.54%。这证明意图头开始开仓，但没有学到跨行情的正确方向。教师门失败，学生门未获准，WFO 未导出 actor。详见 `.artifacts/route-a-multiepisode-student-screen-20261020-run34/screen-report.json` 及 `native-evidence/`。
- 多 episode 训练脚本：`qa/run_route_a_multiepisode_student_screen_20261020.py`。最新修改通过 Ruff 与 `py_compile`；本轮未重跑完整单测。训练/筛选都只使用注册 Train 和情景成本，收益为 `SCENARIO_ONLY`。Apr13/14 已消耗为开发筛选日期，后续须另留未看过的 Train 窗口验证。
- run35 扩展该脚本的 scaler 拟合区间为 2024-02-01 至 2024-03-23，并在 32 个六小时重置 episode 与筛查中复用同一个 Train-only scaler。训练 11,200 个决策后总体意图准确率 93.58%、OPEN 召回 100%；预测 OPEN 86 次而标签仅 24 次。学生 Apr13 四对均胜过教师但仍亏损（学生 -0.291% 至 -0.247%，教师 -0.397% 至 -0.361%；学生每对 17 笔成交）；Apr14 四对均为正但低于教师（学生 +0.437% 至 +0.464%，教师 +0.469% 至 +0.487%；学生 9 笔、教师 5 笔）。因此教师门仍拒绝，学生门未执行，无 WFO actor 导出。报告：`.artifacts/route-a-multiepisode-student-screen-20261020-run35/screen-report.json`。脚本此前尝试直接运行时因缺少工作区 `PYTHONPATH` 无法导入，按项目开发环境明确设置 `packages/*/src;qa` 后正式运行成功；修改脚本 Ruff 与 `py_compile` 均通过。
- 意图采样消融：run37 使用逆平方根损失权重 + 自然随机采样，OPEN 召回 0%，April 两窗均零成交；run38 使用频次逆平方根抽样，OPEN 召回也为 0%，两窗均零成交。新增 recipe 绑定的 `sqrt_balanced_intent_v1` 与 `moderately_balanced_intent_v1` sampler，后者按类别频次的 -0.75 次幂抽样。run39 总体意图准确率 93.76%、OPEN 召回 83.3%（20/24），训练预测 65 个 OPEN；学生在 Apr13/14 各 8 笔成交。Apr13 四对均亏但逐对胜教师（学生 -0.388% 至 -0.352%，教师 -0.397% 至 -0.361%）；Apr14 四对均为正且逐对超过教师（学生 +0.484% 至 +0.503%，教师 +0.469% 至 +0.487%）。教师自身 Apr13 非正，故 gate 继续拒绝，学生不具备晋级资格、没有 actor 导出。证据：`.artifacts/route-a-multiepisode-student-screen-20261020-run39/screen-report.json`。所有回报均为 `SCENARIO_ONLY`，April 数据仍是已用开发样本。
- 反向趋势教师加 0.1% 毛名义仓位止损、止损后冷却 5 分钟，在已看过的 Apr13/14 × Base/Stress × seeds 11/29 共 8 对均为正，平均 +0.209%、最小 +0.002%，最大回撤 0.673%；但平均每对 34 笔成交。更长冷却的收益不稳（15 分钟 0/8 正，30 分钟 4/8，60 分钟 0/8）。研究探针：`.artifacts/route-a-recovery-overlay-probe-20261021-run01/`、`.artifacts/route-a-stop-cooldown-probe-20261021-run02/`。这是对已消耗开发窗口的探索结果，不是独立晋级证据。
- run40 使用该冷却教师生成 11,200 条监督样本后，训练 OPEN 召回 90.4%。教师在同一 April 开发矩阵上 `TRAIN_INNER_SCREEN_PASS`，学生却在 Apr13 的 4 对都正且超过教师（学生 +0.094% 至 +0.121%，教师 +0.002% 至 +0.046%），Apr14 的 4 对全 FLAT、零成交，漏掉教师 +0.357% 至 +0.432%。学生矩阵失败且报告未调用学生晋级门；没有 actor 导出。该结果说明教师候选暂能产生更优案例，但学生未学会跨窗口复现；教师 gate PASS 不应被解释为独立泛化或真实盈利。报告：`.artifacts/route-a-multiepisode-student-screen-20261020-run40/screen-report.json`。
- run41 在同一教师下改用全平衡 intent sampler，训练 OPEN 召回 94.8%，但 Apr13/14 八个 student 配对全部无成交。runner 已新增正式 StudentScreen 计算、actor/source/schema/execution 身份绑定与 `verify_student_screen()` 收据复算；实际学生门拒绝原因为 `STUDENT_NONPOSITIVE_NET_RETURN`、`STUDENT_DOES_NOT_BEAT_TEACHER_IN_EVERY_PAIR`、`STUDENT_HAS_NO_NATIVE_FILLS`。没有导出 actor。证据：`.artifacts/route-a-multiepisode-student-screen-20261020-run41/screen-report.json`。
- 5 分钟冷却教师进一步在 12 个滚动 Train 日期 × Base/Stress × 两种子共 48 对做复核：只有 24/48 为正，平均回报 -0.156%，最差 -1.325%，最大回撤 1.468%，平均 56 笔成交；教师门拒绝。Apr08、Mar08/15/22 明显亏损且频繁交易，说明它在 Apr13/14 的短期 PASS 是过拟合开发窗口。报告和 96 份教师/基准 M2.2 证据：`.artifacts/route-a-rolling-stop-cooldown-probe-20261021-run01/rolling-stop-cooldown-report.json`。所有日期均为已使用 Train 开发数据，仍 `SCENARIO_ONLY`。
- run42 将宿主已因果提供的 `market.zscore_1m` 加入学生 observation schema，并在新 schema 身份下重训。训练意图准确率 93.74%、OPEN 召回 86.1%，但学生 April 8 对全部负回报；Apr13 每对 -0.028% 至 -0.016%，Apr14 每对接近零且负，教师分别为正。StudentScreen 收据以 `STUDENT_NONPOSITIVE_NET_RETURN` 和 `STUDENT_DOES_NOT_BEAT_TEACHER_IN_EVERY_PAIR` 拒绝；未导出 actor。报告：`.artifacts/route-a-multiepisode-student-screen-20261020-run42/screen-report.json`。该特征没有修复学生跨窗策略失效。
- 基于该 z-score 与趋势一致性的 trend-pullback 教师网格也未找到可靠 alpha：8 个方向/阈值候选在 8 个子配对中最多 1 个为正；选出的候选在 12 日完整矩阵 8/48 为正、均值 -0.072%、最差 -0.235%、平均 12.8 笔成交，教师门拒绝。证据：`.artifacts/route-a-trend-pullback-teacher-20261021-run01/trend-pullback-report.json`。这是同一开发窗口的探索结果，不能作为晋级证据。

当前没有教师在未参与调参的 Train 窗口稳定通过的证据，也没有正收益且超过教师的学生模型。真实交易规则、手续费、资金费率、保证金及账户资料齐备前，不得将 scenario-only 回报描述为历史实盘收益；正式经济结论继续 BLOCKED。

## 2026-10-21 因果趋势效率特征与教师筛查

- Route A 的 `causal_mtf_v5` observation 新增 `efficiency_15m` / `efficiency_1h`，映射到宿主 `trend_efficiency_15m` / `trend_efficiency_60m`。宿主定义为已收盘 bar 的净 log-return 除以绝对路径长度，带方向且范围 [-1,1]；教师以绝对值作趋势强弱判断。多周期特征的原生 available-at 时间戳继续进入点时可用性校验。BC 输入 schema 加入两项，样例 host context 随之更新。
- 新增 `qa/probe_route_a_efficiency_pullback_teacher_20261021.py`。在趋势 pullback 网格规则上加入 15m/1h 效率过滤；共评估 64 个 Train 网格 episode，并对 12 个 Train 日期 × Base/Stress × seed 11/29 进行 48 个教师/基准配对。择出的方向、z-score 参数和效率阈值均写入教师配置身份。April 和其他滚动日期都是已看过的开发数据，不是独立晋级数据。
- 结果：48 配对只有 6 个教师回报为正，均值 -0.00797%，最差 -0.03984%，平均成交 2.67。教师门拒绝，原因为 `TEACHER_NONPOSITIVE_NET_RETURN`、`NONPOSITIVE_WORST_PAIR_EXCESS`；48/48 屏障匹配、无强平，96 份原生回放证据齐全。报告：`D:/Program Files/NautilusTrader_RL_ES_IL_Route_A/.artifacts/route-a-efficiency-pullback-teacher-20261021-run02/trend-pullback-report.json`。经济状态仍为 `SCENARIO_ONLY`，没有生成晋级或 actor 收据。
- 运行过程中修复了两处阻止 Python 3.12 导入的异常捕获语法（`teacher_gate.py` 与 Train runner），并改用项目 `.venv` 执行研究脚本。相关源码/runner/QA 探针 `py_compile`、Ruff 和 diff whitespace 检查通过；本轮未运行测试套件。
- 结论：因果效率特征已接通，但该过滤规则没有形成正收益教师；不应将其升为默认教师。当前仍无稳定正收益教师，也无正收益且超过教师的学生，Route A 的目标未完成。下一步需从独立 Train 机制和执行经济建模继续开发，保持筛选门 fail-closed。

## 2026-10-21 v6 原生因果成交流特征接线

- 新增独立 `causal_mtf_v6` FeatureSpec profile，在 v5 全部因果价格/状态特征上追加 5m/15m taker-buy quote-volume imbalance 和 5m log trade count。特征由已保留的官方 Binance kline 字段在可见、已闭合 1m bars 上聚合，availability 继承窗口内最新 bar；旧 snapshot 缺任一流字段时新 profile fail closed。`causal_mtf_v5` 特征与默认 identity 保持原样。
- WFO profile 校验、FeatureSpec/transition identity、Route A host observation 映射和 BC inference 映射均支持 v6。多 episode student runner 可通过 `ROUTE_A_FEATURE_PROFILE=causal_mtf_v6` 选择该 schema；默认仍为 v5。Flow 特征和 observation/action recipe 会随原有 FeatureSpec、schema、actor 和 execution receipt 哈希绑定。
- 验证：针对 9 个改动 Python 文件的 Ruff `--no-cache` 检查通过；内存 `compile()` 语法检查通过。直接 `py_compile` 因环境拒绝创建包目录 `.pyc` 临时文件失败，未修改已有 cache 后改用不落盘语法检查。未运行测试、未读取市场 payload、未训练或评估新候选。
- 边界：这是将此前离线探针中唯一有正向增量的主动流输入接入原生教师/学生共用 observation 的工程准备。既有离线探针仍在各折负收益，不能据此称为教师改进、原生证据或盈利；新 profile 尚需在授权 Train 环境做因果/availability 回归与原生教师门筛查，结果仍须 `SCENARIO_ONLY`，直到费率/规则依据和正式经济审计齐备。

## 2026-10-21 run43 学生效率特征复现消融

- 将 `market.efficiency_15m` 与 `market.efficiency_1h` 纳入 run43 的学生 observation schema；继续使用 32 个注册 Train 六小时 episode（11,200 决策）、温和意图采样及 5 分钟冷却教师。April 13/14 × Base/Stress × seed 11/29 共 8 对仅用于已看过的开发诊断。
- 训练总体意图准确率 94.79%，OPEN 召回 87.83%（实际 OPEN 标签 115 条：85 LONG、30 SHORT）。学生 8/8 回报为负，平均 -0.1672%、最差 -0.3946%、平均 16 笔成交；Apr13 平均 -0.3292%，Apr14 平均 -0.0052%。教师 8/8 为正，平均 +0.2092%。学生没有一对超过教师。8/8 原生屏障匹配且无学生强平。
- 教师门在这组复用的 April 开发矩阵通过；StudentScreen 因 `STUDENT_NONPOSITIVE_NET_RETURN` 与 `STUDENT_DOES_NOT_BEAT_TEACHER_IN_EVERY_PAIR` 拒绝，无 actor 导出。完整哈希报告：`.artifacts/route-a-multiepisode-student-screen-20261020-run43/screen-report.json`（SHA-256 `940446d8265d9afef3e0f142dd21de13ab8a7c76323e0223fd7a1938e89c8573`）。经济状态仍为 `SCENARIO_ONLY`。
- 该消融未改善学生，且日期已用于开发；不得作为独立晋级证据。持续目标未完成。

## 2026-10-21 run44 教师请求监督整改

- run43 原生证据显示：学生预测 OPEN_LONG 时大量请求仓位仅约 0.01，planner 以数量舍入/死区拒绝；这源于 BC 暴露头的回归权重受 native fill/effective-position 权重控制，已核验但未成交的教师命令只监督意图，不监督请求仓位大小。
- 新增 recipe 字段 `exposure_target_mode`：新训练显式用 `teacher_request_v2`，通过 `LabeledStep` 中已对齐 teacher advice 与 native requested_action 的请求目标进行暴露回归监督；该标签表达策略命令，不表示成交。执行事实与实际仓位继续留在 native evidence。旧配置解析时默认 `executed_only_v1`，保持旧 recipe 身份。配置字段进入训练 recipe hash；缺少对应监督权重的数据集 fail closed。
- run44 使用相同 32 个注册 Train 六小时拟合窗、11,200 决策、5 分钟冷却教师、温和意图采样和 April 评估矩阵。OPEN 请求平均仓位从 run43 的约 0.01 提升到约 0.58，说明命令尺度修复生效。8 对中 4 对为正、均值 -0.0258%、最差 -0.4213%、平均 103.1 笔成交；教师 8/8 为正、均值 +0.2092%。学生 0/8 超过教师，StudentScreen 仍以净收益非正和未逐对超过教师拒绝，无 actor 导出。报告：`.artifacts/route-a-multiepisode-student-screen-20261020-run44/screen-report.json`，SHA-256 `9218259e79f2daf11c6c078f3ce492f49035ccbee9887e72b9bfa16fb6ab5cf1`。经济状态为 `SCENARIO_ONLY`；该 April 矩阵已用于开发。
- 源码、旧配置迁移和 runner 经 `py_compile`、Ruff、diff whitespace 检查通过；本轮未运行测试套件。后续要解决意图过采样造成的虚警/高换手，同时仍以独立 Train 窗口验证教师质量、学生正收益及逐对超越。

## 2026-10-21 run45/46 意图抽样与重入冷却消融

- run45 在 run44 的 `teacher_request_v2` 上仅将 sampler 从类别频次 -0.75 次幂改为 -0.5 次幂。April 开发矩阵 8 对中学生 4 对正、2 对超过教师，均值 -0.01234%、最差 -0.4575%、平均 34.4 笔成交；教师 8/8 正、均值 +0.2092%。总成交接近教师（275 vs 272），但学生仍有亏损配对且没过 StudentScreen。学生训练 OPEN 召回 78.26%，预测数量仍高于标签（213 vs 115）。报告：`.artifacts/route-a-multiepisode-student-screen-20261020-run45/screen-report.json`，SHA-256 `985986e4bcbd3149400246befbe9eb66258e1d6f92ab81e9392a44c199d2b0d3`。
- run46 增加 5 分钟平仓后重入冷却并写入 BC recipe identity。其 8 对中学生 4 对正、3 对超过教师，但均值 -0.05692%、最差 -0.64858%、平均成交 51.5 笔，未过门。该冷却每次从有仓转空仓都会触发，不能区分教师的止损平仓与一般平仓，已判定不适合作为生产策略并移除该代码选项。报告：`.artifacts/route-a-multiepisode-student-screen-20261020-run46/screen-report.json`，SHA-256 `b131b9cb9f2758af3dd7a9adb365339e73831ab321811378f2eca68600498e0b`。
- 证据指向下一步：保留 `teacher_request_v2` 与较温和的平方根 intent sampler 作为当前较好训练候选（仍未通过门）；优先改善教师对不同行情的方向/退出判断，并降低少数类虚警。April 13/14 已被反复用于开发，run44–46 不能作为独立晋级证据。

## 2026-10-21 run48/49 时间后置验证消融

- 为控制 32 个 episode 直接拟合后可能的过拟合，runner 新增 `ROUTE_A_INNER_VALIDATION_DATES`：只允许从冻结 Train 拟合日中选择严格后置日期；Scaler 只在训练日期的时间范围拟合，训练/验证的 GRU history 通过独立 episode 边界隔离，`train_bc()` 根据 Train-inner 验证损失 early-stop。训练窗、验证窗、Scaler 区间和 BC recipe 均写入报告。
- run48 以 2024-03-15、03-22 作 2,800 决策的后置验证；前 24 个 episode 共 8,400 决策拟合。20 轮中验证损失最低于第 15 轮。April 8 配对平均 +0.0269%、最差 -0.3512%、每对 3 笔成交；学生 4/8 正、2/8 超过教师。教师均值 +0.2092%、8/8 正。StudentScreen 因非正配对和未逐对超过教师而拒绝。报告：`.artifacts/route-a-multiepisode-student-screen-20261020-run48/screen-report.json`，SHA-256 `b437ecc615e652e15ffd22d0e18d0ab096c2ca041e7e2c8b8365fcb9de8c1960`。
- run49 只用 2024-03-22（1,400 决策）验证，前 28 个 episode 共 9,800 决策拟合，40 轮上限、patience 8。验证损失在第 9 轮最低；April 8 个学生配对都没有成交、净收益为零，正式 StudentScreen 以 `STUDENT_HAS_NO_NATIVE_FILLS` 及收益/超越失败拒绝。仅一个后置日期的验证触发了不交易策略，说明单日验证损失不能替代策略矩阵。报告：`.artifacts/route-a-multiepisode-student-screen-20261020-run49/screen-report.json`，SHA-256 `d04ae9455a5318ac9f88f80f9128b4e2774b4416690acf3d740043de2010cb0a`。
- run47 在训练窗选择前因验证日期格式和窗口 ID 格式不一致，于数据读取前失败；没有实验结果。修正后另用 run48 输出，保留空的 run47 目录。
- runner/训练器改动经 `py_compile`、Ruff、diff whitespace 检查；run43/44/45/46/48/49 报告 SHA-256 sidecar 与文件重算值一致。本轮未运行测试套件。验证日期和 April 矩阵都已参与开发，不能当独立晋级证据；目标仍未达成。

## 2026-10-21 run51 新窗口结果

使用此前冻结的 run50 训练与止损冷却参数，在首看 Train 日期 2024-04-15 的两个非重叠 12 小时窗口复核。Teacher gate 因 8 对全负收益被拒；Student 虽逐对少亏，仍只有 2/8 为正、平均 -0.1091%、最差 -0.2732%，因此不具备晋级资格且没有 actor 导出。seed 11/29 的结果在各窗口/场景中重复，实际只有 4 个不同窗口 × 场景观察。报告 SHA-256 `479ea8f9f66d695dd2ab9dd96897c3d52f25f9ce5156919cd351e93e7957d7fd`，完整报告位于 `.artifacts/route-a-multiepisode-student-screen-20261020-run51/screen-report.json`。结果仍为 `SCENARIO_ONLY`。本次 Route A runner 固定读取 `binance_ethusdt_20231230_20240415`（数据末端约 2024-04-16 00:01 UTC）。仓库还有后续行情分片及 April–August WFO union，既有 WFO 已消费其中部分 Validation/Test/Holdout。使用前必须按宿主注册账本核实各时间区间角色及消费记录，只读取明确获准的 Train；不能把后续分片整体视为未触碰数据，也不能复用 run51 日期作独立确认。

## 2026-10-21 Train-fold扩展（run55）

既有 WFO manifest 将 2024-04-16 至 2024-06-15 00:00 UTC 记为 Train score 区间。基于校验过的原始 Binance archives 重建物理隔离快照 `data/routea_train_20231230_20240531/`，只含 2024-05-31 之前的数据；更早部分作因果历史预热。run55 使用 32 个六小时拟合窗、11,200 条决策，在 5/20 与 5/27 两个 12 小时 Train 窗口做 Base/Stress × seed 11/29 对照。8/8 屏障匹配、无清算。Teacher 4/8 正、平均 -0.1291%，拒绝；Student 0/8 正、平均 -0.0841%、最差 -0.1006%，只在教师亏损的 4/8 配对少亏，未取得 gate 资格、无 actor 导出。8 行结果只有 4 个不同窗口 × 场景；May20 教师为正而学生亏损，May27 学生较少暴露、亏损低于教师。报告 SHA-256 `9790dd1ae3a4b7d7f3eb5ef6bb9e2f472d4009a6c47f76e119b3420e6e6508df`，完整工件 `.artifacts/route-a-multiepisode-student-screen-20261021-run55/screen-report.json`。收益仍为 `SCENARIO_ONLY`；这些日期先前用于其他 WFO 的 Train 训练，不属于全项目独立新样本。

原生轨迹揭示学习缺口集中在持仓持续与换手：May20 学生 172–173 持仓 bars、15–16 fills，教师 293 bars、4 fills；May27 学生 19–59 bars、15–27 fills，教师 568–598 bars、59–76 fills。训练准确率 97.20% 仍未教会学生保持高质量教师持仓；后续先改进状态/持有期监督，再重新评估。Route A 目标仍未完成。

## 2026-10-21 run59：扩展 Train source 与六月窗口复核

- 新增 checksum-verified Train source `data/routea_train_20231230_20240614/`：726,765 events、511 个官方来源档案；只保留事件时间严格早于 2024-06-15 00:00 UTC 的数据。marker SHA `3d9f75e99ff89f77f5015d6d0b14dd8885c28cb4cb3f84d1d66369734960ff7f`，import SHA `d48044a64ce0921969bbc69dd47e0513b954b42ca8b2efb7cc381756129187d1`。6/3、6/10 窗口仅属此前冻结 Train score interval 的内部开发样本。
- run59 复用 run58 的 32 个 Train 拟合 episode/11,200 决策、DAgger 11,200 状态、70% 教师/30% DAgger 微调，在两日 × Base/Stress × seed 11/29 下做 8 组原生对照。8/8 barriers 匹配、无清算。Teacher 0/8 正、均值 -0.02435%、最差 -0.06306%，Teacher gate 拒绝；Student 4/8 正但均值 -0.07388%、最差 -0.22332%，184 fills 对 Teacher 40 fills。seed 重复，因此仅 4 个独立窗口×场景结果。Student gate 不具资格，无 actor 导出；经济状态 `SCENARIO_ONLY`。
- 报告 SHA-256 `6eb653e0e69cb0a35b9bc572fe89a9f95a6f55d4c2fc3e1dffab2f1fc9eac99e`：`.artifacts/route-a-multiepisode-student-screen-20261021-run59/screen-report.json`。run59 加上 run58 的结果没有改变结论：DAgger 增加持仓覆盖，但盈利仍不稳定且换手高；下一阶段应优先改善教师选择性和成本约束，而非继续放大 DAgger 样本量。

### 2026-10-21 顺势与逆势方向探针

方向对照在 6/4、6/7、6/12、6/14 四个 Train-inner 窗口进行。顺势方向 16 对全非正，均值 -0.01754%；逆势方向 4/16 正，均值 +0.00914%，且全部盈利集中在 6/14，其他三日无交易。两种方向 TeacherScreen 均拒绝；因此方向翻转不构成教师整改完成或晋级依据。完整报告：`.artifacts/route-a-trainfold-trend-direction-20261021-run01/trend-direction-report.json`，SHA-256 `051220a8364214f8429867731bf8bd82561757a58d708c7f07279749ebf7d323`。经济状态 `SCENARIO_ONLY`。

### 2026-10-21 均值回归教师探针

6/5、6/13 两日对 4 组既有 z-score/趋势阈值进行 Base/Stress 原生筛查；每组 8 对均没有正收益，最佳均值 -0.02447%、最差 -0.03629%，最高换手组合均值 -0.12147%、472 fills。各组均被 TeacherScreen 拒绝，8/8 barrier 匹配、无强平，seeds 在同窗重复，经济结论为 `SCENARIO_ONLY`。报告：`.artifacts/route-a-trainfold-mean-reversion-20261021-run01/mean-reversion-report.json`，SHA-256 `9e02a5800c0190045f77b2a6eed2641b785c1ebd3a226c9bfeefc8c0ca601142`。趋势方向和均值回归规则都未找到稳定教师；下一阶段以严格因果、费用后预测的 teacher 为研究方向。

## 2026-10-21 因果净成本预测教师原型

新增 `RidgeForecastTeacher` 和 Train-only 60m Ridge fit/native replay。模型仅使用因果 `causal_mtf_v5` 六特征，5m 采样、8,340 个 April–May Train 标签、June 4 个后置 Train 窗口原生回放；30bps Stress round-trip hurdle、15bps exit hurdle。模型拟合分数标准差 4.11bps，拟合期达到 30bps 的样本占比仅 0.024%，命中样本 gross 平均 28.74bps，低于门槛。June 16 组原生配对全部没有成交，因此教师门以 `TEACHER_HAS_NO_NATIVE_FILLS` 等原因拒绝；无 actor 导出，`SCENARIO_ONLY`。报告 SHA-256 `4f0e1ab96ae9eaf7ec61e9fa2e6f1b3a7c4a04acc77018cd0421f7a784981656`：`.artifacts/route-a-forecast-teacher-20261021-run01/forecast-teacher-report.json`。成本门控正确阻止低置信命令，但该模型不是有 alpha 的教师，Route A 目标仍未完成。

### 因果预测教师 240m horizon 消融（run02）

把 60m Ridge label horizon 改为 240m 后，训练拟合输出变大，但 June 原生 screen 16 对全部亏损，均值 -0.01051%、最差 -0.02884%、16 fills；8 个独立窗口×情景 cells、seed 重复；barriers 匹配、无清算。TeacherScreen 拒绝、无 actor 导出，经济状态 `SCENARIO_ONLY`。训练期入选样本 gross 均值不能证明泛化。报告 SHA-256 `0f886cc5b8f03317de048066e2a55c7e3e2948dc0627f009f78cbd0beb92e5b8`：`.artifacts/route-a-forecast-teacher-20261021-run02/forecast-teacher-report.json`。结论：当前因果 ridge 教师过拟合，需新预测输入/训练协议。


### 因果预测教师扩展 v6 特征消融（run03）

- 在不改动冻结的 `causal_mtf_v5` 宿主 FeatureSpec 的前提下，新增显式 `CAUSAL_MTF_V6_ROUTE_A_FEATURES` 投影：v5 六项之外加入 1/5/15/60m returns、5/15/60m volatility、5m volume、basis、last-settled funding，共 16 项；默认 v5 映射及既有 bundle 输入身份保持不变。仅用 4/16–5/15 Train 数据拟合 240m forward label，5m 抽样并 purge 标签终点，共 8,304 样本；Native replay 仍在 6/4、6/7、6/12、6/14 × Base/Stress × seed 11/29。
- 训练期诊断：预测标准差 19.34 bps；+30/−30 bps 入选率 4.17%/5.48%；入选 long/short 标签 gross 均值 +89.08/−24.43 bps。均属拟合诊断，不能作为泛化证据。
- 原生 16 对只有 2/16 净收益为正，均值 −0.01645%，最差 −0.02500%，28 fills；只有 8 个独立窗口×场景 cell，两个 seed 的经济结果重复。屏障全匹配、无强平。TeacherScreen 以 `TEACHER_NONPOSITIVE_NET_RETURN`、`NONPOSITIVE_WORST_PAIR_EXCESS` 拒绝；没有启动学生训练或导出 actor，经济状态 `SCENARIO_ONLY`。报告 SHA-256 `f46429fb856fbd388b8023a2bf9e7836964b3487f178df55c18fb0c746dcbf5c`，sidecar 一致；`.artifacts/route-a-forecast-teacher-20261021-run03/forecast-teacher-report.json`。
- 结论：扩展因果输入提升了拟合分数幅度，但未改善原生筛查结果，不能晋级。六月窗口已作为 Train-inner 开发数据消费；后续需要滚动样本外校准与交易选择机制。


### Ridge 教师时间顺序验证与低频候选原生复核（run04）

- 新增 `qa/probe_route_a_forecast_walkforward_20261021.py`：在 4/16–5/15 的四个递增时间折上，以 240m label、240m 非重叠采样、训练 label 终点不晚于验证起点，比较 causal v5 六特征/v6 十六特征与 Ridge alpha 0.1/1/10/100。非重叠时序样本每折约 23–29 条验证、59–149 条训练。v6 各档未形成稳定费用后入选；v5 alpha=1 在 4 折仅 2 笔入选（fold02 long +93.69 bps、fold04 short +32.73 bps after 30 bps hurdle），另外两折零交易。该结果仅两例，且全属已消费 Train-inner 数据，不能证明稳定优势。报告 SHA-256 `1fe57d8ae9d4652cfd7e2188eb13ee8fb9633f9e866a115b9e180b09a14f6d7a`；`.artifacts/route-a-forecast-walkforward-20261021-run02/forecast-walkforward-report.json`。
- 将 v5 alpha=1、240m 非重叠拟合配方（173 个 Train 样本）带入 June 4/7/12/14 原生 replay。尽管拟合样本仅 3/173 超过成本门，分钟级原生预测仍造成 96 fills；16 对全部非正，均值 −0.01470%，最差 −0.05737%。8 个窗口×场景 cell、双 seed 结果重复；barriers 全匹配、无强平。TeacherScreen 拒绝（`TEACHER_NONPOSITIVE_NET_RETURN`、`NONPOSITIVE_WORST_PAIR_EXCESS`），无学生训练或 actor 导出，`SCENARIO_ONLY`。报告 SHA-256 `30dadaa02c66238f48fac2d698deb4026dc1e86f3ff135aad8d869b3fe224f2b`，sidecar 一致；`.artifacts/route-a-forecast-teacher-20261021-run04/forecast-teacher-report.json`。
- 结论：重叠标签确实夸大了样本数，但去重叠的稀疏参数仍未通过原生执行；频繁重新评估与仓位进出进一步放大成本。下一步验证与 240m 标签频率一致的 4 小时决策/持仓锁定策略，之后仍须过相同原生门。


### 四小时决策节流原生诊断（run05）

- 在 run04 同一个 v5 六特征、240m 标签、240m 非重叠样本、alpha=1、173 条 Train 样本 Ridge 上，增加 240m UTC 对齐的仓位决策间隔：只在四小时边界重新评估预测，其余决策保持当前原生仓位。该设置让模型决策节奏与训练标签/验证采样一致，入口/退出成本门没有变化。
- June 4/7/12/14 × Base/Stress × seed 11/29 的 16 组原生回放全部屏障匹配、无强平，但零成交；TeacherScreen 因 `TEACHER_HAS_NO_NATIVE_FILLS`、非正收益及最差配对原因拒绝。结果为 `SCENARIO_ONLY`，未训练学生或导出 actor。报告 SHA-256 `24ff789cf43a0b84ddcb682be90a9ffb57986afbb4c1fca082ae6d0f0a17a74a`，sidecar 一致；`.artifacts/route-a-forecast-teacher-20261021-run05/forecast-teacher-report.json`。
- 这说明分钟级 run04 的交易不是四小时采样预测信号可持续兑现的机会；按期限降低决策频率能消除重复交易，却同时暴露出 30 bps 入口门下没有有效触发。预测教师仍未达到 StudentScreen 前置条件。


### 60 分钟非重叠 Ridge 时间顺序验证（run06）

- 对同一 4/16–5/15 训练区间与四个递增时间折，将 60m forward-return 标签按 60m 非重叠采样，比较 v5/v6 特征和 alpha 0.1/1/10/100；训练标签终点均早于验证起点。所有设置都没有跨折稳定费用后入选：v5 alpha=0.1 仅 2 笔，费用后均值 −74.41 bps；v6 alpha=0.1 10 笔，均值 −61.44 bps；v6 alpha=1 一笔 −53.80 bps；其余参数无交易。无任何设置在含交易的验证折中费用后为正。报告 SHA-256 `d8f170095401d5e3bde55fd161da1bdcb8ff36ea8a62d46175e55485411ff5e0`，sidecar 一致；`.artifacts/route-a-forecast-walkforward-20261021-run03/forecast-walkforward-report.json`。
- 汇总 v5/v6、60m/240m 非重叠期限、四档正则及 run01–05 原生结果：当前 Ridge 输入/目标/正则范围均未找到可供学生学习的正收益教师。暂时停止 Ridge 调参，转向审计中已验证的原生教师/学生交易频率与持仓标签缺口；Ridge 结果均为开发诊断，June 日期保持已消费状态，`SCENARIO_ONLY`。


60m 非重叠诊断已固化可重复入口 `qa/probe_route_a_forecast_walkforward_60m_20261021.py`；通过独立 run04 重跑后，聚合结果与 run03 一致，报告 SHA-256 同为 `d8f170095401d5e3bde55fd161da1bdcb8ff36ea8a62d46175e55485411ff5e0`（sidecar 一致）。


## 2026-10-21 student 最短持仓约束诊断（run60/61）

- 在 `BCConfig` 增加 `minimum_position_hold_ns`（默认 0，旧配置解析补默认）；`BCInferencePolicy` 跟踪原生非零仓位起始时间，最短持仓期内将输出限制为当前实际仓位。当前仓位的 stop-loss 和 stop-cooldown 仍优先覆盖并立即输出平仓。该字段随 BC config 与 student recipe 序列化/哈希。Train-fold runner 用 `ROUTE_A_MINIMUM_POSITION_HOLD_NS` 固定实验值。静态检查 `py_compile`、Ruff 通过；本轮未运行测试套件。
- 比较组 run59（0 hold）学生 4/8 配对为正但均值 −0.07388%、最差 −0.22332%、184 fills；8 行结果只有 4 个窗口×场景 cells，seeds 重复。run60（15m）学生 0/8 为正，均值 −0.01610%、最差 −0.03954%、20 fills；run61（5m）学生 0/8 为正，均值 −0.02236%、最差 −0.06134%、36 fills。两约束版均显著降低 Jun03 的 student 换手和损失，但也未在 Jun10 发出原本 run59 的盈利交易；5m 比 15m 更活跃但亏损更大。StudentScreen 未取得教师门资格，未导出 actor。
- 教师保持 run59 原样：0/8 正收益、均值 −0.02435%、最差 −0.06306%、40 fills；TeacherScreen 仍拒绝。run60 报告 SHA-256 `dee7f1e26abcbdbf95963e9c1d779824853e9efd014a4fbad2ff2877a767363a`，run61 `0b9df337f4f5aea8b99cc9ab25bb86f153dc9194fdeb8f096e24630e98fdf862`，sidecar 一致；工件分别为 `.artifacts/route-a-multiepisode-student-screen-20261021-run60/screen-report.json` 与 `...run61/screen-report.json`。
- 结论：最短持仓是有效的换手/尾部风险约束，但它压制了正确开仓和退出之前的方向识别问题，没有解决教师无 alpha，也没有使 student 稳定正收益。结果均为 `SCENARIO_ONLY`，June Train-inner 样本已消费；Route A 目标未完成。


## 2026-10-21 OPEN 意图置信门控诊断（run62）

- `BCConfig` 新增 `open_intent_min_confidence`（默认 0、解析旧 bundle 时补默认；recipe 哈希包含此参数）。`BCInferencePolicy` 在 intent-aware 推理中，只对 OPEN_LONG/OPEN_SHORT 使用 argmax 类别概率门控；低于门槛时保持当前仓位，已持仓的 HOLD/REDUCE/EXIT 仍按模型语义，原生 stop/cooldown 优先。学生诊断增加预测 OPEN 置信度分位数及 0.5–0.9 阈值下过滤计数。配置设 0.8，stop-cooldown 教师、32 个 4/16–5/14 拟合窗口、11,200 DAgger 访问状态、70/30 混合采样均沿用 run59/60。代码 `py_compile` 与 Ruff 通过；未运行测试。
- 训练集 1,691 次预测 OPEN 的置信度均值 0.959、中位数 0.984；91 次（5.4%）低于 0.8，训练 OPEN recall 93.17%。六月原生回放 16 对均零交易、零收益；teacher 保持 0/8 正、均值 −0.02435%、最差 −0.06306%，TeacherScreen 拒绝。Student 无交易且无资格，未导出 actor；经济状态 `SCENARIO_ONLY`。报告 SHA-256 `a905eaaf5f88e161fbbb70786fa8d51c1ef4dcb855793cc34557d590a866eae1`，sidecar 一致；`.artifacts/route-a-multiepisode-student-screen-20261021-run62/screen-report.json`。
- 门控阻止了 run59 在 Jun03 的 84 次高损失成交，但也过滤 Jun10 四个配对中的盈利行为。训练置信度很高却在 June 全部选择空仓，说明此单一全局阈值只产生 OOD abstention，没有解决概率校准或方向泛化。将 0.8 候选拒绝；下一步先用严格后置 Train 时间块校准多类概率/OPEN 决策，再做原生配对筛选，不能提高阈值来把“不交易”当成成功。


### OPEN 置信门槛对照（run63，0.6）

使用与 run62 相同的 Train/DAgger/原生流程，仅将 OPEN 最低置信度从 0.8 降至 0.6。训练期 predicted OPEN 数 2,045、均值置信度 0.915，74/2,045 低于 0.6；June Student 恢复 4/8 正收益，但平均 −0.08417%、最差 −0.26585%、168 fills，Jun03 两场景损失扩大，Jun10 两场景虽盈利仍不足抵消。对照 run59 无门槛的均值 −0.07388%、最差 −0.22332%、184 fills；run62 0.8 全部零交易。三个学生均未具备教师晋级门资格，未导出 actor。报告 SHA-256 `85a4a02e6ea1f455dd37093a7bda42978a888706a3e999e89b6c789130758c3f`，sidecar 一致；`.artifacts/route-a-multiepisode-student-screen-20261021-run63/screen-report.json`。结论：全局 softmax 截断在 0.6–0.8 之间只是在灾难性亏损与不交易之间折衷，没有净收益改善；停止阈值扫参，改用后置 Train 时间块做概率校准/拒绝决策验证。


## 2026-10-21 后置 Train OPEN 校准尝试与 run65

- 新增 `intent_temperature` 推理温度字段（旧配置默认 1.0），OPEN argmax 概率按温度缩放。训练/DAgger 与较晚 Train 校准块分离；校准流程对 teacher OPEN 标签过少不再异常退出，而是显式记录 `insufficient_teacher_open_labels` 并不拟合温度/阈值，保留默认值。该回退保证筛选报告完整，不代表概率已校准。静态 `py_compile`、Ruff 与 `git diff --check` 通过；未运行测试套件。
- run64 原在 2024-05-14 校准处异常终止，原因为教师 OPEN 标签少于 5，且没有生成完整 screen report。修正错误处理后 run65 完成同配置重跑：训练 28 个窗口/19,600 决策，DAgger 9,800 状态；校准 1,400 决策中教师 OPEN 为 0，模型却预测 OPEN 20 次，故温度维持 1.0、置信门槛维持 0.0，未进行拟合。训练集虽有 1,780 个 OPEN、意图准确率 96.77%，但后置块零个 OPEN 表明可交易正类在相邻 Train 时间段极稀疏/分布变化，无法用该块估计可靠门槛。
- run65 June Train-inner 回放 8 对（两窗口 × Base/Stress × seed 11/29；同一窗口/场景双 seed 经济结果相同）：Teacher 0/8 正收益，均值 −0.02435%、最差 −0.06306%、40 fills，TeacherScreen 以 `TEACHER_NONPOSITIVE_NET_RETURN` 与 `NONPOSITIVE_WORST_PAIR_EXCESS` 拒绝；Student 0/8 正收益，均值 −0.09080%、最差 −0.22528%、120 fills、零强平。学生在 Jun03 Base/Stress 分别亏损 −0.13793%/−0.22528% 并成交 28/32 单；Jun10 四对均零交易，错过了教师的亏损持仓。学生 screen 不计算晋级资格（`NOT_ELIGIBLE_TEACHER_GATE_REJECTED`），actor 未导出；经济状态 `SCENARIO_ONLY`。
- 报告 SHA-256 `973e5e55ae0dd4f96ef4d0d88b7ec1c41376917945e3878c13a7b021cbfe6131`，sidecar 重算一致；完整产物 `.artifacts/route-a-multiepisode-student-screen-20261021-run65/screen-report.json`。所有 June 日期仍是已消费的 Train-inner 开发数据，不能充当独立 OOS 验证。
- 结论：这轮没有产生可晋级策略，也没有改善收益；它确定了失败链的两个位置：教师自身经济门未通过，以及后置校准块没有正类，学生在 Jun03 错误高频交易而 Jun10 abstain。停止对当前教师作温度/置信门扫参；Route A 的目标仍未完成，需要先找到能在连续时间块产生足量、费用后有收益 OPEN 标签的教师，再训练学生并保留真正未消费的 OOS 窗口。


### run65 后续时序稀疏度核对

从已校验的 fit 原生轨迹按日期计数，28 个模型训练窗口共有 80 个教师 OPEN 动作；May14 四个后置校准窗口为 0。训练窗各日期 OPEN 数为 Apr16 34、Apr20 0、Apr24 7、Apr28 7、May02 13、May06 12、May10 7。教师训练期经济结果也随时间块明显变化：Apr24 的 4 窗口均值 +0.05456%、Apr28 +0.01347%，但 May02 −0.05538%、May06 −0.00507%、May10 −0.01780%；这些是拟合/开发窗统计。后续若继续验证概率校准，应让训练截止早于一段连续、含足够 OPEN 的校准窗，并事先固定日期；不能只靠单日 May14 的零事件，也不应继续扫描置信阈值来补救教师的时间漂移。


## 2026-10-21 因果预测教师非线性与原生调度复核（run06–08）

- 新增 `qa/probe_route_a_nonlinear_forecast_teacher_20261021.py`：只读取冻结 Train 源，使用 causal MTF v6 的 16 项特征；以 240m 前瞻收益作监督，5m 训练采样、每 240m 一个互不重叠的评估锚点、训练 label 终点不晚于评估起点。固定两隐层 MLP（32/16）、seed 31、60 epochs、Huber/SmoothL1 损失和 30 bps 情景门槛，对照 v6 Ridge alpha=1、同周期动量/反转。4 折总计 110 个非重叠评估点：MLP 40 次入选、费用门后均值 −16.13 bps、仅 1/4 折均值为正；动量 79 次、−17.38 bps，反转 −42.62 bps。Ridge alpha=1 仅 7 次入选、加权 +10.86 bps，但仅 2/4 折出现交易，另两折分别为亏损与零交易。该小样本 Ridge 发现值得作机制诊断，不足以证明稳定预测力，也未经过原生下单。报告 SHA-256 `e2ef66cd9d987963936f37202b4844386c176c97d03d63e8f446f361a1cd9725`，sidecar 一致；`.artifacts/route-a-nonlinear-forecast-teacher-20261021-run01/nonlinear-forecast-teacher-report.json`。
- 原生审查发现 `RidgeForecastTeacher` 的 240m 调度按 `decision_ts_ns % interval_ns == 0` 判断，但市场决策时戳是整分钟后 1ms；run06 的 700/700 步都不可能满足条件，故其零成交是调度缺陷造成，不能解释为模型未越过成本门。修正为按 UTC 分钟桶对齐，并将非零间隔规则的 expert ID 版本提升为 v4。`py_compile`、Ruff、`git diff --check` 通过；随后 run07 原生行为确认修复后模型可发出订单。
- run07 将 v6 Ridge alpha=1、240m horizon/decision、5m 训练样本与 30 bps hurdle 带入 6/4、6/7、6/12、6/14 × Base/Stress × 双 seed。16 对中仅 4 笔成交，均发生在 6/7（两个 seed 结果重复）：Base −0.01877%、Stress −0.02235%；0/16 正收益、零强平、所有屏障匹配。TeacherScreen 拒绝（`TEACHER_NONPOSITIVE_NET_RETURN`、`NONPOSITIVE_WORST_PAIR_EXCESS`），`SCENARIO_ONLY`。报告 SHA-256 `0869989980da085e9454e11b762ce0fa182f4dd68d3806309736514d79f1de33`，sidecar 一致；`.artifacts/route-a-forecast-teacher-20261021-run07/forecast-teacher-report.json`。
- 旧 run05 的零成交不能证明 4h 限频有效；以修复后的规则、其原 v5/alpha=1/240m 样本与决策频率完整复跑为 run08。16 对零成交、零收益、零强平且屏障全匹配，教师门仍拒绝 `TEACHER_HAS_NO_NATIVE_FILLS`。因此正确结论是：该冻结 June 窗口上没有可执行的越门信号，不是已证明限频消除了有 alpha 的重复交易。报告 SHA-256 `dbdebafe7cb9f82c8042ad63a0a569f27092e238e302b4c9948912a20272cf2a`，sidecar 一致；`.artifacts/route-a-forecast-teacher-20261021-run08/forecast-teacher-report.json`。
- 总结：MLP 非线性拟合没有超过 Ridge/成本门，Ridge 的少量正折收益在原生回放中没有转为正收益教师。修复了一个会抹掉限频策略全部决策的执行时序错误，但目前无合格教师；不要继续学生拟合或阈值搜参。下一步需要改进预测目标/教师案例筛选并在严格时间折中证明足量费用后正例，再原生筛查；当前 June 开发样本继续标记已消费，经济结果保持 `SCENARIO_ONLY`，项目目标仍未完成。


## 2026-10-21 高质量前瞻事件标签分类诊断

- 新增 `qa/probe_route_a_event_classifier_teacher_20261021.py`：按预先固定规则生成离线案例标签：4 小时 future close log-return ≥ +60 bps 标为 OPEN_LONG，≤ −60 bps 标为 OPEN_SHORT，其余 FLAT；因果 v6 特征、5 分钟训练采样、训练 label 终点严格早于 4 小时不重叠的评估锚点。两隐层 32/16 classifier、逆平方根类频率权重、seed 31、60 epochs；固定 OPEN 概率阈值 0.60，评估净值每笔扣 30 bps 情景成本。
- 四个递增时间折共有 110 个独立评估锚点；分类准确率约 34.5%–52.2%。固定阈值选出 14 个候选，费用后加权平均 −42.01 bps，仅 1/4 折平均为正（fold03 的单笔空头 +91.16 bps）；fold01、02、04 均为负。该事件标签 classifier 没有提供可用的高质量、可预测交易案例，不进入原生教师或学生训练。
- 报告 SHA-256 `269d6e449a48a9c402cf98091079a88bf48ac16da3cfbbb82dbcc76fa0795214`，sidecar 一致；完整结果 `.artifacts/route-a-event-classifier-teacher-20261021-run01/event-classifier-teacher-report.json`。`py_compile`、Ruff、`git diff --check` 通过；未运行测试套件。
- 结论：把高质量标签明确为超过费用门的前瞻事件，仍未解决因果特征对事件的泛化预测。当前已实测的线性预测、非线性收益回归、事件分类与规则教师都未形成稳定正收益教师。保留失败结果，不据单折单笔正收益推进模型或下调门槛；Route A 经济目标继续未完成，且外部真实费率/保证金和独立 OOS 认证仍未具备。


## 2026-10-21 时序事件教师上下文长度消融

- 对 4 小时、±60 bps 的同一事件标签及 30 bps 情景成本门，固定 GRU（hidden 32、seed 31、40 epochs）、5 分钟历史采样、0.60 OPEN 概率阈值、因果 v6 输入和同一 purged 四折，只比较历史长度：60 分钟（12 个快照）与 240 分钟（48 个快照）。训练标签严格截止于验证起点，评估锚点每 4 小时一个。
- 60 分钟上下文选出 57 个交易候选，费用后加权均值 −14.64 bps，2/4 折为正；逐折均值 +0.52、−35.52、+17.56、−53.70 bps。240 分钟上下文选出 52 个候选，均值 −57.08 bps，0/4 折为正；逐折均值 −132.73、−22.03、−27.94、−50.46 bps。序列上下文略优于点状事件分类（−42.01 bps），但没有稳定净优势；延长上下文反而恶化。
- run01 报告 SHA-256 `56d06acb7c9b6e6210767f9bada47fe38edf72d4f9e4627bef38977ebf5e9d41`；run02 `75b0d9a36eed6f0ad281633730314cda2d8c53ebd003b1c77bd18d529aeaf0b7`，sidecar 均一致。完整产物分别位于 `.artifacts/route-a-temporal-event-classifier-teacher-20261021-run01/temporal-event-classifier-teacher-report.json` 与 `...run02/temporal-event-classifier-teacher-report.json`。静态 `py_compile`、Ruff、`git diff --check` 通过；没有原生回放、没有学生晋级，也未运行测试套件。
- 结论：仅扩大历史窗口不能修复费用后事件的跨折漂移；保留 60 分钟候选仅作低容量机制诊断，不以 2 个正折宣称可用教师。当前所有价格预测/事件分类教师都未满足稳定正收益和高质量案例条件，Route A 目标仍未完成。

### 嵌套训练期置信门校准（run04，修正汇总字段后复核）

- 对既有 60m GRU 事件分类器按四个递增时间折作严格内层校准：每折训练尾部 5 天选 OPEN 阈值、此前数据拟合，下一时间折只作外层评估；4h 标签 purge、外层成本门 30 bps。阈值需至少 5 个校准候选，且近似单侧 90% 均值下界大于 0。
- 四折校准各 29 个 4h 锚点；最优候选阈值的近似下界依次为 −65.71、−42.53、−97.64、−26.44 bps，全部小于 0。因此 0/4 折可选门，外层按预注册规则不交易（0 笔），不是外层收益为零或已证明无风险。固定 0.60 对照为 57 笔、费用后加权 −14.64 bps、2/4 折正，与时序 GRU run01 一致。
- 该实验否定了“仅靠训练尾段校准 OPEN 概率阈值即可让当前 GRU 成为可用教师”的当前假设；小校准样本的均值下界只是筛选规则，并非统计认证。数据仍为已消费 Train-inner，未作原生回放或学生训练，状态 `SCENARIO_ONLY`。
- 初次执行发现输出目录固定导致中断后无法重跑；随后又修复 JSON 中 NumPy bool 序列化及外层收益汇总键名。run04 是修正后的权威报告，SHA-256 `57bd571cd32a2dfc4350a431534354f79416e9c47afdd76133baf10b813041a4`，sidecar 一致；`.artifacts/route-a-nested-event-calibration-20261021-run04/nested-event-calibration-report.json`。脚本 `py_compile` 与 Ruff 通过；本次未运行测试套件。
- 后续应停止重复搜索此 GRU 的概率门，转向以可执行进出场路径和费用后结果构造 teacher 案例标签，再做严格时序验证；必须仍通过既定原生教师门后，才训练学生。

### 成本后路径标签教师诊断（close-path run02）

- 新增 `qa/probe_route_a_barrier_label_teacher_20261021.py`，用 240 分钟未来 1m close 路径构造 LONG/SHORT/FLAT 案例：分别以有利 +60 bps 和不利 30 bps 触发止盈/止损；首次触发后按该分钟实际 close 退出，未触发则按 240m close 退出，并扣除 30 bps 往返成本。只在当 side 的情景净收益为正时标记方向，否则 FLAT。训练仍用 causal v6、60m GRU、5m 训练锚点、4h 非重叠评估和 purged 四折。
- 为避免假设止损能恰好在阈值成交，run01 的精确 barrier fill 结果不作权威依据；run02 改成触发分钟的实际收盘价并复跑。阈值 0.0/0.5/0.6/0.7/0.8 下分别有 85/74/60/45/34 笔，费用后加权均值分别为 −28.85/−30.20/−28.05/−28.00/−33.28 bps；所有置信门均 0/4 折为正，分类准确率逐折约 24.1%–41.4%。提高模型置信门没有让预测交易转正。
- 报告 SHA-256 `70d7806713074cbc3298b6dfce21e14a8382179d43a571b79de2a29c6b4a3bfb`，sidecar 一致；`.artifacts/route-a-barrier-label-teacher-20261021-run02/barrier-label-teacher-report.json`。结果为 Train-inner 研究，close-only 止盈止损仍不等于原生撮合，状态 `SCENARIO_ONLY`。`py_compile`、Ruff 通过；未运行测试套件。
- 结论：将标签改为更接近可执行退出路径，并按实际触发分钟收盘计价，仍未得到费用后可预测的教师交易案例。当前 GRU/close-path 标签组合不应进入原生晋级或学生训练；下一步需先提升信息输入或重新提出有可验证机制的教师假设，之后再做独立时间折和原生复核。

### 训练尾段嵌套置信门校准（run04）

- 对既有 60m GRU 事件分类器按四个递增时间折作严格内层校准：每折训练尾部 5 天选 OPEN 阈值、此前数据拟合，下一时间折只作外层评估；4h 标签 purge、外层成本门 30 bps。阈值需至少 5 个校准候选，且近似单侧 90% 均值下界大于 0。
- 四折校准各 29 个 4h 锚点；最优候选阈值的近似下界依次为 −65.71、−42.53、−97.64、−26.44 bps，全部小于 0。因此 0/4 折可选门，外层按预设规则无交易。固定 0.60 对照为 57 笔、费用后加权 −14.64 bps、2/4 折正，与原时序 GRU 结果相符。
- 该实验否定了“仅靠训练尾段校准 OPEN 概率阈值即可让当前 GRU 成为可用教师”的当前假设；小校准样本的均值下界只是筛选规则，并非统计认证。数据仍为已消费 Train-inner，未作原生回放或学生训练，状态 `SCENARIO_ONLY`。
- run04 是修复输出 ID、NumPy bool 序列化与聚合收益字段后的权威报告，SHA-256 `57bd571cd32a2dfc4350a431534354f79416e9c47afdd76133baf10b813041a4`，sidecar 一致；`.artifacts/route-a-nested-event-calibration-20261021-run04/nested-event-calibration-report.json`。脚本 `py_compile`、Ruff 与 `git diff --check` 通过；未运行测试套件。

### OHLC 屏障标签与因果成交流增量（run03–04）

- 将路径标签从收盘路径升级为已登记 snapshot 的 1m OHLC：+60 bps 止盈、−30 bps 止损；同一根 bar 同时触发按止损优先，止损跳空按更差 open 计价，超过 240m 未触发则按期末 close 退出，并扣 30 bps 往返成本。训练和评估仍采用 4h 非重叠锚点及四个 purged Train-inner 时间折。
- 原 snapshot 的 USD-M klines 导入器此前丢弃 quote volume、trade count 和 taker-buy volume。保留四个官方 kline 字段后，使用项目已有 checksum-verified raw archive 重建独立快照 `data/routea_train_20231230_20240614_flow`；Train 截止仍为 2024-06-15，726,765 个事件、3 个边界事件裁剪，无新增 Validation/Test/Holdout 访问。snapshot hash `f684548ffa39dd8e13af6b389d3d979edaab56c3c9629d5d251003cbd16a8bef`，marker SHA-256 `b8cc7cc876a3dba727a2cd9b3c782f78b17308abaff03350919cc293839d47fb`。
- 基于同一 OHLC 标签、同一 GRU/时间折，在 16 项价格/状态特征上增加因果 `taker_buy_imbalance_5m`、`taker_buy_imbalance_15m`、`log_trade_count_5m`。新增特征将 0.0/0.5/0.6/0.7/0.8 置信门的费用后加权均值从 OHLC-only run03 的 −33.41/−35.27/−32.51/−36.59/−33.70 bps 提升至 −27.09/−26.04/−24.02/−25.96/−21.14 bps；相应交易数为 88/80/68/56/44。提升幅度约 6.3–12.6 bps，但五种门全部 0/4 折正收益，各折仍为负，不能晋级教师。
- 权威模型报告 SHA-256 `1ff27ba3b7e4f76c56d920c9096fd363442f3837a1780dfba28401ece6402ae0`，sidecar 一致；`.artifacts/route-a-barrier-label-teacher-20261021-run04/barrier-label-teacher-report.json`。`economic_status=SCENARIO_ONLY`，仍未原生回放或训练学生。数据解析及实验脚本 `py_compile`、Ruff 通过；没有运行测试套件。
- 结论：成交主动方向和活动强度带来同一 Train 内验证分数改善，是目前少数有正向增量的信号来源；其费用后交易期望仍显著为负。下一步应扩充可因果观察的主动成交信息，或优化持仓/执行机制；若继续，必须先在未消费的 Train 时间块验证，再将固定教师送入原生屏障门。所有 June 已用窗口仍不得被称为独立 OOS。

### OHLC 屏障净收益回归教师诊断（run01）

- 使用已登记 flow Train snapshot 和固定 19 项 causal 特征，沿用 240m OHLC 止盈/止损/超时路径标签；GRU 双输出直接拟合多头与空头的费用后净路径收益（Huber/SmoothL1），避免分类类权重将“盈利事件”与实际收益幅度混为一类。训练标签严格 purge 至各评估折起点之前，4h 非重叠评估锚点共 110 个。
- 预先列出的预测净值门 0/5/10/15/20 bps 下，选出 33/29/24/19/13 笔，费用后加权均值为 −16.36/−10.34/−11.25/−17.37/−32.31 bps；每个门仅 1/4 折平均为正。相较同样特征的分类器有所改善，但最佳门仍亏损，且后三折中两到三折为负，没有可用的跨折正收益教师。
- 报告 SHA-256 `b56e25fff4ac182832dfcd73d543174804006fc4267533d473a9a6e45bd0e182`，sidecar 一致；`.artifacts/route-a-barrier-value-teacher-20261021-run01/barrier-value-teacher-report.json`。该模型未原生回放、未训练学生、未导出 actor，经济状态 `SCENARIO_ONLY`；`py_compile`、Ruff 通过，没有运行测试套件。
- 结论：直接预测路径净值比事件分类更接近交易目标，且减少了低置信的亏损候选，但剩余净期望仍为负，不能降低教师晋级门或据此启动学生蒸馏。下一步应扩充时点一致的微观结构输入并在未消费 Train 时间块重新检验，必要时再走原生 Base/Stress/seed 门。

### 单仓位顺序执行复核：路径收益 GRU（run02）

- 为避免每 4 小时一个锚点过度稀疏，固定同一训练 recipe 与 4 个外层时间折，改为每 5 分钟评估一次模型；按预测净值门 0/5/10/15/20 bps 决策，仅持有一仓，模拟 OHLC 首次止盈/止损（同 bar 冲突止损优先、跳空止损按更差 open），退出后只在之后的 5m 决策重新入场。实际标签已扣 30 bps 情景往返成本；交易不重叠。
- 每折有 1,104–1,392 个候选决策锚点。五档门槛一共选出 153–393 笔，费用后加权均值为 −28.84 至 −30.97 bps；所有阈值均为 0/4 折正。每折每门收益都为负，中位数通常为 −60 bps，与“止损后付费”的损失结构一致。此前 4h 稀疏锚点的单折正收益/−10.34 bps 汇总没有复现为可连续执行的策略。
- 权威顺序执行报告 SHA-256 `0380d9af9f552a291a1187868b06fc1bf4998703b6e3206032d82ec93e68870d`，sidecar 一致；`.artifacts/route-a-barrier-value-teacher-20261021-run02/barrier-value-teacher-report.json`。该回放仍是 OHLC 标签模拟，非 Nautilus 原生撮合；成本/规则仍为情景假设，状态 `SCENARIO_ONLY`。
- 结论：直接净收益回归在连续 5m 单仓位执行约束下没有可交易优势，路径标签 GRU 不具备原生教师资格；不训练学生，也不把稀疏评估正折作为策略证据。下一步转向新的、可因果观察的微观结构信息和明确交易机制；在找到稳定正收益的顺序 Train 回放前，不做原生教师或学生晋级。

### 成本敏感性复核：run02 固定交易序列

- 从 `configs/ethusdt_20240101_real.json` 核对情景配置：Base 为 taker 手续费 5bp/边加滑点 1bp/边，共 12bp round trip；Stress 为手续费乘 2、滑点 5bp/边，共 30bp。该文件名中的 `real` 不代表这里的历史成交已被认证，结果仍属 `SCENARIO_ONLY`。
- 对 run02 每个阈值已选定的同一组交易，只把每笔成本由 30bp 改为 12bp（收益统一加 18bp），不改变预测、交易时点、方向或退出路径。五档预测门在 Base 成本敏感性下的交易加权均值仍为 −12.97/−12.21/−10.73/−12.05/−10.84bp，所有门仍为 0/4 折正；Stress 对应均值为 −30.97/−30.21/−28.73/−30.05/−28.84bp。此计算是同序列成本敏感性，不是重新训练或 Base 重新选门策略。
- 结论：亏损无法仅由 30bp 压力成本解释；即使按登记 Base 情景回算，同一条顺序交易路径仍跨折为负。当前 GRU 方案的主要问题是路径/方向选择质量，继续增加相同 recipe 的训练轮数没有已知依据；应暂停该候选，后续换数据/机制后再训练并在 Base、Stress 下重评。

### 新 Train-inner 时间块复核：屏障净收益 GRU（run03/04）

- 窗口盘点在 Route A 报告、脚本和工件索引中未发现 2024-05-28 至 2024-06-03 00:00 UTC 的既有评估记录；该块避开已列明的 5/20、5/27、6/3、6/4、6/5、6/7、6/10、6/12、6/13、6/14 窗口，并处在冻结 Train score 区间。预注册文件 `configs/route_a_barrier_value_may28_jun03_prereg_20261021.json` SHA-256 `94ac3002e689096da9411af3aea1519ffe7432c3fb0690b8daaaa2634b0f75b7`。它仍是 Train-inner 开发筛选，不是独立 OOS。
- run03 在评估特征行为空时，于布尔索引处中止，未拟合 GRU、未生成报告。修复后 run04 将因果特征工厂延伸至评估结束，标准化 scaler 仍仅由 4/16–5/15 拟合数据产生；训练路径标签终点早于评估开始。固定同一 19 项输入、GRU 32 hidden、seed 31、40 epochs、5m 决策、单仓位顺序退出；primary 预测净值门为 10bp，其他门只作预先固定的描述性诊断。
- 评估分成连续的两个 3 日子段（5/28–5/31、5/31–6/3 00:00），10bp 主门总计 104 笔，第一/第二段分别 64/40 笔。压力成本净均值分别 −31.88/−36.85bp，加权总均值 −33.79bp；同一批交易改按 Base 12bp 成本敏感性为 −15.79bp。其余固定阈值 0/5/15/20bp 的加权均值也全为负（−30.90/−32.60/−33.49/−31.74bp），两个子段均无正收益。预注册门槛未通过；没有原生撮合、学生训练或 actor 导出，状态仍为 `SCENARIO_ONLY`。
- run04 报告 SHA-256 `42b370b8633b175c331014f1685d7d1395cc3c9f196389b5426ee4ceead9cccb`，sidecar 重算一致；`.artifacts/route-a-barrier-value-teacher-20261021-run04/barrier-value-teacher-report.json`。静态 `py_compile` 与针对两个研究脚本的 Ruff 检查通过；未运行测试套件。run03 失败目录保留以便追溯，没有用它替代 run04 结果。
- 结论：同一屏障路径收益 GRU 在新 Train 时间块仍明显亏损；增加相同模型训练轮数或再扫这些阈值不构成下一步。应转向新的因果信息或可验证的入场/退出机制；当前注册 snapshot 仍没有逐笔 `aggTrades` 或 `bookTicker` 事件，任何新增市场流需先按项目研究登记和 Train scope 建立来源身份，再评估高频微观结构信号。

### 2026-10-22 案例标签边界与训练并行复核

- 最新 aggTrades GRU 诊断为 run21，目标训练损失 `0.005`、4 个独立 fold 并行、每 worker 8 个 Torch 线程；目标在四折均未达到。校准最佳 checkpoint 训练损失约 `0.0881–0.0896`，校准损失约 `0.0887–0.1001`，均因 calibration plateau 提前停止；并行拟合墙钟 30.24 秒，折级累计 CPU 时间 540.27 秒，按 32 逻辑核计算利用率约 55.8%。报告 `.artifacts/route-a-barrier-value-teacher-20261021-run21_agflow_loss005/barrier-value-teacher-report.json` 明确 `SCENARIO_ONLY`、`promotion_evidence=false`，不是原生教学案例。不能为满足更低训练损失而覆盖校准最优权重或继续追训；下一步应改善可泛化目标/输入，不是单纯增加 epoch。
- 当前最高编号学生矩阵 run65 也未证实学生能学会盈利交易：8 对中教师/学生均为 0/8 正回报；教师均值 −0.02435%、合计 40 fills，学生均值 −0.09080%、合计 120 fills，学生最低 −0.22528%，最高 0。教师门因 `TEACHER_NONPOSITIVE_NET_RETURN` 与 `NONPOSITIVE_WORST_PAIR_EXCESS` 拒绝，student gate 未执行，未导出 actor。报告 SHA-256 `973e5e55ae0dd4f96ef4d0d88b7ec1c41376917945e3878c13a7b021cbfe6131`：`.artifacts/route-a-multiepisode-student-screen-20261021-run65/screen-report.json`。它再次说明“可产生开仓和成交”不等于“学会正确交易”，当前学生不应晋级。
- 教师标签链修复：完整原生负收益往返保留诊断；仅在平仓入场点生成标为 `INFERRED_NEGATIVE_TRADE_ENTRY_AVOIDANCE` 的 FLAT 反事实监督，不改写原生敞口或其他持仓动作。`LabeledStep` 校验其负案例状态、零目标、意图监督权重与显式 reason/origin 一致；案例采样只纳入完整正向原生案例和此类显式负向入场反例。WFO 在打包前把唯一案例摘要与实际训练标签复核，缺少正向原生案例/可执行标签或摘要不一致时拒绝候选，并在原生屏障证据摘要中分别报告正、负、持平及反事实标签数量。
- 定向回归覆盖完整负往返只监督入场、案例均衡采样以及 WFO 缺少正向教师案例拒绝；完整 `packages/route_a_imitation/tests` 通过（2 项 CUDA-only skipped），相关 Ruff 通过。`tests/test_historical_factory.py::test_route_a_t1_t2_t3_teachers_run_registered_native_gym_and_m22_path` 在两个允许临时根目录均未进入业务断言，Windows 对 `rles_data.io.write_curated()` 的 `os.link(staging, target)` 返回 `PermissionError [WinError 5]`；未改动快照原子写入逻辑。该端到端合成集成仍待可创建硬链接的 runner 重试，合成通过也不替代宿主 L6/M2.2 真实权威。
- 当前尚无由真实宿主 verifier 产生并通过多窗 Base/Stress/seed 的正向原生案例，故教师与学生仍不能晋级；Train aggTrades 只证明成交流特征可用，不能证明策略订单的撮合、队列、滑点或冲击成本。学生目标也未达成。续做应接入/验证真实注册 Train 的 M2.2 FillEvent 与 L6 经济 authority，再积累独立完整案例；无此证据时继续修复可本地验证的审计/标签/学生接口缺口，但不得声称生成了可靠经济教学案例。
