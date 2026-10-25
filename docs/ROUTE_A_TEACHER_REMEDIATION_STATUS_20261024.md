# Route A 教师整改状态（2026-10-24）

> 2026-10-25 后续整改及完整原生对照见 [继续开发记录](ROUTE_A_TEACHER_CONTINUATION_20261025.md)。已修正动态退出标签、回调恢复确认、入场 cadence 与资金费时钟配置；但教师仍未取得经济资格，未晋级正式学生。更重要的是，后续原始来源审计发现整月 funding 归档在导入时物理覆盖正式 Test/Holdout：有关 8–10 月及 7–8 月来源的历史结果现均**取消教师资格用途**，只保留为工程诊断。下文“未读取/下载 Test 区间”只针对目标日期的每日价格归档，不适用于整月资金费归档；原表述不完整，在此更正。旧 240m 标签实验也不构成动态 P0 的校准证据。

> 本表与下方此前记录反映截至 2026-10-24 的状态；2026-10-25 实际 Train 研究进展见后附补充节，后者为当前状态。

## 执行边界

本轮只维护规范项目目录的 Ubuntu/WSL 版本：
`/mnt/d/Program Files/NautilusTrader_RL_ES_IL_Route_A`，Python 环境为 `.venv-ubuntu`。
没有运行 Windows Python、Windows 虚拟环境或 Windows 测试。

## 对照研究方案的当前状态

| 阶段 | 当前证据 | 状态 |
|---|---|---|
| T0 / E0：数据、经济权威与旧基线复现 | `docs/audit_evidence/20261016/binance_economics_sources.json` 明确记录：未查询账户凭证，缺少账户历史费率、全时段合约过滤器/保证金参数及实证成交成本。E0 后研究登记库有 **620 条 development/recorded、2 条 Holdout/reserved、3 条 Test/reserved**；本次 E0 增加 6 条开发读取收据（3 个 `train_inner` + 3 个 `train-signal-baselines`），没有 Test/Holdout 查询。另对仓库 `.tmp` 下 29 份 `config*.json` 做只读复核：23 份已有完成报告，其中 19 份指向已归档 `tests/fixtures/real_ethusdt_20240101` 至 `20240108`，对应 38 个 Train/evaluation 区间全部找到精确匹配且状态为 `development/recorded` 的项目登记；另 4 份是 synthetic factory 测试报告。该检查只证明当前可见的这批配置/报告已登记，不能证明仓库外、未留报告或历史未登记访问均已穷尽。旧 `train_signal_baselines` 配置原先硬编码到 `NautilusTrader_RL_ES` 的 Windows 路径，现已改为本项目相对路径；本项目内 bundle/market/rules 的 SHA 均匹配配置。Ubuntu E0 已按冻结配置完成 36 个 Train-inner Base/Stress 回放，与既有六组收益表逐项一致、非数值/决策字段无差异；最大数值差约 `9.9e-7`。**候选门禁仍失败**：冻结 RL 六组净收益均为 0，未优于空仓/全部固定基线。完整摘要见 `docs/audit_evidence/20261016/route_a_teacher_e0_reproduction_20261025.json`，36 份逐项报告及 SHA sidecar 保存在 `.artifacts/route-a-teacher-e0-20261025/`。经济仍为 `SCENARIO_ONLY`；本次运行前的历史 access identity 指向父项目 `.tmp` 快照，不等于当前冻结身份。 | **E0 可复现，但 T0 经济权威/全项目窗口清单未通过；按方案不进入 ML 教师训练或资格晋级。** |
| T1：因果候选及来源审计 | `causal_mtf_v7` 提供已完成 1m bar 的 60-bar breakout；G-A 突破和 G-B 回调要求 1d/1h 慢周期同向。事件绑定 decision/availability 时间、fold、注册 source、可见状态 snapshot SHA-256、FeatureSpec SHA-256 和规则版本。Host provider 在 M2.2 对账后报告候选到 teacher intent / planner / fills 的映射；报告明确标记为 provenance-only。 | **工程实现完成；真实候选覆盖和结果分布未评估**。 |
| T2 / E1 / E2：净价值标签、校准与机会门 | 新增 `teacher_research/opportunity_labels.py`：把外部冻结退出 replay 结果转换成完整候选全集的 counterfactual 标签；按 side 调整 log-price move，显式扣双方费用/滑点、只计一次 funding；正负案例完整保留、未平仓只记 censor clock，不造零收益；候选或 replay 缺失/错配时整批失败。新增 `opportunity_models.py`：固定 Logistic 胜率 + Ridge 净收益基线，标准化只由 Train-fit 计算，Platt 校准只读后置 Train-calibration；标签必须与实际生成的候选事件逐字段一致（方向、家族、决策/可用时钟、来源、fold、规则、snapshot、feature-spec），原始特征向量另有事件绑定 SHA；跨入 calibration 的 fit 结果自动 purge；q10 用后置 calibration 残差估计；只接受完整候选全集 `MODEL_COUNTERFACTUAL`，拒绝 hindsight/选择性 native 样本，并强制输出 `SCENARIO_ONLY`。输出包括费用情景净值估计、q10 残差尾部、拟合支持范围、校准身份；不执行市场 IO。新增独立 `evaluation/teacher_research_gate_v2.py`：只接受 Train-inner screen、至少 4 个独立时间块、每块全部冻结成本情景、Base/Stress 复用同一候选集合/选择；按“每个时间块的最差成本情景配对净增量”做 block bootstrap，并检查候选与负对照的配对差异，显式报告 all-flat 为非 Alpha。此研究门与既有生产 `screen_teacher()` 分离。 | **T2 工程原型已实现、尚未接授权历史结果运行；E1 固定退出候选策略对照、E2 实际模型校准/G1 时间块门均未完成**。须先提供可用 Train-fit/calibration/screen 窗口、经冻结的费用/退出配方与同一候选全集的结果。 |
| T3 / E3：P0 动态持仓规则 | `CausalPositionTeacher` 按候选开仓，持仓期间因果 HOLD/趋势失效 EXIT；没有正常 240 分钟强平，年龄取 native `decision_ts_ns - Position.ts_opened`，300 秒 cooldown 只限制空仓再入。测试覆盖 10/60/240/480/1440/2880 分钟和先平仓、确认后反向开仓。合成 >24h 测试跨 3 次资金费仍 HOLD；另新增真实 Binance 公共 Train 市场的 35h Ubuntu native Gym 回放，M2.2 核验 2,100/2,100 个屏障，资金费非零、fees/slippage/funding 入账、liquidations=0、状态为 `SCENARIO_ONLY`，报告见 `.artifacts/route-a-teacher-native-train-smoke-20261025-run03/native-smoke.json`。该段最长实际持仓约 7.1h，证明本段无 240m 定时强平，不代表 24h 实际持仓或盈利。另有 native MFE/MAE 在线与重建一致、`OPEN_CASE_CENSORED`、跨进程 fail-stop/replay 篡改失败测试。 | **P0/原生工程 smoke 已覆盖；真实账户费率/保证金、授权历史实付和经济签核未覆盖**。随机故障与实际运行卷损坏也未测试；当前恢复验证是进程退出/篡改，不等同断电/文件系统损坏。 |
| T4：学习型继续价值 | 没有启动 continuation/Q 学习。 | **按方案暂不启动**；需先有 G1/G2 正向证据和可识别反事实。 |
| T5：正式教师资格 / G2 | 35h Train native engineering smoke 已覆盖 funding；但费用/滑点/合约规则仍是 scenario assumptions，无历史账户费率/保证金证据、无正式经济权威签核；E1/E2 为 all-flat。 | **未通过**；不能晋级 Alpha Teacher 或导出合格教师声明。 |
| T6：学生 | 当前工作未启动正式学生 promotion。 | **按方案暂不启动**；需先通过 G2。 |

## 本轮 Ubuntu 验证

- Route A imitation 全测试：**237 passed**。
- 历史桥接 NativePosition、MFE/MAE、T3 Recovery、因果 v7 和 >24h synthetic funding replay 定向测试：**6 passed**。
- 组合结果：**243 passed**；本次改动涉及源码 Ruff 检查通过。
- 最新完整 `tests/test_historical_factory.py` + 绑定候选的 WFO 回归：**70 passed, 18 warnings**（5:04）。警告来自 EvoTorch / PyTorch / NumPy API 弃用提示，不是失败。
- 对相关源码和测试运行 Ruff：通过。
- 加入 episode replay 重启恢复后，完整 `tests/test_historical_factory.py`：**70 passed, 18 warnings**（5:06）；定向 MFE/MAE tracker、>24h P0/funding/reconstruction 和 replay 恢复测试均通过。
- 加入 fill-age/cooldown 后再次合并重跑 `packages/route_a_imitation/tests` 与 `tests/test_historical_factory.py`：**308 passed, 18 warnings**（5:31）；相关 Ruff 检查通过。
- Ubuntu 上运行附件 `route_a_teacher_feasibility_smoke.py`：**7 passed**。它只验证独立 stdlib 示意状态机；不导入项目代码、不使用 ETH 数据、不证明费用或经济有效性。checkpoint/replay 集成测试进一步改为 JSON 落盘、关闭原环境后再读入恢复；该定向测试通过。
- 将 replay 恢复升级为独立 Ubuntu 进程验证：写进程 `fsync` checkpoint 后 `os._exit(0)`，新 PID 恢复并逐项核对原生持仓状态和 boundary token；篡改 action 后恢复按 boundary mismatch 拒绝。定向用例 **1 passed in 23.81s**；完整 `tests/test_historical_factory.py` **70 passed, 18 warnings**（6:19，运行于本次篡改拒绝断言加入之前）；当前定向用例再次通过，Ruff 与该文件 `git diff --check` 通过。
- 针对 T0 再运行只读访问注册器复核：`.artifacts/research-access.sqlite` 以 SQLite `mode=ro` 查询；`.tmp` 下完成报告所对应的 38 个已归档 fixture 区间均有精确 `development/recorded` 登记。另用 Ubuntu `.venv-ubuntu` 跑访问注册器、scoped reader 和 Binance cashflow 合约测试：**81 passed in 22.84s**；全部测试使用临时合成数据。
- 按仓库 release lane 在规范 `.venv-ubuntu` 重跑 `tests/` + `packages/route_a_imitation/tests`：**966 passed, 18 skipped, 58 warnings in 1463.04s (24:23)**。18 个 skip 是仅支持 Windows Job Object/API 的用例；未运行 Windows 测试。弃用/FutureWarning 来自 EvoTorch、PyTorch、NumPy 和 SB3。首次全量运行揭示测试缓存的 `co_filename` 指向兄弟项目/Windows 路径；只删除规范仓库 `packages/`, `tests/`, `qa/` 中 536 个生成 `.pyc` 缓存（保留源码及规范 Ubuntu 缓存），并将 `tests/test_compute_device.py` 的 spawn 等待上限从 15s 调整为 60s；定向测试 **1 passed in 36.63s**，之后全量通过。该改动只改善 Ubuntu 上的测试启动余量并报告子进程退出码，不改变生产 GPU lease 逻辑。
- 本轮再次在 Ubuntu `.venv-ubuntu` 运行附件 smoke：**7 passed**；项目内 `tests/test_signal_baselines.py`、`tests/test_research_metrics.py`、P0 因果教师、pending/risk selector 与 native economic projection 组合：**108 passed in 20.83s**；相关 Ruff 通过。
- 加入 `OPEN_CASE_CENSORED` 后在 Ubuntu 专属 `.venv-ubuntu` 重跑相关标签/WFO 回归：**63 passed**，相关 Ruff 通过；测试覆盖 episode 结束未平仓标签、fit-end 校验以及 outcome mask。
- 加入 T2 的固定 Logistic/Ridge + later-block Platt 校准原型及独立 G1 研究门后，并收紧候选方向/家族/来源/fold/规则/snapshot/feature-vector 身份绑定与后置 q10 校准，Ubuntu `.venv-ubuntu` 的整个 `packages/route_a_imitation/tests` 与 `tests/test_historical_factory.py`：**317 passed, 18 warnings in 361.74s**；上传的 smoke 在 Ubuntu 再跑 **7 passed**；项目相关 Ruff 检查通过。warning 来自 EvoTorch/PyTorch/NumPy 弃用提示。全仓 `git diff --check` 被当前 dirty checkout 中大量无关 CRLF/尾随空格改动阻断；没有批量改写其他文件。
- 最后将 G1 负对照门从“负对照绝对收益必须非正”改为“候选相对负对照的配对 block-bootstrap 差异下界为正”，避免把普遍上涨误当 Alpha、也不错误否定有正收益但较弱的对照。修订后的完整 `packages/route_a_imitation/tests` **247 passed**，附件 smoke **7 passed**，相关 Ruff 通过。历史 factory 70 项未因这一独立 G1 统计判定改动重复运行；它与上一条 317 项 combined run 同属本轮验证覆盖。
- 新增固定退出候选结果的 full-denominator label adapter 后，Ubuntu `.venv-ubuntu` 整套 `packages/route_a_imitation/tests` **249 passed**，附件 smoke **7 passed**，相关 Ruff 通过；此前 317 项 combined run 覆盖历史 factory，但未包含此后新增的 label-adapter 两项测试。
- 按研究方案 G1/G2 诊断矩阵扩展 `teacher_research_gate_v2` 的逐 block/scenario 输入与输出：要求配对记录风险调整超额、最大回撤、尾部损失、校准 log-loss、换手，以及分成本情景的费用、资金费、filled-order 数和 exposure time；对独立时间块 bootstrap 报告最坏成本情景下的风险调整超额增量、最大回撤/尾损改善、校准改善和换手变化区间。G1 的 block 现在必须提供合法 `[start, end)` 时间边界；Gate 按实际边界验证 block 不重叠，并拒绝同一 block 的不同成本情景边界不一致。每个情景还必须提供完整候选结果与拒绝原因，报告正/负/删失数、被拒候选结果分布及 missed-positive regret（仅诊断）；机会标签也携带独立 block 边界，跨出 block 末端的结果被删失而不是当作完整标签；Logistic/Ridge 拒绝 fit/calibration 内部重叠、完整标签跨 block 边界及 calibration 时间块不晚于 fit 的输入；候选特征向量哈希对完整和删失样本一并复核。风险/校准/换手诊断不擅自设资格阈值，也不替代负对照、净增量门、既有生产 gate 或 L6 经济权威。Ubuntu `.venv-ubuntu` 整套 `packages/route_a_imitation/tests` **254 项全通过**；上传的 feasibility smoke **7/7 passed**；本次涉及的 Ruff 检查通过。以上仍是合成工程验证，没有可代替真实 E1/E2/G1 样本的结果。
- 为使 T0 旧 Train 基线具备 Ubuntu 执行路径，将 `configs/train_signal_baselines_20261016.json` 的 bundle/行情/规则改为仓库相对路径；`qa/train_signal_baselines.py` 现在仅在 Linux/Ubuntu 读取 `/proc/meminfo` 的 `MemAvailable`，并在任何访问前把 plan/output/source 限定于 canonical 项目根，拒绝 Windows 绝对路径和路径逃逸。重算配置 sidecar SHA；bundle、market 和 rules SHA 与本地文件一致。新增路径/内存解析测试后，Ubuntu `.venv-ubuntu` 的 `tests/test_signal_baselines.py` **9 passed**；Ruff、sidecar SHA 和 `git diff --check` 均通过。
- 按上述配置完成 E0：`qa/train_signal_baselines.py --plan configs/train_signal_baselines_20261016.json --output .artifacts/route-a-teacher-e0-20261025`；Ubuntu 24.04 / Python 3.14.7，3 个固定 Train 日 × 2 个成本模式 × 6 个冻结策略，共 **36 runs**。source-verified 报告校验 4,403,867 个市场事件且 dataset hash 匹配冻结模型；新旧六组 net-return 表完全一致，非数值/决策字段差异 0，最大绝对数值差 `9.881575913084362e-7`；36/36 报告 SHA sidecar 通过，formal queries 0，scaler 未重拟合，项目开发读取登记新增 6 条。最终 `FAIL_NO_CROSS_WINDOW_BASELINE_ADVANTAGE`，冻结 RL 六组均为 0 净收益。详细机器可读摘要 [route_a_teacher_e0_reproduction_20261025.json](audit_evidence/20261016/route_a_teacher_e0_reproduction_20261025.json)；完整报告保留于本地 `.artifacts/route-a-teacher-e0-20261025/`。结果仅为 Train-inner scenario-only 复现，不支持盈利、独立 OOS 或正式教师资格。
- 上述都是工程/合成测试，不是 ETH 历史收益、原生经济认证或独立 OOS 证据。

## 2026-10-25 Train 数据与预注册补充

- 用户已授权我自行选择未消费的连续 Train 窗口并下载数据。检查 `.artifacts/research-access.sqlite` 后，已知 ETHUSDT formal Test/Holdout 预约不与本次区间重叠；但仓库清单自身仍声明无法推断账本外/未登记历史读取，因此本区间只称 Train-inner，不称独立 OOS。
- 首次下载前已登记 `[2026-08-12T00:00:00Z, 2026-10-02T00:00:00Z)` development scope；正式分析数据裁剪至 `[2026-08-12T00:00:00Z, 2026-10-01T00:00:00Z)`。因 8 月 11 日首分钟与既有 Test 预约重叠，扩展 warm-up 的登记被拒绝，未读取/下载该区间；随后首次因果特征扫描又证明 `causal_mtf_v7` 的 1d return 需要 2 根完整日线，故只用 8/12–8/13 预热，正式评分从 8/14 开始。最终 prereg SHA-256 `8aa6b58c9c6f3afdd10687a59e8b8fedf5d6f22262bae3815b52fc8b1a076a72`；18d fit（4 个不重叠块）/10d later calibration（2 个后置块）/20d screen（4 个互不重叠 5d block）。退出预测跨度固定为仅供标签研究的 240m，不是运行时强平。两次未产生成果标签/模型输出的预检失败均保留在对话过程，本配置修订依据仅为 Test 边界和特征可用性。
- Binance 官方归档 importer 获得 `data/routea_teacher_train_20260812_20261002/` 数据：72,000 根 ETHUSDT USDⓈ-M 1m K 线，另含各 72,000 条 mark/index、150 条 funding update 和 150 条 funding settlement；bar 无重复且无分钟断档。Snapshot SHA-256 `9f07c4e0db1922b176c0f1186b511de8c9b41130ea1cbb3a69443234af708fab`，COMPLETED marker SHA-256 `8cc5e6057e4099c3ff7fef7b5fa181a4a9e527bf7266c07544d52dd065f69375`。0.4.1 行组级读取回执验证 `registered_before_read=True`。
- 因当月 2026-10 funding API 下载超时且月包未发布，研究日期收束到完整 2026-09；没有合成/填充缺失费率。数据不含账户手续费等级、全历史合约/保证金参数或账户真实 fills，固定 12bp Base / 30bp Stress 仍只是场景成本。故公开数据和模型 counterfactual 均只能标记 `SCENARIO_ONLY`。
- 2026-10-25 在 Ubuntu `.venv-ubuntu` 重跑 Route A imitation 全套 **254 passed**、附件 smoke **7 passed**、相关 Ruff 检查通过；另对 G1 gate 增加 SHA-256 candidate event ID 强制校验，避免 Gate 接受与真实 `OpportunityEvent` 不同的任意字符串身份。
- 按冻结 prereg 执行 `qa/run_route_a_teacher_e1_e2_g1_20261025.py`，Ubuntu `.venv-ubuntu` 产生 169 个 fit、161 个 calibration、138 个 screen 候选；Base/Stress 完整标签分别为 fit 166/169、calibration 147/161、screen 134/138（其余均为 block-end censor）。两个成本情景完成固定 Logistic/Ridge + 后置 Platt/q10 校准；有效 fit 样本覆盖 **3 个独立 blocks**。随后仅读到 screen 前的 row groups 0–5 做 fit-only 候选计数审计，四块候选数为 **66 / 0 / 52 / 51**；第二块为冻结候选规则下没有信号，不是标签删失或模型漏样本。冻结模型在 screen 选中 **0** 个事件，G1 返回 `NO_ELIGIBLE_ALPHA_TEACHER` / `ALL_FLAT_NO_TRADE`。相对“每个候选都交易”基线的正增量来自零交易避开亏损，不构成 alpha；负对照也为零交易。回放属于 `MODEL_COUNTERFACTUAL` / `SCENARIO_ONLY`，不是 Nautilus 原生 fills。
- 首次 runner 审计发现跨 block 标签可能在删失前读到块外退出价格/资金费。首份报告保留但不作为最终证据；修正后重跑报告在 `.artifacts/route-a-teacher-e1-e2-g1-20261025-run02/teacher-e1-e2-g1-report.json`，SHA-256 `4ef93bb29e3543187157184596d2ab2731973dafee43e26040081d4433864436`。修正前后数值相同。新增 guarded-read 回归测试确认跨块退出不读取块外价格/资金费、块末后入场不读取块外 open；Ubuntu `.venv-ubuntu` **2 passed**，Ruff 通过。fit-only 审计证明少掉的第四块是零候选块，故三块有效训练覆盖是数据结果，不是模型静默丢弃样本；当前 runner 已补充逐块覆盖诊断供后续实验审计。
- 已固化 `qa/audit_route_a_teacher_fit_block_coverage_20261025.py`：通过行组 0–5 的精确 development scope、`registered_before_read=True`，扫描只截止 Train-fit end，row group scope 在 screen 起点前结束；未读取 screen/Holdout，不训练模型、不改阈值。机器报告 `.artifacts/route-a-teacher-fit-block-coverage-audit-20261025/fit-block-coverage.json` SHA-256 `c70d965de37aeb46e173bbd3638756cd55bbdebeb5c33e5963cf28843aeb4ed7`。E1/E2 runner 另新增 fit-block coverage 字段供后续实验显式报告；本次既有 run02 不重写。
- 将 T3 native smoke 固化为 `qa/run_route_a_teacher_native_train_smoke_20261025.py`，并在 Ubuntu `.venv-ubuntu` 实际运行：ETHUSDT USDⓈ-M 已登记 Train 源，连续 2,100 个 native decision barrier（35h），M2.2 对 2,100/2,100 屏障核验，跨 funding 且账户报告 `total_funding=0.01083726`、`total_fee=0.24833622`、`total_slippage_cost=0.05016`、liquidations=0，`economic_status=SCENARIO_ONLY`。最长实际持仓 25,620 秒（约 7.1h）；没有声称本段出现 >24h 单笔持仓或盈利。摘要 `.artifacts/route-a-teacher-native-train-smoke-20261025-run03/native-smoke.json` SHA-256 `90d0da5d5d1482e28c9029ec37994f9bf532c2bab6434487310e1b5b047e34e4`，sidecar 已核验。

## 继续前必须补齐

1. 本次区间已授权并登记为 Train，但宿主未能证明账本外/未登记历史访问穷尽；不能宣称全项目绝对未触碰或独立 OOS。若出现新的冲突证据，应停止并废弃该区间。
2. 提供或授权可核验的 Binance 账户历史费用/规则/保证金与资金费证据；否则成本和回放只能保持情景级。
3. E0 已在三个固定已登记 Train 窗口完成复现，且冻结 RL 未胜过固定基线。新的 Train 区间 E1/E2/G1 scenario-only screen 结果是 all-flat / `NO_ELIGIBLE_ALPHA_TEACHER`，没有晋级依据。fit-only 审计确认 prereg 四块中一块按冻结候选规则无事件、其余三块各有候选；不得通过事后放宽规则补造该块。禁止把已用日期重命名为 OOS，或根据已读 screen 调整阈值。任何新实验必须新建并先冻结配置/数据角色；本次 screen 不得复用于调参。
4. T3 的 35h 原生 Train 工程 smoke 已完成，但使用 scenario-only 规则，不是实际账户费用/保证金对账；需要可核验的账户/规则证据才能做经济签核。跨进程 fail-stop/replay 篡改测试已通过，随机故障/实际运行卷恢复仍未覆盖。

在以上证据到位前，正确结论是 **`NO_ELIGIBLE_ALPHA_TEACHER` / `INSUFFICIENT_EVIDENCE`**，而不是“教师整改已经济合格”。
