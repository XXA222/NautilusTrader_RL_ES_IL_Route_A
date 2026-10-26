# 教师整改继续开发与原生复核（2026-10-25）

当前结论：`NO_ELIGIBLE_ALPHA_TEACHER`，未完成用户要求的经济合格教师目标，未开启正式学生晋级。以下为已经执行的整改和测试，不是未来计划。

## 环境与边界

- 唯一代码目录：`D:\Program Files\NautilusTrader_RL_ES_IL_Route_A`。
- 执行：WSL Ubuntu-24.04，`/mnt/d/Program Files/NautilusTrader_RL_ES_IL_Route_A/.venv-ubuntu/bin/python`，Python 3.14.7，Linux 6.6.87.2-microsoft-standard-WSL2。
- 曾使用已注册 Train 源，评分仅为 2026-08-14 至 2026-09-01 四个连续 4.5 天 fit 块，未在正式 Test/Holdout 日期评分。但事后原始来源审计发现导入时解码了**整个 2026-08 月度 funding ZIP**，该归档物理包含已登记的 Test。下列旧 fit 实验只能用于工程诊断，**不能作为教师资格证据**；不能把 fit 研究称为全新 OOS。
- 未改旧 screen/G1/G2/学生生产资格门，未提交、推送 GitHub，未改其他项目目录。

## 修正了什么

1. **退出标签与运行规则不一致**：旧 E1/E2 的固定 240 分钟标签只能保留为期限预测诊断，不能当成动态 P0 的校准证据。新增共享 `TrendInvalidationExit`，运行教师与研究标签端点均调用它；没有正常持仓时间上限。模型反事实仍不等于 native 成交，研究端不模拟账户硬风险覆盖，明确限制用途。
2. **回调缺少恢复确认、效率方向不对称**：新增显式 `route_a_candidates_v3_recovery` 机制。沿大周期方向，要求 15 分钟反向回调和最近 5 分钟恢复；效率取幅度。旧 v2 默认行为保留供对照。没有改变突破距离、zscore、最低效率、费用或经济通过门槛。
3. **决策频率不一致**：增加入场 cadence 配置。新对齐实验每 5 分钟允许入场，已持仓每分钟继续管理/退出；非入场时刻不能阻挡风险或退出。默认 1 分钟保留旧实验可复现性。
4. **现金流和边界**：新动态标签使用 constant-quantity / entry-notional 简单回报，退出费用按退出名义价值换算，资金费按结算时最近已收盘 mark 代理计入，避免把 log 价格回报与名义本金现金流混称同一口径。标记价只是研究代理，不是交易所结算权威。未闭合交易保留 censor，不补零收益；无块内可执行入场时整体拒绝，不静默丢候选。
5. **资金费时钟误报**：第 4 块有 13/13 次结算，偏移为 13、26、0、1、0、0、2、0、0、1、2、2、1 毫秒。旧辅助脚本继承的 15ms 容差误拒绝 26ms。新的源专属实验冻结 1s 时钟匹配容差，与既有 funding API ingress 的秒内抖动约定一致；保留真实事件时间，1ms 发布延迟上限、结算数量和缺失检查不变。不是修改通用生产 gate 或补造资金费。前 3 块 18 组已有效结果在配置修正前后完全相同。

## 已执行测试

- Route A 全套、runner 边界和经济 coverage 回归：**295 passed，36.98s**。
- 覆盖共享退出、15 分钟提前退出、241/480 分钟仍可持有、方向对称、恢复缺失/未来时点拒绝、5 分钟入场但分钟退出、块末退出成交截尾、现金流不重复、1ms funding 抖动、过期 mark、缺失结算、过界时钟和过慢发布。
- 相关 Ruff 检查通过（最终新增文件另做收尾检查）。
- 完整原生实验 **24 组**：4 块 × Base/Stress × FLAT/v2/v3；每组 6,477 个 barrier，合计 **155,448**，M2.2 覆盖全部，零强平，全为 `SCENARIO_ONLY`。
- 最长原生已成交持仓 **6,425 分钟（约 107.1 小时）**，证明可以跨日、跨 funding 且不在 240 分钟机械退出；这不是收益合格证据。
- 另保留 1 分钟旧时钟和 5 分钟旧时钟实验，各完成前 3 块 18 组后在第 4 块 DQ-D 停止。它们不是两个完整 4 块试验，也不是额外独立市场样本。

## 真实结果

新 v3，5 分钟入场、分钟退出，账户净收益：

| Train-fit 块 | Base | Stress |
|---|---:|---:|
| 1 | -0.055431% | -0.068907% |
| 2 | +0.679745% | +0.674647% |
| 3 | -0.113318% | -0.131332% |
| 4 | -0.149321% | -0.158181% |

四块平均回报 Base +0.090419%、Stress +0.079057%，但都主要依赖第 2 块。枚举四块经验 bootstrap 的探索性均值 95% 区间分别为 [-0.131319%, +0.481480%]、[-0.144757%, +0.473153%]，包含零。四块都来自开发 fit，不是独立 screen，不能据均值为正发证。旧 v2 在对齐 cadence 后为 0/4 正收益；FLAT 全部为零。

动态候选标签也不能替代账户结果：v2 为 169 候选、140 完整、29 截尾；v3 为 74 候选、47 完整、27 截尾，其中 Base/Stress 各 8 个盈利完整标签，全部集中第 1 块。第 2 块 v3 的 12 个候选全部截尾，而连续 native 有正的终局权益；必须保留这种区别，不把窗口终局结算收益伪称普通策略主动闭合的完整监督案例。

## 可复查产物

- v2 动态标签：`.artifacts/route-a-causal-exit-fit-20261025-run02/`。
- v3 恢复机制标签：`.artifacts/route-a-causal-exit-recovery-fit-20261025/`。
- 时钟失败根因：`.artifacts/route-a-causal-teacher-fit-native-dq-20261025-run02/block-04-base-quality.json`。
- 完整原生实验：`.artifacts/route-a-causal-teacher-fit-native-5m-20261025-archive-clock/`。
- 该目录包含冻结 `recipe.json`、24 份逐 arm 报告及 SHA、24 份压缩逐 barrier 订单/权益证据、`report.json` 和 `verification.json`。校验脚本逐一核验报告 sidecar、解压 trace SHA、M2.2 数量及与此前 18 组结果的一致性。
- 原生 recipe SHA-256：`a24dfc36057d4e01b48a3aeeeddc956d63aa6663c55aa9f5a13d03989ea26a3a`。
- 首次动态标签探针因资金费带 1ms 偏移、精确 mark 时间 join 失败而停止；原 recipe 保留。修正为向后匹配且 mark 年龄严格小于 1 分钟后 run02 完成，没有改写旧结果。

## 尚不能跨越的门

目前仅得到工程正确性和 fit 机制诊断，**没有合格 Alpha Teacher**。固定多/空 beta 归因已完成，见下节；仍未建立全新校准/screen 的可重复净增量，也未取得历史经济权威。学习型 continuation 和学生正式晋级没有启动；不能重复用已消费 screen 选阈值、用盈利块替代全样本或忽略截尾。

## 后续冻结实验：beta 归因及退出闭环

- 在同一已消费 Train-fit 四块、Base/Stress、原生账户和 0.3 初始仓位下，追加固定多/空各 8 组，共 16 组、103,632 个 M2.2 已核 barrier；配方及全量证据在 `.artifacts/route-a-beta-attribution-fit-native-20261025/`，`attribution.json` 给逐块比较。它们可描述市场方向暴露，但持仓时间不完全匹配，不能直接当风险匹配的 alpha 负对照。
- 第 2 块 v3 的 Base +0.679745%，固定多 +0.677295%，净差仅约 **0.245 个基点**；Stress 净差约 0.245 个基点。v3 在其他三块的 Base/Stress 均落后于固定多、固定空两者，原先唯一盈利块几乎可以用市场做多解释；不是可泛化择时证据。
- 事前冻结新的 `either` 退出政策：15m/1h 以外不偷读未来，持仓时 1h **或** 1d 逆向趋势超过原 0.002 阈值即普通 EXIT，仍无最大持仓时间；候选、仓位、入场 cadence、风险、成本及原生执行不变。默认 `both` 政策和已有产物不变。研究标签与运行教师使用同一版本化 `TrendInvalidationExit`。
- 新政策在四块 Base/Stress 跑 8 组、51,816 个已核原生 barrier，证据在 `.artifacts/route-a-either-scale-exit-fit-native-20261025/`。Base 逐块净收益为 -0.055393%、+0.217016%、-0.078732%、-0.027589%；Stress 为 -0.109168%、+0.172074%、-0.128029%、-0.094678%。仍只有 **1/4 块盈利**，不能发 G1/G2 合格证。每种成本情景中分别有 11/10/11/15 条原生成交支持的普通 flat-to-flat **情景路径**，另第 1 块有 1 条块末未平仓；两种成本情景不是双倍独立案例。情景路径不是 `OBSERVED_NATIVE_CASE`，不送正式学生。
- 修正审计缺口：`SCENARIO_ONLY` 的块末 L6 强制结算不能把原持仓当作教师自主平仓，也不能让它从截尾计数消失；现在保留 `OPEN_CASE_CENSORED`，收益为空、无可执行教育标签。新增终局回归测试并通过。真正 `OBSERVED_NATIVE_CASE` 的原生开平仓、M2.2 与经济权威要求未放松。

## 新 Train-inner 复核：未通过

- 先核对研究账本，2026-10-03 至 10-23 没有登记 Test/Holdout 或正式查询意向重叠；下载前冻结 `configs/route_a_teacher_oct_fresh_train_prereg_20261025.json`（SHA-256 `55470520c9b0d4997913701164342dddb62c5dfc1198b42b1ab4c9dff42d66ef`）。10 月 2–22 日每日 OHLC/mark/index 归档已下载验签，但当月资金费月度归档尚未存在，官方公开 funding API 从本机超时。导入严格失败并保留 `data/routea_teacher_train_20261002_20261022/IMPORT_FAILED.json` 与价格源回执；**未补造资金费、未跑不完整的市场源**。Binance 官方归档说明每日文件次日可用、月度文件在次月首个周一发布；此处月度资金费尚未出版与 API 超时是两项不同事实。
- 随后另选账本中评分日期无登记访问、避开既有 Holdout/Test 的 2026-07-17 至 08-05 区间；在下载前冻结 `configs/route_a_teacher_july_fresh_train_prereg_20261025.json`。公开归档每日 OHLC/mark/index 及 7、8 月资金费月度文件全部验签导入，82,194 个裁剪后事件，源 marker SHA-256 `6a6682632a55975f3a247ec96e3134bd70d96b0b079994be5021507ebfa054ab`。**事后发现此来源不合规**：解码的整月 7 月 funding ZIP 含已登记 Holdout，8 月 ZIP 含已登记 Test；裁剪后的事件数和评分日期不能消除物理读入违规。
- 初次配方 7 月 18 日评分起点缺少 1d return 所需的两个完整 UTC 日桶，第一块 `application.prepare()` 在任何 arm 回放前拒绝；`.artifacts/route-a-july-fresh-train-native-20261025/failure.json` 记载 `completed_runs=0`。保留失败，不以绩效为理由重试。仅因特征暖机，将评分起点后移至 7 月 19 日，保持源、候选、阈值、成本、臂及 4 个连续块；事前冻结 `configs/route_a_teacher_july_fresh_train_warmup_errata_20261025.json`（SHA-256 `083c55f5df97b36f165159c4f8c410ed73b04d8aa1041da4bb8c2723bef67f7a`），并绑定原失败回执 SHA。
- 新矩阵 4 块 × Base/Stress × Flat/固定多/固定空/v3 双尺度退出/v3 任一尺度退出 = **40 组**，每组 6,117 barrier，共 **244,680**；M2.2 逐组核对，零清算。冻结 `recipe.json`、40 份逐组回执、40 份压缩逐 barrier 证据、总报告及 `verification.json` 在 `.artifacts/route-a-july-fresh-train-native-20261025-run02/`。这些回执证明工程运行和窗口内绩效，不证明来源合规；**整套实验因原始归档跨入 Test/Holdout 而取消资格**。
- 旧 v3 双尺度退出 Base 净收益逐块约 **+0.071485%、+0.041126%、-0.159882%、-0.027976%**；Stress **+0.066910%、+0.027654%、-0.182266%、-0.045968%**。仅 2/4 块为正，且 Base/Stress 均 **0/4 块同时胜过固定多与固定空**；平均块收益 Base -0.018812%、Stress -0.033418%，探索性四块均值 bootstrap 区间均覆盖零。完成的情景级 flat-to-flat 路径每个成本情景各 9 条，其中仅 1 条净正、8 条净负，4 条块末仍持有而删失。
- 新 v3 任一尺度退出 Base 四块 **-0.024611%、-0.014094%、-0.147597%、-0.061334%**；Stress **-0.073919%、-0.054415%、-0.223804%、-0.110706%**，均 0/4 块正收益。情景级完整路径每成本情景各 46 条，Base 6 正/40 负，Stress 2 正/44 负，另 2 条块末删失；新退出增加闭环但不能制造可用 Alpha。
- **正式 `OBSERVED_NATIVE_CASE` 教育标签仍是 0**：当前数据与规则是 `SCENARIO_ONLY`，且两种教师均未证明费用后稳定增量。Flat 可以继续作风险/无 Alpha 对照，但不是 Alpha Teacher；未过 G1/G2，正式学生晋级没有启动。不得把两种成本情景当成两套独立市场样本，或将完整的情景循环当成可教的经核验经济案例。

## 原始归档物理范围复核与隔离

- `.artifacts/route-a-physical-source-scope-audit-20261025/audit.json` 只读导入清单和正式研究账本，不再打开市场值。它复现：旧 8–10 月来源的整月资金费归档与正式 Test/Holdout 范围有 2 项重叠；新 7–8 月来源有 3 项。结论 `DQ_PHYSICAL_TEST_HOLDOUT_ARCHIVE_OVERLAP`。此前“没有读取 Test/Holdout 市场数据”的说法错误，在此更正。
- 导入器现在按**原始归档覆盖的整月**在下载和解码之前与 `access`、`query_intent` 中正式 Test/Holdout 比对；重叠或项目研究账本不可用即拒绝，不创建目标目录。已导入来源的 Train 读取也检查 `import.json` 的整月来源，在预留或解码 curated 市场文件之前拒绝。这两个隔离点都有 Ubuntu 单元回归；对上述两个真实已导入来源的只读调用也都在打开 `market.parquet` 前实际返回 `DataError`。旧来源不因已存在而获得豁免。
- 最后一轮相关 Route A、研究读取、归档、现金流、经济覆盖与 runner 边界 Ubuntu 回归 **336 passed**；用户附件 smoke **7 passed**；改动的 Python 文件 Ruff 检查通过。额外的全项目 1,042 项套件因一小时截止主动中断于 **805 passed、0 已报告失败，237 项未完成**，不能写作全套通过。测试通过只证明安全拦截和工程契约，并非 G1/G2 通过。
- 保留原有实验和失败回执作为审计诊断，不重命名为 OOS、不作为 G1/G2 资格材料，不据此启动正式学生。要继续教师资格研究，需要能在**物理源级别**避开保护窗口的新 Train 数据、事前冻结实验，以及经核验的原生经济权威。

公开市场数据可自行下载，但账户历史费用/保证金权威不能由代码猜测。本轮再次核对 [Binance 官方账户 API](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/account)：用户佣金是 USER_DATA 账户端点，公开 K 线和 funding 归档不包含账户历史费率证明。未访问账户凭证、未调用交易接口。即使后续情景收益改善，也必须保持 `SCENARIO_ONLY`，直至项目既有 verifier 真正核验经济证据。

后续研究需要新的可检验机制和预注册未消费时间段，并补齐成本权威；本报告不能被当作继续训练学生的授权收据。

## 2026-10-25 后续工程整改追加

- 本轮将 packages 范围 mypy 从 20 项错误修复到 0，增加无收益输入的单仓位因果准入与闭仓现金流对账模块，修正固定 240 分钟旧 screen runner 对完整结果的未来筛选、删失仓位的块末记账、空 screen block 索引，并封闭 Binance importer 临时输出路径绕过物理保留区守卫的问题；同时增加冻结四门的 16 组合/顺序/分组诊断代码。
- Ubuntu-24.04 `.venv-ubuntu` 最终 mypy 为 223 个源文件无错误，Ruff 全仓检查通过，Route A + 历史工厂 + 注册 Train 读 + 归档测试合计 **387 passed**。18 个依赖兼容告警不影响退出码。
- 本轮未读取/下载新市场数据，未复用旧 screen 调参，也未启动新市场实验；现有 E1/E2 报告仍为零选中，不能修改或覆盖成新结果。教师资格状态维持不合格，正式学生开发仍未晋级。
- 研究账本只读汇总显示 development access 最新到期为 2026-10-02；本轮未打开市场价格数据。旧 E1/E2 screen 已消费，且其原始 source 有 Test/Holdout 月度 funding 物理重叠，不能重跑。现存报告不含逐候选 score/四门位图；新增 runner 现已连接预测分布、Q10/support、screen outcome 校准、F0–F5 的 Base/Stress 配对审计和账户回放，但没有用旧聚合值伪造缺失表。完整执行结果仍待合法新 source 与账本登记。详见 [ROUTE_A_TEACHER_REMEDIATION_EXECUTION_20261025.md](ROUTE_A_TEACHER_REMEDIATION_EXECUTION_20261025.md)。状态仍为 `BLOCKED_BY_SOURCE`。
