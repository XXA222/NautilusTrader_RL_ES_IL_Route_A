# Route A 教师整改执行记录（2026-10-25）

## 结论与执行边界

- 代码整改和 Ubuntu 工程验证完成本轮；**没有训练出通过正式 G1/G2 的教师**，不启动正式学生晋级。
- 唯一项目根目录：`D:\Program Files\NautilusTrader_RL_ES_IL_Route_A`；所有命令均在 WSL Ubuntu-24.04 的项目 `.venv-ubuntu` 中执行。执行前核对 Git top-level 与 `.canonical-project-root`。
- 本轮没有读取 Test/Holdout，没有下载市场数据，没有重命名数据窗口，没有改资格阈值、成本或 preregistration，没有运行市场资格实验，没有推送 GitHub。
- 旧的 E1/E2/G1 报告保留原样，仅作为已消费 Train-inner 窗口的工程诊断。其结论仍为 `NO_ELIGIBLE_ALPHA_TEACHER` / `ALL_FLAT_NO_TRADE`，不能作为 OOS 或资格证据。

## 本轮改动

1. 清除 packages 范围原有的 20 项 mypy 错误（类型字面量、意图类型、NumPy 输入、可删失收益的显式校验，以及 Gate reason 类型推断）。不通过放宽 mypy 配置解决。
2. 新增 `portfolio_replay.py` 的单仓位因果准入与账户账本：准入 API 不接收收益标签；仅按候选时点、下一开盘成交时点、上一仓位退出/删失状态和独立块边界入场。闭仓收益必须与绑定标签在费用、滑点和资金费口径上精确对账。删失仓位只按块内末端 mark 计值，不伪造退出成交或退出费用；realized、terminal mark-only、费用、滑点、资金费、持仓分钟和成交数分开记录。
3. 修正旧 screen runner 的组合记账：不再因 `net_return_bps is None` 将候选从入场阶段静默删除；完整结局只在准入后用于核对和账户结果记账。跨块退出拒绝，无块内 entry 独立计数。该 runner 仍是固定 240 分钟的历史预测诊断，不是动态 P0 的经济权威。
4. 新增四门 funnel 诊断：逐门计数、顺序 funnel、全部 16 种布尔组合，以及按独立 block / family / side 的分组，不改冻结比较阈值。空 block 被写入独立空块记录并跳过 G1 候选分母构造；无块内入场与仓位占用使用不同拒绝原因。
5. Binance 归档物理保留区守卫现在与输出目录无关：即使目标指向临时目录，也必须先读取项目研究账本并检查下载归档的完整物理时间跨度。合成导入测试各自使用隔离的空白账本。
6. 动态退出仍使用共享的 `TrendInvalidationExit` / `CausalExitSchedule` 和 `make_causal_exit_replay`，label/runtime 一致性由合成回归验证。当前冻结的 E1/E2 prereg 明确绑定 fixed-240m；本轮未篡改 prereg 来重跑为动态标签试验。

## 验证结果（WSL Ubuntu-24.04）

使用 `.venv-ubuntu/bin/python` / `.venv-ubuntu/bin/pytest`：

```text
.venv-ubuntu/bin/python -m mypy packages
Success: no issues found in 223 source files

.venv-ubuntu/bin/ruff check .
All checks passed!

.venv-ubuntu/bin/pytest -q packages/route_a_imitation/tests tests/test_historical_factory.py tests/test_registered_train_window_read.py tests/test_binance_archive_range.py
387 passed, 18 warnings in 350.64s
```

18 项 warning 来自安装依赖在 Python 3.14/NumPy 2 下的弃用或未来兼容提示；测试退出码为 0。单独的 portfolio ledger/admission/funnel 合成测试为 11 项，全量套件也包含这些测试。

## 资格判断与后续阻塞

- 工程测试通过不是教师经济资格。当前已知旧 screen 的 138 个候选全部被冻结模型拒绝；正式 Gate 结果仍为 `NO_ELIGIBLE_ALPHA_TEACHER`。
- 本轮新增了 runner 会输出 16 组合 funnel 的代码，但没有重跑已消费的 screen，因此旧报告仍无本轮 funnel 数表；不从旧聚合字段臆造 16 组合分布。
- 研究账本只读汇总中已登记 development access 的最大 end 为 `2026-10-02T00:00:00Z`；当前可用后续日期不足冻结的 fit/calibration/screen 连续窗口长度。没有用此汇总打开任何市场价格文件。
- `configs/route_a_teacher_e1_e2_g1_train_prereg_20261025.json` 的成本和窗口是已消费的诊断配方，退出政策是 fixed-240m；既有源的 monthly funding 物理归档还曾覆盖 Test，因此不可用于教师资格或独立 OOS。
- 下一阶段状态：`BLOCKED_BY_SOURCE`。只有在研究账本预先登记一段新的、未消费且连续的 Train-fit/calibration/screen 窗口，获取该窗口可合法读取的来源与可审计资金费/合约/保证金/费用证据，并另行冻结动态退出 recipe/prereg 后，才能启动新的市场实验。此条件满足前不训练正式学生、不放宽规则、不重用旧 screen 调参。

## 文件索引

- 单仓位选择：[portfolio_replay.py](../packages/route_a_imitation/src/rles_route_a/teacher_research/portfolio_replay.py)
- 四门冻结筛选诊断：[gate_funnel.py](../packages/route_a_imitation/src/rles_route_a/teacher_research/gate_funnel.py)
- 因果准入测试：[test_portfolio_replay.py](../packages/route_a_imitation/tests/test_portfolio_replay.py)
- 归档物理范围守卫：[binance_archive.py](../packages/data/src/rles_data/binance_archive.py)
- 固定退出历史诊断 runner：[run_route_a_teacher_e1_e2_g1_20261025.py](../qa/run_route_a_teacher_e1_e2_g1_20261025.py)
- 动态退出共享实现：[causal_exit.py](../packages/route_a_imitation/src/rles_route_a/teacher_research/causal_exit.py)；[causal_exit_replay.py](../packages/route_a_imitation/src/rles_route_a/teacher_research/causal_exit_replay.py)
- 旧的诊断结果：[teacher-e1-e2-g1-report.json](../.artifacts/route-a-teacher-e1-e2-g1-20261025-run02/teacher-e1-e2-g1-report.json)

## 2026-10-25 账户级研究回放补充

- 在同一冻结基线和唯一 Ubuntu 环境中继续 R2。新增 `ResearchPortfolioRecipe`、按分钟 bar/point-in-time observation/funding settlement 输入的 `simulate_causal_research_portfolio`，使用实际模拟数量、单仓位、严格晚于决策时点的 next-open 执行、动态 `TrendInvalidationExit`、逐笔 fee/slippage、结算时 long/short 带符号资金费和账户权益 MTM。函数没有标签/outcome 输入，结果固定标记 `SCENARIO_ONLY` / `RESEARCH_SCENARIO_NOT_NATIVE_FILLS`。
- 两种终局处理显式区分：`mark_only` 保留 open-censored 仓位，不补虚构退出费/成交；`scenario_force_flat` 只作为命名清楚的终局情景。Funding mark 必须在授权窗口内、严格早于结算，且新鲜度严格小于上限。
- 新增合成回归覆盖：前缀因果不受截短未来影响、239/241/480/1440/6400 分钟持仓不触发 240m 强平、long/short funding 符号及单次结算、账户费用/滑点/净值守恒、终局 MTM/情景强平分离、共享动态退出、反向同时候选拒绝、重复 bar/越界候选/陈旧资金费 fail-closed。账户回放对应测试 21 项通过。
- 新增 `opportunity_diagnostics.py`：分离预测时的全候选门槛/分组分位数/fit 极值覆盖审计与事后的 screen outcome 校准、完整/删失分布和 Q10 恒等式审计；缺失预测保留 `null` 和明确拒绝码。F0–F5 固定消融统一标记 `DIAGNOSTIC_REUSE_NOT_OOS`，不读 outcome、不参与训练。后续整改已将该 API 接入旧 runner，并新增 Base-gate 配对 Base/Stress 的 F0–F5 结果审计与账户回放输出；旧 138 候选报告没有逐候选分数，旧污染行情源未重跑，因此尚无这些新表的真实结果。
- 按独立组并行验证：Route A 全测试组退出码 0（pytest 进度到 100%，无 failure）；历史工厂 + 注册 Train 读 + 归档组 **96 passed, 18 warnings in 339.08s**。结合后续诊断回归，相关指定组的最新测试事实见本报告的 R1/工程追加，不把前一轮总数当成本轮新跑结果。`.venv-ubuntu` 没有 `pytest-xdist`，通过彼此独立的 pytest 进程并行执行，未安装依赖。
- 全仓 Ubuntu 检查再次通过：`.venv-ubuntu/bin/python -m mypy packages`（225 source files）和 `.venv-ubuntu/bin/ruff check packages qa tests`。`git diff --check` 通过。HEAD 仍为任务书固定 SHA `6e277d77c2619f756771191cedcee93159412b59`；工作区未提交变更保留。
- 曾尝试更大的 `tests + packages/route_a_imitation/tests` 集合：脚本式 `pytest` 首次因 `qa` 导入路径缺失而 3 项收集错误；改用 `python -m pytest` 后，在约 47% 的非 Route-A 慢测试区段运行约 15 分钟后主动停止。该全仓集合结论为 **NOT COMPLETED**，不是 PASS；未重复运行。`.venv-ubuntu` 未安装 `pytest-xdist`，也未改锁文件或安装依赖。下次若多个独立测试组需重跑，将先确认共享副作用，再通过并行独立进程执行；不会把存在顺序依赖/共同写目标的测试强行并发。
- A3 当前仅有合成情景账户账本验证；与 M2.2/native 逐项对照尚未完成，故不能声称 native reconciliation PASS。A4 的旧报告仍缺逐候选分数与 16 组合，新增 runner 输出路径已接上完整诊断/消融 API，但受旧来源污染与账本未登记影响，尚未执行；R3.3 真实统计仍未交付。账户 simulator 尚未在新的合法 Train 数据上运行；A6/A7 不运行、不发合格教师，也不启动正式学生。

## 2026-10-25 R1 funding 来源边界复核

- 再次在 WSL Ubuntu-24.04 核验 Git root、`.canonical-project-root`、固定基线 SHA 与冻结分支一致。SQLite 以只读 URI 检查时，`2026-10-02` 至 `2026-10-23` 区间没有对应的 development access、query intent 或 trial attempt 记录；没有据此打开任何行情文件。现有 frozen prereg 不能替代缺失的研究账本访问记录。
- 仅检查了已有导入目录的文件名、大小、source receipt 与 `IMPORT_FAILED.json`。失败记录显示 funding API 请求因 25 秒超时失败，并且请求 `startTime` 落在 `2026-10-01`，比 prereg 的 source start `2026-10-02` 多读了一天。未读取 ZIP 内容、funding 响应或 curated market 数据，也没有重试网络请求。
- 修正 `binance_archive.py`：REST fallback 从精确的 prereg `start_day` 起请求，结束于 `end_day 23:59:59.999 UTC`；解析器现在逐条验证响应 fundingTime 在同一包含边界内。新增请求 URL 起止时间与越界响应拒绝测试。
- 修复后并行验证：Binance archive 范围测试 **24 passed**；Gate funnel / opportunity diagnostics / portfolio replay 组退出码为 0（最新为 **30 个测试点**）；相关 Ruff 检查通过；`mypy packages` 为 **225 source files，无错误**；runner `py_compile` 通过。一个初始测试命令引用了仓库中不存在的 `tests/test_binance_archive.py`，该命令没有运行测试；随后已对实际存在的 archive 测试文件成功重跑。
- 这些工程修复不补写或追认历史访问记录。现有 October raw 文件不因此变成已授权、未消费的研究来源；本轮不从其训练或做资格回放，教师资格仍未通过，状态保持 `BLOCKED_BY_SOURCE`。
