# F00 — 基线与本地证据清单

状态：`ENGINEERING_PASS`（清点完成；S07 仅为已消费 D1 诊断）  
日期：2026-10-26（Asia/Shanghai）  
指定基线：`bd313c5241c31f1ccff96fa4bf76a7966be48767`

## 仓库与工作区

- 规范根：Windows `D:\\Program Files\\NautilusTrader_RL_ES_IL_Route_A`；WSL `/mnt/d/Program Files/NautilusTrader_RL_ES_IL_Route_A`。机器标记 `.canonical-project-root` 一致。
- WSL2 `Ubuntu-24.04`；`.venv-ubuntu/bin/python --version` → `Python 3.14.7`。
- 已从 origin 获取指定 SHA，并建立本地分支 `codex/route-a-teacher-v2-20261026`；HEAD 精确为指定 SHA。旧分支 `codex/route-a-1m-multiteacher-v1` 保留。
- 指定提交 parent=`6e277d77c2619f756771191cedcee93159412b59`，tree=`5c835bc33fd64df6b32c32cf7b3b905da1e27dd9`。此前 HEAD `10516aa709cc4949867ac18719b46c58dd1897f2` 与之同 parent、同 tree。切换前工作区干净；切换后也干净。未 reset、clean、stash、覆盖旧记录或推送。
- 本阶段未读行情原始值或 Test/Holdout。研究访问账本只查表结构与行数，不读取保护内容。

## 冻结文件（未修改）

| 文件 | SHA-256 |
|---|---|
| `configs/route_a_1m_teacher_native_prereg_20261026.json` | `96a5f9f8c5eaee9c65830a4bf9e2e44a79b0b6bede721b4813522a859070c036` |
| `docs/route_a_1m_teachers/S07_FINAL_REPORT.json` | `33a2a520009f2267b838b3c41c1875e1538747c74187407ae8a956f0fca238e9` |
| `docs/route_a_1m_teachers/S07_FINAL_SUMMARY.md` | `5525b4101cdbf661002f31e8c6b2194657efbac78128d4c07dc73278ee02fd98` |

本地 `.artifacts/research_1m_4teachers_v1/report.json` 与完整报告 SHA 一致；`summary.md` 与摘要 SHA 一致。

## S07 本地证据清点

`.artifacts/research_1m_4teachers_v1/` 共 589 个文件：

| 证据 | 数量 | 内容 |
|---|---:|---|
| 教师 `native-run.json` | 48 | 4 教师 × 2 退出 × 3 时间块 × Base/Stress |
| 教师逐分钟 `decision-rows.csv` | 48 | 决策、信号与状态 |
| 教师逐笔 `native-trades.jsonl` | 48 | 成交、方向、持仓、PnL、fees、slippage、funding、案例状态 |
| 固定控制 `control-report.json` | 18 | 固定方向 / FLAT；3 个时间块 × Base/Stress × 3 控制 |
| Exposure-matched 控制 `control-report.json` | 96 | 4 教师 × X1/X2 × 3 时间块 × Base/Stress × LONG/SHORT 匹配控制 |
| 全部控制 `control-decisions.csv` / `account-rows.csv` | 各 114 | 逐分钟决策 / 账户权益 |

共 162 份账户/控制报告，全部标记 `SCENARIO_ONLY`；48 份教师报告标记 `D1_DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_G1_G2`。S07 汇总：48/48 教师账户完成，Base 24/24 负收益，Stress 24/24 负收益，控制账户 114/114 完成。三个已消费窗口：2024-02-01—03-01、03-01—04-01、04-01—04-15。不得用于重新选优或独立 OOS 声明。

M2.2 核验计数存在于各 run receipt（20,158、41,757、44,637）。其跨账户总和 5,753,808 不是独立市场样本，因为同块账户共享行情时钟。单独逐步 M2.2 barrier 轨迹在该目录未找到，状态为 `NOT_LOCALLY_AVAILABLE`；计数回执不能替代原始轨迹。

已抽查 T02 ROC X1 Base 的真实 `native-trades.jsonl` 样例，含入/出场成交价和时间、持仓分钟、净 PnL、手续费、滑点、资金费及案例状态；这不是全量重算。F02 必须从本地原始逐笔文件重算，不能从汇总报告补造成交。

S07 为已消费的 Train-inner 诊断，成本/账户规则没有账户历史权威，因此仍为 `SCENARIO_ONLY`，不满足 G2。F05 需单独核实新 Train 的未消费状态、授权和物理原始文件覆盖区间；未证明避开 Test/Holdout 前，不启动正式候选选择。

## F00 执行

命令均在 WSL 项目根：`git rev-parse --show-toplevel`、`git rev-parse HEAD`、`git status --short`、`.venv-ubuntu/bin/python --version`、`find .artifacts/research_1m_4teachers_v1 -type f`、上述冻结文件的 `sha256sum`。结果与本清单一致；本阶段只读，没有运行测试或回测。新增清单文件为本目录。
