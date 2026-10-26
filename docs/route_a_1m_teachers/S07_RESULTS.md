# S07 — 四教师 × 双退出 Train-inner 原生对照

状态：**COMPLETED_D1_DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_G1_G2**。截至 2026-10-26，冻结矩阵 162/162 个账户已完成；所有结果仍为场景化收益（SCENARIO_ONLY），没有教师通过 G1/G2，也未启动学生。正文中“正在运行/尚未完成”的段落是当时的历史 checkpoint，最终结果见文末。

## 冻结实验身份与边界

- 预注册：`configs/route_a_1m_teacher_native_prereg_20261026.json`，SHA-256 `96a5f9f8c5eaee9c65830a4bf9e2e44a79b0b6bede721b4813522a859070c036`。
- 市场/账户：ETHUSDT-PERP.SIM；10,000 USDT 起始权益，1,000 USDT 风险预算，目标 requested exposure 0.3，最大杠杆 1；同一 Nautilus Gym/M2.2/L6 原生账户路径。
- 仅使用预注册 Train-inner 区间：2024-02-01 至 2024-04-15 00:01Z，3 个独立 block。development-fit 为 2024-01-02 至 2024-02-01；校准为固定规则下 `NOT_APPLICABLE_FIXED_RULES_NO_PROBABILITY_MODEL`。
- 原始合法来源仍是预注册 legacy 0.4.0 Train 快照；物理月度 Test/Holdout overlap 开关为 false，保护月 guard 在每个 block 行解码前执行。未打开受保护月度原始 funding 包。
- 每个 teacher × exit × scenario × block 独立重置账户；每次最多一个原生 episode 并行。Base/Stress 仅是场景成本假设，所有结果为 `SCENARIO_ONLY`。
- 固定 LONG/SHORT 使用同一 0.3 名义敞口预算；额外 exposure-matched LONG/SHORT 根据对应教师同一逐分钟 `requested_exposure` 的绝对值复播方向固定敞口，事件日程 SHA-256 随基线结果记录。该 beta 控制为离线方向消融，不是可部署策略。

## 已实测的 S07 前置

1. 第一个 block 源审计命令：

   ```bash
   .venv-ubuntu/bin/python qa/run_route_a_1m_teacher_native.py \
     --dry-validate-source --window inner-screen-1 \
     --teacher T01_EMA_9_21 --exit X1 --scenario base \
     --prereg configs/route_a_1m_teacher_native_prereg_20261026.json \
     --output .artifacts/route-a-1m-teacher-native-s07-source-audit-block1
   ```

   结果：exit 0；`DQ-B`，唯一 finding `conservative_availability`；registered Train grant 和 protected-month guard 通过。完整矩阵执行器会在三个 block 各自重新审计。

2. 先前一次 screen 试跑因 host 要求“evaluation 必须复用冻结训练 feature scaler”而 fail-closed 拒绝，未产生可接受结果。修正为只在 development-fit 区间拟合并固化 scaler，再让 screen factory 以 `fit_scaler=False` 复用；没有从任何 screen block 拟合 scaler。

3. 冻结的 development-fit scaler：SHA-256 `3f9a076e6d62ff152c0ccec11b2dfc9ccfb91d56b57ad0152dc71582caee1a1e`；fit window hash、feature hash、sample count 和训练质量收据见该研究输出的 `preparation/development-fit-scaler/development-scaler.json`。

4. 全长试跑命令：

   ```bash
   .venv-ubuntu/bin/python qa/run_route_a_1m_teacher_native.py \
     --window inner-screen-1 --teacher T01_EMA_9_21 --exit X1 --scenario base \
     --prereg configs/route_a_1m_teacher_native_prereg_20261026.json \
     --output .artifacts/route-a-1m-teacher-native-s07-pilot-t01-x1-base-block1-run02
   .venv-ubuntu/bin/python qa/verify_route_a_1m_teacher_results.py \
     --output .artifacts/route-a-1m-teacher-native-s07-pilot-t01-x1-base-block1-run02
   ```

   实际：41,757/41,757 决策及 M2.2 barrier 通过；T01 EMA/X1/Base，978 个完整 flat-to-flat 案例，0 右删失、0 强平；raw `LONG_ENTRY=978`、`EXIT_LONG=978`、`NONE=39,801`；1,956 个提交/成交订单。账户 10,000 → 9,767.46944106 USDT，净收益 -2.3253056%，最大回撤 2.426046%，费用 243.19037546 USDT，滑点 49.60255 USDT，资金费 -1.71224348 USDT。验证器状态 `VERIFIED_SCENARIO_ONLY_RESEARCH_BLOCK`。

5. 控制策略 smoke 命令：

   ```bash
   .venv-ubuntu/bin/python qa/run_route_a_1m_teacher_research.py \
     --smoke-control --output .artifacts/route-a-1m-teacher-native-s07-control-smoke-03
   ```

   实际：FLAT-window 1,437 决策 M2.2 通过；固定 LONG 1,437/1,437，0 强平；Exposure-matched LONG/SHORT 各 1,437/1,437 决策、各 57 个 native fill events、0.50174 exposure-time fraction，mean absolute scheduled exposure 0.125605、native realized mean absolute exposure 0.124574。结果均为场景诊断，非资格结论。

## 全矩阵运行状态

运行命令：

```bash
.venv-ubuntu/bin/python qa/run_route_a_1m_teacher_research.py \
  --prereg configs/route_a_1m_teacher_native_prereg_20261026.json \
  --output .artifacts/research_1m_4teachers_v1
```

`--dry-plan` 实际列出 48 个教师账户（4 × 2 × 2 × 3）、18 个固定控制账户和 96 个 exposure-matched native 控制账户，共 162 个顺序独立 episode。当前运行进度见 `progress.json`；用户要求的源/账户/逐分钟 CSV、交易案例和 matrix/report 文件以最终报告为准。

### 全矩阵试跑与资源停止证据

计划展开为 48 个教师账户（4 × 2 × 2 × 3）、18 个固定控制和 96 个 exposure-matched 控制，共 162 个串行、独立 native episodes。输出目录 `.artifacts/research_1m_4teachers_v1/` 内保留最初两个完整教师账户（T01 EMA X1/X2、Base、inner-screen-1）及其四个完整方向匹配控制。它们均有 41,757/41,757 决策，M2.2 对账通过；本段不是全矩阵资格结论。

完整账户恢复命令（只读注册 Train 来源；保护月份仍先行拒绝）：

```bash
.venv-ubuntu/bin/python qa/run_route_a_1m_teacher_research.py \
  --prereg configs/route_a_1m_teacher_native_prereg_20261026.json \
  --output .artifacts/research_1m_4teachers_v1 --resume
```

资源阻断实际情况：

- 最初 S07 full matrix 在已完成 2 个账户与 4 个控制后持续运行；当时 live RSS 约 4.397 GiB，超过预注册 `max_memory_mb=4096`，操作者中断，未把未完成账户计入完成数。
- 第一次恢复尝试同样超过上限，live `VmRSS`/`VmHWM` 为 4,895,532 kB（约 4,780.8 MiB）。操作者立即 Ctrl-C；未写入第三个账户的 `native-run.json`，未把失败账户计入完成数。
- 曾在 `.artifacts/route-a-1m-teacher-native-s07-control-smoke-04` 对 compact report snapshots 做短窗诊断；它因省略中间 barrier 的 native order identities 而未通过 M2.2 order reconciliation，明确拒绝，不进入矩阵结果。正式部分账户和后续复测始终保留完整快照。
- 为保留完整快照证据并减少完全重复的原生订单对象，改为共享内容未变的 detached native order row、任一字段变化时仍保存新快照副本；M2.2 status 映射在 Route A 事实对象中使用只读共享实例。未启用 `compact_report_snapshots`，没有省略快照字段，也未改订单、成交、账户或 gate 语义。
- 上一轮无损优化后恢复时 RSS 达到 4,900,904 kB（约 4,786.0 MiB），超过冻结的 4,096 MiB 单进程上限；该轮在未完成的 T02 episode 中 Ctrl-C，未产生该账户验收结果。此约束是实验协议的单进程 cap，并非当时 WSL 宿主可用 RAM 不足；当前 Ubuntu `free -h` 显示约 23 GiB 可用。后续在不缩短窗口、不关闭核验且保留全部字段的前提下，进一步实现 append-only 历史视图共享及精确 revision 查找。
- 当前恢复命令与上方相同；已完成 12/48 教师账户（包括 T01 EMA/X1/X2 的 Stress block1 及其 exposure-matched LONG/SHORT 控制），开始 T02 ROC/X2/Stress。此前新增的精确 policy-order cache lookup 回归解决了 T01 Stress matched SHORT 在 `2024-02-28 22:13Z` 新提交订单无 after-status 的 fail-closed。当前 live `/proc/<pid>/status` 的峰值 VmHWM 为 2,886,680 kB（约 2,752 MiB），低于冻结上限。账户只有在完整 native runner 和全 M2.2 验证完成后才计入 `progress.json`。恢复尚未完成，不得将这个中间状态宣称为 S07 验收。
- 完整快照小窗 controls 在优化前后都通过：`--smoke-control --output .artifacts/route-a-1m-teacher-native-s07-control-smoke-fullsnap-06` 和 `...-fullsnap-07` 均 exit 0；固定 LONG、exposure-matched LONG/SHORT 各完成 1,437/1,437 次 M2.2 barrier。两次进程峰值分别 1,616.63 MiB、1,601.16 MiB；方向匹配实盘路径均 57 个 native fill events，mean absolute actual exposure 0.124574、scheduled 0.125605。它只能证明小窗链路，不解除长窗资源阻断。

账面部分结果（不是经济选优）：T01 EMA X1/Base 回报 -2.3253%，最大回撤 2.4260%，费用 243.1904 USDT、滑点 49.60255 USDT、资金费 -1.71224 USDT；X2/Base 回报 -3.8745%，最大回撤 4.0130%，费用 333.1646 USDT、滑点 67.94303 USDT、资金费 -0.27556 USDT。连续组合退出对照包含退出时点引发的后续再入场路径差异，且 opportunity-level exit isolation 仍是 `EXIT_ISOLATION_BLOCKED`，不能把差异解释为单次退出的因果收益。

### 本次 Ubuntu 命令和测试

- 本阶段新增/修改源码：`qa/run_route_a_1m_teacher_research.py`（完整账户预验签/安全续跑）；`packages/gym_bridge/src/rles_gym_bridge/historical.py`（完整 native snapshot 的共享历史视图）；`packages/nautilus_adapter/src/rles_nautilus_adapter/session.py`（只按精确 session-owned client-order ID 补查 native cache，以保留 account-scoped reader 漏掉的提交状态）；`packages/route_a_imitation/src/rles_route_a/integration/host_gym_projection.py`（只读共享 status mappings 与 fail-closed 诊断）；相关 Route A 测试。快照字段没有被精简。
- `git rev-parse --show-toplevel` → `/mnt/d/Program Files/NautilusTrader_RL_ES_IL_Route_A`；本阶段所有相关命令均在该 WSL Ubuntu 24.04 canonical root 内运行。
- `.venv-ubuntu/bin/python -m py_compile packages/gym_bridge/src/rles_gym_bridge/historical.py qa/run_route_a_1m_teacher_research.py qa/run_route_a_1m_teacher_native.py` → PASS。
- `.venv-ubuntu/bin/ruff check packages/gym_bridge/src/rles_gym_bridge/historical.py packages/route_a_imitation/src/rles_route_a/integration/host_gym_projection.py packages/route_a_imitation/tests/test_v03_l3_gym_adapter.py`（以及此前 QA runner 涉及文件）→ PASS。
- `.venv-ubuntu/bin/mypy --follow-imports=skip --ignore-missing-imports packages/route_a_imitation/src/rles_route_a/integration/host_gym_projection.py qa/run_route_a_1m_teacher_research.py qa/run_route_a_1m_teacher_native.py qa/verify_route_a_1m_teacher_results.py` → `Success: no issues found in 4 source files`（较新聚焦检查覆盖 6 个变更源文件，结果同为 PASS）。
- 本轮聚焦 `.venv-ubuntu/bin/pytest -q -o addopts='' packages/route_a_imitation/tests/test_v03_l3_gym_adapter.py packages/route_a_imitation/tests/test_v03_native_gym_rollout.py packages/route_a_imitation/tests/test_v03_host_projection.py packages/route_a_imitation/tests/test_v03_transition_builder.py packages/route_a_imitation/tests/test_v03_wfo_train.py qa/test_route_a_1m_teacher_native.py` → `76 passed in 19.39s`。
- 最新 account-scoped native-order regression：`.venv-ubuntu/bin/pytest -q -o addopts= qa/test_route_a_1m_teacher_native.py` → `6 passed in 17.59s`；Ruff 对 adapter session、runner、Route A projection 与测试通过；聚焦 MyPy 对 3 个源文件 `Success: no issues found in 3 source files`。
- 完整快照共享 smoke：`.venv-ubuntu/bin/python qa/run_route_a_1m_teacher_research.py --prereg configs/route_a_1m_teacher_native_prereg_20261026.json --output .artifacts/route-a-1m-teacher-native-s07-control-smoke-shared-history-08 --smoke-control` → exit 0，固定 LONG 与 exposure-matched LONG/SHORT 均 1,437/1,437 M2.2；RSS 峰值 1,604.79 MiB；`native_snapshot_compaction=false`、`native_snapshot_history_sharing=true`。
- S06 已记录的全包 MyPy：`Success: no issues found in 244 source files`；它不代表本次 S12 全面回归已执行。

结论：S07 尚未签收，也不能判定 `NO_ELIGIBLE_ALPHA_TEACHER` 或 G1/G2 资格。历史运行触发的是冻结的 4,096 MiB 单进程 cap；当前对完整快照表示作无损共享后的恢复仍在跑，须等全矩阵、全 M2.2 核验及原定验收门完成后再定阶段结论。不应缩短窗口、关闭核验或改变交易方案来绕过上限。

S08–S12 均未开始；S09/G1 未评估，S10 不允许启动 Position Value Model，S11 学生训练未启动，S12 全面回归未执行。全部结果继续标注 `D1_DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_G1_G2` / `SCENARIO_ONLY`。不得据此启动 Position Value Model 或 BC/DAgger/IQL/PPO 学生。



## 最终矩阵结果（2026-10-26）

- 原生矩阵完成 162/162：48 个教师账户、114 个控制账户；全部账户通过本研究的 M2.2 回放核验。
- 4 个固定规则教师 × 2 种退出 × 3 个 Train-inner 窗口：Base 的 24/24 教师账户净收益为负，Stress 的 24/24 也为负。没有从这些结果中选教师。
- 资格状态：D1_DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_G1_G2；G1 为 NOT_ASSESSED_DEVELOPMENT_DIAGNOSTIC_ONLY；G2 为 NOT_ASSESSED_SCENARIO_ONLY；经济口径为 SCENARIO_ONLY。退出隔离仍为 EXIT_ISOLATION_BLOCKED。
- 逐账户摘要见 [最终摘要](S07_FINAL_SUMMARY.md)，机器可读审计见 [完整报告](S07_FINAL_REPORT.json)。
