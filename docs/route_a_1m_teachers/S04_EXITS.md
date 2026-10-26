# S04 — X1/X2 退出与原生净持仓生命周期

执行日期：2026-10-25（Asia/Shanghai）

## 阶段结论

`S04 PASS`（工程/合成与原生 DTO 生命周期级）。四位教师可显式构造 X1 指标退出或 X2 ATR 跟踪退出；两种 variant 复用完全相同的教师入场事件、目标敞口、risk recipe 和 Base/Stress cost IDs。没有新增订单状态机、撮合器、账户账本或资金费引擎。

## X1 / X2 隔离

- `build_teacher_1m(teacher_id, exit_variant="X1"|"X2")` 是唯一 variant 选择点；未知 variant fail closed。
- X1 继续使用各教师自己的 `indicator_exit_rule_id`。
- X2 将退出身份冻结为 `X2_ATR_TRAIL_3_V1`，入场 recipe 不变。
- 自动测试以相同 EMA 交叉观测证明 X1/X2 均发 `OPEN_LONG(0.3)`，reason 和 target 完全一致。
- X2 激活时不再叠加 X1；两种退出贡献可独立比较。

## X2 ATR 语义

- 只读取当前已完成且已可用的 `bar_high_1m/bar_low_1m/bar_close_1m/atr_14`。
- 多头：`peak_high=max(持仓后已完成 bar high)`，`trail=max(prev_trail, peak_high-3*ATR14)`。
- 空头：`trough_low=min(...)`，`trail=min(prev_trail, trough_low+3*ATR14)`。
- 多头 trail 只升不降，空头只降不升；ATR 放大不能放松既有保护线。
- 只有已完成 bar 的 close 低于/高于 trail 后才发 `EXIT`；没有假设在未知的同 bar 内按 trail 价成交。
- 同一 close timestamp 重复推进 ATR exit 会报错，避免同一 bar 更新两次极值。
- profit-protect 默认关闭，没有把额外规则混入 X2。

## 原生生命周期复用

教师只发目标建议；`teachers/protocol.py`、`FrozenTeacherSelector`、`RecoveryTeacher`、`NativeActionTrace` 和 `integration/exit_lifecycle.py` 继续负责既有事实链。

测试轨迹：

```text
FLAT
  -> 指标 entry: OPEN_LONG(0.3)
  -> native pending: PENDING_ACK，仍不假设已成交
  -> account exposure=0.3: HOLD / X1 或 X2 检查
  -> EXIT(0)
  -> pending close: HOLD + PENDING_ACK，仍不假设 flat
  -> native close fill + pending cleared + exposure=0: confirmed flat
```

`verify_native_flat_lifecycle()` 只有在 flat target 始终保持、相关 close order 确有原生 fill、最终 pending 为空且 actual exposure 为 0 时返回 true。仅 target=0、仅 cancel/WAIT、仅 exposure 被写成 0 而无 close fill，都不能证明退出。

硬风险优先级也已回归：账户 drawdown 超限且存在普通 pending order 时，`RecoveryTeacher` 仍先返回 `EXIT/MAX_DRAWDOWN`，不会被普通 pending guard 吞掉。

## 长持仓与终局

- 合成持仓连续运行到 241、481、1441 分钟检查点，均返回普通 `HOLD/POSITION_HOLD`。
- 新教师代码没有 `max_holding_minutes` 普通退出。
- 反向信号在持仓中只请求 EXIT；不在同一分钟反手开仓。
- 既有 host projection 的 terminal force-flat 与 partial fill 定向测试继续通过；terminal flatten 保持 censored/terminal 分类，不变成自然 X1/X2 教学案例。
- 本阶段未改资金费逻辑；资金费仍由原生 L6/历史 factory 计入。真实市场资金费对账属于 S06/S07 native smoke。

## Ubuntu 实测

```text
.venv-ubuntu/bin/python -m pytest -q -p no:cacheprovider \
  packages/route_a_imitation/tests/test_teachers_1m_lifecycle.py
exit 0 — 7 passed

.venv-ubuntu/bin/python -m pytest -q -p no:cacheprovider \
  packages/route_a_imitation/tests/test_contracts_teachers.py \
  packages/route_a_imitation/tests/test_v03_host_projection.py \
  -k "terminal_force_flat or partial_fill or executable_exit or pending"
exit 0 — 4 passed

.venv-ubuntu/bin/python -m mypy \
  packages/route_a_imitation/src/rles_route_a/teachers_1m
exit 0 — Success: no issues found in 9 source files

.venv-ubuntu/bin/python -m ruff check \
  packages/route_a_imitation/src/rles_route_a/teachers_1m \
  packages/route_a_imitation/tests/test_teachers_1m_lifecycle.py
exit 0 — All checks passed!
```

## 停止条件核对

- 用当前 1m 已发生的 trail 价假成交：否，只有 close-confirmed EXIT 建议。
- 资金费另起账本或错误补零：否，本阶段未建账本、未改资金费。
- terminal flatten 伪装自然退出：否，既有分类回归通过。
- pending open/close 重复普通订单：否，统一 protocol guard 回归通过。
- 结论：S04 不触发停止条件，允许进入 S05。

## S05 集成时补充的 X2 时钟回归

同一 epoch-ns 精度问题也影响 ATR trailing 的新 bar 检查。`AtrTrailingExit`
现在从 `PointInTimeObservation.market_clock_ns("latest_1m_close_ts_ns")`
读取严格整数时钟，不再将收盘时间作为 float indicator。新增
`test_x2_uses_exact_epoch_nanosecond_close_clock`；S05 扩展矩阵中 X1/X2
lifecycle 全部通过。普通持仓时间规则没有变化。
