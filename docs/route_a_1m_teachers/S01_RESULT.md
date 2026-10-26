# S01 — Ubuntu 代码可靠性与旧研究前瞻缺口整改

执行日期：2026-10-25（Asia/Shanghai）

## 阶段结论

`S01 PASS`。任务书列出的 20 项 Ubuntu MyPy 错误已经由进入本任务前保留在工作树中的未提交修改修复；本轮逐文件审计、实测并补齐“窗口末盈利/亏损未平仓都必须入账”的双向回归。旧冻结配方和历史报告未覆盖，生产 `teacher_gate.py`、student gate 与经济阈值未修改。

## 类型整改审计

已核对的现有前置修改：

- `candidate_events.py`：候选方向缩窄为 `Literal[-1, 1]`。
- `teachers/selector.py`、`teachers/causal_position.py`：意图变量显式使用契约 `Intent`。
- `opportunity_models.py`：NumPy array/`Sequence[float]` 桥接；完整收益通过 `_complete_return()` 对 `None` 和非有限值 fail closed。
- `teacher_research_gate_v2.py`：`reason_codes` 使用明确 `tuple[str, ...]`，避免临时变量推断冲突。
- `opportunity_labels.py`、`causal_exit_replay.py`：删失时钟、退出时钟和退出价格均显式收窄；缺失必需事实时报错。

没有通过全局 MyPy 放宽或批量 `type: ignore` 压制问题。

## 未来结果筛选整改

旧 `_portfolio` 在入场前遇到 `net_return_bps is None` 或 `exit_ts_ns is None` 就 `continue`，会按未来是否完整退出筛选当前交易。现有整改采用下列边界：

1. `admit_single_position_candidates()` 的入场 API 不接收 outcome/return，只按事件时钟、下一合法开盘、账户占用及独立 block 边界顺序准入。
2. outcome 只在准入之后用于绑定校验和结果记账；不能反向改变入场集合。
3. 已平仓路径核对费用、滑点、资金费和模拟收据；未平仓路径只用 block 内最后合法 mark 计入账户结果。
4. 未平仓路径保持 `TERMINAL_MARK_ONLY` / censored，只计 entry fill，不虚构 exit fill 或 exit fee。
5. `qa/run_route_a_teacher_e1_e2_g1_20261025.py::_portfolio` 复用该序贯准入与账本；旧脚本身份仍为 fixed-240m historical diagnostic，不冒充动态原生账户权威。
6. terminal force-flat 与 mark-only 分开；自然教师退出与终局处理不混淆。

本轮新增的唯一 S01 代码变化是将删失 mark-only 测试参数化为 `terminal_price=103` 和 `97`，分别覆盖窗口末盈利与亏损。两条都被准入、都进入账户结果、都仍为 censored，不会因为未来盈亏符号被剔除。

## Ubuntu 实测

所有命令均从规范项目根在 WSL2 `Ubuntu-24.04` 中使用 `.venv-ubuntu/bin/python` 执行。

```text
.venv-ubuntu/bin/python -m mypy packages
exit 0 — Success: no issues found in 225 source files

.venv-ubuntu/bin/python -m ruff check packages qa tests
exit 0 — All checks passed!

.venv-ubuntu/bin/python -m pytest -q -p no:cacheprovider packages/route_a_imitation/tests
exit 0 — 311 个测试点，进度 100%，无 failure

.venv-ubuntu/bin/python -m pytest -q -p no:cacheprovider tests/test_historical_factory.py
exit 0 — 70 passed, 18 warnings in 342.11s

.venv-ubuntu/bin/python -m pytest -q -p no:cacheprovider \
  packages/route_a_imitation/tests/test_portfolio_replay.py \
  packages/route_a_imitation/tests/test_gate_funnel.py \
  packages/route_a_imitation/tests/test_opportunity_diagnostics.py \
  tests/test_binance_archive_range.py \
  tests/test_registered_train_window_read.py \
  tests/test_directional_quality_scope.py
exit 0 — 60 passed in 24.61s

git diff --check
exit 0
```

定向回归覆盖：future outcome 不参与 admission、末尾盈利/亏损 mark-only、无 block 内 entry、持仓占用、标签/现金流对账、资金费一次计入、完整/删失分母、16 组合 gate funnel、归档物理保护范围、registered Train 读取与 directional-only 禁止 RL training。

## 合同与停止条件

- MyPy 保留现有错误：否。
- 旧数据保护被破坏：否；保护源守卫定向测试通过。
- 原生经济/student gate 被放宽：否。
- 历史冻结配方或报告被覆盖：否。
- 结论：S01 不触发停止条件，允许进入 S02。
