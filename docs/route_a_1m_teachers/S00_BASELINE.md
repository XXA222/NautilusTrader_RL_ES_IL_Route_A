# S00 — Ubuntu 工作树、环境与旧版回归基线

执行日期：2026-10-25（Asia/Shanghai）

## 阶段结论

`S00 PASS`。唯一规范项目根、固定备份分支、固定审查 SHA、WSL2 Ubuntu 24.04 与项目 `.venv-ubuntu` 均已实测确认。现有工作树包含前置未提交源码和研究文件；这些文件已原样保留，没有 reset、clean、checkout 覆盖、stash、提交或推送。为避免移动固定备份分支，已在同一个规范工作树从同一 HEAD 创建并切换到 `codex/route-a-1m-multiteacher-v1`。

## 源码身份与工作树保护

- Windows 规范根：`D:\Program Files\NautilusTrader_RL_ES_IL_Route_A`
- WSL Git top-level：`/mnt/d/Program Files/NautilusTrader_RL_ES_IL_Route_A`
- `.canonical-project-root`：`D:\Program Files\NautilusTrader_RL_ES_IL_Route_A`
- 基线与当前 HEAD：`6e277d77c2619f756771191cedcee93159412b59`
- 固定备份分支：`backup/20261025-current-project`，仍指向固定 SHA
- 当前开发分支：`codex/route-a-1m-multiteacher-v1`，从固定 SHA 创建
- 未执行 GitHub push。

开始时已有 14 个 tracked 修改文件：

```text
docs/ROUTE_A_TEACHER_CONTINUATION_20261025.md
packages/data/src/rles_data/binance_archive.py
packages/gym_bridge/src/rles_gym_bridge/historical.py
packages/orchestrator/src/rles_orchestrator/application.py
packages/route_a_imitation/src/rles_route_a/evaluation/teacher_research_gate_v2.py
packages/route_a_imitation/src/rles_route_a/teacher_research/__init__.py
packages/route_a_imitation/src/rles_route_a/teacher_research/candidate_events.py
packages/route_a_imitation/src/rles_route_a/teacher_research/causal_exit_replay.py
packages/route_a_imitation/src/rles_route_a/teacher_research/opportunity_labels.py
packages/route_a_imitation/src/rles_route_a/teacher_research/opportunity_models.py
packages/route_a_imitation/src/rles_route_a/teachers/causal_position.py
packages/route_a_imitation/src/rles_route_a/teachers/selector.py
qa/run_route_a_teacher_e1_e2_g1_20261025.py
tests/test_binance_archive_range.py
```

开始时还已有未跟踪配置、文档、研究模块和测试，包括 2023 Train prereg/errata、`gate_funnel.py`、`opportunity_diagnostics.py`、`portfolio_replay.py` 及其测试。它们被视为用户/前置工作，不冒充本轮新建；后续阶段只在审计后补缺口。

## Ubuntu 与 Python

实际命令及结果：

```text
wsl.exe --list --verbose
Ubuntu-24.04  Running  VERSION 2

pwd
/mnt/d/Program Files/NautilusTrader_RL_ES_IL_Route_A

uname -a
Linux ... 6.6.87.2-microsoft-standard-WSL2 ... x86_64 GNU/Linux

python3 --version
Python 3.12.3

ls -l .venv-ubuntu/bin/python
.venv-ubuntu/bin/python -> /home/freqai/.local/share/uv/python/cpython-3.14.7-linux-x86_64-gnu/bin/python3.14

.venv-ubuntu/bin/python --version
Python 3.14.7

.venv-ubuntu/bin/python -m pytest --version
pytest 9.1.1

.venv-ubuntu/bin/python -m mypy --version
mypy 2.3.0 (compiled: yes)

.venv-ubuntu/bin/python -m ruff --version
ruff 0.17.0
```

系统 `python3` 不是项目规定的 3.14.7，因此后续 pytest、mypy、ruff、compileall 和 QA 脚本一律显式使用 `.venv-ubuntu/bin/python`。未创建新 venv，未执行 `pip install -e`，未降级 Python。

`uv.lock` SHA-256：`25c47e24de0cc1ba19a88457d6afd2d0f5ddee166f9898674ad58b9b6bf22957`。

## 旧 CI 失败与当前基线实测

任务书记录的旧 Ubuntu CI 基线为 `mypy packages` 在 219 个源文件中有 20 项错误、涉及 7 个文件。当前工作树的前置未提交修改已经改变该状态，因此本轮重新实测而不沿用旧报告：

```text
.venv-ubuntu/bin/python -m mypy packages
exit 0 — Success: no issues found in 225 source files

.venv-ubuntu/bin/python -m ruff check packages qa tests
exit 0 — All checks passed!

.venv-ubuntu/bin/python -m pytest -q -p no:cacheprovider packages/route_a_imitation/tests
exit 0 — 进度 100%，310 个测试点，无 failure 输出

.venv-ubuntu/bin/python -m pytest -q -p no:cacheprovider tests/test_historical_factory.py
exit 0 — 70 passed, 18 warnings in 342.11s

git diff --check
exit 0
```

18 项 warning 均来自安装依赖在 Python 3.14/NumPy 2 下的弃用或未来兼容提示；无项目测试 failure。S01 仍须审计上述前置修改是否满足合同，而不能仅凭零类型错误宣布完成。

## Train 来源只读盘点

本阶段没有读取 `market.parquet` 行情值，没有读取 Test/Holdout 原始包，没有下载数据。只读检查 `.artifacts/research-access.sqlite`、prereg、导入清单和完成收据得到：

- 注册簿表计数：`access=697`、`inventory_import=2`、`query_intent=0`、`trial_attempt=0`、`trial_attempt_event=0`。
- 已登记新 Train 协议：`route-a-teacher-core-train-20230601-20231210-v1`。
- access 状态：`recorded/development`；市场 `ETHUSDT`；family `perpetual-market-v1`；物理授权范围 `2023-05-29T00:00:00Z` 至 `2023-12-10T00:00:00Z`；`pre_read_train_scope=true`。
- 冻结 prereg 角色：fit `2023-06-01..2023-09-01`，calibration `2023-09-01..2023-10-01`，五个顺序 inner-screen 块覆盖 `2023-10-01..2023-12-10`。
- 已有导入：`data/routea_teacher_train_20230529_20231209/market.parquet`；完成收据记录 `843570` 个事件、52 个 row groups、snapshot hash `e789dc7f7ff21b240c35b46c0a36f1938bf24ec2e36315c83696748b229e73ef`。
- 数据流清单含 USD-M 1m klines、markPriceKlines、indexPriceKlines 与 2023-05..12 monthly fundingRate；这些月份早于注册簿中的保留 Test/Holdout 时段。
- 该数据先前的旧教师原生试跑在 rollout 前因 `DQ-C permits directional-only, low-leverage research` 停止，`completed_runs=0`；已有 errata 只授权 directional、禁止 RL training，经济身份仍为 `SCENARIO_ONLY`。
- 当前 access 记录可以支持工程/Train-inner 研究来源审计，但新 1m 多教师实验仍必须在真实读取前另建与本任务 recipe 对应的 prereg/query intention，并经过 dry source validation；不能沿用旧 5m/fixed-240m arms 冒充新实验授权。

已知旧 `2026-08-12..2026-10-02` 教师 Train 来源的 monthly funding 物理范围曾与保护区间重叠，不能用于本任务资格结论。S00 没有打开该市场文件。

## S00 停止条件核对

- 源码身份：已确认。
- 现有虚拟环境：可用且版本正确。
- 本地工作保护：固定备份分支未移动，dirty files 未覆盖。
- 结论：不触发 `BLOCKED_ENV_OR_WORKTREE`，允许进入 S01。
