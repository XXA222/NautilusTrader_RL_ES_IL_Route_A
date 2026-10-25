# Route A -> NautilusTrader-RL-ES 快照 80194a 集成手册

**基线**：`80194a587444bf4a11f06f6e273723e5754f2483`（branch `codex/archive-20261018-development-snapshot`）。该分支只读；请在新的 Route A 集成分支上操作。

## 代码归属与用户原项目的边界

本交付的源码只包含 `rles_route_a` 新模块、测试、配置、示例和文档。**不包含 NautilusTrader、NautilusTrader-RL-ES 现有 12 个包、数据集、已有实盘凭据、历史经济报告、训练权重。**

源码所用的 `PointInTimeObservation`、`NativeActionTrace` 是 **Route A 只读、最小投影视图**，不替换原项目 L1 DTO 或 L3 原生事件。实际 L3/风险、ExecutionPlanner、撮合与账户事实始终由原项目提供。

## 以最小变更集成

1. 在原仓库以固定 SHA 新开研发分支，不修改 archive 分支或 NautilusTrader 上游源码。
2. 复制本源码包**整个目录**到原仓库 `packages/route_a_imitation/`。现有 `pyproject.toml` 的 `members = ["packages/*"]` 自动发现新增包；不要把 `rles_world_model` 移走或覆盖。
3. 在原仓库 3.14.7 环境执行 `uv lock` 更新新 package 的 workspace lock，然后在原仓库执行 `uv sync --locked --all-packages --no-editable --extra application`（extra 是否可用取决于实际安装入口；不适用时使用项目根已有安装命令）。不要在本方案中写入官方仓库现有包的源码副本。
4. 为 Route A 的读取和回测各实现一个显式 host adapter（见下）。所有真实数据先由 `ProjectResearchRegistry` 登记并获取合法 Train grant，然后交给 L2 `read_registered_train_window()` 或当前真实受限 Reader。Route A **不直接读取本地市场 Parquet**；这防止绕过研究权限控制。
5. 从 `rles_data` 的已授权 feature rows 构造 `project_visible_observation(...)`，`available_at` 必须覆盖全部市场字段；账户来自 L3 native snapshot，不在新包计算资金费用或生成账户事实。
6. `TrendTeacher/WaitTeacher/RecoveryTeacher` 的 advice 先经现有 SoftRisk、HardRisk、ExecutionPlanner 和 native engine 回放，再用 `native_teacher_step(...)` 生成监督样本。若 `NOOP/WAIT/CANCEL/UNFILLED`，绝不能把 requested target 当已实现敞口。
7. 使用 `teacher_quality(...)` 过滤没有真实可执行成交的 alpha 教师；空仓和风险教师可作为 baseline 数据，但不能宣称具有 alpha。必须固定 `zero_target_policy="close"`，并通过 `verify_native_flat_lifecycle(...)` 校验取消/ACK/平仓完整流程。
8. GRU/BCTrainer 仅接收已授权 Train、且真实风控转换得到的 `LabeledStep`；模型保存为 NPZ + JSON，不加载任意 Python pickle。IQL/TD3+BC 的离线 Actor 推断 bundle 使用安全 NPZ，只导出 Actor 参数。v0.2 另外新增独立 `training.checkpoints`，可以恢复 Actor/Critic/Optimizer/目标网络/采样器 RNG；**训练 Checkpoint 与 Actor-only 模型不能混用**。
9. DAgger 只能从真实 Native student rollout `VisitedStudentState` 收集样本，教师必须在**学生实际**账户/订单状态重新决策。`NativeExpertProjection` 是 host 负责的无副作用反事实标签投影；其输出必须是 `ProjectedTeacherTarget` 且状态标记 `PROJECTED`，**不能误标为 FILLED**；不可以从旧教师历史标签复制。
10. Offline IQL/TD3+BC 先通过 RA3、经济 reward 守恒和 replay 覆盖门，调用 `fit_offline(..., readiness=OfflineReadiness(...))` 时所有状态必须由原研究系统审核通过。使用预先冻结的 BC Encoder，状态为其定长序列编码；动作是**学生选择的请求目标**，不是下游经过 Risk/订单改写后实际仓位。所有经济 transition 来自 Engine 原生账户，标准统计使用 L6 账本。
11. Route A 不打开正式数据集；注入 `AuthorizedFormalEvaluator` 只能在原 L6 完成注册与 query reservation 以后。Validation/Test/Holdout 的一次性控制不能靠本包自行颁发权限。

## 必要 Provider 对接清单

| 新接口 | 原仓库权威来源 | 尚需写的桥接代码 |
|---|---|---|
| `HostRegisteredReader` | `rles_research.access.ProjectResearchRegistry` + L2 scoped Train reader | 将真实 L6 注册收据映射为 `HostTrainGrant`；不能自造来源 |
| `project_visible_observation` | `rles_data` 已闭合 K线/特征与 L3 observation/schema | 仅投影字段及 available_at；特征值必须用现有 scaler |
| `teacher_native_rollout` | `rles_gym_bridge` / L3 原生回放 | 迭代产出真实 TeacherFact，不另建引擎 |
| `L3ActorAdapter` | `rles_execution.planner` + risk + adapter | 只转发 target [-1,1]，sink 调用原有执行链 |
| `NativeExpertProjection` | L3 action/risk/planner | 学生状态下的专家目标合法性投影，输出 ProjectedTeacherTarget（带执行配方/PROJECTED 状态），不能伪造成交 |
| `AuthorizedFormalEvaluator` | `rles_research` 选择账本、WFO、统计与 Base/Stress | Host 先原子预留 query，再把 real facts 交由原 metrics |

### 原仓库中的已知能力现状

- 当前 Gym + L4 PolicySpec 仍为单维 `Box([-1,1], shape=(1,))`。
- ETHUSDT NETTING 为主线；BTC/ETH/SOL 泛化和 Hedge Mode 属 M4.1/M4.2 延后工作，不能仅在 Route A 发明第二套双侧账本。
- L3 `ExecutionPlanner` 在 `zero_target_policy=close` 且目标 0 时可越过通常 deadband，但仍可能遇到挂单取消确认和 dated min_qty/min_notional 限制。
- M2.2 已有逐决策订单/执行/权益 trace，RA0 不应复制这一体系。
- M3.3 最近 Train-inner 报告中的部分策略完全无成交，因此新教师必须证明真实 action reachability。
- Route B 世界模型属于独立 Python/JAX 研发环境；本包无 JAX 与 DreamerV3 依赖。
- 历史合约规则、maker/taker 等仍有 scenario-only 限制。没有真实数据授权时不允许将本代码的通过测试表达为收益提高。

## 不能以当前源码包独立证明的工作

- 真实 Binance K 线的获取、DataTruth 质量、历史规则的来源与完整性；
- NautilusTrader 原生 Engine 的下单、取消、账户/权益与资金费正确性；
- 参数与真实交易风控配置联动后的完整通过门槛；
- `uv.lock` 在目标 Windows/WSL2 Python 3.14.7 + GPU/CUDA 的依赖复现；
- WFO 统计显著性与同风险教师收益比较、Validation/Test/Holdout 结果；
- RA4 IQL 在真实交易数据的收益提升。

因此交付只承诺 **核心算法 / 数据契约 / 合成专项测试** 已实现；正式项目集成与经济验证是用户原仓库的独立工作。


## v0.2 新增注意事项

- 配置 `max_gap_ns` 必须与真实决策屏障的允许间隔一致；训练数据集及在线推断在 gap 时统一截断历史。
- `NativeTeacherStep` 的 `WAIT/CANCEL/UNFILLED` 和与教师目标不一致的 `NOOP` 不再提供有效 BC 标签；不能使用学生的旧持仓当作教师应模仿的 EXIT 目标。
- DAgger 同迭代+policy+episode 单独形成 trajectory；原生学生持仓必须由宿主回放事实提供。
- 正式 IQL/TD3+BC 数据必须填 `fold_id, episode_id, decision_ts_ns, next_decision_ts_ns, native_execution_recipe`，并在 `OfflineReadiness.registered_fold_id` 绑定；`allow_synthetic_unit_only=True` 只能用于隔离的合成算法测试。
- 训练续跑须同时恢复 `ReplaySampler.state()` 与 TD3 target-noise RNG；`fit_offline(..., sampler=...)` 可消费已恢复采样器。
- `native_action_trace` 不提供唯一 FillEvent ID 时不要用订单 ID 跨 barrier 强行去重；资金流经济和真实成交仍必须由 L3/L6 审核。

- **GPU 显存控制**：RA4 的 `fit_offline` 在 CPU 持有冻结经验特征，按随机 mini-batch 拷给 learner 的 CUDA 设备；宿主不要再把整份 encoded replay 预先上传到单 GPU。
