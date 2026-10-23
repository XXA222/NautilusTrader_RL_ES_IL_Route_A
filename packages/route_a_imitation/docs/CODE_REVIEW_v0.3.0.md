# Route A HCE-IL v0.3.0 — 源码审查与第三轮功能交付

**基线**：`XXA222/NautilusTrader-RL-ES@80194a587444bf4a11f06f6e273723e5754f2483`。范围仅为 `packages/route_a_imitation` 新增源码，不修改 NautilusTrader 和项目现有 L1–L6。

**工程结论**：v0.3 增强了真实宿主连接边界、教师标签可达性、DAgger 训练来源隔离、离线转移的账本/时间语义和教师筛选；并不等于已完成宿主 Engine 运行或获取交易收益。现有验证全部使用**合成的宿主字段与账户证据**。

## 1. 审计发现及实际修复

| 严重度 | 发现 | 代码修复 | 仍需宿主证明 |
|---|---|---|---|
| P0 | v0.2 没有直接消费 snapshot `Gym.step()` 字段的安全投影；`decision_ts_ns` 易将执行后屏障当成执行前观测时钟 | 新增 `integration/host_gym_projection.py`，强制分离决策前/推进后时间、识别实际 Gym 字段、检查 native 订单 ID/费用后权益/风险调整，并拒绝终局双重语义 | 对照项目中的完整 L3/M2.2 原生事实和一次真实 Train 回放；不能把 `execution_client_order_id` 当成成交证据 |
| P0 | `SUBMIT` 搭配任意关联旧单成交 ID，或部分/未结订单，原版本可能错误当成成功教师标签 | `native_teacher_step` 只在**当前新订单交集有成交**、实际敞口到达 planner target、无剩余 pending 时标 `FILLED`；否则标 `UNFILLED`，权重 0 | 数量精度、完整订单生命周期和同订单多次 FillEvent 仍由宿主 L3 负责 |
| P0 | RA1 `teacher_quality()` 曾把任意 `filled_order_ids` 计为本教师成交，WAIT/CANCEL 关联的旧单成交可误放行 alpha 教师 | 当前资格门只计本栏位新提交订单的 fill，且须实际到达 planner 目标、无残留挂单；部分成交不计作完整可执行样本 | 最终 FillEvent 数量与手续费仍以独立 M2.2/L3 事实为准，真实 Train 内教师经济门尚未运行 |
| P0 | Gym 的订单状态摘要与 M2.2 执行账本若未对账，Gym 内部 SHA-256 只能证明摘要未被改写，不能证明独立来源 | provider 每个 barrier 调用 `native_order_facts_verifier(info, source, facts)`；`rollout_with_m22_trace()` 结束后，还把 Gym 状态/待处理订单/策略步 fills 与相同决策时钟的完整 `factory.snapshots` 对账，启用 compact snapshots 时失败关闭 | 合成同次 rollout 和状态/fill mismatch 用例已覆盖；真实宿主 verifier 与真实授权 M2.2 receipt 仍需宿主提供 |
| P0 | 训练标签与宿主账户事实缺少统一的 RA4 转移转换 | 新增 `integration/native_transition_builder.py`；使用命令 action、原生 reward、下一真实账户观测、Fold/episode/encoder/source/recipe 与时间戳生成 `OfflineTransition` | **终局强制结算**未在此通用转换器支持；需要宿主 L6 专项终局事实接口 |
| P0 | native equity ratio 不能排除外部转账 | 宿主投影接收 `no_external_cashflow_receipt`，且 provider 在每个非终局决策前调用 `no_external_cashflow_verifier(info, source, receipt)`；终局 receipt provider 同样要求独立 verifier | Route A 只执行宿主验证器并要求严格返回 `True`；实际资金流水读取和 verifier 权限仍归属 L6，synthetic verifier 不能构成真实账户证明 |
| P1 | 零目标平仓链只要看到旧开仓单成交 + 后续仓位变零，可能误判完整主动平仓 | 平仓生命周期要求 `SUBMIT` 的**本次订单**有成交 ID，且最后无 pending、敞口为零 | 原生平仓若来自强制清算/风险指令，必须通过另一套专属流程验证，不冒充政策 EXIT |
| P1 | 审计多 Episode 时输出首尾权益易被误读为同一条复利回报，且允许 Episode A/B/A 交错 | `audit_teacher_barriers` 拒绝交错 Episode，保存 `episode_equity_segments`；区分订单引用 mentions 和唯一引用数 | 不同 Episode 回报需要 L6 使用正确分母与账户重置语义聚合 |
| P1 | DAgger 某次 `NOOP` 投影仍与教师要求的 EXIT 不一致；跨批次可能更换 L3 执行配方 | `DAggerCollector` 屏蔽被阻断的 NOOP，`DAggerSet` 跨迭代校验 `execution_recipe`，原子预检 | DAgger `PROJECTED` 只表假设执行可行，不代表已经产生真实成交或收益 |
| P1 | Offline RL 的 gamma 步长和账本真实分钟/5 分钟时钟不一定一致；样本可能跳过未观察的同 Episode 中间步 | `OfflineReadiness.decision_period_ns` 可绑定时间基准，检查 `dt == discount_steps × period`；拒绝重复决策与隐式 bootstrap 跳空 | 若环境采取不规则决策节拍，应使用明确的合法分段，不要硬编码 1m 假设 |
| P1 | 在线 GRU 推断异常提前写入历史，导致同一决策不能重试 | 先构造候选序列和数值检查，成功推断后才提交 `_series/_last_ts` | 宿主下单失败后的策略回放/事务恢复属于宿主集成测试 |
| P1 | “教师优于同样亏损更大的基准”可能被误写成有效 alpha | `evaluation/teacher_gate.py` 强制费用后**自身正收益**及基准超额、多窗口、Base/Stress、原生匹配、无清算、回撤与最低交易数等 Train-inner 门槛；报告明确 `SCENARIO_ONLY` | 窗口不重叠的事实应经 L6 验证；结果不自动晋级正式 Validation |
| P2 | Replay 数值越界测试抛出可避免的 NumPy overflow warning | 对 float64 源先比较 float32 上界，再做转换 | 更大规模数据/Arrow/Parquet 主链需要在原 L2 测试 |

## 2. 推荐的真实宿主接线方式

1. 原仓库固定 commit 开独立工作分支，复制本包到 `packages/route_a_imitation`。
2. 从宿主 **ProjectResearchRegistry 注册后的** Train fold 和 L2 causal feature rows 创建**决策前** `PointInTimeObservation`。特征的 `available_at_ns` 必须属于可见时刻；账户状态只取 L3 的前态。
3. 以教师命令送入原有 `soft risk → hard risk → planner → native engine`。读取 Gym 单步返回的 `info`；同时从 M2.2 的完整执行轨迹读取订单提交/成交/pending 身份，构造 `L3OrderFacts`。这些真实 L3 订单结果**不在 Gym info 的 `execution_client_order_id` 中自行推断**。
4. 原 L6 原生资金流账本先验证该政策屏障无外部转账，传入 `no_external_cashflow_receipt`。调用 `project_nonterminal_teacher_barrier(...)`。如果 `terminal_settled=True`，**此转换器会拒绝**，由宿主单独提供终局归因。
5. 把返回的 `teacher_barrier` 送入 `audit_teacher_barriers` / `native_teacher_step`，生成有执行证据的 BC 标签。`UNFILLED`、`WAIT`、`CANCEL` 以及被死区阻断的教师目标不得用于有效 BC 监督。
6. 如需 RA4，独立冻结同一编码器对该前态和下一时刻**真实观测**编码，调用 `native_nonterminal_transition(...)`。输出 action 是**原始命令**，不是成交后仓位；包含官方账户 reward（不再减手续费/滑点）。
7. 在宿主授权 Train-only 的多个隔离内折内，用 `NativePair` + `screen_teacher` 评估教师，验证两窗口确实不重叠、场景/seed 矩阵完整且经济基准一致。正式验证仍在 L6。

## 3. 可量化阶段门槛与下一步

- **RA0-S1**：真实 Gym `info` × M2.2 order trace 样例至少覆盖 `NOOP/deadband`、`WAIT`、`CANCEL/ACK`、`SUBMIT+FILLED`、`SUBMIT+UNFILLED`、`policy_flat` 与终局分离。每步通过原生经济守恒，任何成交只有 L3 证据才能被标为可训练。
- **RA0-S2**：1m、1h、1d 完成 bar 可见性、60s 或合法 5m/15m 决策时间基准、Train scaler 与线上推断的观测一致；故意注入未来特征必须失败。
- **RA1-S1**：候选趋势/突破/均值回归教师与空仓、固定方向等基线至少跨 2 个 Train-inner 隔离窗口、Base+Stress、>=2 种 seed 比较；若教师自身经济收益或最差场景超额不为正，不晋级 BC 训练的大规模投入。
- **RA2-S1**：BC-GRU 与 BC-MLP 同数据同费用下有独立错误率、动作 reachability、持仓时长、换手率与 native PnL 归因。部署异常/断档不得污染 GRU 状态。
- **RA3-S1**：只接收真实学生 Native rollout 状态；DAgger 不能跨 Fold、来源、执行配方、trajectory 混合。
- **RA4-S1**：IQL/TD3+BC replay 需要可追踪原生 `(s, command, r_native, s')`，并且终局结算、折扣时钟、encoder/source 一致。Stage RA4 不应因更多更新预算而自动开启。

## 4. 明确不做的事情

- 不复制或重构 NautilusTrader 原生 Engine。
- 不为 1m bar 主线设计订单簿/交易队列。
- 不改变 M4.1/M4.2 仍延期的多标的/Hedge Mode 主线。
- 不在 Route A 内部打开正式 Validation/Test/Holdout，也不声明实盘盈利。
- 不把合成订单回放/后验收益充当真实 Binance 历史撮合结论。
