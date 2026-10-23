# v0.3 新增宿主接入契约（基于 snapshot 80194a）

原始完整集成指导见 `INTEGRATION_SNAPSHOT_80194a.md`。本补充仅讲解新增接口和严格边界，不复制原项目代码。

## A. Snapshot Gym 信息正确时钟

在该快照，`Gym.step()` 返回 `info["decision_ts_ns"]` 对应**推进后**原生 barrier，而教师决策属于之前的稳定观测时点。`project_nonterminal_teacher_barrier` 因而要求两个时钟严格上升，并将教师 `NativeActionTrace` 绑定**执行前**时钟。

必需宿主注入：

- 决策前 `PointInTimeObservation` 和 `ExpertAdvice`；
- Gym 返回的非终局 `info`，包含 `policy_equity_before/after/change`、`r_econ`、`requested_action`、`risk_adjusted_target`、`quantity_rounded_target_exposure`、`policy_executed_action_pre_terminal`、`execution_client_order_id` 等；
- 独立 M2.2 `L3OrderFacts`，包含已提交、已成交、关联旧单、仍在途单与非空原生证据收据；
- `native_order_facts_verifier(info, source, facts)`，必须从独立 M2.2 执行 trace 对账并严格返回 `True`；
- 冻结 `execution_recipe/source_identity`、L6 签发的 `no_external_cashflow_receipt`，并通过 `no_external_cashflow_verifier(info, source, receipt)` 逐步验证；verifier 必须严格返回 `True`。

**不可偷换**：`execution_client_order_id` 只是提交 ID；`execution_plan_action=="SUBMIT"` 不代表成交。`project_nonterminal_teacher_barrier` 只处理非终局步。终局步必须走 `project_terminal_teacher_barrier` 并注入由宿主 L6 签发的 `TerminalL6Receipt`；使用 `terminal_receipt_provider` 时还必须提供独立 `terminal_receipt_verifier`，逐次验证来源与收据并严格返回 `True`。收据绑定 Gym 终局结算摘要、两个 barrier 时钟、来源、币种和结算权益。终局强平 fill 保留在结算事实中，不并入教师动作的成交 ID。Route A 只调用宿主 verifier，不签发或验证真实账户流水；synthetic verifier 只用于合成测试。

`NativeGymTeacherProvider.from_prepared_factory()` 是应用接入入口：它读取 `prepare()` 保存的注册 Train 回执，先通过宿主 verifier 回查 ProjectResearchRegistry 中的 key/token/owner/窗口/身份，再生成 `HostTrainGrant` 并从已准备 factory 创建 Gym 环境。provider 每步执行后先从 Gym info 投影 order facts，再要求独立 M2.2 verifier 对同一栏位核验，之后再核验 L6 资金流与终局收据。所有 verifier 必须严格返回 `True`。此外，正式 BC 训练使用 `rollout_with_m22_trace()` 将每步 Gym 的订单状态、pending IDs 和 FillEvents 按准确决策时钟与 `factory.snapshots` 中保留的 M2.2 原生快照交叉核对；启用 `compact_report_snapshots=True` 会被拒绝，缺快照或差异都会失败。快照交叉核对加强了同次运行的证据关联，但不取代宿主独立 M2.2 verifier，也不签发 L6 收据。当前包不提供 M2.2/L6 的真实权威实现，synthetic 回调仅能证明拒绝路径，不构成真实订单或账户证据。没有注册读取回执或登记记录不匹配的 factory 会被拒绝；Route A scope DTO 本身不签发项目权限。

教师案例的经济分类默认始终为 `SCENARIO_ONLY`。只有宿主额外注入 `native_economic_evidence_verifier(info, source, order_facts)` 并逐步严格返回 `True`，才可将本次屏障标为非情景证据；该 verifier 必须证明该来源的市场事件、成交与费用账本确属该案例允许的原生证据，不能只复述 Gym 的订单状态。通过后，案例审计仍要求同一 episode 从平仓出发、经 M2.2 已核验的开仓成交，并最终由已核验成交平仓或 L6 终局结算；单步账户收益、情景回放、未完成仓位、原子翻仓和现金流未调整样本都不成为正向 BC 标签。完整案例收益按开仓前权益至平仓/结算后权益计算，收据摘要绑定轨迹内的决策、订单、FillEvents、权益和 verifier 身份。负收益往返仍作为独立原生诊断案例；不把整段老师动作简单反向标注。另会对其平仓入场点生成显式来源为 `INFERRED_NEGATIVE_TRADE_ENTRY_AVOIDANCE` 的反事实标签：只教学生在相同平仓状态下保持 `FLAT`，该标签不是老师实际说过的话，也不改写该步的原生实际敞口；持仓、减仓和退出步骤仍不监督。采样器只接受此来源类型及其负收益案例收据，WFO 输出分开记录原生案例计数和该类反事实标签数。若训练窗没有至少一个正向完整原生案例，BC 拟合与 WFO 候选打包均失败关闭。案例 `case_end_ts_ns` 还必须留在注册拟合窗口内，避免退出结果越界泄漏。

DAgger 同样要求注入 L6 `train_grant_verifier(grant)`，且必须严格返回 `True`。`DAggerCollector.collect()` 在调用学生状态 provider 前核验授权；RA3 `run_native_recovery_iteration()` 在调用 host recovery runner 前先核验同一 grant，随后每个 episode 仍须通过独立 native receipt verifier。两层校验用途不同：grant verifier 证明当前 Train reservation/source 有效，episode receipt verifier 证明这次 native recovery 确实绑定指定 prefix、fault、source 与执行配方。合成测试中的 verifier 只覆盖控制流和拒绝行为，生产接入必须绑定宿主 L6 权威。

`HostTrainGrant.from_registered_read()` 还会校验读回执中的 marker 是 64 位十六进制 SHA-256，并拒绝非字符串 scope 身份或非布尔的 `registered_before_read`。`DAggerSet` 要求初始专家样本都绑定同一个非空 Train source，后续每条聚合标签也必须匹配该 source。这样即使某个调用跳过 collector 的来源门，也不能通过聚合 API 混入无来源或异源样本。这些 DTO/聚合检查用于规范化边界输入，不能替代宿主 verifier 本身。

收集一轮 DAgger 标签后，可用 `fine_tune_bc_on_daggerset(policy=student, aggregate=daggerset, grant=grant, train_grant_verifier=l6_verify, expert_fraction=0.7)` warm-start 更新现有 GRU BC。它会在训练前核验 Train grant、Fold 和 source，以 `DAggerSet.training_steps()` 的完整专家+学生聚合轨迹构造严格序列 Dataset，并用确定性 `DAggerMixtureSampler` 在每个训练 epoch 中按目标比例抽取可监督的专家/学生样本；默认以 70:30 作为样本比例，`0.9` 和 `0.5` 支持方案要求的 90:10、50:50 消融。若某一池样本不足，会有放回抽取并在结果中记录实际比例和每 epoch 样本数。训练从当前 actor 权重副本开始；仅成功时返回新策略，原策略保持不变，新策略的 episode 历史为空。`RouteAConfig` 现在会严格解析并校验 DAgger 的轮数、专家比例和 Train-only split。训练和样本来源仍由宿主授权 verifier 决定。

需要执行配置冻结的 K 轮时，可调用 `run_native_dagger_iterations(...)`，并传入 `RouteAConfig.dagger` 与长度为 `K+1` 的唯一策略 ID 序列。它会逐轮复用同一 Train aggregate，把上一轮返回的新 GRU 对象传给下一次原生 runner，再收集和 warm-start；成功时返回每轮 recovery/fine-tune 报告、最终模型和聚合集。若某一轮宿主授权、回放或拟合失败，异常会停止后续轮次，已追加到传入 aggregate 的已验证样本仍可通过单轮续跑 API重新训练。

需要把 RA3 恢复采样与训练更新作为单轮执行时，可调用 `run_native_dagger_round(...)`。该入口先核验旧/新 student policy identity、GRU policy 类型、aggregate 的 Fold/source 和 Train grant，再运行 `run_native_recovery_iteration()`：宿主必须按 wrong-direction、oversized-position、missed-exit 三类场景提供 Train prefix、故障配方和可独立验证的 native receipt；Route A 把本轮确切的 `BCInferencePolicy` 对象和其 ID 一并交给 host runner，并在每个场景前用场景 ID 重置模型历史。runner 不能只按 ID 重新解析另一个模型；访问状态回传时仍须匹配该策略 ID 和场景 episode。通过校验的学生访问状态才由 DAgger 教师投影并原子追加。入口可接收已校验的 `DAggerConfig`，只接受配置范围内的迭代并把 `expert_fraction` 传入 warm-start。新增标签数大于零后，入口再次核验 Train grant 与聚合来源并调用 BC warm-start。调用方传入不同的旧/新 student policy identity；函数只返回新策略，不会自动替换线上 actor。授权预检失败时不会启动宿主恢复回放；宿主授权、回放和 receipt 验证器仍是可信边界。

RA3 晋级比较可使用 `compare_recovery_iterations(...)`。报告保留每个场景的 `kind/native_prefix_id/fault_recipe_id`；比较器要求基线和候选有相同 Fold、Train source、执行配方、seed、scenario ID 及完整场景规格，候选必须是更晚且身份不同的学生。调用方传入预注册 `RecoveryGateThresholds`，并提供同一自然轨迹指标的基线/候选值与方向。候选出现任何硬风险违规、超过恢复率/时间/回撤/成本退化界限、自然轨迹退化超限，或恢复指标完全没有严格改善时，结果为拒绝。该门只比较宿主回放提供的恢复报告，不代替正式 L6/WFO 经济验收。

## B. Offline native transaction

```python
from rles_route_a.integration.native_transition_builder import (
    EncodedNativeState, native_nonterminal_transition,
)

# `before`, `after` 为已授权 L2/L3 观察；encoder 由训练折冻结。
# `projected` 为上一步 host_gym_projection 返回的 HostGymProjection。
# t = native_nonterminal_transition(
#     EncodedNativeState(before, tuple(before_embedding), encoder_id, source_id),
#     EncodedNativeState(after, tuple(after_embedding), encoder_id, source_id),
#     projected, decision_period_ns=60_000_000_000, sample_id=native_step_id)
```

`requested_action` 保留策略在决策屏障发出的行为命令，也是离线 IQL/TD3+BC 的动作 `a`。`executed_target` 保留 SoftRisk、HardRisk 与数量规划后的 native `planner_target`，连同完整 L3 trace 用于审计动作映射。学习转移仍使用引擎实际产生的奖励和后态；`next_state` 保留原生账户/市场事实及活动订单，因此未成交目标不会被当成实际仓位。该动作语义遵循 v1.1 快照适配说明 S3 的 RA4 优先条款。投影入口还要求教师观察的 pending IDs 与 Gym 预动作快照一致，并要求 Gym 后动作快照与独立 L3 order facts 一致。

每次 Gym teacher rollout 会绑定确定性 `trajectory_id`（由 fold/source/执行配方/seed/teacher 类型派生）；可传入 `rollout_id` 区分同 seed 下有意独立的实验运行。该身份从 `AuditedTeacherBarrier` 传至 `LabeledStep`，序列 Dataset 按 fold/episode/trajectory 切分上下文，避免相同 episode 窗口的不同运行串接隐藏状态。

## C. 教师 Gate

`NativePair` 需要冻结 teacher/baseline/source，覆盖完整 `window × scenario × seed` 矩阵，`matched_native_barriers=True`，并由 L6 **实际验证**窗口互不重叠后设置 `nonoverlap_verified=True`。默认必须具备 `base` 和 `stress` 两类成本情景、至少两组独立训练随机种子。`screen_teacher` 返回 `TRAIN_INNER_SCREEN_PASS/REJECT`，无论 PASS 与否均不表示独立市场获利证明。

## D. 与既有 M2.2 决策 trace 对账

使用 `provider.rollout_with_m22_trace(factory=factory, host_rollout=rles_orchestrator.application.rollout, grant=grant, seed=...)` 跑同一环境和同一教师策略。宿主 `rollout` 的 predictor factory / decision observer hooks 在每步收集 Route A facts，并在调用返回 M2.2 报告后自动执行一一核对；核对前 barriers 不会返回给训练流程。决策 trace 校验匹配决策前 Route A 观察与 M2.2 记录的执行后 barrier 时钟，并核对计划/幂等 ID、请求/SoftRisk/HardRisk/取整目标、动作、原因、订单引用、执行敞口和政策屏障权益。随后，订单核验器将每个 Gym barrier 前后状态、pending identities 和策略步 fills 与 `factory.snapshots` 的 M2.2 原生 account snapshot 以相同 decision timestamp 对齐，并核对 client order ID、status、side、quantity、filled quantity、fill event ID/数量/价格/费用；历史工厂必须保留完整 `all_orders` 与 fills。这个后置核对不能在逐 barrier provider 模式下单独替代 host `native_order_facts_verifier`，因为它需要整次 rollout 完成后取得独立 snapshot 序列。

Route A BC 的 WFO fit_candidate 必须先调用 train_bc_from_registered_native_rollout(...) 在拟合 Train-inner factory 上采集并训练，再调用 evaluate_bc_on_registered_train_inner_rollout(...) 评估同一模型。评估在同一独立 scoring 窗口分别执行一次教师回放和一次冻结 BC 学生回放；teacher selection 与 student replay 使用独立准备、仍属 registered Train 的 factory/provider/grant，fit、selection、student replay 可持有不同读预留，但必须绑定同一 owner、source 与 Fold。selection start_ns 至少晚于 fit end_ns + features.max_lookback_ns，使用三个不同 rollout seed，并要求相同执行配方/特征哈希/冻结教师/市场字段映射。每次回放各自执行完整 M2.2 与 L6 对账；BC 学生的请求动作与实际可执行敞口单独统计，同时保留原生权益、费用和奖励报告，不更新模型权重。结果收据绑定来源、Fold、配方、教师配置哈希、purge 与窗口；make_route_a_wfo_train 缺少收据、身份不匹配或 fit/selection/student 任一动作全落入 deadband 时拒绝导出或启动 Validation。WFO 在 Train 输出目录写入 fit-teacher、selection-teacher、selection-BC-student 三份逐决策 barrier 证据并在 bundle lineage 绑定各自 SHA-256；证据包括请求/风险目标、planner/订单/fill 费用、原生权益和奖励，不复制逐步 market_features 或 account_rows。该 Train-inner 选择诊断不是正式经济结果。

WFO publication 现在还要求 `NativeBCTrainingResult.student_screen` 携带 `screen_student()` 生成的完整多窗口配对收据。筛选结果在进入 bundle 前会从教师/学生矩阵和冻结阈值重算；再核对当前模型权重 hash、BC observation/action schema、Train source 与 execution recipe。WFO 写出 `student-promotion.json` 保存完整配对矩阵与筛选阈值，actor 序列化后重新加载并核对相同的 actor-state hash。单窗口 evaluator 不会自动生成此收据，因此生产 fit_candidate 必须在多个已登记 Train-inner disjoint window、Base/Stress 和至少两个 seed 上完成原生 teacher、baseline 和 student replay；收据缺失或失败时拒绝写 bundle 和启动 Validation。该晋级仍是 Train-inner scenario-only 筛选，不代替正式 L6 经济证明。

同条件四臂工程比较由既有 IL Core `compare_four_native_arms(...)` 执行：规则教师、纯 BC、纯 PPO 与 BC→PPO 共用冻结 PPO 配方与原始 policy observation 域（L4 VecNormalize 关闭），纯 PPO 与 BC→PPO 使用相同 Train 样本预算，最终由成对 seeds 在新建 native Train-inner 账户上回放。调用必须提供与 fit 窗口分离的 Train-inner selection trajectories、精确 selection window 和 purge；两个 BC 初始化 actor 均在该独立窗口进行冻结推理，并须通过非恒定平仓检查与冻结 `parity_tolerance` 动作误差门。未过门时在任何 PPO 更新前抛出 `BCFinetuneBlocked`。报告保留 fit/selection 数据身份哈希、窗口、BC fit/selection 行为与 parity receipts，以及固定的 policy input domain。该工具报告标记为 Train-inner scenario-only，不构成历史经济认证；真实历史运行仍须通过授权读登记、教师质量和经济资格门槛。

## E. 尚待真实运行验证

- 目标系统 Python 3.14.7、Torch 2.14、CPU/CUDA/WSL2；
- Binance 历史撮合/费用/保证金实际完整性；
- 非终端与终端外部现金流事实仍必须由宿主 L6 verifier 提供；原生 AccountState/权益变化本身不能证明不存在存取款；
- 原生 terminal drain/flat 独立 step 完整归因；
- L3 多条 FillEvent、订单取消/ACK/限价路径；
- L6 项目级 Train/Validation/Test/Holdout 的真实数据权限和独立评价。
