# Route A / HCE-IL — 独立源码包 v0.3.0

**冻结工程基线**：NautilusTrader-RL-ES `codex/archive-20261018-development-snapshot`，commit **`80194a587444bf4a11f06f6e273723e5754f2483`**。

## 目标与范围

这是独立新增的 `packages/route_a_imitation`，**不含原 NautilusTrader-RL-ES 的任意模块源码、上游 NautilusTrader、真实行情数据、密钥或预训练收益模型**。代码覆盖 A0/A1 教师、A2 时序 BC、A3 DAgger、A4 IQL/TD3+BC，以及 A5 注入式研究接口。`docs/CODE_REVIEW_v0.3.0.md` 记录精确审计结论；新增宿主契约参见 `docs/INTEGRATION_V03.md`。

目前属于 **ENGINEERING PROTOTYPE / NATIVE_INTEGRATION_PENDING / ECONOMIC_UNVERIFIED**，不是可运行于交易所的完整应用。训练/测试使用的全部行情账户事实须由宿主工程 L2/L3/L6 提供；不得伪造资金费、撮合、风控或研究访问收据。

## v0.3 深度开发（新增）

- **RA0 实际宿主字段适配**：`integration/gym_rollout.py` 从原生 Gym 执行前上下文收集教师样本；`integration/host_gym_projection.py` 区分执行前决策和执行后 barrier，核对订单状态、策略步成交、Risk/Planner/原生权益。终局步要求宿主 L6 的 `TerminalL6Receipt` 绑定结算事实，强平成交与教师动作成交分开保存。当前适配由合成集成测试覆盖，尚未通过真实授权历史数据验收。
- **教师训练标签可达性**：`SUBMIT` 只有当前提交订单确有 fill、实际敞口已到达 planner 目标且无残留 pending 才变成监督正样本；历史其他订单成交、部分挂单、取消/等待和 deadband 阻断不产生虚假正标签。
- **RA1 alpha 教师资格**：`teacher_quality()` 只计当前 SUBMIT 的订单成交，并要求敞口到达 planner 目标且无残留挂单；WAIT/CANCEL 上旧订单的 fill 和未完成部分成交不能把教师判为可执行。
- **RA4 原生 transition 转换**：`integration/native_transition_builder.py` 从前后真实观察及原生收益构造 Fold/episode/source/encoder/recipe 绑定的 offline transition，动作始终是原始命令，拒绝跨时钟或账户状态不符。
- **DAgger 与时钟更严谨**：跨轮次拒绝执行配方漂移，`NOOP` 改写学生旧仓位时屏蔽监督；Offline RL 可校验 `decision_period_ns` 与 `discount_steps` 一致且不会隐式跨丢失样本 bootstrap。
- **RA1 经济筛选**：`evaluation/teacher_gate.py` 提供完整窗口×成本情景×随机种子 Train-inner 配对筛选，要求教师自身正收益、基准超额、无强平、最大回撤和真实交易可达；不提供 OOS 或实盘认证。
- **RA5 配对矩阵**：`build_preregistered_matrix()` 固定 B0/E1-E8 的全部实验臂、五个种子和 zero/base/stress 三种成本情景；`evaluate_frozen_matrix()` 只允许在 Validation 上运行，要求 L6 authority verifier 确认每臂独立授权和注册来源，授权绑定精确配方后再运行，并校验原生费用、资金费、滑点、终局、清算、HardRisk 和订单诊断字段，保留策略构建或回放失败项。Test/Holdout 仍由宿主 WFO 冻结选择并遵守一次性访问门。它只负责编排与记录，不授予数据权限，也不宣称经济通过。
- **数值/状态质量**：GRU 只有成功推断后才推进历史，CPU replay 拒绝 float32 溢出而不抛多余 NumPy RuntimeWarning，episode 报告分离账户重置片段；原生动作诊断包含请求敞口离散度、空仓请求占比、直接反向比例和实际敞口离散度，帮助识别 BC 输出退化。

**本版本仍未在真实 NautilusTrader 端到端运行，已实现的是合成验证过的严格接入接口。**

## 主要优化

| 优先级 | 修复内容 | 代码位置 |
|---|---|---|
| P0 | NOOP/deadband 与未成交 EXIT 不再错误标成可学习的持仓目标；监督屏蔽 `WAIT/CANCEL/UNFILLED` | `integration/native_rollout.py`, `datasets/sequence_dataset.py`, `models/bc.py` |
| P0 | DAgger 数据聚合按学生轨迹隔离、同批次原子预校验、禁止跨 Fold/来源串接 | `datasets/daggerset.py`, `training/dagger_collector.py` |
| P0 | Offline replay 携带 fold/episode/clock/执行配方；正式训练必须绑定研究 Fold；不把请求动作替换为事后成交仓位 | `datasets/transitions.py`, `training/offline_train_loop.py` |
| P1 | 离线 value 更新按实际 `discount_steps` 折扣；奖励仅用于学习时进行冻结数值缩放；多年 replay 只驻留 CPU、按 batch 拷入 GPU | `offline_common.py`, `iql_trainer.py`, `td3bc_trainer.py` |
| P1 | IQL 边界动作使用稳定的 tanh Jacobian，TD3BC 目标动作噪声使用独立确定性 RNG；所有优化器增加数值和梯度检查 | `training/` |
| P1 | IQL/TD3+BC 可恢复**优化器、目标网络、replay 采样器、TD3 target noise RNG** 的 pickle-free 训练检查点 | `training/checkpoints.py`, `training/offline_train_loop.py` |
| P1 | GRU 在断档处显式清除序列上下文，且训练与部署使用同一 `max_gap_ns`；冻结可用时间对象不可被外部字典突变 | `models/`, `datasets/`, `contracts.py`, `export/inference.py` |
| P1 | 只读观测对象在 Windows `spawn` 进程仍可 pickle 序列化；监督特征单块连续矩阵，避免逐行数组额外开销 | `contracts.py`, `datasets/sequence_dataset.py` |
| P2 | 模型导出前拒绝 NaN/Inf；加载前验证 NPZ 成员、压缩/解压大小、张量列表与架构形状 | `export/archive_safety.py`, `export/bundle.py`, `export/offline_actor_bundle.py` |
| P2 | 专家参数拒绝非有限或越界值；新增原生 teacher barrier 和 reward 守恒审计接口 | `teachers/`, `integration/trace_audit.py` |

## 目录

```text
packages/route_a_imitation/
  src/rles_route_a/
    contracts.py, config.py
    teachers/             # 趋势/突破/反转/等待/风险恢复/选择器/质量门
    datasets/             # 序列、DAgger、Train label 索引、Native offline transitions
    models/               # 多时间尺度 GRU，BC heads，MLP 对照
    training/             # BC、DAgger、IQL、TD3+BC、样本权重、优化器恢复
    integration/          # L2/L3/L6 协议与 native trace audit
    evaluation/           # 恢复案例、实验矩阵、原生动作诊断
    export/               # NPZ/JSON 模型与安全推断
  configs/
  tests/
  examples/
  docs/
  pyproject.toml
```

## 独立测试

本容器中的专项测试运行环境是 **Python 3.13.5、PyTorch 2.10 CPU、NumPy 2.3.5、pytest 9.0.2**；源码目标依赖仍按工程锁声明 `Python == 3.14.7` 与 `torch>=2.14,<2.15`，**尚未完成该目标运行时验证**。

```bash
cd route_a_imitation
PYTHONPATH=src python -m pytest tests -q
PYTHONPATH=src python examples/synthetic_smoke.py
python -m compileall -q src tests examples
```

PowerShell：

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m pytest -q tests
python .\examples\synthetic_smoke.py
```

配置样例 `configs/route_a_v1.json` 的 `max_gap_ns` 默认按 1m 决策连续性设定。**仅当确实逐分钟决策时使用**；若宿主选择 5m/15m 决策，必须同时修改采样器和推断的允许间隔。

`RouteAConfig.load()` 对 market、teacher、IQL 和 evaluation 子对象执行完整键集及值域检查：只接受支持的因果周期和教师，必须保留 Train-inner 经济教师门、RA3 晋级后才可启用的冻结编码器 IQL、zero/base/stress 全情景、L6 研究账本与一次性 Holdout。market 与 BC 的 `context_bars` 也必须一致。工程 smoke 配置不授予数据权限或经济验收资格。

## 模型用途严格分开

- `save_bc_bundle/load_bc_bundle`：BC 模型推断参数与配置，JSON+NPZ（不含 Optimizer）。
- `save_offline_actor/load_offline_actor`：IQL 或 TD3+BC **Actor-only 推断**，不包含 value/critic，**不可用于续训**。
- `save_training_checkpoint/load_training_checkpoint`：IQL/TD3+BC **Train-only** 完整模型/优化器状态、可选 `ReplaySampler`，绑定 Fold、数据、执行配方与编码器身份，并要求相同 Python minor 和 PyTorch 版本；它**不授予正式数据访问权限**。
- `new_bc_policy/new_iql/new_td3bc`：从零初始化时的固定种子工厂；DAgger 接着微调已有模型时，不调用从零工厂重置权重。

## 需由宿主工程接线的内容

复制新目录到从固定 SHA 新建的 Route A 分支中的 `packages/route_a_imitation/`；保留已有 `packages/world_model`。项目自动发现 workspace packages，随后需在 Python 3.14.7 环境更新 lock 并安装新包、绑定 `HostRegisteredReader`、`NativeRolloutProvider`、`NativeExpertProjection`、`NativePolicySink` 和 `AuthorizedFormalEvaluator`。每一步由原 L6 `ProjectResearchRegistry` 控制 Train/Validation/Test/Holdout 的数据许可；本包中的 scope DTO 不是安全凭证。

具体接线与剩余阻断见 [集成手册](docs/INTEGRATION_SNAPSHOT_80194a.md)、[v0.3.0 源码审查](docs/CODE_REVIEW_v0.3.0.md)、[v0.3.0 QA 记录](docs/QA_RESULT_v0.3.0.md) 和 [宿主契约增补](docs/INTEGRATION_V03.md)。项目级阶段状态见仓库 [HCE-IL 实施状态](../../docs/route_a_hce_il/IMPLEMENTATION_STATUS.md)。


## v0.3 接线说明

详细约束、时钟证明及代码接口见 [`docs/INTEGRATION_V03.md`](docs/INTEGRATION_V03.md)。请勿把本包自带的合成 `L3OrderFacts` 测试样例用作真实订单证明。
