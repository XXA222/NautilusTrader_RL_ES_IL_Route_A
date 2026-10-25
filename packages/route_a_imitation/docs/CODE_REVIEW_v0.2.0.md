# Route A 源码深度优化与审计报告 v0.2.0

**审计对象**：仅 `rles_route_a` 新增源码。**宿主固定提交** `80194a587444bf4a11f06f6e273723e5754f2483`。**判定**：单元级功能通过，原生集成及经济效果未验证。

## 1. 发现与修复（按交易研究危害排序）

| 发现 | v0.1 问题 | v0.2 修复 | 残余风险 |
|---|---|---|---|
| F01 错误教师监督 | Teacher 请求 EXIT，但 Planner NOOP 或订单 CANCEL/WAIT 时实际仓位不变，却可能被监督为“正确持有”，导致模型偏好不止损 | Native label 根据执行状态设置 `supervision_weight=0`；GRU 多头损失加权，拒绝全无可执行标签的数据 | 只有原生交易可证明某命令何时完整执行；订单在后续 barrier 结算要再评估 |
| F02 DAgger 序列污染 | 不同迭代/Student checkpoint 的观测可能拼接成一个伪造状态序列，导致循环隐藏状态错误 | `trajectory_id=dagger:iteration:policy:episode`，Dataset 独立窗口；batch append 先验全体核验后提交 | 完整历史状态仍由宿主 Engine 真正重放，不在本包人工变更账户 |
| F03 观测 TOCTOU | dataclass frozen 但原始 dict 可被外部修改，突破 `available_at` 一次性校验 | `FrozenMap` 拷贝封装，保持 Windows spawned worker pickle 支持 | 宿主必须确保账户时点与 L2 已知 barrier 一致 |
| F04 长时序断档污染 | 只有 Episode ID 边界，没有 Kline gap 截断 | Dataset 与 Inference 同时使用 `max_gap_ns`，超时清空历史 | 正常的 5m/15m 决策需要准确配置决策节拍 |
| F05 Offline replay 混折 | 离线转换未带折、样本来源、时间/执行身份，值函数可能把多个 scaler 混在一起 | 正式训练需注册 Fold 和执行来源，每笔可携带 episode/ts，拒绝身份错配 | `HostTrainGrant` 与 `OfflineReadiness` 不是独立数据库认证，宿主 L6 是权限权威 |
| F06 折扣定义错误 | 只按一步 gamma，未考虑环境一步对应多个决策间隔 | 每步 `discount_steps`，目标 `r+gamma^Δ * bootstrap * V/Q` | Gamma 的基本时钟须与冻结研究目标一致，连续费率仍是原生收益 |
| F07 数值脆弱性 | optimizer 步前未系统检查所有 loss/gradient；TD3 Noise 依赖全局 RNG | `safe_update` + clip + finite validation、独立 TD3 noise RNG、边界稳定 tanh log-prob | GPU 浮点差异不承诺跨设备 bitwise 复现 |
| F08 不能持续训练恢复 | 只有 Actor-only bundle，无法重载 Q/V、Optimizer/target/policy-delay/随机数 | JSON/NPZ 非 pickle train-only Checkpoint + sampler state，检查 runtime、fold、recipe、dataset、encoder | 尚未测试本机 3.14.7 + Torch 2.14 的真实长跑、驱动故障、CUDA 恢复 |
| F09 权重载入边界 | 模型压缩数据未做完整资源上限预检；非有限权重可能保存 | 限制 NPZ 单文件/成员/解压总量、形状、tensor 列表；导出拒绝 NaN/Inf | SHA256 仅表完整性，不等同可信签名；不要加载不可信来源工件 |
| F10 性能 | 每一条监督特征为 Python 小 ndarray、访问时频繁堆叠，且离线经验集可能整体上传 GPU | 监督特征转换为连续只读矩阵；离线全量经验驻 CPU，仅 sampled mini-batch 拷到 learner GPU | 多年 Train 仍需 L2 Parquet/Arrow 分块，不应一次把整份数据复制到多个 worker |

## 2. RA0–RA5 目前实际功能深度

**RA0**：时点观测、配置与单维 NETTING、native action trace、原生权益 reward 守恒，新增 `audit_teacher_barriers` 合规核对；不创造新的撮合和账户账本。

**RA1**：候选趋势/突破/均值回归/等待/恢复教师、冻结选择、实际可达性计数。重要限制：候选教师还没有通过真实经济筛选，“专家”只是规则角色，非已证实盈利策略。

**RA2**：GRU/MLP、加权 intent+exposure 损失、训练/Train-inner、可复现从零初始化、时序断档重置和推断 bundle。重要限制：没有分布外 regime 诊断/大规模真实演示实验。

**RA3**：学生 native 状态重标、执行可行性投影、轨迹隔离、批量聚合。重要限制：未实现可在没有宿主的情况下构造真实账户扰动/成交（这是设计边界，不应复制引擎）。

**RA4**：IQL expectile 双 Q + 权重克隆、TD3+BC 基线、训练循环、Folds/成本 provenance 与安全恢复。重要限制：没有自然市场样本支持域证明、未验证正收益或 critic 的样本外泛化。

**RA5**：经济/成本/动作诊断接口和预注册矩阵；正式 WFO、ResearchRegistry、Test/Holdout 在宿主 L6，当前不运行。

## 3. 新增的训练可恢复契约

`ReplaySampler` 以独立 CPU 随机生成器驱动 mini-batch 样本，`TD3BCTrainer` 使用单独噪声生成器；完整检查点能保存两个 RNG state 以及 Actor/双 Q/V/目标网络/Adam 状态和 TD3 policy delay 计数。读取时需明确要求原 `fold_id`、data source、execution recipe、frozen encoder 与 reward_scale，且**完全相同 Python minor + PyTorch 版本**。安全存取不加载 Python Pickle，任何不存在的 tensor key / 非法 NPZ / checksum 不符均拒绝。

独立对照测试先完整跑 6/7 次更新，再跑 3 次→保存→恢复→跑剩余更新，确认 IQL/TD3+BC 中各 Actor/Q/V、目标网络以及 TD3 RNG 对应参数值逐项一致。此证据只表明 **CPU 合成数据同运行时恢复**，不证明生产 GPU 跨硬件 bitwise 一致。

## 4. 公开 API 的向前兼容

- `LabeledStep` 增加默认 `supervision_weight`、`trajectory_id`、`source_identity`，原有非关键测试调用保持可构建；但 WAIT/CANCEL/UNFILLED 强行非零监督现会拒绝。
- `BCConfig` 添加 `max_gap_ns=None`，读取旧 BC 配置仍允许缺失此字段。模型推断可保持旧行为；正式 1m 数据应明确设定间隔。
- `OfflineTransition` 添加 `fold_id/episode_id/decision_ts_ns/next_decision_ts_ns/discount_steps/native_execution_recipe`。合成测试可显式 `allow_synthetic_unit_only=True`；**正式 RA4 默认必须有 Fold**，不再默认放行无身份的 replay。
- `train_bc` 对 `EpisodeSequenceDataset.max_gap_ns` 与配置进行一致性检查；如添加 gap 参数，训练/推断必须同步更新。
- checkpoint 与 actor-only bundle 明确分离，不能混用。

## 5. 尚不能通过的门槛

1. **Python 3.14.7 + Torch 2.14 / CUDA** 的目标依赖、设备和真实运行复现。
2. **真实 Nautilus 原生 Engine** 的 Teacher/Risk/Execution 导出、`zero_target_policy=close`、CANCEL/ACK/限价成交事件边界。
3. L2 真实已授权 Train 数据、冻结 Scaler、完整 1m/1h/1d `available_at` 的时序一致性。
4. L6 的项目级查询账本、独立 Base/Stress、未消费 Validation/Test/Holdout 与研究有效试验数。
5. 历史手续费/资金费、合约规则/保证金等真实经济证据的不足不能由程序测试填补。
6. 合成测试通过不代表教师存在 alpha，也不代表 BC/DAgger/IQL 策略费用后正收益。

## 6. 后续开发优先级（供 Codex）

**第一步**：RA0 原生 Provider 联调，在新分支独立测试 `NativeActionTrace` 与 L3 `M2.2` 实际 trace 的一致性，尤其检查平仓取消链、ZeroTargetPolicy、真实 `SUBMIT`/`FILLED`、资金费/权益守恒。只在项目授权 Train 运行，不消费正式验证窗口。

**第二步**：将实际 L2 feature schema/Scaler 映射为 `PointInTimeObservation`，验证闭合 1h/1d 的 `available_at_ns`，以真实 ETHUSDT Train-only 数据建立可复用教师示范缓存。教师每个状态都要说明其因果理由与 raw/feasible/realized action。

**第三步**：以同一 native Engine、同一费用/风险，先比趋势/空仓教师与 BC-MLP/GRU，保留实际交易行为和持有时长报告；若教师经成本后没有价值，不增加 IQL 训练预算。

**第四步**：仅在 BC 的自然持仓行为合理时推进合法账户扰动和 DAgger，随后对冻结轨迹学习 IQL/TD3+BC；最终按 L6 正式 WFO/查询账本执行一次性经济评价。
