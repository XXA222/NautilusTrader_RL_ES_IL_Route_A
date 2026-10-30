# Route A Meta 拟合：重叠标签权重（2026-10-30）

分类：`D1_DEVELOPMENT_DIAGNOSTIC_ONLY`。本次只改研究模型的样本权重，不改预注册市场范围、成本、最低样本数或入场门槛；没有读取市场负载，也没有产生盈利或教师资格结论。

## 本次源码改动

- `entry_meta_policy.py` 和 `opportunity_models.py` 新增同一套 `AVERAGE_INVERSE_LABEL_CONCURRENCY_V1` 权重：对每个完整标签 `[decision_ts, label_end)`，按其覆盖时段的平均并发标签数取倒数；无重叠样本权重为 1，完全重合的样本均分同一时间影响。
- Logistic、Platt 校准、Ridge、fit 标准化和校准 MAE/Brier 都使用该权重；分位残差和线性模型方差也使用加权口径。删失标签仍不能变成零收益或拟合样本。
- `train_meta.py` 保留原始标签数及原有 overlap/max-concurrency 统计，并增加唯一性权重总量（时间覆盖等价计数，非标准 Kish ESS）；原始计数不再被解释成独立样本量。
- 冻结 Meta policy schema 升为 v2，序列化训练与校准有效样本量和权重方法 ID。v1 文件不会被当作 v2 模型载入。`opportunity_models` 返回的冻结模型也保留同一权重身份与有效样本量。

## 验证

- Ubuntu 24.04 / Python 3.14.7：定向 Meta、候选、标签、policy IO、Gym 与原生引擎合成整链回归 **49 passed**。
- Ruff：本次改动源码与测试检查通过；MyPy：4 个改动源码模块通过。
- 回归包含同一未来区间重复添加候选后，Logistic、Ridge、Platt 参数和唯一性权重总量保持不变；非重叠样本的唯一性权重总量等于原始样本数；重叠标签按实际并发时长分摊；右删失标签拒绝进入完整标签权重。
- `qa/run_route_a_meta_study_20261029.py` 只读预检：`REGISTERED_FUTURE_SOURCE`，protocol `route-a-meta-x1-train-20261201-20261219-v1`，已完成评分窗口 0，`market_payload_read=false`。没有执行真实市场拟合/校准或原生经济评分。

## 资格状态

八教师、Meta 筛选器与 Nautilus 原生账户的工程整链已有合成验证；本次重叠权重修正仍属工程质量提升。注册 Train 时间范围从 2026-12-01 UTC 开始，真实模型训练、校准、Base/Stress 原生对照和新旧策略经济比较仍未完成。不得把本报告、合成收益或已消费历史资料宣称为独立费用后盈利。
