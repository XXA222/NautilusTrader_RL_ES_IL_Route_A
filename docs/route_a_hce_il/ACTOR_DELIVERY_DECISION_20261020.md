# Route A student actor 交付决策（2026-10-20）

## 决策

标准 Nautilus/L4 原生回放与后续 WFO 的候选交付路线，统一采用四臂比较中的 **SB3 `bc_ppo` actor**：先用 Train-inner BC 初始化，再通过冻结的 PPO 预算更新，最后以标准 L0/SB3 actor 产物进入 native replay 和 WFO。Pure BC 与 pure PPO 保留为对照臂；规则教师不是学生 actor。

Route A GRU BC 继续作为独立学生候选和研究模型，不直接替代 SB3 交付物，也不把 GRU 权重转换或描述成四臂中的同一 actor。GRU 当前使用有序因果特征序列和 episode 内 recurrent history；四臂 SB3 policy 使用 L4 的固定原生 observation 向量与 SB3 actor mapping。两者可共享底层原生环境及 `[-1, 1]` 请求敞口语义，但输入表示、权重、状态和模型身份不同，数值动作域一致不构成 actor parity。

这一决定是工程交付路线，不是模型优劣结论，不授予真实数据访问权，也不构成经济通过或实盘准入。

## 冻结 actor 身份链

每个可交付 `bc_ppo` 候选必须保留并逐项绑定以下来源；缺项或重算出的身份不一致时拒绝发布：

1. Route A Train-inner 数据 manifest 与来源 marker；fit 窗口、purge/lookback、独立 selection 窗口及各自 transition identity。
2. 冻结的 observation schema hash、原始 L4 输入域（四臂要求关闭 VecNormalize）、action schema 和 `[-1, 1]` requested-exposure 语义。
3. L3 scaler、feature/config、risk、execution recipe、PPO fingerprint、BC 初始化 receipt 与初始化 actor hash。
4. PPO arm 的冻结 seed/预算及最终 SB3 actor 参数 SHA-256、L0 mapping 身份和导出文件 hash。
5. selection 上 BC 初始化 parity/collapse gate 的 receipt，以及导出后由**同一最终 actor hash**完成的独立 native Train-inner replay/M2.2/L6 evidence。

四臂比较报告已有 fit/selection 窗口、PPO 输入域、训练身份和 BC 初始化 parity 字段；只有四臂报告本身并不等于最终 SB3 actor artifact。最终导出应沿标准 L4/L0 actor 路径完成，导出后冻结其哈希，并让 native replay、WFO candidate 与 bundle 均引用该同一哈希。禁止将 GRU bundle hash、SB3 initialization hash 和 PPO 训练后 actor hash混为一个身份。

## GRU 后续资格

若后续要把 GRU 升为标准交付 actor，需先实现并独立审查明确的 L4-compatible adapter：固定 feature order、原生 observation 到序列张量的因果投影、episode/gap reset、同一 action schema，以及绑定 GRU bundle/config/Train source 的来源 receipt。随后在完全相同的冻结观测样本上完成重复确定性、输入域边界和动作 parity 检查，再由同一个冻结 GRU hash 完成 native replay 与 WFO artifact 绑定。完成这些门槛前，GRU 结果只作为独立研究候选，不参与 SB3 actor 导出。

## 学生准入门

`rles_route_a.evaluation.screen_student()` 已实现多窗口筛选器，并由 WFO 导出路径强制消费其收据：它绑定通过的教师筛选收据，要求两个以上不重叠窗口、Base/Stress 和多个 seed，并要求学生逐对产生正净回报且高于教师。配对收据固定同一个 actor SHA-256、schema、执行配方、市场源和双方原生回放证据。详见 [学生筛选器说明](STUDENT_PROMOTION_GATE_20261020.md)。独立 Train-only runner 已实际评估 4 月两个窗口；教师门拒绝，BC 学生各配对回报为负，因此没有学生 PASS 收据，也没有导出 GRU actor/bundle。当前多窗收集器仍是研究/QA runner，尚未接入为通用 WFO `fit_candidate` 编排。

## 本轮审计待办状态

- `MeanReversionTeacher` 非零原生交易路径已有合成覆盖：在 Train-inner 区间开空并归零退出，保留 fill/commission 与 M2.2 证据。
- `RecoveryTeacher` 非零原生纠正路径已有合成覆盖：先建立仓位，再触发 oversize 风险规则并成交减仓，保留 native exposure/fill 证据。
- MeanReversionTeacher 的开仓/归零退出与 RecoveryTeacher 的 oversize 减仓均已由合成 native rollout 覆盖 M2.2、成交及暴露变化；针对性测试 `route_a_t1_t2_t3_teachers_run_registered_native_gym_and_m22_path` 本轮复跑通过。该结果仅证明机制链路。
- 四臂比较和其他 native Train-inner teacher/BC-student 回放仍为 scenario-only 工程证据；本轮未重新执行四臂比较。
- 真实 Train 情景诊断未执行。当前历史规则、完整费用/保证金证据和获批 Train 数据来源不足；仅可在宿主注册和授权后使用 Train 数据补充 scenario-only 诊断。Validation/Test/Holdout 不用于训练证明，历史费用后收益认证保持 BLOCKED。
