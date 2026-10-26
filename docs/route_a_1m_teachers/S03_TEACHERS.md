# S03 — 首批四名独立 1m 教师与冻结配方

执行日期：2026-10-25（Asia/Shanghai）

## 阶段结论

`S03 PASS`。T01 EMA、T02 ROC、T03 Donchian、T04 Z-score 已实现为四个独立、可显式注册的新教师；每个实现 `advise(PointInTimeObservation) -> ExpertAdvice` 并统一调用现有 `teachers/protocol.py::advice()`，不直接下单、不访问行情文件、不调用另一个教师作为入场门槛。

## 冻结共同行为

- `entry_decision_every=1m`：每个新的完整且已可用 1m bar 最多推进一次。
- `target_exposure=0.3`：含义为原生 planner 风险预算名义值的 30%。
- `htf_mode=FEATURE_ONLY`：5m/15m/1h/4h/1d 内容不参与 raw entry 或方向判断。
- `NEW_EVENT_ONLY`：持续状态不重复生成开仓；未成交交叉不会在后续每分钟重发。
- 空仓无事件为 FLAT；持仓无退出为 HOLD；第一轮不 ADD/REDUCE。
- 单币 NETTING；反向持仓仍由现有 protocol 先 EXIT、待原生确认 flat 后下一决策再考虑新方向。
- pending native order 由现有 protocol 统一返回 `PENDING_ACK`，不重复普通订单；raw signal 仍保存在教师旁路状态，供 S05 漏斗使用。
- 新 episode 自动清空递推事件状态；同 bar 重放返回 `NO_NEW_BAR`，缺失分钟返回 `MISSING_BAR` 并切断跨缺口交叉。
- 缺指标或未成熟返回 `INDICATOR_NOT_READY` / `INDICATOR_PREVIOUS_NOT_READY`，不以 0 制造信号。

## 四教师配方

| teacher | 入场 | X1 指标退出 | warmup |
|---|---|---|---:|
| T01 `EMA_9_21` | EMA9/21 新交叉；多空镜像 | 反向 EMA 交叉 | 21 |
| T02 `ROC_20_60` | 两 ROC 首次同正/同负 | 任一 ROC 破坏原方向一致 | 61 |
| T03 `DONCHIAN_20` | 当前 close 突破不含当前 bar 的 prior-20 通道；margin=0 | close 突破对向 prior-10 通道 | 21 |
| T04 `ZSCORE_60` | 从 ±2.0 外回到 `[1.8,2.0)` / `(-2.0,-1.8]` | 到达均值或出现对向确认 | 60 |

T04 一步越过 reentry 带明确记录 `FAST_REVERSION_MISSED`，不追单。T03 没有继承旧 1.5% breakout margin。四位教师各自配置完整包含 `teacher_id`、indicator recipe、参数、warmup、entry/X1 rule、HTF mode、target exposure、cooldown/order/risk recipe 和 Base/Stress cost IDs。

冻结配置为 `configs/route_a_1m_teacher_library_v1.json`。文件同时预声明 T05–T08，但 S03 registry 只允许显式构造 T01–T04；未知 id 直接失败，不 silent fallback。

## 实际合成事件轨迹

以下轨迹由 `test_teachers_1m_signals.py` 在 Ubuntu 实跑：

```text
T01 LONG:  EMA9<=EMA21 → EMA9>EMA21 => OPEN_LONG(0.3)
    HOLD:  持多且无反向交叉 => HOLD(0.3)
    EXIT:  持多后 EMA9<EMA21 => EXIT(0), EMA_REVERSE_CROSS
T01 SHORT/EXIT: 完全镜像，已断言

T02 LONG:  前分钟非同正 → ROC20>0 且 ROC60>0 => OPEN_LONG(0.3)
    EXIT:  持多且 ROC20<=0 => EXIT(0), ROC_CONSENSUS_BROKEN
T02 SHORT/EXIT: 同负及破坏条件镜像，已断言

T03 LONG:  prior close<=prior U20 且 current close>current U20 => OPEN_LONG(0.3)
    EXIT:  持多 current close<prior L10 => EXIT(0), DONCHIAN_EXIT_DOWN
T03 SHORT/EXIT: 下破/上通道退出镜像，已断言

T04 LONG:  Z -2.2 → -1.9 => OPEN_LONG(0.3)
    EXIT:  持多 Z>=0 => EXIT(0), ZSCORE_MEAN_OR_OPPOSITE
T04 SHORT/EXIT: Z 2.2 → 1.9 及回到<=0，已断言
T04 MISS:   Z -2.2 → -1.7 => FLAT, FAST_REVERSION_MISSED
```

另对四位教师逐一断言：相同 1m 输入、任意翻转 1h/4h/1d 值，raw entry 与 `ExpertAdvice` 完全相同；入场事件存在 pending order 时返回 `PENDING_ACK` 而不是提交新单；同一 bar 第二次调用不产生新信号。

## Ubuntu 实测

```text
.venv-ubuntu/bin/python -m pytest -q -p no:cacheprovider \
  packages/route_a_imitation/tests/test_teachers_1m_signals.py
exit 0 — 14 个测试点，100%

.venv-ubuntu/bin/python -m pytest -q -p no:cacheprovider \
  packages/route_a_imitation/tests/test_teachers_1m_signals.py \
  packages/route_a_imitation/tests/test_contracts_teachers.py \
  packages/route_a_imitation/tests/test_causal_position_teacher.py \
  packages/route_a_imitation/tests/test_teacher_rework.py
exit 0 — 41 个测试点，100%

.venv-ubuntu/bin/python -m mypy packages
exit 0 — Success: no issues found in 239 source files

.venv-ubuntu/bin/python -m ruff check packages qa tests
exit 0 — All checks passed!
```

## 停止条件核对

- 任一教师只能 LONG 不能 SHORT：否，四位均双向通过。
- 普通退出强制 240m：否，代码没有持仓时间退出字段；长持仓生命周期在 S04 专测。
- 不能每分钟评估：否，新 bar、重复 bar、缺口和未就绪均有确定动作。
- 结论：S03 不触发停止条件，允许进入 S04。
