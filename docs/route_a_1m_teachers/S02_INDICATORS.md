# S02 — 版本化 1m 指标提供层

执行日期：2026-10-25（Asia/Shanghai）

## 阶段结论

`S02 PASS`。新增单一共享的因果 1m 递推内核、冻结 recipe 契约、授权 L2 输入投影、完整 UTC 高周期聚合与自动回归。未改变现有 `causal_mtf_v5/v6/v7` 字段和语义；未读取真实行情值，所有本阶段数值验证使用合成确定性序列或内存 `CuratedStore`。

## 数据与时钟合同

- `CausalBar` 明确保存 `open_time_ns`、`close_time_ns`、`available_at_ns`、OHLCV 与 timeframe。
- 1m bar 必须 UTC epoch 对齐、价格有限且为正、volume 非负、OHLC 区间一致，且 `available_at_ns >= close_time_ns`。
- 只有 `available_at_ns <= decision_ts_ns` 才推进指标；未来 bar fail closed。
- 同一 bar、同一 decision 的完全相同重放幂等；重复时间但事实冲突、不同 decision 重放、倒序、缺失分钟全部拒绝。
- 每个合法 bar 最多推进一次；状态只保留固定上界队列及递推量，不回扫全历史，不形成 `O(T²)`。
- `IndicatorSnapshot` 只含 finite 已就绪值；未成熟或零方差 Z-score 不补零，以 recipe `NOT_READY` 表示。
- `feature_available_at_ns <= decision_ts_ns` 在 snapshot 和 `PointInTimeObservation` 投影两层校验。
- 5m/15m/1h/4h/1d 聚合只从连续 1m 构成完整 UTC 桶，并以所有组成 bar 的最大 availability 为本桶可用时间；部分桶和未到达桶不暴露。

## L2 一源化

`rles_data.features.causal_1m_indicator_inputs()` 是版本化输入 profile `causal_1m_indicators_v1`：

- 只从调用方已经授权并构造好的 `CuratedStore` 获取数据；不开文件、不请求网络、不创建影子行情库。
- 复用现有 `("ohlcv_closed", instrument, 1)` feature cache。
- 输出已关闭且在 decision 前可用的 typed 1m 输入；重复、缺口、陈旧 latest bar 和非有限值 fail closed。
- `StreamingIndicators1m` 是纯计算器；它不持有 source path 或读取 API。
- `integration/l2_adapter.py` 新增独立 `CAUSAL_1M_INDICATORS_V1_ROUTE_A_FEATURES` 和 `project_indicator_snapshot()`；v5/v6/v7 map 未修改。

## 冻结 recipe 表

| recipe | warmup | 输出 | 单位/数值语义 |
|---|---:|---|---|
| `route_a.ema_9_21.v1` | 21 | `ema_9`, `ema_21` | 价格；第 n 根 SMA 初始化，之后 alpha=`2/(n+1)` |
| `route_a.roc_20_60.v1` | 61 | `roc_20`, `roc_60` | decimal ratio；`C_t/C_(t-n)-1`，不乘 100 |
| `route_a.donchian_20_10.v1` | 21 | entry 20 与 exit 10 上下轨 | 价格；严格只含当前 bar 以前的完整 bars |
| `route_a.zscore_60.v1` | 60 | `zscore_60` | 无量纲；总体标准差 `ddof=0`，sigma=0 保持 NOT_READY |
| `route_a.atr_14.v1` | 14 | `atr_14` | 价格；TR 三项 max，首 TR 为 high-low，Wilder 递推 |
| `route_a.macd_12_26_9.v1` | 34 | DIF/DEA/HIST | 价格；严格成熟的 DIF 后初始化 DEA，HIST=`DIF-DEA`，不乘 2 |
| `route_a.rsi_14.v1` | 15 | `rsi_14` | 0..100；14 组涨跌初值后 Wilder；flat=50，单边=0/100 |
| `route_a.bollinger_20_2.v1` | 20 | mid/upper/lower | 价格；SMA20 ± 2×总体标准差 `ddof=0` |

每个 snapshot 还带当前已完成 bar 的 open/high/low/close/volume，供后续教师和 ATR 生命周期使用；`latest_1m_close_ts_ns` 在 host observation 投影时显式加入。

## Synthetic fixture 与断言

`packages/route_a_imitation/tests/test_indicators_1m.py` 使用 1ms availability delay 的 UTC 1m bars，覆盖：

- 常价、线性涨/跌、突变、正弦趋势的 EMA 与独立 NumPy 实现逐步比对，容差 `1e-10`。
- MACD 严格成熟点和 `HIST=DIF-DEA`。
- RSI 连涨=100、连跌=0、完全不动=50。
- ROC 21/61 根要求和 decimal 单位。
- Donchian 当前 bar 创高时仍不进入自己的阈值集合。
- Z-score/Bollinger `ddof=0`，常价 Z-score 不产生伪值。
- ATR 跳空 TR 三项 max 和 Wilder 初值。
- 5/15/60/240/1440m 完整桶、partial bucket 与 delayed availability。
- duplicate/out-of-order/gap/future/non-finite fail closed。
- uninterrupted 与重启前缀重放完全一致。
- 任意修改未来 bars 不改变此前 snapshot（prefix invariance）。
- L2 input profile 只返回 decision 前已到达 bars 并实际命中现有 cache。
- 投影后的每个 feature availability 均不晚于 decision。

## Ubuntu 实测

```text
.venv-ubuntu/bin/python -m pytest -q -p no:cacheprovider \
  packages/route_a_imitation/tests/test_indicators_1m.py
exit 0 — 21 个测试点，100%

.venv-ubuntu/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_l2_gym_bridge.py \
  packages/route_a_imitation/tests/test_v03_host_projection.py \
  packages/route_a_imitation/tests/test_v03_host_context.py
exit 0 — 57 passed in 18.13s

.venv-ubuntu/bin/python -m mypy packages
exit 0 — Success: no issues found in 230 source files

.venv-ubuntu/bin/python -m ruff check packages qa tests
exit 0 — All checks passed!
```

## 停止条件核对

- 未来扰动改变过去指标/信号：否，prefix invariance 通过。
- warmup 错误或未成熟补零：否，逐 recipe 成熟点自动断言。
- host 无法携带合法 availability：否，L2 input 和 Route A projection 均 fail closed 校验。
- 结论：S02 不触发停止条件，允许进入 S03。

## S05 集成时补充的时钟精度回归

在接入原生教师侧路时发现 epoch 纳秒时间戳不能通过 `float` 保真。已
将 `latest_1m_close_ts_ns` 从指标浮点值改为 observation 的
`exact_market_clocks_ns` 整数元数据；`project_indicator_snapshot` 不再把
ns 值塞进 `market`，并新增 epoch 量级测试。S05 扩展矩阵中
`test_indicators_1m.py` 与 `test_teachers_1m_signals.py` 通过。旧小时间戳
fixture 仅在 `2**53` 精确范围内兼容；超范围没有精确元数据会拒绝。
