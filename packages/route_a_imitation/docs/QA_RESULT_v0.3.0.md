# Route A v0.3.0 独立源码 QA（2026-10-18）

**冻结宿主** `80194a587444bf4a11f06f6e273723e5754f2483`。

## 本轮实际执行

```text
Python: 3.13.5
PyTorch / NumPy: 2.10.0+cpu / 2.3.5
pytest: ........................................................................ [ 93%]
.....                                                                    [100%]
python -m compileall -q src tests examples: PASS
synthetic_smoke.py: PASS (详见下方)
```

```json
{
  "status": "SYNTHETIC_ENGINEERING_SMOKE_ONLY",
  "train_rows": 60,
  "epochs": 3,
  "last_train_loss": 0.626440183321635,
  "synthetic_action": 0.11461956799030304,
  "real_market_data_read": false,
  "native_engine_run": false,
  "economic_approval": false
}
```

本轮新增 `test_v03_host_projection.py`、`test_v03_quality_and_risk.py` 和 `test_v03_transition_builder.py`，逐项检查 Gym 前后时钟、风险目标、订单身份、NOOP/UNFILLED、终局拒绝、模拟资金流合约、DAgger Recipe、离线时钟/折扣与配对教师质量门。

## 代码交付边界

- Python **3.14.7** 和 Torch **2.14** 的目标环境本次**无法运行**；本容器使用 Python 3.13.5、Torch 2.10 CPU，仅可说明本地逻辑和兼容性烟测通过。
- **没有读取真实交易数据、启动 Nautilus 原生 Engine、使用真实 L6 ResearchRegistry 或执行 CUDA/WSL2 验证**。
- 所有假冒 L3/L6 的对象都仅是 `tests/` 中的隔离**合成 mock 字段**；不是签名经济事实。
- 原生 Gym 终局强制平仓（和跨终局的奖励、清算后零权益）、完整订单生命周期及实盘资金流水仍留给宿主验证。
- 未进行正式 WFO、Validation/Test/Holdout 或收益/泛化评价，不能据此宣称交易盈利。

**声明**：本测试数字只表本包独立功能测试通过，不能把 `ENGINEERING_PASS` 混作经济准入。
