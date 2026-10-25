# Route A 源码 v0.2.0 验证记录（2026-10-18）

## 实际执行结果

- `PYTHONPATH=src python -m pytest -q tests --disable-warnings`：**50 passed（合成测试）**。
- `PYTHONPATH=src python examples/synthetic_smoke.py`：合成 60 Train 行、3 epoch BC-RNN、JSON/NPZ bundle 读写；不访问真实市场或账户。
- `PYTHONPATH=src python -m compileall -q src tests examples`：**PASS**。
- 新增验证包括：不可变因果数据对象、可序列化 DataLoader、断档重新建历史、按监督有效数归一化的 BC loss、NOOP/deadband 平仓被屏蔽、DAgger 批量原子提交/轨迹隔离、Fold/执行合约一致性、IQL 边界动作稳定性、TD3 噪声固定种子、离线训练 checkpoint 断点续训精确对照、权重篡改拒绝。

## 运行时/系统边界

实际验证 Python 3.13.5 / Torch 2.10.0+cpu / NumPy 2.3.5 / pytest 9.0.2。目标 Python 3.14.7 / Torch 2.14.x / RTX 5070 CUDA 尚未实测。`ruff`、`mypy`、目标 wheel 安装、Windows spawned native engine、WSL2/CUDA 亦尚未实测；所有实际未执行的检验不得写作 PASS。

任何交易经济结论均为 `ECONOMIC_UNVERIFIED`。L6 正式验证、真实历史规则和新数据权限由宿主项目独立管理。

## 构建限制

尝试 `uv build --offline`：当前容器仅 Python 3.13.5，项目 `requires-python ==3.14.7`，报告 `No interpreter found for Python ==3.14.7`，故没有声称 wheel 构建通过。本次交付为纯源码 ZIP，不打包旧项目。

## 边界统计

- 50 项专项测试均在合成数据和本包模块上运行；不读取真实 Binance market/account、Validation/Test/Holdout。
- 48 个包内 Python 模块完成导入 smoke。
- `examples/synthetic_smoke.py` 完成 60 行 synthetic Train / 3 epoch BC / bundle 恢复，返回 `SYNTHETIC_ENGINEERING_SMOKE_ONLY`。
- IQL/TD3+BC 合成 optimizer 和 sampler 恢复测试均通过，CPU 相同 runtime 下结果逐参数相同。
