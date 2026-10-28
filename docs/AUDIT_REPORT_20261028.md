# 项目深度审计与改进建议（2026-10-28）

审计对象：D:\Program Files\NautilusTrader_RL_ES_IL_Route_A。已执行 git rev-parse --show-toplevel，并与 .canonical-project-root 核对一致。基线 HEAD 为 bd313c5241c31f1ccff96fa4bf76a7966be48767；本报告针对包含未提交、未跟踪变更的当前工作树，不能直接当作该提交的审计结论。审计开始时有 14 个已跟踪文件修改，另有教师、分析工具、预注册配置与测试等未跟踪文件。

## 判断

项目已经具备较完整的研究基础设施：原生订单/成交与经济账本、数据时点与来源契约、Train/Test/Holdout 访问登记、BC/DAgger/IQL/TD3+BC、教师/学生晋级门禁、恢复与交付检查。当前最重要的工作是保证“每份研究结果确实对应同一套代码、数据和执行条件”，以及补齐推理错误处理和自动回归覆盖。

工程通过与经济有效性必须继续分开。docs/route_a_teacher_v2/08_FINAL_DECISION.md 记录的历史 S07 为 Base 24/24、Stress 24/24 负收益，且都是已消费的 SCENARIO_ONLY 开发诊断；该阶段没有合格教师，也没有学生超越教师的证据。这些数字是项目文档记载，本审计没有重新运行这些市场实验，不能据此推出所有未来策略都无效。

## 范围与证据等级

检查重点：教师及退出状态机、Gym/原生执行桥、BC 训练与推理、离线 transition/训练器、研究访问/筛选/统计、导入与 wheel 交付、CI 以及当前开发文档。不是逐行覆盖全仓的形式化验证，也没有完成独立安全渗透、远程 Hosted CI 或实际交易认证。

DYNAMIC 表示当前环境中已用合成输入复现；STATIC 表示由代码路径直接推导；DOCUMENT 表示项目既有记录而非本次重新验证。所有问题均未在本次审计中修改实现。

### 本次验证结果

环境：Ubuntu-24.04 / WSL，项目既有 .venv-ubuntu，Python 3.14.7。实际导入 rles_route_a 来自 canonical root 下 packages/route_a_imitation/src。以下均为实际执行结果，两个测试选择存在重叠，不能相加为唯一用例总数。

| 检查 | 实际结果 |
|---|---|
| Route A 全包测试 + 教师筛选/退出归因/隔离回放/费用归因 + Binance archive range | 515 passed，69.84 秒 |
| 根目录 PR selector：contracts or unit or micro_golden or algorithm_smoke | 626 passed，7 skipped，168 deselected，47 warnings；570.52 秒 |
| 四个新增根目录文件的 PR/nightly marker 并集收集 | no tests collected (30 deselected)，非零退出，确认覆盖缺口 |
| ruff check --no-cache packages ci qa tests | All checks passed |
| 内存合成复现 | NaN 意图仍输出 0.46211716532707214；跨身份保留 2 行历史；旧代码/成本报告被恢复接受 |

7 项 skipped 全部来自 test_windows_worker_limits.py，要求 Windows Job API。47 条 warning 包含 EvoTorch 的 NamedTuple/NumPy 兼容警告、torch.jit.script 在 Python 3.14+ 的支持警告，以及 MLP PPO 使用 GPU 的效率提示；没有把这些警告视为通过的兼容性认证。建议单列上游兼容跟踪与冻结环境 smoke，不根据本次警告直接改动锁定依赖。

复现脚本：docs/audit_evidence/20261028/deep_review/probe.py；在项目根目录用 .venv-ubuntu/bin/python 执行。脚本只创建内存模型、合成 observation 与 mocked report，不访问市场、不创建真实研究运行。

本次未运行 MyPy，也未运行完整 nightly/release 或远程 CI。

## F01 — P1：恢复与汇总未绑定完整运行身份（DYNAMIC + STATIC）

位置：
- qa/run_route_a_teacher_student_train_screen_20261026.py:500，_completed_run
- 同文件 :611，_write_screen_summary
- 同文件 :863，expected 生成
- qa/run_route_a_1m_teacher_native.py:951，native-run 结果生成

现状：恢复比较 protocol_id、teacher/teacher_recipe、市场 marker、窗口、scenario 名称和 seed；未比较完整代码身份、预注册内容哈希、源勘误哈希、特征配方哈希、scaler 身份或实际执行成本。预检确实检查当前代码与当前冻结配置匹配，但这并不能证明已有 native-run.json 由当前版本产生。汇总只要求同窗口各报告的 marker 一致，没有从可信运行清单为每个槽核验完整来源与执行身份。

复现：以现有单元测试的合成 native-run 为基础，将预注册/特征身份标为旧版本，将成本改为零并附上旧代码身份；_completed_run 仍返回该报告。没有访问市场，也没有修改真实运行报告。

影响：工程修复后继续执行相同目录时，旧实现与新实现的结果可能混合；仅有相同教师参数和 scenario 名字不足以支持配对比较。这不证明当前已经产生了混合经济结果，而是已确认存在接受路径。

改进：引入版本化 RunSpec/RunReceipt；把 code manifest、协议及勘误、市场与规则、feature/scaler、execution/fill/cost、seed、窗口、依赖锁统一摘要。每个槽运行前冻结，报告回写同一身份；恢复和汇总共享一个验证函数。旧报告缺字段时标为 legacy/incompatible，保留原始证据，不补填当前身份。

验收：独立改变上述任一字段，恢复与汇总都在使用结果之前拒绝；相同完整身份可以恢复；代码变化必须生成新研究运行身份。

## F02 — P1：NaN 意图输出可绕过置信度门槛并生成开仓目标（DYNAMIC）

位置：packages/route_a_imitation/src/rles_route_a/export/inference.py:133。

现状：仅检查 exposure 输出是否有限；intent_aware_v1 的 logits 直接用于 argmax 和 softmax。NaN 概率与最小置信度比较时，NaN < threshold 为 False，因此没有阻止开仓。

复现：真实 BCGRUPolicy 的 OPEN_LONG 意图 bias 置为 NaN，exposure bias 为 0.5，最小开仓置信度设为 0.9。predict_exposure 返回 0.46211716532707214，并把内部时间推进到 100。返回值本身有限，L3ActorAdapter 的有限区间检查也无法识别意图头异常。

影响：数值异常会被解释成有效指令，且破坏“失败不推进推理状态”的设计约定。这里验证的是异常注入路径，未宣称现有模型发生过 NaN。

改进：使用 logits 前检查 shape、dtype/有限性及概率有限性；全部检查通过后才提交 history/clock/cooldown 状态。出现异常应向现有宿主风险/故障路径明确报错，不把异常分类结果当作交易信号。

验收：逐类 NaN、正负 Inf、异常 shape 均拒绝；历史、时钟与风控状态不变；合法有限输出保持一致。

## F03 — P2：学生推理历史没有按 source/fold/feature schema 隔离（DYNAMIC）

位置：packages/route_a_imitation/src/rles_route_a/export/inference.py:60、:76。

现状：BCInferencePolicy 只绑定 episode_id 与递增时间。教师 BaseTeacher1m 已经新增 source/fold 切换重置，学生路径尚未形成对应约束。

复现：相同 episode、递增时间，从 fold-A/source-A/schema-A 切到 fold-B/source-B/schema-B；推理成功，history 长度为 2，保留旧来源的观测。

影响：调用方复用 episode ID 或拼接不同来源时，模型上下文会跨越研究边界。正确重置且 episode 唯一的宿主流程未必触发，但公共推理接口本身未防御该输入。

改进：绑定 episode、fold、source 和 feature_spec_hash；明确区分合法 reset 与身份错误。feature schema 改变应拒绝，不应仅清空序列后继续。同步考虑 cooldown、position_since 等状态，并核对其他导出推理接口是否采用相同契约。

验收：同 episode 的 source/fold/schema 变化被拒绝或按明确契约 reset；合法连续序列行为不变；训练序列与在线推理的一致性回归覆盖这些边界。

## F04 — P1：30 个新增关键回归没有进入 PR/nightly（DYNAMIC）

位置：.github/workflows/l6-ci-reusable.yml:61、:67；tests/ 下的四个文件：
- test_route_a_teacher_train_screen_runner.py
- test_route_a_isolated_exit_replay.py
- test_route_a_exit_loss_attribution.py
- test_route_a_x1_x2_exit_economics.py

现状：这些文件没有 unit/contracts 等 CI marker。PR/nightly 根目录测试都用 -m 筛选；单独完整执行的 Route A 套件位于 packages/route_a_imitation/tests，覆盖不到这些根目录文件。

复现：对上述四文件使用 PR/nightly 的 marker 并集执行 --collect-only，输出 no tests collected (30 deselected)。这是预期的缺口复现，命令为非零退出；不能记录为检查通过。

改进：为用例添加准确分类，同时加入“所有测试至少进入一个适用 CI lane”的收集门禁。README 当前推荐 pytest tests，根 pytest testpaths 也只有 tests，默认本地运行又会漏掉包内 Route A 测试；建议提供统一入口，让文档、开发与 CI 使用同一份集合定义。

验收：四文件全部进入适用 lane；新建无分类测试会明确使覆盖检查失败；不能只靠手工记住额外测试目录。

## F05 — P2：Binance 导入登记库路径依赖源码目录布局（STATIC）

位置：packages/data/src/rles_data/binance_archive.py:28、:42。

现状：PROJECT_ROOT = Path(__file__).resolve().parents[4] 只适用于 packages/data/src/rles_data 布局。若正常安装到 /project/.venv/lib/python3.14/site-packages/rles_data，推导根为 /project/.venv，登记库查找将落到错误位置。Windows wheel 安装的祖先布局也不同。

影响：README 推荐 --no-editable，CI 也安装构建后的 wheels；模块能够 import 不代表导入命令可用。若错误位置没有登记库，导入会 fail-closed。这是可用性/交付缺陷，本审计没有证实它绕过保护，也未实际重装当前环境。

改进：通过明确的研究上下文注入 registry 路径/能力，或复用一个有 canonical-root 验证的根解析器；不要为解决路径问题自动新建另一份研究账本。保持包依赖方向，避免 data 直接反向依赖 orchestrator。

验收：源码、editable、wheel 三种安装方式；从根目录与子目录调用；都定位同一登记库。错误/缺失登记库必须在网络 IO 前拒绝。测试应覆盖真实安装布局而不是始终 monkeypatch PROJECT_ROOT。

## F06 — P2：release 工作流缺少外部 gate 证据的传入步骤（STATIC，交付未闭环）

位置：.github/workflows/l6-ci-reusable.yml:105、:117。

现状：release 在独立 runner checkout 后立即验证 artifacts/gates/trusted_producers.json 与 Q0–Q9；没有对应的输入获取/下载步骤，也没有 tracked gate 文件。needs: [cpu] 只表达依赖，不传递 CPU job 文件。工作流中的上传仅针对 nightly 测量，不是 release gate 报告。

影响：即使 CPU 检查通过，按当前仓库工作流也无法自行取得要求的外部证据。fail-closed 是正确行为，缺的是交付流程，不应放宽 verifier 来“修绿”。

改进：定义受信的外部审核包输入，按 revision/run identity 下载并验证；内部可自动生成的测试/构建报告走 artifact 传递，外部认可保持独立签发和摘要绑定。没有证据时保留明确 BLOCKED。

验收：有效且匹配 revision 的证据可被完整获取/验证；缺失、过期、被篡改及其他 revision 的证据拒绝。本审计没有触发远程 release。

## 应保留的设计

- 经济指标来自原生 equity/fill/cashflow，而非直接把 reward 当收益。
- 数据可见时间、split、fold、窗口与来源身份已经在多个边界检查。
- 订单提交不等于成交，终端结算与 bootstrap 有显式契约。
- 教师、学生、scenario-only、正式经济资格已经分开；不要用工程回归通过替代策略资格。
- 大型离线 replay 放 CPU，仅把 minibatch 移到 GPU，方向合理。
- 已有教师生命周期和恢复回归覆盖较丰富；优先补漏洞，不宜重写整个执行引擎。

## 改进顺序与验收

### 第一阶段：封住会污染结果解释的缺口

先修 F01、F02、F04，再修 F03。全部可用合成数据验证，不需要消费新的市场窗口。每项做负例复现及正例回归，尤其要证明“错误时状态未改变”。

产物：一个统一的运行身份 schema、一份覆盖完整 lane 的测试清单、一套推理身份/数值契约。不要顺手更改已冻结的教师阈值、费用或筛选条件。

### 第二阶段：使当前研究入口可以复现和交付

解决 F05/F06。将 qa 的日期脚本中通用的协议校验、来源校验、恢复、配对汇总逐步抽成包内模块，日期脚本只保留冻结配置和入口；不一次性大搬迁。对标准化组件使用共享测试，避免 restore 与 summary 各自实现一套近似检查。

统一当前状态入口：docs/DEVELOPMENT_GOAL_CURRENT.md:8 仍指向旧 D:\Program Files\NautilusTrader_RL_ES，而 AGENTS 与根标记指向现项目。旧文件应明确标为历史并链接到新游标；从机器可读根标记生成路径说明，避免继续出现多个“当前目标”。

当前工作树有大量未提交修改。研究结论与交付应绑定完整工作树清单/补丁身份，不能仅记录 HEAD。历史报告和失败产物保持不变。

### 第三阶段：提高研究效率和解释能力

先执行既有冻结、完整窗口的教师对照方案，再决定是否进入学生训练。当前日期下未来窗口不能假装已完整；不改动现有验收门槛来适应结果。

在现有授权和新实验预注册下，把研究问题拆开：
1. 方向信号是否存在费用后的动作价值；
2. 固定入场、单次交易的退出效果，与允许重入的完整策略效果分别报告；
3. BC 是否学会教师行为，是否因 HOLD/FLAT 比例掩盖入场/退出错误；
4. 学生在同源、同成本、同窗口的原生回放中是否真正超过教师。

正收益案例筛选、负入场回避、完整教师行为是不同训练目标。使用已有消融框架在冻结条件下比较，不因 BC loss 下降就认定策略变好。报告逐类动作错误、交易频率、每次交易成本和净收益、按方向/窗口的分解、原生清算与回撤；不只看一个聚合 Sharpe。

对于确定性的教师/执行场景，应先检查不同 seed 是否真正改变轨迹；不要把重复的相同轨迹当作新的独立市场证据。保留现有种子矩阵，但在解释统计独立性时明确区别。

性能优化以测量为先：分别记录数据准备、特征构造、原生 rollout、训练、报告序列化耗时和峰值 RSS/VRAM。只有确认瓶颈后再做缓存、流式报告或并行；缓存键必须使用 F01 的完整身份。避免用更多实验并行放大同一版本/来源问题。

## 审计边界

本轮未修改业务代码、教师参数、预注册或市场数据；只新增审计报告、合成复现脚本和证据摘要。未下载新行情，未开启正式教师/学生训练，未消费新的 Validation/Test/Holdout，未提交或推送。回归中使用的是测试自带的合成/夹具路径。

已完成的本地测试不能替代 Windows 原生执行、Linux/Windows Hosted CI、依赖漏洞扫描、真实账户/流动性认证或策略盈利验收。F05/F06 为静态发现，其完整安装/外部证据闭环仍需后续专项验证。

## 核心文件字节身份（SHA-256）

| 文件 | SHA-256 |
|---|---|
| packages/route_a_imitation/src/rles_route_a/export/inference.py | 388f4498503dcf2328dfbb03137b84b220d8d00334f183b0d4b26b8e2ea31392 |
| qa/run_route_a_teacher_student_train_screen_20261026.py | 07a6aa9d4eabddf71bf43f48921310e791d97c06767e2a488887193bb21527ce |
| packages/data/src/rles_data/binance_archive.py | 06ce56358ca04bb4b22209e99328b6149a234ca67302b97ee090fca072ba2259 |
| .github/workflows/l6-ci-reusable.yml | 4bb1b960307c960e3443cde4227af5b9043a1cb6caa043c7253cd0deca2e395e |
| pyproject.toml | cef9121fad49f8b82ce11002a386d8e9e4c7f578f1f9eed285fc0fa888dcc611 |
| packages/route_a_imitation/src/rles_route_a/training/bc_trainer.py | 5a5be71da3d4a35e2100ffead412f076ff3890ab3a40f83a8049ff38758b590f |
