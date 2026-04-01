# 反思与计划文档 (Reflection and Planning)

本文件系统性地记录远程主环境中每次训练后的深入分析、模型表现剖析、超参数策略及下一步的具体优化方案。本机只负责把这些关键节点整理成可追溯文档。

## 0. 优化 Agent 定义 (Optimization Agent)

- **Agent 名称**: `optimization_agent`
- **Autoresearch 角色**: 外环反思器，负责吸收审查结论、更新研究判断，并生成下一轮最小必要改动方案。
- **核心职责**:
  - 读取 `Target_Metrics.md` 的最新审查结论和 `Training_Log.md` 的实验细节。
  - 判断当前问题属于远程训练稳定性故障、类别短板、数据问题、配置不当，还是架构天花板。
  - 按 CRI 最大化原则给出下一轮 1 到 2 个最高优先级改动，避免一次叠加过多变量导致无法归因。
  - 明确写出实验假设、预期收益、风险、验证标准、失败后的回退动作。
  - 将下一轮计划移交给 `training_agent`，形成 Autoresearch 的外环到内环闭环。
- **主要输入**:
  - `docs/dev_management/Target_Metrics.md`
  - `docs/dev_management/Training_Log.md`
  - `cvprw26/manage_cri.py` 输出的指标结论
- **主要输出**:
  - 本文件中的下一轮实验计划
  - 对根因的判断
  - 对 `training_agent` 的可执行改动指令
- **决策优先级**:
  - 先修复训练失败、收敛异常、数据错误、环境问题。
  - 再修复拖累 `CRI` 的最弱类别短板。
  - 再推进结构性增强，例如损失函数、采样策略、骨干网络升级。
  - 若连续多轮 `CRI` 无提升，必须显式执行 `deepen / broaden / pivot` 判断，禁止盲目扫参。

### 0.1 标准外环流程

1. 读取 `review_agent` 的最新 verdict 和阻断项。
2. 归纳本轮哪些改动有效、哪些无效、哪些结论仍不充分。
3. 判断下一步是继续加深验证、扩展新假设，还是中止当前路线并 pivot。
4. 在本文件写清下一轮实验计划、预期收益、风险与止损标准。
5. 将计划移交给 `training_agent` 执行，不留模糊口头结论。

### 0.2 优化 Agent 当前状态

- **当前阶段**: 远程基线闭环阶段。
- **当前优先级**: 先完成 `exp001` 双复现，建立初始 `mAP_confirm` 与 `CRI`。
- **审查后的默认优化方向**:
  - 若 `damaged` 为最弱类别，优先尝试 `model.loss: focal` 或采样均衡策略。
  - 若整体收敛不稳定且无明显类别短板，优先排查 `lr / batch_size / warmup`。
  - 若基线稳定但提升停滞，再评估 Backbone 升级方案。

## 1. 实验回顾模板 (Reflection Summary)

### 实验编号: exp001_run1 - Remote Baseline (Mask R-CNN ResNet-50)

#### 1.1 表现分析 (Performance Analysis)
- **优势**:
  - 远程主环境已证明可以稳定完成 `12` 个 epoch 的基线训练并导出结构化摘要。
  - `intact` 类别 AP 达到 `0.3178`，是当前三类中最强的一类。
  - `experiment_sync` 已形成 `latest_summary.md`、`metrics.jsonl`、`decision_log.md` 等可追溯材料。
- **劣势**:
  - 当前 `segm_AP=0.1694`，低于官方公开基线 `0.1854`。
  - `damaged=0.0541` 是当前最明显短板，严重拖累后续 `CRI`。
  - 目前只有单次 `run1`，还没有 `mAP_confirm` 与 `CRI`。

#### 1.2 剖析原因 (Root Cause Analysis)
- **模型架构**: 标准 Mask R-CNN ResNet-50 + FPN，基准线评估。
- **超参数策略**: LR=0.002, Epochs=12, Steps=[8, 11]，旨在建立 1x schedule 稳定基线。
- **数据与执行环境**: 当前主线以远程 `/remote-home/lizilong/bright_cvprw26` 为准，训练与验证均围绕 `data/instance_annotations/train.json` 与 `val.json` 展开。当前已同步摘要中没有证据支持 “CPU + 200 张子集” 仍是主线事实，因此不再沿用该表述。
- **当前判断**:
  - 现阶段问题首先不是远程训练无法完成，而是基线精度偏低且弱类明显失衡。
  - 在只有单次运行的前提下，任何结构性优化都应先建立 `run2` 复现，再决定是否继续沿当前配置拓展。

#### 1.3 调优心得 (Lessons Learned)
- 单次结果只能作为远程基线快照，不能直接做主线升格决策。
- 当前最需要的是补齐复现实验，而不是一次叠加多种策略。
- 同步必须坚持“远程为主，本机备份，关键节点才同步”，否则日志会被高频噪声污染。

---

## 2. 下一步计划 (Next Steps)

### 2.1 具体优化方案 (Specific Optimization Plan)
- **短期计划 (Next Run)**:
  - [ ] 由远程 watchdog 按 `training_queue.json` 自动补齐 `exp001_run2`。
  - [ ] 由远程 watchdog 调用 `python manage_cri.py exp001 --json-out experiment_sync/evaluations/exp001.json`，建立首个 `mAP_confirm` 与 `CRI`。
  - [ ] 将 `run2` 结果和 `CRI` 同步到本机 `experiment_sync`、`Training_Log.md` 和论文日志。
- **中期计划 (Strategic Plan)**:
  - [ ] 若 `damaged` 仍为最弱类别，优先尝试 `Focal Loss` 或 `Balanced Sampling`，一次只改一个变量。
  - [ ] 若双复现后整体仍显著低于官方基线，再评估更长 schedule 或更强 backbone。

---

## 3. 超参数调整策略记录

| 实验 ID | 参数名称 | 原值 | 新值 | 调整理由 | 预期效果 | 实际结果 (↑/↓) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| exp001_run1 | lr | 默认基线 | 0.002 | 建立远程 1x schedule 基线 | 稳定收敛 | `segm_AP=0.1694`，低于官方基线 |
| exp001_run1 | epochs | 默认短训 | 12 | 建立可复现的远程完整基线 | 得到完整曲线与 best epoch | 已完成，best epoch=11 |
| exp001_run1 | batch_size | 未固定 | 1 | 适配当前基线配置 | 完成训练并沉淀摘要 | 已完成，但弱类 AP 偏低 |

---

## 4. 优化日志 (历史关键节点)
- **2026-03-29**: 远程 `exp001_run1` 完成，`segm_AP=0.1694`，best epoch=11，当前低于官方公开基线。
- **2026-04-01**: 明确“远程为主、本机为备份”的规则，并将同步策略与论文日志结构纳入项目文档。
- **2026-04-01**: 上线远程 `remote_training_watchdog.py`、`training_queue.json` 和用户级 cron，后续 `exp001_run2` 的补跑与 CRI 评估不再依赖本机常开。
