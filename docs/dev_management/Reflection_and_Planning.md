# 反思与计划文档 (Reflection and Planning)

本文件系统性地记录每次训练后的深入分析、剖析模型表现、超参数策略及下一步的具体优化方案。

## 0. 优化 Agent 定义 (Optimization Agent)

- **Agent 名称**: `optimization_agent`
- **Autoresearch 角色**: 外环反思器，负责吸收审查结论、更新研究判断，并生成下一轮最小必要改动方案。
- **核心职责**:
  - 读取 `Target_Metrics.md` 的最新审查结论和 `Training_Log.md` 的实验细节。
  - 判断当前问题属于稳定性故障、类别短板、数据问题、配置不当，还是架构天花板。
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

- **当前阶段**: 基线闭环阶段。
- **当前优先级**: 先完成 `exp001` 双复现，建立初始 `mAP_confirm` 与 `CRI`。
- **审查后的默认优化方向**:
  - 若 `damaged` 为最弱类别，优先尝试 `model.loss: focal` 或采样均衡策略。
  - 若整体收敛不稳定且无明显类别短板，优先排查 `lr / batch_size / warmup`。
  - 若基线稳定但提升停滞，再评估 Backbone 升级方案。

## 1. 实验回顾模板 (Reflection Summary)

### 实验编号: 001 - Baseline (Mask R-CNN ResNet-50)

#### 1.1 表现分析 (Performance Analysis)
- **优势**: [待记录]
- **劣势**: [待记录]

#### 1.2 剖析原因 (Root Cause Analysis)
- **模型架构**: 标准 Mask R-CNN ResNet-50 + FPN，基准线评估。
- **超参数策略**: LR=0.002, Epochs=12, Steps=[8, 11]，旨在建立 1x schedule 稳定基线。
- **数据质量**: 原始 BRIGHT 数据集。由于 Intel Arc A380 显卡驱动在处理大图时不稳定 (UR_RESULT_ERROR_DEVICE_LOST)，已强制切换至 **CPU** 训练，并使用 **200张图像子集** 建立 CRI 优化流水线。

#### 1.3 调优心得 (Lessons Learned)
- [待记录]

---

## 2. 下一步计划 (Next Steps)

### 2.1 具体优化方案 (Specific Optimization Plan)
- **短期计划 (Next Run)**:
  - [ ] 完成 Exp 001 Run 1 & Run 2 建立 mAP_confirm。
  - [ ] 计算基准 CRI，识别当前模型在 Damaged 类的瓶颈。
- **中期计划 (Strategic Plan)**:
  - [ ] 引入 Focal Loss 或 Balanced Sampling 解决损毁类别分布不均问题。
  - [ ] 探索 Swin-Transformer 或 ConvNeXt 作为 Backbone 提升特征表征能力。

---

## 3. 超参数调整策略记录

| 实验 ID | 参数名称 | 原值 | 新值 | 调整理由 | 预期效果 | 实际结果 (↑/↓) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 001-R1 | lr | 0.02 | 0.002 | 线性缩放规则适配 Batch 2 | 稳定收敛 | [待记录] |
| 001-R1 | epochs | 1 | 12 | 建立标准训练时长 | 性能提升 | [待记录] |

---

## 4. 优化日志 (历史关键节点)
- **2026-03-25**: 成功搭建基线环境并生成初步标注，准备开始首次实验。
- **[日期]**: [描述关键优化进展]
