# 训练结果记录文档 (Training Results Log)

本文件用于详细记录每次训练后的关键性能指标、资源消耗及量化评分。

## 0. 训练 Agent 定义 (Training Agent)

- **Agent 名称**: `training_agent`
- **Autoresearch 角色**: 内环执行器，负责把已锁定的实验假设变成可复现的训练结果。
- **核心职责**:
  - 训练前先读取 `Reflection_and_Planning.md` 中最新一轮实验假设、改动点、风险和验证标准。
  - 优先使用 `cvprw26/run_confirm.py` 完成同一实验的两次独立复现，形成 `mAP_confirm` 的原始输入。
  - 仅在排障或快速冒烟时允许直接调用 `python -m src.train --config ...`，但该结果不得直接作为达标依据。
  - 训练结束后，立即整理 `best_model.pth`、`train.log`、`eval_results_epoch*.json`、训练时长、资源占用和异常信息，并回写本文件。
  - 将实验编号、配置差异、种子、关键指标和异常移交给 `review_agent` 审查。
- **主要输入**:
  - `cvprw26/config/disaster.yaml`
  - `cvprw26/run_confirm.py`
  - `cvprw26/src/train.py`
  - `docs/dev_management/Reflection_and_Planning.md`
- **主要输出**:
  - `cvprw26/outputs/{exp_id}_run1/`
  - `cvprw26/outputs/{exp_id}_run2/`
  - 本文件中的实验结果记录
- **禁止事项**:
  - 未在 `Reflection_and_Planning.md` 锁定本轮实验计划前直接启动正式训练。
  - 只跑单次结果就宣称达到 `mAP_confirm`、`CRI` 或阶段目标。
  - 训练失败后不记录异常原因、恢复动作和下一步处理。

### 0.1 标准执行顺序

1. 在 `Reflection_and_Planning.md` 锁定本轮实验假设、配置变更、风险和验证标准。
2. 在 `cvprw26` 下确认配置文件路径；必要时复制出本轮专用 `config_{exp_id}.yaml`。
3. 执行 `python run_confirm.py <config_path> <exp_id>`，产出 `outputs/{exp_id}_run1/` 与 `outputs/{exp_id}_run2/`。
4. 若当前目标仅为排障，允许先执行 `python -m src.train --config <config_path>` 做单轮调试，但不得替代双复现确认。
5. 训练完成后立即回写本文件，并把完整实验信息移交 `review_agent`。

### 0.2 标准命令

```bash
cd cvprw26
python run_confirm.py config/disaster.yaml exp001
python manage_cri.py exp001
```

### 0.3 训练 Agent 当前状态

- **状态**: 已定义，等待按 `Exp 001` 基线闭环执行。
- **当前默认任务**: 保留现有 `exp001_run1` 产物，补齐 `exp001_run2`，生成可用于审查的 `mAP_confirm`。
- **移交条件**: `run1`、`run2` 均产出 `best_model.pth`，且本文件完成关键指标回写。

## 1. 训练结果总览表

| 实验编号 | 日期 | 模型描述 | mAP (Val) | AP50 (Val) | 训练时间 | 资源占用 | **量化评分 (100分制)** | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 000 | 2026-03-25 | 官方基线 (Mask R-CNN) | 0.1854 | 0.3540 | - | - | 50.0 | 基准线 |
| 001 | 2026-03-25 | 基线首次运行 (CPU 稳定性回退) | [待记录] | [待记录] | [运行中] | CPU (Subset 200) | [待打分] | XPU 驱动崩溃，切换 CPU 建立 CRI 流水线 |

---

## 2. 详细评分与记录 (实验记录模板)

### 实验编号: [ID]
- **日期**: [YYYY-MM-DD]
- **模型架构**: [简述，例如: ResNet-50 + FPN]
- **改进策略**: [例如: 调整 Learning Rate, 引入 Mixup 增强]

#### 核心指标:
- **mAP**: [数值] (对比上次: [↑/↓ 数值])
- **AP50**: [数值]
- **AP75**: [数值]
- **Loss**: [Total / Box / Mask / Class]
- **F1-Score / AUC**: [可选]

#### 资源与时间:
- **训练时间**: [小时/分钟]
- **显存消耗 (Peak VRAM)**: [GB]
- **GPU 型号**: [例如: RTX 4090]

#### **量化评分 (10分制或100分制)**:
- **当前总分**: [分数]
- **对比上次评分**: [上升 / 下降]
- **评分标准与依据**:
  - 基础评分: 基于 mAP 相对于目标的达成度。
  - **扣分项**: 
    - [ ] 性能下降扣分 (若低于上一次 mAP，扣 [下降幅度/0.01 * 5] 分)。
    - [ ] 资源消耗增加且性能无明显提升 (扣 [X] 分)。
    - [ ] 稳定性问题 (训练过程中出现 Loss 震荡或梯度爆炸)。
  - **改进建议**: [此处详细记录后续优化方案]

---

## 3. 量化评分标准 (详细定义)
- **100分 - 顶级表现**: mAP 超过理想目标值 (0.35+)，且推理延迟低于基准。
- **80-99分 - 优秀表现**: 性能稳步上升，主要指标 (Damaged 类) 显著提升。
- **60-79分 - 合格优化**: 性能有一定提升或持平，训练成本可控。
- **< 60分 - 不及格或倒退**: 发生明显扣分。

**扣分执行机制**:
1. 若 `mAP_current < mAP_previous`，基础扣除 10 分，并按下降比例 (每 0.01 扣 5 分) 累加。
2. 若 `Train_Loss_current > Train_Loss_previous` 且 `mAP` 未提升，扣除 5 分。
3. 若改进策略未达到预期效果，必须在“改进建议”中给出具体原因剖析。
