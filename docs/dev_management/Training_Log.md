# 训练结果记录文档 (Training Results Log)

本文件记录已同步回本机的远程训练结果。远程服务器是主事实源，本机只保留备份和论文写作所需摘要，不以本机运行态替代远程结论。

## 0. 训练 Agent 定义 (Training Agent)

- **Agent 名称**: `training_agent`
- **Autoresearch 角色**: 内环执行器，负责把已锁定的远程实验假设变成可复现的训练结果。
- **核心职责**:
  - 训练前先读取 `Reflection_and_Planning.md` 中最新一轮实验假设、改动点、风险和验证标准。
  - 正式训练、推理、评估默认在远程主环境 `/remote-home/lizilong/bright_cvprw26` 执行。
  - 优先使用 `cvprw26/run_confirm.py` 完成同一配置的两次独立复现，形成 `mAP_confirm` 的原始输入。
  - 训练结束后先回收远程 `experiment_sync/runs/{exp_id}` 摘要，再回写本文件。
  - 将实验编号、配置差异、关键指标和异常移交给 `review_agent` 审查。
- **主要输入**:
  - 远程 `config/disaster.yaml`
  - 远程 `run_confirm.py`
  - 远程 `src/train.py`
  - `docs/dev_management/Reflection_and_Planning.md`
- **主要输出**:
  - 远程 `outputs/{exp_id}_run1/`
  - 远程 `outputs/{exp_id}_run2/`
  - 本机 `cvprw26/experiment_sync/runs/{exp_id}_run*/`
  - 本文件中的实验结果记录
- **禁止事项**:
  - 未在 `Reflection_and_Planning.md` 锁定本轮计划前直接启动正式训练。
  - 只跑单次结果就宣称达到 `mAP_confirm`、`CRI` 或阶段目标。
  - 训练失败后不记录异常原因、恢复动作和下一步处理。
  - 用本机临时运行替代远程主环境的正式结果。

### 0.1 标准执行顺序

1. 在 `Reflection_and_Planning.md` 锁定本轮远程实验假设、配置变更、风险和验证标准。
2. 登录远程主环境，确认配置文件和输出目录。
3. 执行 `python run_confirm.py <config_path> <exp_id>`，产出远程 `outputs/{exp_id}_run1/` 与 `outputs/{exp_id}_run2/`。
4. 若当前目标仅为排障，允许先执行 `python -m src.train --config <config_path>` 做单轮调试，但该结果不得替代双复现确认。
5. 在关键节点把远程 `experiment_sync/runs/<experiment_id>` 摘要同步到本机备份层。
6. 回写本文件，并把完整实验信息移交 `review_agent`。

### 0.2 标准命令

```bash
ssh -p 21427 lizilong@146.56.220.99
cd /remote-home/lizilong/bright_cvprw26
python run_confirm.py config/disaster.yaml exp001
python manage_cri.py exp001
```

### 0.3 训练 Agent 当前状态

- **状态**: 已定义，当前远程主线只有 `exp001_run1` 已确认完成。
- **当前默认任务**: 在远程补齐 `exp001_run2`，再计算 `mAP_confirm` 与 `CRI`。
- **移交条件**: `run1`、`run2` 均形成已同步摘要，本文件与 `Reflection_and_Planning.md` 已完成闭环回写。

## 1. 训练结果总览表

| 实验编号 | 日期 | 模型描述 | mAP (Val) | AP50 (Val) | AP75 (Val) | 资源占用 | 状态 | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 000 | 2026-03-25 | 官方公开基线 (Mask R-CNN) | 0.1854 | 0.3540 | - | 官方公开结果 | published | 用作最低可接受阈值 |
| exp001_run1 | 2026-03-29 | 远程基线首次完整运行 | 0.1694 | 0.3139 | 0.1718 | 远程服务器；GPU 型号与训练时长待回收 | completed | 单次结果，低于官方基线，尚未形成 `mAP_confirm`/`CRI` |

## 2. 已确认远程实验记录

### 实验编号: exp001_run1

- **来源环境**: `lizilong@146.56.220.99:21427:/remote-home/lizilong/bright_cvprw26`
- **同步时间**: `2026-03-29T00:19:54Z`
- **同步方式**: 远程 `experiment_sync/runs/exp001_run1/` 摘要回收至本机备份层
- **输出目录**: `/remote-home/lizilong/bright_cvprw26/outputs/exp001_run1`
- **配置摘要**:
  - 配置文件：`config/disaster.yaml`
  - `epochs=12`
  - `batch_size=1`
  - `lr=0.002`
  - `lr_steps=[8, 11]`
  - `seed=42`
  - `output_dir=outputs/exp001_run1/`
- **状态**: `completed`
- **最佳 epoch**: `11`
- **核心指标**:
  - `segm_AP=0.1694`
  - `segm_AP50=0.3139`
  - `segm_AP75=0.1718`
  - `intact=0.3178`
  - `damaged=0.0541`
  - `destroyed=0.1364`
  - `train_loss=0.8327`
- **已确认产物**:
  - `best_model.pth` 存在
  - `latest.pth` 存在
  - `train.log` 存在
  - `metrics.jsonl` 和逐 epoch 指标已同步
- **当前不能确认的字段**:
  - GPU 型号
  - 峰值显存
  - 精确训练时长
  - `mAP_confirm`
  - `CRI`
- **结论**:
  - 本轮证明远程主环境可以完成 12 epoch 基线训练并导出结构化摘要。
  - 当前结果低于官方公开基线 `0.1854`，不能作为主线最佳方案。
  - 当前仅有 `run1`，不得宣称形成双复现确认。
- **下一步**:
  - 在远程执行同配置 `exp001_run2`
  - 远程运行 `python manage_cri.py exp001`
  - 将 `run2` 结果和 `CRI` 回写本文件与论文日志

## 3. 记录规则

- 只记录已验证并已同步回本机的远程关键结果。
- 若本机与远程冲突，以远程已验证事实为准。
- 新实验继续按“来源环境 -> 配置摘要 -> 核心指标 -> 结论 -> 下一步”的结构续写。
