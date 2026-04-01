# Training and Optimization Log

本文件面向论文写作，汇总远程主环境中已经发生的训练与优化关键节点。本机只做备份、整理和 GitHub 中转，不作为正式实验事实源。

## 1. 使用规则

- 主环境：`lizilong@146.56.220.99:21427:/remote-home/lizilong/bright_cvprw26`
- 备份环境：本机仓库 `C:\Users\lizilong\Desktop\武大人工智能学院\论文专著\CVPR2026BrightChanllenge`
- 事实来源优先级：远程已验证结果 > 本机 `cvprw26/experiment_sync/` 摘要 > 旧计划或口头结论
- 本文件只记录关键节点：
  - 新的 best/reference 结果
  - 一轮实验进入 `completed`、`failed`、`interrupted`、`stalled`
  - 连续停滞，需要调研新算法或切换路线
  - 形成新的稳定论文素材或阶段性结论
- 每条记录必须标注：远程路径、实验编号、同步时间、同步方式、当前能直接用于论文的结论

## 2. 实验总览表

| 实验 ID | 阶段 | 目标 | 远程来源 | 配置摘要 | 关键结果 | 当前结论 | 同步记录 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| exp001_run1 | Baseline | 建立远程主环境的首个稳定基线 | `/remote-home/lizilong/bright_cvprw26/outputs/exp001_run1` | `disaster.yaml`, `epochs=12`, `batch_size=1`, `lr=0.002`, `seed=42` | `segm_AP=0.1694`, `AP50=0.3139`, `AP75=0.1718`, `intact=0.3178`, `damaged=0.0541`, `destroyed=0.1364` | 单次结果，低于官方公开基线 `0.1854`，不能直接升格为主线 | `2026-03-29T00:19:54Z`，远程 `experiment_sync` 摘要回收到本机备份层 |

## 3. 详细实验记录

### 3.1 exp001_run1

#### 背景与目标

- 目标：在远程主环境上建立 Mask R-CNN 基线的首个稳定结果，验证训练、评估和摘要回收链路是否可用。
- 远程项目根：`/remote-home/lizilong/bright_cvprw26`
- 远程输出目录：`/remote-home/lizilong/bright_cvprw26/outputs/exp001_run1`
- 远程摘要目录：`/remote-home/lizilong/bright_cvprw26/experiment_sync/runs/exp001_run1`

#### 关键步骤

1. 在远程主环境使用 `config/disaster.yaml` 运行基线训练，输出至 `outputs/exp001_run1/`。
2. 训练完成后导出 `experiment_sync/runs/exp001_run1/` 的结构化摘要。
3. 将摘要回收到本机备份层，用于更新管理文档、论文日志和 GitHub 可提交信息。
4. 基于 `latest_summary.md`、`training_state.json`、`origin_info.json` 与 `decision_log.md` 交叉确认关键指标和 best event。

#### 配置摘要

- 模型：Mask R-CNN ResNet-50 + FPN
- 配置文件：`config/disaster.yaml`
- `epochs=12`
- `batch_size=1`
- `lr=0.002`
- `lr_steps=[8, 11]`
- `seed=42`
- 训练与验证入口：
  - 训练：`python run_confirm.py config/disaster.yaml exp001` 的单轮前置结果
  - 后续审查：`python manage_cri.py exp001`

#### 结果

- 状态：`completed`
- 同步时间：`2026-03-29T00:19:54Z`
- 最佳 epoch：`11`
- `segm_AP=0.1694`
- `segm_AP50=0.3139`
- `segm_AP75=0.1718`
- `intact=0.3178`
- `damaged=0.0541`
- `destroyed=0.1364`
- `train_loss=0.8327`

#### 结果解读

- 这次实验证明远程主环境能够完成 12 epoch 基线训练，并沉淀可追溯的摘要文件。
- 当前 `intact` 类表现相对较好，但 `damaged` 类 AP 显著偏低，是当前最主要的短板。
- 当前整体 `segm_AP=0.1694` 低于官方公开基线 `0.1854`，因此该轮不能视为竞争性结果。
- 当前只有 `run1`，尚未形成 `mAP_confirm` 和 `CRI`，不能据此宣称阶段达标。

#### 可直接用于论文写作的表述

- 远程主环境中的 Mask R-CNN baseline 在 `12` 个 epoch 后达到 `0.1694` 的 `segm_AP`，最佳结果出现在 epoch `11`。
- 当前基线对 `intact` 类更敏感，但对 `damaged` 和 `destroyed` 类的区分能力不足，尤其 `damaged` 类 AP 仅为 `0.0541`。
- 该结果低于官方公开基线，因此后续优化应优先围绕弱类识别能力和复现实验闭环展开，而不是直接扩大模型复杂度。

#### 当前缺失信息

- 精确训练时长
- GPU 型号与峰值显存
- `exp001_run2`
- `mAP_confirm`
- `CRI`

#### 下一步

- 在远程主环境按相同配置完成 `exp001_run2`
- 运行 `python manage_cri.py exp001`
- 将双复现结果和 `CRI` 回写到本文件与 `docs/dev_management/`

## 4. 图表与表格素材来源

- 单轮关键指标表：`cvprw26/experiment_sync/runs/exp001_run1/latest_summary.md`
- 逐 epoch 曲线：`cvprw26/experiment_sync/runs/exp001_run1/metrics.jsonl`
- 单 epoch 指标快照：`cvprw26/experiment_sync/runs/exp001_run1/metrics_epoch*.json`
- 关键决策时间线：`cvprw26/experiment_sync/decision_log.md`
- 远程路径与同步时间：`cvprw26/experiment_sync/runs/exp001_run1/origin_info.json`、`training_state.json`、`sync_manifest.json`

## 5. 同步节点记录

| 时间 | 节点 | 远程来源 | 同步方式 | 内容范围 | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-03-29T00:19:54Z | `exp001_run1` best + completed | `/remote-home/lizilong/bright_cvprw26/experiment_sync/runs/exp001_run1` | 远程摘要回收到本机备份层 | `latest_summary.md`, `metrics.jsonl`, `metrics_epoch011.json`, `training_state.json`, `origin_info.json`, `decision_log.md` | 当前首个稳定远程基线快照 |
| 2026-04-01 | 规则与论文日志收敛 | 本机备份层 | 文档更新，待通过 GitHub 共享 | `agent.md`, `PROJECT_CONTEXT.md`, `docs/dev_management/`, 本文件 | 不引入新的训练结果，只整理已确认事实 |
| 2026-04-01 | 远程 watchdog 与本机 Frontier Loop 升级 | `/remote-home/lizilong/bright_cvprw26` | 本机分支实现并部署到远程 | `manage_cri.py`, `scripts/remote_training_watchdog.py`, `training_queue.json`, `watchdog_state.json`, 本机自动化配置 | 目标是让 `exp001_run2` 的续训和后续量化门禁不再依赖本机关机状态 |
| 2026-04-01T04:37:33Z | 远程 watchdog 启动 `exp001_run2` | `/remote-home/lizilong/bright_cvprw26/outputs/exp001_run2` | 远程 watchdog 状态与运行目录回收到本机备份层 | `watchdog_state.json`, `watchdog_events.jsonl`, `outputs/exp001_run2/config_seed123.yaml`, `outputs/exp001_run2/train.log` | 证明远程自驱续训链路已实际拉起训练，而不是只停留在文档或脚本层 |

## 6. 系统级里程碑

### 6.1 远程自驱续训机制上线

- 时间：`2026-04-01`
- 远程环境：`lizilong@146.56.220.99:21427:/remote-home/lizilong/bright_cvprw26`
- 变更内容：
  - 新增远程 `scripts/remote_training_watchdog.py`
  - 新增 `experiment_sync/training_queue.json`
  - 新增 `experiment_sync/watchdog_state.json` 与 `watchdog_events.jsonl`
  - 扩展 `manage_cri.py`，使其可以输出 `evaluations/<exp_id>.json`
  - 将本机 `BRIGHT Frontier Loop` 改为读取远程 watchdog 状态，而不是本地直接判断是否开训
- 设计约束：
  - 远程 watchdog 只执行已批准训练步骤
  - 不达标但没有下一步批准计划时，状态必须停在 `awaiting_local_codex`
  - 本机自动化只在 `awaiting_local_codex`、`stalled`、`critical_blocker` 时保留 inbox 和进入调研
- 可直接用于论文方法部分的表述：
  - 为减少本机关机对实验连续性的影响，项目引入了一个运行在远程服务器上的轻量 watchdog。该模块通过读取显式训练队列与结构化评估结果，在训练中断时自动恢复已批准实验，并在达到量化门禁或进入阻塞状态时将状态回传到本机备份层。
