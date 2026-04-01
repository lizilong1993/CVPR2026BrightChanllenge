# PROJECT_CONTEXT

## 1. 项目目标与当前阶段

- 项目目标：围绕 CVPR 2026 BRIGHT Challenge 建立可复现的实例级建筑损毁识别训练、评估、优化和论文材料沉淀流程。
- 当前阶段：远程主环境的基线闭环阶段。
- 当前唯一已确认的远程实验：`exp001_run1`。

## 2. 环境主从关系

- 远程主环境：
  - 主机：`lizilong@146.56.220.99`
  - 端口：`21427`
  - 项目根：`/remote-home/lizilong/bright_cvprw26`
  - 职责：正式训练、推理、评估、监控、实验推进、状态判定
- 本机备份环境：
  - 路径：`C:\Users\lizilong\Desktop\武大人工智能学院\论文专著\CVPR2026BrightChanllenge`
  - 职责：接收远程关键摘要、维护项目文档、沉淀论文材料、通过 GitHub 共享轻量信息
- 事实优先级：远程已验证事实 > 本机备份快照 > 人工回忆或旧计划

## 3. 关键目录与职责

- 远程 `/remote-home/lizilong/bright_cvprw26/config/`：正式实验配置
- 远程 `/remote-home/lizilong/bright_cvprw26/outputs/`：训练输出、权重和原始日志，不直接入 GitHub
- 远程 `/remote-home/lizilong/bright_cvprw26/experiment_sync/runs/<experiment_id>/`：单轮实验的结构化摘要
- 本机 `docs/dev_management/`：训练管理、指标门禁、反思与下一步计划
- 本机 `docs/paper_writing/`：面向论文写作的训练与优化总日志
- 本机 `cvprw26/experiment_sync/`：Git-safe 的远程实验摘要备份层

## 4. 同步策略

- GitHub 中转优先：代码、配置、结构化实验摘要、决策日志、论文日志、可复用脚本、说明文档
- 直连同步兜底：数据集、权重、原始输出目录、缓存、临时排障文件、大文件、密钥和其他敏感信息
- 只在关键节点同步：
  - 新的 best/reference 结果
  - 训练进入 `completed`、`failed`、`interrupted`、`stalled`
  - 连续停滞，需要调研新算法或切换路线
  - 形成新的稳定论文素材或阶段性结论
- 非关键节点不同步：普通 epoch 递进、短时波动、临时调试、尚未确认的中间结果、高频状态刷新

## 5. 当前已确认事实

- 远程项目根已确认存在：`/remote-home/lizilong/bright_cvprw26`
- 远程 `experiment_sync/runs/` 当前可见实验目录只有 `exp001_run1`
- 远程 `experiment_sync/runs/exp001_run1/latest_summary.md` 已确认：
  - `Status=COMPLETED`
  - `Best segm_AP=0.1694`
  - `Best epoch=11`
  - `segm_AP50=0.3139`
  - `segm_AP75=0.1718`
  - `intact=0.3178`
  - `damaged=0.0541`
  - `destroyed=0.1364`
  - `train_loss=0.8327`
- 远程 `training_state.json` / `origin_info.json` 已确认：
  - 输出目录：`/remote-home/lizilong/bright_cvprw26/outputs/exp001_run1`
  - 远程同步目录：`/remote-home/lizilong/bright_cvprw26/experiment_sync/runs/exp001_run1`
  - 导出时间：`2026-03-29T00:19:54Z`
- 远程顶层 `experiment_sync/latest_summary.md` 当前不存在；本机顶层摘要是后续整理出的备份视图，不是远程原生事实文件
- `exp001_run2` 仍未形成已同步结果，因此当前没有 `mAP_confirm`，也没有 `CRI`

## 6. 常用命令

```bash
ssh -p 21427 lizilong@146.56.220.99
cd /remote-home/lizilong/bright_cvprw26
python run_confirm.py config/disaster.yaml exp001
python manage_cri.py exp001
cat experiment_sync/runs/exp001_run1/latest_summary.md
```

```bash
git status -sb
git push -u origin codex/docs-experiment-sync
```

## 7. 当前限制

- 本机未安装 `gh`
- 本机到 GitHub `22` 端口当前超时，因此 GitHub 发布默认走现有 HTTPS remote
- 本轮只提交“备份层和文档层”内容，不混入当前未确认训练代码改动
