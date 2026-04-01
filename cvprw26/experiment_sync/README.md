# Experiment Sync

本目录是远程主环境实验摘要在本机的 Git-safe 备份层，不是远程训练产物目录本身。

## 1. 角色定义

- 远程主环境：`lizilong@146.56.220.99:21427:/remote-home/lizilong/bright_cvprw26`
- 本机备份层：当前仓库中的 `cvprw26/experiment_sync/`
- GitHub：仅用于共享适合 Git 的轻量、结构化、可公开信息

结论：正式训练、推理、评估和状态判定都以远程为准；本目录只负责保存回收到本机的关键摘要。

## 2. 目录内容

- `experiments.yaml`：实验索引与当前 reference 状态
- `latest_summary.md`：本机整理后的顶层摘要视图
- `decision_log.md`：关键 best event、状态变化和决策记录
- `training_queue.json`：远程 watchdog 可执行的已批准训练步骤
- `watchdog_state.json`：远程 watchdog 当前状态
- `watchdog_events.jsonl`：远程 watchdog 关键动作日志
- `evaluations/<exp_id>.json`：`manage_cri.py --json-out` 产出的结构化评估结果
- `runs/<experiment_id>/`：单轮实验的结构化备份
  - `latest_summary.md`
  - `metrics.jsonl`
  - `metrics_epoch*.json`
  - `training_state.json`
  - `origin_info.json`
  - `remote_runtime.json`
  - `sync_manifest.json`
  - `decision_log.md`
  - `repo_snapshot/`

注意：远程未必存在顶层 `latest_summary.md`。若本机存在该文件，应理解为本机备份层整理出的视图，而不是远程原生事实文件。

## 3. 同步规则

- 优先使用 GitHub 中转同步：
  - 代码
  - 配置
  - 实验摘要
  - 决策日志
  - 论文日志
  - 可复用脚本和文档
- 不适合进入 GitHub 的内容，必须直接在本机和远程之间同步：
  - 数据集
  - 模型权重
  - 原始输出目录
  - 缓存
  - 临时排障文件
  - 大文件
  - 密钥和敏感信息

## 4. 关键节点同步策略

只在以下关键节点同步：

- 产生新的 best/reference 结果
- 一轮实验进入 `completed`、`failed`、`interrupted`、`stalled`
- 连续停滞，需要调研新算法或切换路线
- 形成新的稳定论文素材或阶段性结论

以下情况不主动同步：

- 普通 epoch 增长
- 临时调试
- 尚未确认的中间结果
- 高频状态刷新

## 5. 推荐工作流

1. 由远程 `scripts/remote_training_watchdog.py` 读取 `training_queue.json`，自动恢复或继续已批准训练。
2. 训练关键节点由远程 watchdog 写入 `watchdog_state.json`、`watchdog_events.jsonl` 和 `evaluations/<exp_id>.json`。
3. 在关键节点把 `experiment_sync/runs/<experiment_id>/` 的结构化摘要与 watchdog 状态回收到本机。
4. 在本机更新 `docs/dev_management/` 和 `docs/paper_writing/`。
5. 若内容适合 Git，则通过 GitHub 作为中转共享；若不适合 Git，则仅保留直连同步。

## 6. 远程 watchdog 职责边界

- 只能执行 `training_queue.json` 中 `approved=true` 的训练步骤
- 只能根据量化评估结果判断“达标 / 继续训练 / 等待本机 Codex”
- 不能自行发明新的实验路线、骨干网络、损失函数或优化策略
- 默认 cron 入口：
  - `*/15 * * * * cd /remote-home/lizilong/bright_cvprw26 && /usr/bin/flock -n .watchdog.lock /usr/bin/python3 scripts/remote_training_watchdog.py >> experiment_sync/watchdog_cron.log 2>&1`

## 7. 当前已确认的备份对象

- `exp001_run1` 已完成同步，当前是唯一已确认的远程基线实验备份
- `exp001_run2` 仍未形成已同步结果
