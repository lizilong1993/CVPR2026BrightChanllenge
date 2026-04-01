# 目标指标文档 (Target Metrics)

本文件定义 BRIGHT 挑战赛项目的唯一总控指标、阶段目标、训练闭环和项目结束条件。后续所有训练、调参、数据处理、模型改进和后处理，均以提升 CRI 为唯一目标。正式结果以远程主环境为准，本机只保存回收后的摘要和文档闭环。

## 0. 审查 Agent 定义 (Review Agent)

- **Agent 名称**: `review_agent`
- **Autoresearch 角色**: 内环结果审查器和门禁裁决者，负责判定训练结果是否允许进入下一轮优化或主线扩展。
- **核心职责**:
  - 读取 `Training_Log.md` 的最新实验记录，以及本机 `cvprw26/experiment_sync/runs/{exp_id}_run*/` 摘要和远程 `outputs/{exp_id}_run*/` 中的必要产物。
  - 运行 `python manage_cri.py {exp_id}` 或等价逻辑，计算 `mAP_confirm`、类别 AP 均值和 `CRI`。
  - 对照本文件第 2、3、6 节，给出 `通过 / 有条件通过 / 不通过` 的明确结论。
  - 识别阻断项，包括类别短板、文档未闭环、单次结果冒充复现结果、训练异常未解释等。
  - 将审查结论和主要瓶颈移交给 `optimization_agent`，作为下一轮外环优化输入。
- **主要输入**:
  - `docs/dev_management/Training_Log.md`
  - `cvprw26/experiment_sync/runs/{exp_id}_run*/`
  - `cvprw26/manage_cri.py`
  - 远程 `outputs/{exp_id}_run1/`
  - 远程 `outputs/{exp_id}_run2/`
- **主要输出**:
  - 最新实验的 `mAP_confirm`
  - 最新实验的 `CRI`
  - 审查结论与阻断原因
- **硬性阻断条件**:
  - 没有形成同一配置的双复现实验结果。
  - `Training_Log.md` 或 `Reflection_and_Planning.md` 未完成闭环更新。
  - 指标退化但未给出根因和修复动作。
  - 关键类别 AP 明显拖累 `CRI`，却被忽略并继续推进主线。

### 0.1 标准审查顺序

1. 确认 `training_agent` 已完成双复现实验并回写 `Training_Log.md`。
2. 执行 `python manage_cri.py <exp_id>`，得到 `mAP_confirm` 与 `CRI`。
3. 对照本文件第 2、3、6 节判断是否达到基线、M1、M2、Target，以及是否满足 `R = 1`。
4. 输出明确 verdict，并写明主阻断项或通过理由。
5. 将 verdict 交给 `optimization_agent`，禁止无审查直接进入下一轮大改。

### 0.2 审查结论模板

- **最新实验编号**: `exp001_run1`
- **mAP_confirm**: `未建立`
- **CRI**: `未建立`
- **审查结论**: `当前不通过主线升格`
- **主阻断项**: `缺少 exp001_run2，且当前 segm_AP=0.1694 低于官方公开基线 0.1854`

### 0.3 审查 Agent 当前状态

- **状态**: 已定义，当前只收到远程 `exp001_run1` 的已同步结果。
- **当前默认阻断条件**: 在 `mAP_confirm` 未建立前，不得宣称达标、不得切换主线最佳方案。
- **当前已知 best**: `exp001_run1` 的 `segm_AP=0.1694`，低于官方公开基线 `0.1854`。

## 1. 总控指标与决策原则

- **唯一总控指标**: `CRI`
- **禁止事项**: 不得仅以单次最高 `mAP` 作为最佳方案标准；所有方案选择必须以 `CRI` 是否提升为准。
- **事实优先级**: 远程已验证结果 > 本机备份摘要 > 旧计划或口头结论。

### 1.1 CRI 定义

`CRI = 100 × [0.70 × (mAP_confirm / 0.3500) + 0.20 × min(AP_intact / T_intact, AP_damaged / T_damaged, AP_destroyed / T_destroyed) + 0.10 × R]`

其中：
- `mAP_confirm`: 同一配置连续两次独立复现实验的平均 `mAP`。
- `AP_intact`、`AP_damaged`、`AP_destroyed`: 三个类别各自的验证集 AP。
- `T_intact`、`T_damaged`、`T_destroyed`: 取自本文档第 3 节的分类目标阈值。
- `R`: 仅当本次实验完整更新 `Training_Log.md` 与 `Reflection_and_Planning.md`，且配置、资源消耗、结论闭环完整时取 `1`，否则取 `0`。

### 1.2 决策硬规则

- 所有实验必须同时关注 `mAP_confirm`、三类 AP 的短板项和文档闭环完成度。
- 任何导致 `CRI` 下降的改动，不得作为默认主线方案继续扩展。
- 遇到报错、退化、过拟合、类别失衡、数据问题、显存瓶颈、收敛异常时，必须继续定位根因、提出修复方案并推进任务，禁止绕开问题，禁止只汇报不处理。
- 禁止主动中断任务，禁止将本应完成的分析、判断、排错和方案选择转交给他人。

## 2. 性能基准设定

| 性能阶段 | mAP (Validation) | AP50 (Validation) | 备注 |
| :--- | :--- | :--- | :--- |
| **最低可接受阈值** | 0.1854 | 0.3540 | 基线水平 (Baseline) |
| **短期里程碑 (M1)** | 0.2200 | 0.4000 | 初步优化，引入多模态融合改进 |
| **中期目标 (M2)** | 0.2800 | 0.4800 | 强化学习策略或数据增强策略见效 |
| **理想目标值 (Target)** | 0.3500+ | 0.5500+ | 具备竞争力的顶级模型水平 |

## 3. 分类性能细分目标

- **Intact (完好)**: `T_intact = 0.4000`
- **Damaged (受损)**: `T_damaged = 0.2000`
- **Destroyed (毁坏)**: `T_destroyed = 0.2500`

说明：
- CRI 中的类别项采用 `min(AP_intact / T_intact, AP_damaged / T_damaged, AP_destroyed / T_destroyed)`，因此最弱类别决定该项得分。
- 任何只提升整体 `mAP`、但放大类别短板的方案，都不应视为优先方案。

## 4. 训练执行闭环

1. **训练前先看计划**: 启动新一轮训练前，必须先查看 `Reflection_and_Planning.md`，并严格按其中当前计划执行。
2. **训练前先补计划**: 每次训练前，必须先在 `Reflection_and_Planning.md` 记录本轮变动原因、预期收益、风险与验证标准。
3. **训练后先验证**: 每次训练后，必须先在远程主环境对 `data/instance_annotations/val.json` 运行验证，得到 `mAP`、各类 AP 及其他必要指标。
4. **关键节点再同步**: 只有在产生 best/reference、实验结束、进入阻断态或形成稳定结论时，才把 `experiment_sync/runs/<experiment_id>` 摘要、`evaluations/<exp_id>.json` 与 `watchdog_state.json` 回收到本机备份层。
5. **训练后立即记日志**: 验证和同步完成后，立即更新 `Training_Log.md`，至少记录 `mAP`、各类 AP、配置变化、同步时间、结论和下一步。
6. **训练后同步反思**: 同一轮实验结束后，必须回写 `Reflection_and_Planning.md`，补全结果分析、问题定位、是否达预期、下一轮计划，形成闭环。
6. **退化必须处理**: 若当前 `mAP` 低于上一次结果，必须执行既定扣分规则，并明确给出下一步修复方案，不得跳过。
7. **结束前再查目标**: 每轮训练闭环完成后，必须回看本文档，依据第 6 节判断是否达到项目结束条件。

## 5. 评估流程规范

1. **远程验证**: 每次训练后优先在远程主环境运行 `src.test` 或等价评估流程。
2. **本机只做备份与复核**: 本机保留 `experiment_sync` 摘要、文档和论文材料，不用本机临时结果替代远程正式结果。
2. **Holdout 提交**: 达到 M1 阶段后，每周至少提交 1 次至 CodaBench。
3. **资源限制**: 模型应在单张 24GB/40GB GPU 上可训练及推理。

## 6. 评分与项目结束规则

### 6.1 训练记录评分要求

- 若当前 `mAP` 低于上一次，必须执行 `Training_Log.md` 中既定扣分规则。
- 若本轮实验未完整更新 `Training_Log.md` 与 `Reflection_and_Planning.md`，则本轮 `R = 0`。
- 若配置、资源消耗、结论记录不完整，即使 `mAP` 上升，也不得判定为闭环完成。

### 6.2 项目成功判定

只有同时满足以下条件，才可判定项目成功并结束训练：
- `mAP_confirm ≥ 0.3500`
- `CRI ≥ 100`
- `docs/dev_management` 下三份文档全部完成闭环更新

未同时满足以上条件时，不得判定项目成功。
