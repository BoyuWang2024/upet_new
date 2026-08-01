# UPET ConfidenceHead 九组单分支消融实验设计

## 目标

在 bywang 服务器上为 UPET ConfidenceHead 准备九组可独立调参和提交的全数据 GPU 实验：一组只监督力，八组只监督能量并遍历累积矩阶数 1–8。实验沿用当前 UPET 的 early-stopping、逐原子力均值和逐原子能量定义，不迁移旧结果。

## 实验矩阵

| 配置 | force coefficient | energy coefficient | energy order |
| --- | ---: | ---: | ---: |
| `full_gpu_energy0.yaml` | 1 | 0 | 3（能量分支不参与 loss，仅作为配置占位） |
| `full_gpu_force0_order1.yaml` | 0 | 1 | 1 |
| `full_gpu_force0_order2.yaml` | 0 | 1 | 2 |
| `full_gpu_force0_order3.yaml` | 0 | 1 | 3 |
| `full_gpu_force0_order4.yaml` | 0 | 1 | 4 |
| `full_gpu_force0_order5.yaml` | 0 | 1 | 5 |
| `full_gpu_force0_order6.yaml` | 0 | 1 | 6 |
| `full_gpu_force0_order7.yaml` | 0 | 1 | 7 |
| `full_gpu_force0_order8.yaml` | 0 | 1 | 8 |

每份配置都是完整 YAML，便于后续单独修改超参数，不依赖模板展开或 Slurm array。

## 输入与输出

- checkpoint：`/home/bywang/code/UQ/upet/pet-omatpes-l-v0.1.0.ckpt`
- train：`/home/bywang/code/UQ/mace/UQ_orb_post_train_force/data/matpes_train.extxyz`
- validation：`/home/bywang/code/UQ/mace/UQ_orb_post_train_force/data/matpes_val.extxyz`
- test：`/home/bywang/code/UQ/mace/UQ_orb_post_train_force/data/matpes_test.extxyz`
- 输出根目录：`/home/bywang/code/UQ/upet_new/Uncertainty_Quantification/ConfidenceHead/outputs`

配置记录并校验上述四个输入文件在目标服务器上的 SHA-256。九份配置继承服务器基准配置的 `trainer.batch_size: 32`。

## 监督与 early stopping

- 力分支保持 `target_mode: atom_mean`，即逐原子对三个笛卡尔分量绝对误差求平均。
- 能量误差保持逐原子定义。
- 两条 ConfidenceHead 继续使用各自独立的 UPET readout 特征。
- 调度器、最佳 checkpoint、EMA 和 early stopping 均继续监督 `val/total_loss_ema`。
- force-only 运行的 total loss 等于 force loss；energy-only 运行的 total loss 等于 energy loss。

当前配置模型要求两个 loss 系数都严格大于零，与单分支消融冲突。实现将把单项约束调整为大于等于零，并新增组合校验，禁止两个系数同时为零；其他 loss 语义不变。

## 命名规则

九份配置统一使用 `name_prefix: upet_full_rtx5090`。沿用 UPET 的派生命名器，使输出目录名和 W&B run 名包含分箱、误差上限、目标模式、loss 权重、MLP 宽度和能量阶数。例如：

- `upet_full_rtx5090_linear_f50-fmax0.5-ftarget-atommean-fw1-fmlp256x256x256_e50-emax0.3-ew0-emlp256x256x256-order3`
- `upet_full_rtx5090_linear_f50-fmax0.5-ftarget-atommean-fw0-fmlp256x256x256_e50-emax0.3-ew1-emlp256x256x256-order8`

实际运行目录保持 UPET 当前布局：`outputs/runs/<派生名称>`。`ftarget-atommean` 必须保留，以免与历史逐分量实验混淆。

## 提交脚本

新增 `submit_energy0.sh` 和 `submit_force0_order1.sh` 至 `submit_force0_order8.sh`。九份脚本保留现有 Slurm 资源参数，统一激活 `upet_new` 环境并设置 W&B endpoint。Python 工作流按以下顺序执行：

1. `build_cache.py`
2. `train.py`
3. `evaluate.py`
4. `verify.py`

九份脚本之间仅配置文件路径不同。完整验证步骤不包含绘图，也不自动提交全数据 GPU 作业。

## 验证范围

- 配置模型测试覆盖单项系数为零及双零拒绝。
- 九份 YAML 均进行严格解析。
- 校验九份派生 run 名互不重复且符合预期。
- 校验九个提交脚本分别引用唯一且正确的配置。
- 执行 ConfidenceHead 相关单元测试和静态检查。
- 服务器端只生成和验证文件，不启动全数据 GPU 训练。
