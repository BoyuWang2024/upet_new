# ConfidenceHead 配置化工作流

本目录为 uPET 置信度头提供缓存、训练、评估和完整验证的配置化入口。所有脚本都是薄分派层：它们只读取 YAML 配置并调用工作流接口，不在脚本中重新实现数据发现、特征缓存、训练或验证逻辑。

## 三份配置与数据路径

从仓库根目录执行命令。配置中的相对路径会相对于仓库根目录解析；先把本地数据放到与配置一致的位置，或把 checkpoint.path、data.train.path、data.validation.path 和 data.test.path 改为可访问的绝对路径。

- configs/n20_local_cpu.yaml：本机 CPU 冒烟配置，使用 data/checkpoint/pet-omatpes-l-v0.1.0.ckpt 和 data/dataset/matpes_n20.extxyz。
- configs/n20_cpu.yaml：共享文件系统上的 CPU 冒烟配置，路径指向预设的 /HOME、/XYFS01 位置。
- configs/full_gpu.yaml：生产 GPU 配置，使用彼此独立的 train/validation/test 数据集；本次工作不运行完整 GPU 工作流。

每个 checkpoint 和数据 split 都有 expected_sha256。构建缓存阶段会校验文件内容与该 SHA-256 是否一致；不要在未确认数据来源时随意改写哈希。n20 配置刻意让三个 split 指向同一份小数据，并设置 profile: smoke、allow_identical_splits: true，只用于冒烟测试，不能作为生产性能或泛化结论。

## 关键配置字段

- checkpoint 与 data：输入文件路径及其 SHA-256；生产配置要求三个 split 的哈希互不相同。
- readouts：模型读出与最后一层特征字段名称。
- cache：特征缓存的批大小、工作线程与分片原子数上限。
- binning、model、loss：标签分箱、力/能量头结构与损失权重。
- optimizer、scheduler、trainer：优化、学习率调度、epoch、早停和恢复设置。
- run：输出根目录、运行名前缀、随机种子、设备；当前实现不支持 AMP，必须保持 amp: false。
- logging：JSONL 记录和 Weights & Biases 模式。

## 四阶段命令

四个阶段使用同一份配置，必须按顺序执行：

~~~
python Uncertainty_Quantification/ConfidenceHead/scripts/build_cache.py --config Uncertainty_Quantification/ConfidenceHead/configs/n20_local_cpu.yaml
python Uncertainty_Quantification/ConfidenceHead/scripts/train.py --config Uncertainty_Quantification/ConfidenceHead/configs/n20_local_cpu.yaml
python Uncertainty_Quantification/ConfidenceHead/scripts/evaluate.py --config Uncertainty_Quantification/ConfidenceHead/configs/n20_local_cpu.yaml
python Uncertainty_Quantification/ConfidenceHead/scripts/verify.py --config Uncertainty_Quantification/ConfidenceHead/configs/n20_local_cpu.yaml
~~~

build_cache.py 输出匹配配置身份的缓存清单路径；train.py 输出派生的 run 目录；evaluate.py 输出评估目录；verify.py 输出完整性校验结果。输出位于 run.output_root 下：缓存位于 cache/，训练 run、检查点、manifest、日志与评估产物位于由配置派生的运行目录中。训练、评估和验证会选择与输入身份严格匹配的唯一完整缓存。

## W&B、恢复与运行范围

默认的本地 n20 配置使用 logging.wandb_mode: offline，将记录保存在本地，适合无网络环境。确认项目名与账号权限后，将其改为 online 才会上报到 W&B；full_gpu.yaml 已设置为 online。

恢复训练时，设置 trainer.resume_from 为派生运行目录内的检查点，例如：

~~~
trainer:
  resume_from: Uncertainty_Quantification/ConfidenceHead/outputs/runs/<run-name>/checkpoints/last.pt
~~~

恢复训练通常使用 `last.pt`；`evaluate.py` 默认评估 `best.pt`。

恢复路径必须位于该配置派生的 run 目录内，否则训练会拒绝执行，以免串用其他实验的检查点。

本地只执行静态与单元验证；n20 CPU 全链路在远端执行。当前流程不执行 full GPU 训练或绘图步骤；后续分析使用 JSONL、评估输出或 W&B 记录。

## 力与能量的监督目标

三套发布配置都明确使用逐原子力均值模式：

~~~yaml
model:
  force:
    target_mode: atom_mean
~~~

`atom_mean` 对每个原子的三个笛卡尔分量误差先取绝对值，再取算术平均：

~~~text
force_error_i = mean(abs(force_prediction_i - force_reference_i), dim=-1)
~~~

它不是力矢量范数、RMS、最大分量或概率平均。对于 `N` 个原子和 `B` 个力分箱，力 logits、labels 和 observed errors 的形状分别是 `[N, B]`、`[N]` 和 `[N]`。评估清单中的 `force_components=3N` 始终表示原始缓存的笛卡尔分量数，`force_targets=N` 表示实际参与监督和指标计算的逐原子目标数。

能量误差保持逐原子定义：

~~~text
energy_error_s = abs(energy_prediction_s - energy_reference_s) / num_atoms_s
~~~

UPET 的两条读出保持独立：力头只接收 `force_features`，能量头只接收 `energy_features`，不会互换或共享输入张量。

如需复现实验兼容的逐分量目标，可显式设置 `target_mode: component`。此时力 logits/labels/errors 的形状是 `[N, 3, B]`、`[N, 3]`、`[N, 3]`，`force_targets=3N`。缺少力目标元数据的历史 checkpoint、预测文件或评估清单只会按 `component` 解释；它们不能作为 `atom_mean` 产物使用。`atom_mean` 运行名带有 `-ftarget-atommean`，显式 `component` 保留历史运行名格式。

固定线性分箱仍通过 YAML 调整，当前发布配置默认力上限为 `0.5`、逐原子能量上限为 `0.3`，分箱数为 `50`。监督总损失默认保持：

~~~text
total_loss = 1.0 * force_loss + 1.5 * energy_loss
~~~

学习率调度、最佳 checkpoint、EMA 与 early stopping 都只监控 `val/total_loss_ema`。改变力目标模式不会改变这套 early-stopping 逻辑。
