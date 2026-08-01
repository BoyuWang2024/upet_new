# UPET ConfidenceHead 默认逐原子平均力目标设计

## 1. 背景与目标

当前 UPET ConfidenceHead 的 Force 分支把每个原子的三个笛卡尔分量视为三个独立分类样本：误差和标签形状为 `[N_atoms, 3]`，logits 形状为 `[N_atoms, 3, num_bins]`。carnet_new 最新版本已经支持另一种 `atom_mean` 目标：先计算每个原子的三个分量绝对误差，再取算术平均，最后分箱并用每原子一个分类 Head 训练。

本次设计在 UPET 中引入相同的双模式机制，并将 `atom_mean` 设为默认。设计必须同时满足：

- 默认训练目标是逐原子三分量绝对误差的算术平均；
- 显式 `component` 继续支持原有逐分量训练；
- 旧 component checkpoint、run 和 evaluation 可以继续读取；
- 两种模式共享原始 UPET 特征缓存；
- 两种模式的训练、恢复、评估和产物身份严格隔离；
- UPET 的 Force 与 Energy 使用不同 readout features 的约束保持不变；
- 监督总损失、EMA、scheduler 和 early stopping 逻辑保持不变；
- 本次不增加绘图、不运行 GPU 全数据训练、不迁移或覆盖旧结果。

## 2. 已确认的决策

1. 新增 `model.force.target_mode: atom_mean | component`。
2. 代码默认值为 `atom_mean`；未填写该字段的 YAML 自动采用 `atom_mean`。
3. 三份现有 YAML 均显式填写 `target_mode: atom_mean`。
4. 要继续使用逐分量模式，用户必须显式填写 `target_mode: component`。
5. `atom_mean` 定义为三个分量绝对误差的算术平均，不是向量模长或 RMS。
6. 两种模式共用原始 cache；切换模式不重新运行 UPET 基础模型或提取 readout features。
7. `atom_mean` 使用新运行名和新身份链；component 保留旧运行名和旧身份兼容。
8. evaluation 同时记录原始分量数、实际分类目标数和目标模式。
9. scheduler、best checkpoint、EMA 和 early stopping 继续监督 `val/total_loss_ema`。
10. 固定线性分箱和现有默认上限保持不变：Force `0.5`，逐原子 Energy `0.3`。

## 3. 方案比较

### 3.1 配置驱动的双模式单流程（采用）

由一个配置字段统一驱动误差构造、Force Head 类型、损失形状、身份、评估和验证。缓存继续保存原始预测、参考值和 features，因此两种模式共享 cache。

优点是与 carnet_new 最新模式一致、改动集中、不复制训练流程，并能通过身份系统阻止产物混用。代价是训练、评估和 verify 必须显式支持两种 Force 张量形状。

### 3.2 独立 AtomMean 训练流程（不采用）

单独复制模型、训练、评估和校验入口。虽然表面隔离清楚，但会重复 early stopping、W&B、恢复训练和产物验证逻辑，长期容易产生行为差异。

### 3.3 评估时平均 component 概率（不采用）

保留逐分量训练，仅在评估阶段平均三个概率分布。该方案没有改变训练标签，不能实现“先平均误差再分箱”，不满足本次目标。

## 4. 配置契约

Force 分支使用专用配置模型：

```yaml
model:
  force:
    enabled: true
    target_mode: atom_mean
    hidden_dims: [256, 256, 256]
    dropout: 0.0
    num_bins: 50
```

字段定义：

```text
target_mode: atom_mean | component
默认值: atom_mean
```

以下配置必须显式加入 `target_mode: atom_mean`：

- `configs/n20_local_cpu.yaml`
- `configs/n20_cpu.yaml`
- `configs/full_gpu.yaml`

未知值由 Pydantic 在配置加载阶段拒绝。显式 component 配置是访问历史逐分量运行的唯一方式。

## 5. 缓存与特征边界

缓存格式和 cache identity 不加入 `target_mode`，继续保存：

```text
force_prediction: [N_atoms, 3]
force_reference:  [N_atoms, 3]
force_features:   [N_atoms, D_force]
energy_prediction: [N_structures]
energy_reference:  [N_structures]
energy_features:   [N_atoms, D_energy]
```

Force 和 Energy features 必须继续来自不同 UPET readout。模型仍检查二者的原子数一致，并拒绝二者引用同一底层张量。`atom_mean` 只改变 Force 目标和 Head，不改变 Energy adapter、Energy Head 或任一 readout。

缓存 manifest 中的 `force_component_count` 始终表示原始数据量 `3 × N_atoms`。两种模式应解析到同一个 cache ID。

## 6. 误差、分箱与模型数据流

### 6.1 Force 误差

统一的模式感知函数接受有限且形状一致的 `[N_atoms, 3]` 预测与参考力。

component：

```text
abs(force_prediction - force_reference)
→ errors [N_atoms, 3]
```

atom_mean：

\[
e_i = \frac{|\Delta F_{i,x}| + |\Delta F_{i,y}| + |\Delta F_{i,z}|}{3}
\]

```text
abs(force_prediction - force_reference) [N_atoms, 3]
→ mean(dim=-1)
→ errors [N_atoms]
```

语义定义分别为：

```text
component: abs_cartesian_component_v1
atom_mean: abs_cartesian_component_mean_v1
```

### 6.2 固定线性分箱

两种模式继续使用相同的数值阈值生成规则：

- `num_bins` 来自 `model.force.num_bins`；
- 上限来自 `binning.force_max_error`；
- 超过上限的误差进入末箱；
- Energy 分箱保持不变。

虽然阈值数值可以相同，atom_mean 的 binning payload 必须记录其目标模式和 error definition，使其 `binning_id` 与 component 不同。component payload 保留旧结构，以维持历史 `binning_id`。

### 6.3 Force Head

`ConfidenceModel` 显式接收 `force_target_mode`：

- `atom_mean` 创建普通 `ConfidenceHead`，输出 `[N_atoms, B]`；
- `component` 创建现有 `ComponentConfidenceHead`，输出 `[N_atoms, 3, B]`。

两者使用相同的 `force_input_dim`、hidden dimensions、dropout、bin 数和 Shifted Softplus。Energy 分支完全不变。

## 7. 损失、epoch 统计与 early stopping

`confidence_loss` 显式接收模式并验证形状，禁止仅根据张量维数静默推断语义。

```text
atom_mean:
  logits [N,B]
  labels [N]
  force_count=N

component:
  logits [N,3,B]
  labels [N,3]
  force_count=3N
```

Force 和 Energy 均先计算各自分类样本上的平均交叉熵。监督总损失保持：

\[
L_{total}=1.0L_{force}+1.5L_{energy}
\]

训练和 validation 的累计器按当前 `force_count` 加权，避免不同原子数 batch 的平均偏差。以下控制逻辑不变：

- 监控指标：`val/total_loss_ema`；
- plateau scheduler 使用该 EMA；
- best checkpoint 使用该 EMA；
- early stopping 使用该 EMA；
- n20 默认仍因 `max_epochs=1` 停止。

## 8. 身份、命名与旧 component 兼容

### 8.1 atom_mean 身份

atom_mean 运行名增加：

```text
-ftarget-atommean
```

示例：

```text
..._f50-fmax0.5-ftarget-atommean-fw1-fmlp256x256x256_...
```

atom_mean 的 resolved config、checkpoint、run manifest、prediction 和 evaluation manifest 显式记录：

```text
force_target_mode: atom_mean
force_error_definition: abs_cartesian_component_mean_v1
```

其 `config_id`、`binning_id`、`model_loss_id` 和 `run_id` 均不得与 component 相同。

### 8.2 component 兼容规范

显式 component 保持旧运行目录名，不增加 target tag。为了让历史上缺少 `target_mode` 字段的 component 产物继续可用，身份规范化遵循：

- component 的语义配置在计算兼容身份时省略 `target_mode`；
- component 的 binning payload 保持旧结构；
- component 的 Head 结构和 state-dict 参数名保持不变；
- 旧产物缺少模式字段时，仅在兼容读取路径中解释为 component；
- 显式 component 配置可以恢复或评估旧 checkpoint；
- atom_mean 绝不接受缺少模式字段的旧产物。

新 component prediction/evaluation 仍显式写出模式和目标计数。旧 evaluation 缺少 `force_targets` 时，验证器可按 component 语义推导为 `3 × N_atoms`；其他缺失或矛盾必须失败。

### 8.3 混用防护

下列情况必须在写入任何新产物前失败：

- atom_mean 配置加载 component checkpoint；
- component 配置加载 atom_mean checkpoint；
- resume checkpoint 与当前模式不一致；
- binning error definition 与当前模式不一致；
- prediction、manifest 和 resolved config 的模式不一致；
- logits、labels、errors 或 expected errors 的形状与模式不一致。

cache schema 和 cache identity 保持不变。现有 run、checkpoint 和 evaluation schema 采用向后兼容字段扩展，不要求复制或转换旧 component 文件。

## 9. 评估与产物契约

`test_predictions.pt` 根据模式保存：

```text
atom_mean:
  force_logits          [N_atoms, B]
  force_labels          [N_atoms]
  force_observed_errors [N_atoms]
  force_expected_errors [N_atoms]

component:
  force_logits          [N_atoms, 3, B]
  force_labels          [N_atoms, 3]
  force_observed_errors [N_atoms, 3]
  force_expected_errors [N_atoms, 3]
```

prediction 顶层显式记录 `force_target_mode` 和 `force_error_definition`。分类指标和 `force_bin_summary.csv` 直接使用当前目标粒度：atom_mean 每原子一条分类观察，component 每分量一条分类观察；atom_mean 不再平均三份预测概率。

evaluation manifest 同时记录：

```text
structures: N_structures
atoms: N_atoms
force_components: 3 * N_atoms
force_targets:
  atom_mean: N_atoms
  component: 3 * N_atoms
force_target_mode: atom_mean | component
```

verify 必须分别校验缓存原始计数和当前目标计数，不能把 `force_components` 重定义为分类样本数。

## 10. 错误处理

- 未知目标模式：配置加载失败。
- Force 输入不是相同的有限 `[N,3]`：误差构造失败。
- 模式与 Head、logits 或 labels 形状不匹配：损失计算失败。
- 模式与 checkpoint、binning 或 run identity 不匹配：恢复/评估前失败。
- atom_mean 产物缺少显式模式：验证失败。
- `force_targets` 与模式推导值不一致：验证失败。
- cache 的原始 `force_component_count` 不是 `3 × atom_count`：缓存验证失败。
- Force 和 Energy readout features 相同：模型执行失败。

所有失败必须发生在原子写入或锁保护边界内，不留下伪完整 manifest。

## 11. 测试与验收

### 11.1 单元和契约测试

必须覆盖：

1. 配置默认值及三份 YAML 的显式 atom_mean。
2. 未知模式拒绝。
3. 三分量绝对误差算术平均的手算结果。
4. 两种 Head 的输出形状。
5. 两种交叉熵及 `force_count` 的手算结果。
6. 两种模式共享 cache ID。
7. 两种模式的 binning/model-loss/run identity 隔离。
8. atom_mean 新运行名和 component 旧运行名。
9. 历史 resolved config、checkpoint、manifest 缺少模式字段时的 component 兼容。
10. atom_mean 拒绝历史缺省模式产物。
11. 跨模式恢复和评估拒绝。
12. prediction、CSV、metrics 和 evaluation manifest 的模式、形状及计数。
13. verify 接受合法旧 component 产物并拒绝所有矛盾组合。
14. Force 与 Energy 独立 readout features 的回归保护。
15. `val/total_loss_ema` 仍是 scheduler、best 和 early stopping 的唯一监督指标。

已有默认依赖 component 的测试必须显式设置 `target_mode: component`，不得通过改变断言掩盖默认行为变化。

### 11.2 本地验证

在 `upet_new` 环境执行：

- ConfidenceHead 全测试；
- Ruff format/check；
- mypy；
- 三份 YAML 静态解析。

本地不运行大模型、GPU 全数据训练或绘图。

### 11.3 远端 CPU n20 验收

在最终 GitHub 提交上执行：

1. 确认远端 HEAD 与 GitHub SHA 一致。
2. 构建一次或复用现有原始 n20 cache。
3. 用默认 atom_mean 配置依次运行 build_cache、train、evaluate、verify。
4. 确认运行名包含 `ftarget-atommean`。
5. 对 143 个原子的 n20 测试数据，确认 Force prediction 为 `[143, 50]`。
6. 确认 `force_components=429`、`force_targets=143`。
7. 确认 JSONL、best/last checkpoint、manifest 和 W&B offline run 完整。
8. 使用临时显式 component 配置读取现有旧逐分量 run，执行 evaluate/verify，不修改旧 checkpoint。
9. 确认两种模式解析同一 cache，不重新提取基础模型特征。
10. 仅静态解析 full GPU 配置，不启动 GPU 训练。

## 12. 非目标

本次明确不包含：

- 向量模长、RMS 或最大分量等其他 Force reduction；
- 同时训练 atom_mean 和 component 两个 Force Head；
- 修改 Energy 误差、cumulant order 或损失权重；
- 修改 early stopping、scheduler 或 W&B 监督指标；
- 绘图及绘图兼容层；
- 迁移、复制、覆盖或重新计算旧结果；
- GPU 全数据训练。
