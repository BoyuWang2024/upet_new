# 三数据集 Prediction、UQ 与绘图

本工作流只使用九个已完成任务的 `best.pt`，不会重新训练、修改原 run manifest、覆盖现有 evaluation 或初始化 W&B。

## 数据复用边界

- `matpes_test` 直接复用九份现有 `evaluation`，不重复推理。
- `matpes_train` 从每个 run 声明的原训练 cache 只读复用 `train` split，不重新提取 UPET 特征。
- `mad_test` 校验 extxyz 的 SHA-256 后只构建一次公共基础 UPET cache，再供九个 ConfidenceHead 复用。

正式配置是 `configs/predict_external_gpu.yaml`，小数据 CPU 全链路配置是 `configs/predict_external_smoke.yaml`。两份配置之外不提交临时配置副本。

## 执行命令

从仓库根目录运行：

~~~bash
python Uncertainty_Quantification/ConfidenceHead/scripts/predict_external_datasets.py \
  --config Uncertainty_Quantification/ConfidenceHead/configs/predict_external_gpu.yaml

python Uncertainty_Quantification/ConfidenceHead/scripts/plot_prediction_datasets.py \
  --config Uncertainty_Quantification/ConfidenceHead/configs/predict_external_gpu.yaml
~~~

可重复传入 `--dataset` 只处理指定数据集，例如 `--dataset mad_test`。正式 Slurm 模板位于 `run/submit_external_prediction.sh`，通过 `STAGE` 选择 `mad-cache`、`mad-predict`、`matpes-train-predict` 或 `plot`。

## 产物与语义

新预测发布在 `<run>/predictions/<dataset>/`，包含 `predictions.pt`、`metrics.json`、50-bin 统计 CSV 与可完整校验的 `manifest.json`。数据集绘图发布在 `plots_root/<dataset>/`，跨数据集相关性发布在 `plots_root/comparisons/`。

目录采用身份绑定、原子发布和 no-clobber 语义：相同身份重复执行只复用，已存在但身份不同则停止。绘图脚本只读取已经完整发布的预测或 evaluation，不会触发 UPET/ConfidenceHead 推理。

能量 observed error 始终是逐原子 `abs(E_pred-E_ref)/N`，绘图阶段不会再次除以原子数；力 observed error 始终是每个原子的三个 Cartesian component 绝对误差均值。energy head 只读取 energy readout features，force head 只读取 force readout features；UQ 继续使用各 run 训练时固定的 50-bin thresholds 和 representatives。
