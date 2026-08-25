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

## MAD r2SCAN E0 后处理

该实验只修正 UPET 基础模型已经推理得到的总能量，不重新运行训练，也不改变 UPET forward 或八个 energy ConfidenceHead 的输入、logits、分箱 representatives 和 expected error。后处理使用修正后的模型能量重新计算 observed energy error 和原 thresholds 下的 labels；不同 energy order 共用同一份能量修正，但各自保留原来的 expected error。

发布结果包含三个 test 视图：

- `uncorrected`：原始 UPET 总能量基线。
- `direct_mad_e0_test_informed`：直接用 test 的 `energy - atomization_energy` 恢复 MAD r2SCAN E0，再替换 checkpoint E0。该方法不使用 validation，但使用了 test 标签，因此必须解释为 test-informed/oracle 实验。
- `model_aware_reestimated_val_calibrated`：只在 validation 上对总能量残差做不加权 OLS/SVD，得到逐元素 E0 correction 后固定用于 held-out test；不使用 test 调参或重新拟合。

所有视图的 energy observed error 都是 `abs(E_model-E_r2scan)/N_atoms`，单位为 eV/atom；汇总 MAE、RMSE、mean signed error 和 P95 使用 meV/atom。force run、force prediction、force UQ 和历史力图完全不参与本流程，也不在派生结果中重新发布。绘图只生成连续密度图，不生成 boxplot。

正式输入固定为：

- checkpoint SHA-256：`879b1045391d88869522605a8b8b3cedeed74668e7062fdd7487548ab7b08004`。
- validation SHA-256：`4f4d4807592d75cfedda4a157850d60fc1428e44762e8debf37c012a4fc060aa`，16,098 个结构、310,432 个原子、89 种元素。
- test SHA-256：`499b479499eb56d0792360cb8bcb3397b566ac99c4e290ce0e866382c7f4d2ed`，16,072 个结构、311,657 个原子、89 种元素。

正式配置为 `configs/predict_r2scan_e0_gpu.yaml` 和 `configs/e0_postprocessing_gpu.yaml`。只在 bywang 服务器的 `/home/bywang/code/UQ/upet_new` checkout 上运行全量数据，不需要把 checkpoint、validation 或 test 数据拉到本地。可从仓库根目录显式执行一个阶段：

~~~bash
/home/bywang/.conda/envs/upet_new/bin/python \
  Uncertainty_Quantification/ConfidenceHead/scripts/postprocess_r2scan_e0.py \
  --config Uncertainty_Quantification/ConfidenceHead/configs/e0_postprocessing_gpu.yaml \
  --stage predict

/home/bywang/.conda/envs/upet_new/bin/python \
  Uncertainty_Quantification/ConfidenceHead/scripts/postprocess_r2scan_e0.py \
  --config Uncertainty_Quantification/ConfidenceHead/configs/e0_postprocessing_gpu.yaml \
  --stage postprocess

/home/bywang/.conda/envs/upet_new/bin/python \
  Uncertainty_Quantification/ConfidenceHead/scripts/postprocess_r2scan_e0.py \
  --config Uncertainty_Quantification/ConfidenceHead/configs/e0_postprocessing_gpu.yaml \
  --stage plot
~~~

也可以通过 Slurm 顺序执行完整流程：

~~~bash
cd /home/bywang/code/UQ/upet_new/Uncertainty_Quantification/ConfidenceHead/run
STAGE=all sbatch submit_e0_postprocessing.sh
~~~

每个阶段都会先验证配置、SHA、结构身份和已有 manifest。身份相同且产物完整时直接复用；目录已存在但身份冲突、源 artifact 被修改或发布不完整时立即停止，不覆盖旧结果。
