# UPET ConfidenceHead Recompute Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `Uncertainty_Quantification/ConfidenceHead/` 中实现自包含的 UPET ConfidenceHead，使用独立 energy/force readout、逐分量 force 误差、逐原子 energy 误差和 total-loss EMA early stopping，并在远程登录节点完成真实 n20 CPU 全链路验证。

**Architecture:** 先将冻结 UPET 的预测和两个独立 readout 特征写入带身份的分片 raw cache，再仅基于 cache 训练 component-wise force head 与 cumulant energy head。科学计算核心、UPET 集成、产物协议和 workflow 分层；所有动态测试只在远端 `upet_new` 环境执行，本地只运行静态检查。

**Tech Stack:** Python 3.11、PyTorch 2.11.0+cu128、UPET、metatrain/metatomic、ASE、Pydantic 2、PyYAML、SciPy、pytest、tox、Ruff、mypy。

---

## 0. 实施约束

- 设计规格：`docs/superpowers/specs/2026-07-29-upet-confidence-head-recompute-design.md`
- 本地仓库：`/home/lilong/code/UQ/upet_new`
- 本地分支：`ConfidenceHead`
- 远端连接：`ssh -p 55801 bywang@121.48.164.204`
- 远端仓库：`/home/bywang/code/UQ/upet_new`
- 本地只运行静态检查；不得在本地运行 pytest、加载 checkpoint、构建 cache、训练或评估。
- 每个动态红/绿测试都在远端执行。
- 当前阶段只运行 CPU n20；不得提交或运行 GPU/full-data 作业。
- 不添加 plotting 模块，不生成图片。
- 不复制旧 `UQ_orb_post_train_force` 的任何代码或结果。

## 1. 最终文件映射

### 包与配置

- Create: `Uncertainty_Quantification/ConfidenceHead/__init__.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/README.md`
- Create: `Uncertainty_Quantification/ConfidenceHead/requirements-remote.txt`
- Create: `Uncertainty_Quantification/ConfidenceHead/configs/n20_cpu.yaml`
- Create: `Uncertainty_Quantification/ConfidenceHead/configs/full_linear.yaml`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/__init__.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/config.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/identity.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/data.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/checkpoint.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/features.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/cache.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/errors.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/binning.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/adapters.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/heads.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/model.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/losses.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/metrics.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/trainer.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/artifacts.py`

### Workflows 与入口

- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/__init__.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/build_cache.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/train.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/evaluate.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/verify.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/build_cache.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/train.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/evaluate.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/verify.py`

### Tests 与仓库集成

- Create: `Uncertainty_Quantification/ConfidenceHead/tests/conftest.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_errors_binning.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_model_math.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_config_identity_artifacts.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_data_checkpoint_features.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_cache.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_trainer.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_n20.py`
- Modify: `pyproject.toml`
- Modify: `tox.ini`

## Task 1: 仓库骨架、静态范围和远程环境

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/__init__.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/__init__.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/requirements-remote.txt`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/conftest.py`
- Modify: `pyproject.toml`
- Modify: `tox.ini`

- [ ] **Step 1: 创建最小包骨架**

`Uncertainty_Quantification/ConfidenceHead/__init__.py`：

```python
"""UPET post-training confidence heads."""
```

`Uncertainty_Quantification/ConfidenceHead/confidence_head/__init__.py`：

```python
"""Self-contained confidence-head implementation for frozen UPET models."""
```

`Uncertainty_Quantification/ConfidenceHead/tests/conftest.py`：

```python
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    output = tmp_path / "outputs"
    output.mkdir()
    return output
```

- [ ] **Step 2: 声明远端测试依赖**

`requirements-remote.txt`：

```text
pydantic>=2,<3
pyyaml
scipy
pytest
tox
ruff
mypy
sphinx-lint
```

- [ ] **Step 3: 将新目录加入静态检查和 pytest marker**

在 `pyproject.toml` 的 pytest markers 中增加：

```toml
"confidence_head_n20: runs the real UPET checkpoint on the 20-structure smoke dataset",
```

在 `tox.ini` 的 `lint_folders` 中增加：

```ini
"{toxinidir}/Uncertainty_Quantification/ConfidenceHead/"
```

增加：

```ini
[testenv:confidence-head-tests]
description = Run UPET ConfidenceHead tests
deps =
    pytest
    pydantic>=2
    pyyaml
    scipy
changedir = {toxinidir}
commands =
    python -m pytest Uncertainty_Quantification/ConfidenceHead/tests {posargs}
```

- [ ] **Step 4: 本地运行静态检查**

Run:

```bash
tox -e lint
git diff --check
```

Expected: exit 0；不得运行 pytest。

- [ ] **Step 5: 提交骨架**

```bash
git add pyproject.toml tox.ini \
  Uncertainty_Quantification/ConfidenceHead/__init__.py \
  Uncertainty_Quantification/ConfidenceHead/confidence_head/__init__.py \
  Uncertainty_Quantification/ConfidenceHead/requirements-remote.txt \
  Uncertainty_Quantification/ConfidenceHead/tests/conftest.py
git commit -m "chore: scaffold UPET confidence head"
git push origin ConfidenceHead
```

- [ ] **Step 6: 在远端克隆仓库**

从本地 WSL 执行：

```bash
ssh -p 55801 bywang@121.48.164.204 \
  "test ! -e /home/bywang/code/UQ/upet_new && \
   git clone --branch ConfidenceHead \
   https://github.com/BoyuWang2024/upet_new.git \
   /home/bywang/code/UQ/upet_new"
```

Expected: clone 成功，远端 HEAD 为当前 `ConfidenceHead`。

- [ ] **Step 7: 创建精确的远端 conda 环境**

远端执行：

```bash
CONDA=/home/shared/spack/opt/spack/linux-icelake/miniforge3-25.3.0-3-7criefbpaxjacshuyjahvrpo6ppkvsfr/bin/conda
"$CONDA" create -y -n upet_new \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
  python=3.11 pip
```

然后执行：

```bash
PY=/home/bywang/.conda/envs/upet_new/bin/python
"$PY" -m pip install \
  --index-url https://mirrors.aliyun.com/pytorch-wheels/cu128 \
  --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
  "torch==2.11.0+cu128"
"$PY" -m pip install \
  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
  -e /home/bywang/code/UQ/upet_new
"$PY" -m pip install \
  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
  -r /home/bywang/code/UQ/upet_new/Uncertainty_Quantification/ConfidenceHead/requirements-remote.txt
```

阿里云镜像已列出 Python 3.11 x86_64 wheel：
`torch-2.11.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl`。远端 glibc 2.34
满足 manylinux 2.28。

- [ ] **Step 8: 验证远端版本，不接受替代版本**

```bash
/home/bywang/.conda/envs/upet_new/bin/python - <<'PY'
import torch
assert torch.__version__ == "2.11.0+cu128", torch.__version__
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

Expected:

```text
2.11.0+cu128
False
```

登录节点没有 GPU 是预期状态。

## Task 2: Force/Energy 误差与固定线性分箱

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/errors.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/binning.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_errors_binning.py`

- [ ] **Step 1: 先写失败测试**

```python
import pytest
import torch

from Uncertainty_Quantification.ConfidenceHead.confidence_head.binning import (
    expected_error,
    fixed_linear_binning,
    labels_from_thresholds,
)
from Uncertainty_Quantification.ConfidenceHead.confidence_head.errors import (
    energy_per_atom_error,
    force_component_error,
)


def test_force_error_is_componentwise() -> None:
    pred = torch.tensor([[1.0, -2.0, 3.0]])
    ref = torch.tensor([[0.0, 1.0, 1.0]])
    torch.testing.assert_close(
        force_component_error(pred, ref),
        torch.tensor([[1.0, 3.0, 2.0]]),
    )


def test_energy_error_is_per_structure_atom() -> None:
    pred = torch.tensor([10.0, -5.0])
    ref = torch.tensor([6.0, -1.0])
    counts = torch.tensor([2, 4])
    torch.testing.assert_close(
        energy_per_atom_error(pred, ref, counts),
        torch.tensor([2.0, 1.0]),
    )


def test_linear_bins_use_upper_boundary_and_saturate() -> None:
    spec = fixed_linear_binning(num_bins=5, max_error=0.5)
    values = torch.tensor([0.0, 0.099, 0.1, 0.5, 0.9])
    assert labels_from_thresholds(values, spec.thresholds).tolist() == [0, 0, 1, 4, 4]
    torch.testing.assert_close(
        spec.representatives,
        torch.tensor([0.05, 0.15, 0.25, 0.35, 0.45]),
    )


def test_expected_error_uses_all_probabilities() -> None:
    logits = torch.log(torch.tensor([[0.25, 0.75]]))
    reps = torch.tensor([0.1, 0.3])
    torch.testing.assert_close(expected_error(logits, reps), torch.tensor([0.25]))


def test_energy_rejects_zero_atom_count() -> None:
    with pytest.raises(ValueError, match="positive"):
        energy_per_atom_error(
            torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([0])
        )
```

- [ ] **Step 2: 本地静态检查测试文件**

```bash
tox -e lint
git diff --check
git add Uncertainty_Quantification/ConfidenceHead/tests/test_errors_binning.py
git commit -m "test: specify confidence errors and bins"
git push origin ConfidenceHead
```

- [ ] **Step 3: 远端验证红灯**

```bash
cd /home/bywang/code/UQ/upet_new
git pull --ff-only origin ConfidenceHead
/home/bywang/.conda/envs/upet_new/bin/python -m pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_errors_binning.py -q
```

Expected: collection error，`errors` 或 `binning` module 不存在。

- [ ] **Step 4: 实现误差函数**

`errors.py`：

```python
from __future__ import annotations

import torch
from torch import Tensor


def force_component_error(prediction: Tensor, reference: Tensor) -> Tensor:
    if prediction.shape != reference.shape or prediction.ndim != 2:
        raise ValueError("force prediction/reference must share shape [N,3]")
    if prediction.shape[1] != 3:
        raise ValueError("force prediction/reference require three components")
    if not torch.isfinite(prediction).all() or not torch.isfinite(reference).all():
        raise ValueError("force prediction/reference contains non-finite values")
    return torch.abs(prediction - reference)


def energy_per_atom_error(
    prediction: Tensor,
    reference: Tensor,
    atom_counts: Tensor,
) -> Tensor:
    prediction = prediction.reshape(-1)
    reference = reference.reshape(-1)
    atom_counts = atom_counts.reshape(-1)
    if prediction.shape != reference.shape or prediction.shape != atom_counts.shape:
        raise ValueError("energy prediction/reference/counts must share shape [S]")
    if torch.any(atom_counts <= 0):
        raise ValueError("atom counts must be positive")
    if not torch.isfinite(prediction).all() or not torch.isfinite(reference).all():
        raise ValueError("energy prediction/reference contains non-finite values")
    return torch.abs(prediction - reference) / atom_counts.to(prediction.dtype)
```

- [ ] **Step 5: 实现固定线性分箱**

`binning.py`：

```python
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class BinningSpec:
    algorithm: str
    num_bins: int
    max_error: float
    thresholds: Tensor
    representatives: Tensor


def fixed_linear_binning(num_bins: int, max_error: float) -> BinningSpec:
    if num_bins < 3 or max_error <= 0:
        raise ValueError("linear binning requires num_bins >= 3 and max_error > 0")
    width = max_error / num_bins
    thresholds = torch.arange(1, num_bins, dtype=torch.float32) * width
    representatives = (torch.arange(num_bins, dtype=torch.float32) + 0.5) * width
    return BinningSpec(
        algorithm="fixed_linear_v1",
        num_bins=num_bins,
        max_error=max_error,
        thresholds=thresholds,
        representatives=representatives,
    )


def labels_from_thresholds(values: Tensor, thresholds: Tensor) -> Tensor:
    if not torch.isfinite(values).all():
        raise ValueError("cannot bin non-finite errors")
    if torch.any(values < 0):
        raise ValueError("errors must be non-negative")
    return torch.bucketize(values, thresholds.to(values), right=True).to(torch.int64)


def expected_error(logits: Tensor, representatives: Tensor) -> Tensor:
    if logits.shape[-1] != representatives.numel():
        raise ValueError("logit and representative dimensions differ")
    return torch.sum(
        torch.softmax(logits, dim=-1) * representatives.to(logits),
        dim=-1,
    )
```

- [ ] **Step 6: 本地静态检查并提交**

```bash
tox -e lint
git diff --check
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/errors.py \
  Uncertainty_Quantification/ConfidenceHead/confidence_head/binning.py
git commit -m "feat: add component errors and linear bins"
git push origin ConfidenceHead
```

- [ ] **Step 7: 远端验证绿灯**

Run the same remote pytest command.

Expected: `5 passed`。

## Task 3: 独立 readout adapters、heads、model 与 loss

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/adapters.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/heads.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/model.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/losses.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_model_math.py`

- [ ] **Step 1: 写独立 readout 与数学测试**

测试必须包含：

```python
def test_model_never_reuses_force_features_for_energy() -> None:
    model = ConfidenceModel(
        force_input_dim=2,
        energy_input_dim=2,
        hidden_dims=(4,),
        num_bins=5,
        cumulant_order=1,
        signed_root=True,
        dropout=0.0,
    )
    force_features = torch.zeros((3, 2))
    energy_features = torch.full((3, 2), 7.0)
    offsets = torch.tensor([0, 1, 3])

    captured: list[torch.Tensor] = []
    model.energy_head.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    )
    output = model(force_features, energy_features, offsets)

    assert output.force_logits.shape == (3, 3, 5)
    assert output.energy_logits.shape == (2, 5)
    torch.testing.assert_close(captured[0], torch.full((2, 2), 7.0))
```

同时测试：

- `LocalToGlobalCumulantAdapter` order 1–4 与手算标量 cumulants；
- order 范围只能为 1–8；
- `ComponentConfidenceHead` 三个子 head 参数对象互不相同；
- force CE 先展平 `[N,3,B]`；
- total loss 等于 `1.0 * force_mean + 1.5 * energy_mean`。

- [ ] **Step 2: 本地静态检查、提交测试、远端红灯**

```bash
tox -e lint
git add Uncertainty_Quantification/ConfidenceHead/tests/test_model_math.py
git commit -m "test: specify separate readout heads"
git push origin ConfidenceHead
```

远端运行：

```bash
python -m pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_model_math.py -q
```

Expected: import failure。

- [ ] **Step 3: 实现 cumulant adapter**

实现 `LocalToGlobalCumulantAdapter(input_dim, order, signed_root)`：

```python
raw[k] = mean(values ** k, dim=0)
kappa[n] = raw[n] - sum(
    comb(n - 1, m - 1) * kappa[m] * raw[n - m]
    for m in range(1, n)
)
```

一阶保持均值；二阶以上在 `signed_root=True` 时应用：

```python
sign(kappa) * abs(kappa) ** (1 / degree)
```

输入必须是 `[N_atom,D_energy]` 和严格递增、覆盖全部原子的 offsets。

- [ ] **Step 4: 实现 heads**

`heads.py` 定义：

```python
class ShiftedSoftplus(nn.Module):
    def forward(self, values: Tensor) -> Tensor:
        return F.softplus(values) - math.log(2.0)


class ConfidenceHead(nn.Module):
    # Linear -> ShiftedSoftplus -> optional Dropout, final Linear


class ComponentConfidenceHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
        num_bins: int,
    ) -> None:
        super().__init__()
        self.components = nn.ModuleList(
            [
                ConfidenceHead(input_dim, hidden_dims, dropout, num_bins)
                for _ in range(3)
            ]
        )

    def forward(self, features: Tensor) -> Tensor:
        return torch.stack([head(features) for head in self.components], dim=1)
```

- [ ] **Step 5: 实现严格分离的 model**

`model.py`：

```python
@dataclass(frozen=True)
class ConfidenceOutput:
    force_logits: Tensor
    energy_logits: Tensor


class ConfidenceModel(nn.Module):
    def forward(
        self,
        force_features: Tensor,
        energy_features: Tensor,
        offsets: Tensor,
    ) -> ConfidenceOutput:
        if force_features.data_ptr() == energy_features.data_ptr():
            raise ValueError("force and energy readout features must be distinct tensors")
        return ConfidenceOutput(
            force_logits=self.force_head(force_features),
            energy_logits=self.energy_head(
                self.energy_adapter(energy_features, offsets)
            ),
        )
```

force 直接进入 component head；energy 只通过 energy cumulant adapter。

- [ ] **Step 6: 实现 branch-wise loss**

`losses.py`：

```python
@dataclass(frozen=True)
class LossOutput:
    total: Tensor
    force: Tensor
    energy: Tensor
    force_count: int
    energy_count: int


def confidence_loss(
    force_logits: Tensor,
    force_labels: Tensor,
    energy_logits: Tensor,
    energy_labels: Tensor,
    force_coefficient: float = 1.0,
    energy_coefficient: float = 1.5,
) -> LossOutput:
    force = F.cross_entropy(
        force_logits.reshape(-1, force_logits.shape[-1]),
        force_labels.reshape(-1),
    )
    energy = F.cross_entropy(energy_logits, energy_labels.reshape(-1))
    return LossOutput(
        total=force_coefficient * force + energy_coefficient * energy,
        force=force,
        energy=energy,
        force_count=force_labels.numel(),
        energy_count=energy_labels.numel(),
    )
```

- [ ] **Step 7: 静态检查、提交、远端绿灯**

```bash
tox -e lint
git diff --check
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/{adapters,heads,model,losses}.py
git commit -m "feat: add separate force and energy heads"
git push origin ConfidenceHead
```

Expected remote result: all `test_model_math.py` tests pass。

## Task 4: 严格配置、identity 与原子性产物

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/config.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/identity.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/artifacts.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_config_identity_artifacts.py`

- [ ] **Step 1: 写失败测试**

测试以下行为：

The exact test names are:

- `test_production_rejects_identical_splits`
- `test_smoke_allows_identical_splits_only_when_explicit`
- `test_unknown_config_key_is_rejected`
- `test_paths_resolve_from_repo_root_not_cwd`
- `test_stable_id_ignores_mapping_order`
- `test_atomic_json_never_exposes_partial_file`
- `test_incomplete_manifest_is_rejected`

构造 production 配置时使用三个不同的临时文件；smoke 配置显式复用一个文件。

- [ ] **Step 2: 本地静态、提交测试、远端红灯**

Expected: missing modules。

- [ ] **Step 3: 实现 Pydantic 配置**

所有 config model 使用：

```python
class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
```

定义：

```python
Profile = Literal["smoke", "production"]

class ReadoutConfig(StrictModel):
    energy_prediction: str = "energy"
    force_prediction: str = "non_conservative_forces"
    energy_features: str = "mtt::aux::energy_last_layer_features"
    force_features: str = "mtt::aux::non_conservative_forces_last_layer_features"


class BinningConfig(StrictModel):
    algorithm: Literal["fixed_linear_v1"] = "fixed_linear_v1"
    force_num_bins: int = Field(default=50, gt=1)
    force_max_error: float = Field(default=0.5, gt=0.0)
    energy_num_bins: int = Field(default=50, gt=1)
    energy_max_error: float = Field(default=0.3, gt=0.0)


class LossConfig(StrictModel):
    force_coefficient: float = Field(default=1.0, gt=0.0)
    energy_coefficient: float = Field(default=1.5, gt=0.0)
    label_smoothing: float = 0.0
    class_weights: None = None


class TrainerConfig(StrictModel):
    max_epochs: int = Field(default=200, gt=0)
    ema_beta: float = Field(default=0.95, ge=0.0, lt=1.0)
    early_stopping_patience: int = Field(default=15, gt=0)
    min_delta: float = Field(default=1e-4, ge=0.0)
    min_epochs: int = Field(default=3, gt=0)
```

根配置 validator 强制：

- production 不允许相同 split SHA；
- `label_smoothing == 0`；
- `class_weights is None`；
- scheduler monitor 与 early-stopping monitor 都是
  `val/total_loss_ema`；
- device=cpu 时 amp=false。

- [ ] **Step 4: 实现稳定 identity**

`identity.py`：

```python
def canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def stable_id(namespace: str, payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json(payload)).hexdigest()
    return f"{namespace}-{digest[:16]}"
```

分别生成 config/cache/binning/model-loss/run ID。

- [ ] **Step 5: 实现原子性 artifacts**

`artifacts.py` 提供：

```python
sha256_file(path: Path) -> str
atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None
atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None
require_complete_manifest(path: Path, expected_identity: str) -> dict[str, Any]
```

临时文件必须位于目标文件同一目录，成功验证后 `os.replace`。

- [ ] **Step 6: 静态检查、提交、远端绿灯**

Commit:

```text
feat: add strict confidence artifact identities
```

Expected: config/identity/artifact tests pass。

## Task 5: Extxyz、checkpoint 与两个 UPET readout

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/data.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/checkpoint.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/features.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_data_checkpoint_features.py`

- [ ] **Step 1: 写纯单元失败测试**

使用 fake model/fake TensorMap，不加载真实 checkpoint。测试：

- energy label 从 `info` 或 calculator result 读取且必须为标量；
- force label shape 必须为 `[N,3]`；
- dataset identity 统计 structure、atom、force component；
- checkpoint 在反序列化前检查 SHA；
- feature extractor 同时请求四个配置 key；
- energy/force feature 必须是两个不同 tensor；
- 缺失任一 readout 时错误消息包含具体 key；
- prediction/features 的 structure/atom count 不一致时报 structure ID。

- [ ] **Step 2: 本地静态、提交测试、远端红灯**

Expected: missing data/checkpoint/features modules。

- [ ] **Step 3: 实现 deterministic extxyz loader**

复用当前仓库 LLPR 的公开数据规则，但定义新的 `ConfidenceSample`：

```python
ENERGY_LABEL_CANDIDATES = (
    "energy",
    "free_energy",
    "dft_energy",
    "REF_energy",
)
FORCE_LABEL_CANDIDATES = (
    "forces",
    "force",
    "dft_forces",
    "REF_forces",
)


@dataclass(frozen=True)
class ConfidenceSample:
    index: int
    atoms: Atoms
    energy_reference_total: float
    force_reference: np.ndarray
```

`iter_samples(path)` 使用 `ase.io.iread(str(path), index=":", format="extxyz")`，不 shuffle。

- [ ] **Step 4: 实现 checkpoint loader**

```python
def load_upet_checkpoint(
    path: Path,
    expected_sha256: str,
    device: torch.device,
    dtype: torch.dtype,
) -> LoadedCheckpoint:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"checkpoint SHA mismatch: {actual} != {expected_sha256}")
    from metatrain.utils.io import load_model
    model = load_model(str(path)).eval().to(device=device, dtype=dtype)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return LoadedCheckpoint(model=model, sha256=actual)
```

不得导入旧训练代码。

- [ ] **Step 5: 实现 UPET readout extractor**

请求：

```python
keys = (
    config.energy_prediction,
    config.force_prediction,
    config.energy_features,
    config.force_features,
)
supported_outputs = model.supported_outputs()
missing = [key for key in keys if key not in supported_outputs]
if missing:
    raise ValueError(f"checkpoint is missing required outputs: {missing}")
requested = {key: supported_outputs[key] for key in keys}
```

抽取并验证：

```text
raw energy_prediction:      [N,1] atom contributions
internal energy_prediction: [S] total (sum by batch-local system)
raw force_prediction:       [N,3,1] with xyz component and one property
internal force_prediction:  [N,3] (squeeze property axis)
energy_features:            [N,D_energy]
force_features:             [N,D_force]
```

如果联合请求不被 UPET 接受，执行两个明确请求：

```text
energy prediction + energy features
force prediction + force features
```

两个请求必须通过相同 structure IDs、atom counts 和 offsets；禁止 feature 互换。

- [ ] **Step 6: 静态检查、提交、远端单元测试绿灯**

Commit:

```text
feat: extract separate UPET readout features
```

## Task 6: 分片 raw cache 与流式读取

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/cache.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/build_cache.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_cache.py`

- [ ] **Step 1: 写失败测试**

用 3 个 synthetic structures 写两个 shard，验证：

- manifest count 与 shard count 相等；
- offsets 首值为 0、末值覆盖全部原子且严格递增；
- 两套 feature 字段同时存在且不共享 storage；
- duplicate structure ID 被拒绝；
- NaN 错误包含 split/shard/structure ID；
- `CachedSplitDataset` 只加载需要的 shard；
- incomplete cache 不可读取；
- 相同输入重复构建得到相同 cache ID。

- [ ] **Step 2: 本地静态、提交测试、远端红灯**

- [ ] **Step 3: 实现 cache schema**

每个 shard 使用 `torch.save` 保存：

```python
{
    "schema_version": "upet_confidence_raw_cache_v1",
    "split": split,
    "structure_ids": int64[S],
    "num_atoms": int64[S],
    "atom_offsets": int64[S + 1],
    "atomic_numbers": int64[N],
    "force_prediction": float32[N, 3],
    "force_reference": float32[N, 3],
    "energy_prediction": float32[S],
    "energy_reference": float32[S],
    "force_features": float32[N, D_force],
    "energy_features": float32[N, D_energy],
}
```

manifest 保存每个 shard 的 relative path、SHA、structures、atoms、force components、
feature dimensions。

- [ ] **Step 4: 实现流式 dataset/collate**

`CachedSplitDataset` 使用 manifest 的 structure-to-shard index，并通过一个小型 LRU 缓存
读取 shard：

```python
torch.load(path, map_location="cpu", weights_only=True, mmap=True)
```

`collate_cached_structures` 拼接原子字段并重新计算 batch-local offsets。不得把 full cache
一次加载到内存。

- [ ] **Step 5: 实现 build-cache workflow**

阶段顺序：

```text
write incomplete manifest
-> load/verify checkpoint
-> inspect dataset identity
-> extract structures in stable order
-> flush whole-structure shards
-> verify every shard
-> write complete manifest
```

shard 达到 `shard_max_atoms` 前只能在完整结构边界 flush。

- [ ] **Step 6: 静态检查、提交、远端绿灯**

Commit:

```text
feat: add immutable UPET raw cache
```

## Task 7: Epoch 汇总、total-loss EMA、scheduler 与精确续训

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/trainer.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_trainer.py`

- [ ] **Step 1: 写状态机失败测试**

定义可直接测试的 API：

```python
@dataclass
class EarlyStoppingState:
    ema: float | None = None
    best: float | None = None
    best_epoch: int | None = None
    bad_epochs: int = 0
    stopped: bool = False
    stop_reason: str | None = None


@dataclass(frozen=True)
class ControlUpdate:
    state: EarlyStoppingState
    improved: bool
    should_save_best: bool
    should_stop: bool


def update_control_state(
    state: EarlyStoppingState,
    raw_total_loss: float,
    epoch: int,
    beta: float,
    min_delta: float,
    patience: int,
    min_epochs: int,
) -> ControlUpdate:
    raise NotImplementedError("red test: update_control_state is not implemented")
```

测试：

- epoch 0 的 EMA 等于 raw value；
- 后续只做一次 `0.95/0.05` 更新；
- 改善必须满足 `< best - 1e-4`；
- 第 15 个连续 bad epoch 停止；
- 3 epochs 前不停止；
- scheduler `factor=0.5, patience=5, threshold=1e-4, abs`；
- smoke-only `stop_after_epoch` stops after atomically writing `last.pt`, does not enter config identity, and resume keeps the original `max_epochs`;
- force/energy epoch loss 按 sample sum/count，而非 batch mean；
- snapshot/restore 后状态逐字段相同；
- best 与 last 不相互覆盖；
- config/cache/binning/model-loss ID 不符时 resume 失败。

- [ ] **Step 2: 本地静态、提交测试、远端红灯**

- [ ] **Step 3: 实现 sample-weighted accumulator**

```python
@dataclass
class LossAccumulator:
    force_sum: float = 0.0
    force_count: int = 0
    energy_sum: float = 0.0
    energy_count: int = 0

    def result(self, force_coefficient: float, energy_coefficient: float):
        force = self.force_sum / self.force_count
        energy = self.energy_sum / self.energy_count
        return force, energy, force_coefficient * force + energy_coefficient * energy
```

训练 batch 的 CE 使用 `reduction="sum"` 累加；反向传播仍使用 branch mean 的加权 total。

- [ ] **Step 4: 实现 control EMA**

```python
new_ema = raw if state.ema is None else beta * state.ema + (1.0 - beta) * raw
improved = state.best is None or new_ema < state.best - min_delta
```

连续 `patience=15` 次无改善且已完成 `min_epochs=3` 时停止。

- [ ] **Step 5: 实现 scheduler**

只创建：

```python
torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5,
    threshold=1e-4,
    threshold_mode="abs",
    cooldown=0,
    min_lr=1e-6,
)
```

每个 validation epoch 调用一次 `scheduler.step(val_total_loss_ema)`。

- [ ] **Step 6: 实现 checkpoint snapshot**

保存：

```text
model, optimizer, scheduler
epoch, global_step, learning_rate
EMA, best, best_epoch, bad_epochs, stop_reason
Python/NumPy/Torch CPU/Torch CUDA RNG
sampler generator
config/cache/binning/model-loss IDs
schema version
```

`last.pt` 每个完整 epoch 原子性保存；改善时另存 `best.pt`。

- [ ] **Step 7: 静态检查、提交、远端绿灯**

Commit:

```text
feat: add total-loss EMA trainer state
```

## Task 8: Train workflow、evaluation、metrics 与 verify

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/metrics.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/train.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/evaluate.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/verify.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py`

- [ ] **Step 1: 写 workflow 失败测试**

使用 synthetic complete cache 验证：

- train workflow 创建 `resolved_config.yaml`、`binning.json`、best/last、metrics JSONL；
- force labels shape 为 `[N,3]`；
- energy labels shape 为 `[S]`；
- evaluation 默认加载 best；
- test predictions 含完整 logits、labels、continuous/expected errors、IDs、offsets；
- force sample count 为 `3 * N_atom`；
- Brier、NLL、accuracy、Pearson、Spearman 数值与手算一致；
- overflow count 单独报告；
- verify 拒绝文件 hash 被修改的 run；
- outputs 中没有 png/pdf/svg。

- [ ] **Step 2: 本地静态、提交测试、远端红灯**

- [ ] **Step 3: 实现 metrics**

`classification_metrics(logits, labels, observed, representatives)` 返回：

```text
sample_count
accuracy
nll
brier
mean_observed_error
mean_expected_error
mae_expected_vs_observed
pearson
spearman
overflow_count
overflow_fraction
```

Brier：

```python
one_hot = F.one_hot(labels, num_classes=logits.shape[-1]).to(probabilities)
brier = torch.mean(torch.sum((probabilities - one_hot) ** 2, dim=-1))
```

force 在调用 metrics 前展平 `[N,3,B] -> [3N,B]`。

- [ ] **Step 4: 实现 train workflow**

顺序：

```text
load strict config
-> require complete cache
-> create bin specs
-> infer D_force/D_energy from manifest
-> construct ConfidenceModel
-> create deterministic loaders
-> train/validate
-> scheduler/control state
-> save best/last/metrics
-> finalize manifest
```

labels 每个 batch 从连续误差生成，不写回 raw cache。

- [ ] **Step 5: 实现 evaluation**

默认 checkpoint：

```python
run_dir / "checkpoints" / "best.pt"
```

保存 `test_predictions.pt`：

```python
{
    "force_logits": torch.cat(force_logits, dim=0),
    "force_labels": torch.cat(force_labels, dim=0),
    "force_observed_errors": torch.cat(force_observed, dim=0),
    "force_expected_errors": torch.cat(force_expected, dim=0),
    "energy_logits": torch.cat(energy_logits, dim=0),
    "energy_labels": torch.cat(energy_labels, dim=0),
    "energy_observed_errors": torch.cat(energy_observed, dim=0),
    "energy_expected_errors": torch.cat(energy_expected, dim=0),
    "structure_ids": torch.cat(structure_ids, dim=0),
    "atom_offsets": merged_atom_offsets,
    "force_representatives": force_spec.representatives,
    "energy_representatives": energy_spec.representatives,
}
```

另写 `metrics.json` 和两个 bin summary CSV。

- [ ] **Step 6: 实现 verify**

`verify_run(path, full=True)` 检查：

- root/stage manifest complete；
- identity 链一致；
- 所有声明文件存在；
- full 模式重算 SHA；
- prediction shape/count 与 manifest 一致；
- 目录中不存在图片。

- [ ] **Step 7: 静态检查、提交、远端绿灯**

Commit:

```text
feat: add confidence training and evaluation workflows
```

## Task 9: CLI、正式配置与中文 README

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/build_cache.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/train.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/evaluate.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/verify.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/configs/n20_cpu.yaml`
- Create: `Uncertainty_Quantification/ConfidenceHead/configs/full_linear.yaml`
- Create: `Uncertainty_Quantification/ConfidenceHead/README.md`

- [ ] **Step 1: 写 CLI smoke 测试**

在 `test_workflows.py` 增加对四个脚本的 `--help` 测试，以及：

```text
build_cache.py --config configs/n20_cpu.yaml
train.py --config configs/n20_cpu.yaml --run-name n20_continuous
evaluate.py --config configs/n20_cpu.yaml --run-name n20_continuous
verify.py --run outputs/runs/n20_continuous --full
```

参数缺失时 exit code 2；workflow failure 时 exit code 1；成功时 exit code 0。

- [ ] **Step 2: 提交测试并远端红灯**

- [ ] **Step 3: 实现薄 CLI**

每个脚本只包含：

- `build_cache.py`: required `--config`
- `train.py`: required `--config`, required `--run-name`, optional `--resume-from`, and smoke-only optional `--stop-after-epoch`
- `evaluate.py`: required `--config`, required `--run-name`
- `verify.py`: required `--run`, boolean `--full`

Every entry point ends with `raise SystemExit(main())`. Success returns 0, a known workflow failure returns 1 after logging the error, and argparse errors return 2.

数学和文件协议不得出现在 scripts 中。

- [ ] **Step 4: 写 n20 配置**

关键字段：

```yaml
profile: smoke
allow_identical_splits: true
checkpoint:
  path: data/checkpoint/pet-omatpes-l-v0.1.0.ckpt
  expected_sha256: 879b1045391d88869522605a8b8b3cedeed74668e7062fdd7487548ab7b08004
data:
  train: &n20
    path: data/dataset/matpes_n20.extxyz
    expected_sha256: c92161329aab539064a2c2438a395cb01e38bfc91211c558aebbc1ff94702e3d
  validation: *n20
  test: *n20
readouts:
  energy_prediction: energy
  force_prediction: non_conservative_forces
  energy_features: mtt::aux::energy_last_layer_features
  force_features: mtt::aux::non_conservative_forces_last_layer_features
cache:
  batch_size: 2
  num_workers: 0
  shard_max_atoms: 100000
binning:
  algorithm: fixed_linear_v1
  force_num_bins: 50
  force_max_error: 0.5
  energy_num_bins: 50
  energy_max_error: 0.3
model:
  hidden_dims: [256, 256, 256]
  dropout: 0.0
  cumulant_order: 3
  signed_root: true
loss:
  force_coefficient: 1.0
  energy_coefficient: 1.5
optimizer:
  learning_rate: 0.001
  weight_decay: 0.0
scheduler:
  monitor: val/total_loss_ema
  factor: 0.5
  patience: 5
  threshold: 0.0001
  threshold_mode: abs
  cooldown: 0
  min_lr: 0.000001
trainer:
  max_epochs: 3
  ema_beta: 0.95
  early_stopping_patience: 15
  min_delta: 0.0001
  min_epochs: 3
run:
  output_root: Uncertainty_Quantification/ConfidenceHead/outputs
  seed: 1234
  device: cpu
  amp: false
```

- [ ] **Step 5: 写 full 配置**

Use the following exact production data identity. This configuration is written and
statically validated, but is not executed during this task:

```yaml
profile: production
allow_identical_splits: false
data:
  train:
    path: data/dataset/matpes_train.extxyz
    expected_sha256: 12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec
  validation:
    path: data/dataset/matpes_val.extxyz
    expected_sha256: 5b2ce7f0835f0f69d27840116608ee264536d2cc0ac253a33625ece29f985eef
  test:
    path: data/dataset/matpes_test.extxyz
    expected_sha256: 1ffcdcad2fc6f0b0907b91cd29bfee340eb02cddf6b525268290c6329f56182d
trainer:
  max_epochs: 200
run:
  device: cuda
  amp: false
```

- [ ] **Step 6: 写中文 README**

README 必须包含：

- energy/force 独立 readout 图；
- 逐分量 force 和逐原子 energy 公式；
- fixed-linear bin 默认参数；
- branch-wise loss；
- total-loss EMA early stopping；
- 四阶段命令；
- best/last 与 resume；
- 本地静态/远端动态边界；
- n20 是 smoke、不可作为科学结果；
- 当前无 plotting。

- [ ] **Step 7: 静态检查、提交、远端 CLI 绿灯**

Commit:

```text
docs: add confidence head workflows and configs
```

## Task 10: 远端真实 n20 cache、训练、续训、评估和全验证

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_n20.py`
- Runtime only: `data/checkpoint/pet-omatpes-l-v0.1.0.ckpt`
- Runtime only: `data/dataset/matpes_n20.extxyz`
- Runtime only: `Uncertainty_Quantification/ConfidenceHead/outputs/**`

- [ ] **Step 1: 写真实 n20 测试**

标记：

```python
@pytest.mark.confidence_head_n20
def test_real_n20_cache_contract(n20_config):
    manifest = build_cache(n20_config)
    for split in ("train", "validation", "test"):
        assert manifest["splits"][split]["structure_count"] == 20
        assert manifest["splits"][split]["atom_count"] == 143
        assert manifest["splits"][split]["force_component_count"] == 429
        assert manifest["splits"][split]["force_feature_dim"] > 0
        assert manifest["splits"][split]["energy_feature_dim"] > 0
        assert manifest["splits"][split]["force_feature_key"] != (
            manifest["splits"][split]["energy_feature_key"]
        )
```

增加端到端测试，断言 train/evaluate/verify 完成且不存在图片。

- [ ] **Step 2: 本地只做静态检查并提交**

```bash
tox -e lint
git diff --check
git add Uncertainty_Quantification/ConfidenceHead/tests/test_n20.py
git commit -m "test: add real n20 confidence workflow"
git push origin ConfidenceHead
```

- [ ] **Step 3: 远端准备 checkpoint**

```bash
mkdir -p /home/bywang/code/UQ/upet_new/data/checkpoint
cp /home/bywang/code/UQ/upet/pet-omatpes-l-v0.1.0.ckpt \
  /home/bywang/code/UQ/upet_new/data/checkpoint/pet-omatpes-l-v0.1.0.ckpt
sha256sum /home/bywang/code/UQ/upet_new/data/checkpoint/pet-omatpes-l-v0.1.0.ckpt
```

Expected:

```text
879b1045391d88869522605a8b8b3cedeed74668e7062fdd7487548ab7b08004
```

- [ ] **Step 4: 从本地 WSL 传输 n20**

```bash
ssh -p 55801 bywang@121.48.164.204 \
  "mkdir -p /home/bywang/code/UQ/upet_new/data/dataset"
scp -P 55801 \
  /home/lilong/code/UQ/upet_new/data/dataset/matpes_n20.extxyz \
  bywang@121.48.164.204:/home/bywang/code/UQ/upet_new/data/dataset/
ssh -p 55801 bywang@121.48.164.204 \
  "sha256sum /home/bywang/code/UQ/upet_new/data/dataset/matpes_n20.extxyz"
```

Expected:

```text
c92161329aab539064a2c2438a395cb01e38bfc91211c558aebbc1ff94702e3d
```

- [ ] **Step 5: 远端运行全部非 n20 单元测试**

```bash
cd /home/bywang/code/UQ/upet_new
git pull --ff-only origin ConfidenceHead
/home/bywang/.conda/envs/upet_new/bin/python -m pytest \
  Uncertainty_Quantification/ConfidenceHead/tests \
  -m "not confidence_head_n20" -q
```

Expected: all pass, 0 warnings（pytest warnings 按项目配置视为 errors）。

- [ ] **Step 6: 远端构建真实 n20 cache**

```bash
PY=/home/bywang/.conda/envs/upet_new/bin/python
CFG=Uncertainty_Quantification/ConfidenceHead/configs/n20_cpu.yaml
"$PY" Uncertainty_Quantification/ConfidenceHead/scripts/build_cache.py \
  --config "$CFG"
```

Expected: complete cache；每个 split 为 20 structures、143 atoms、429 force components；
device=cpu、amp=false。

- [ ] **Step 7: 远端执行连续三 epoch 基线**

```bash
"$PY" Uncertainty_Quantification/ConfidenceHead/scripts/train.py \
  --config "$CFG" \
  --run-name n20_continuous
```

Expected: best.pt、last.pt、3 个 epoch metrics。

- [ ] **Step 8: 远端执行中断/续训对照**

先通过测试 override 将 `max_epochs=1` 写入独立 resolved config，运行：

```bash
"$PY" Uncertainty_Quantification/ConfidenceHead/scripts/train.py \
  --config "$CFG" \
  --run-name n20_resumed \
  --stop-after-epoch 1
"$PY" Uncertainty_Quantification/ConfidenceHead/scripts/train.py \
  --config "$CFG" \
  --run-name n20_resumed \
  --resume-from \
  Uncertainty_Quantification/ConfidenceHead/outputs/runs/n20_resumed/checkpoints/last.pt
```

`--stop-after-epoch` is smoke-only execution control. It does not change the resolved config or config identity; it stops only after the requested epoch has atomically committed `last.pt`. Resume clears the transient external-stop marker and retains `max_epochs: 3`.

- [ ] **Step 9: 比较连续与续训状态**

使用测试 helper 逐字段比较：

```text
model state tensors
optimizer state
scheduler state
EMA/best/bad_epochs
epoch/global_step
metrics JSONL
```

CPU deterministic 模式下要求 exact equality。若失败，先修复 RNG/sampler/epoch commit
语义，不允许放宽容差掩盖状态错误。

- [ ] **Step 10: 远端评估 best 并 full verify**

```bash
"$PY" Uncertainty_Quantification/ConfidenceHead/scripts/evaluate.py \
  --config "$CFG" \
  --run-name n20_continuous
"$PY" Uncertainty_Quantification/ConfidenceHead/scripts/verify.py \
  --run Uncertainty_Quantification/ConfidenceHead/outputs/runs/n20_continuous \
  --full
```

Expected:

- evaluation 读取 best.pt；
- force logits `[143,3,50]`；
- energy logits `[20,50]`；
- metrics/sample_count force=429、energy=20；
- complete manifests；
- 0 张图片。

- [ ] **Step 11: 运行真实 n20 pytest marker**

```bash
"$PY" -m pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_n20.py \
  -m confidence_head_n20 -v
```

Expected: pass。

- [ ] **Step 12: 记录远端验收摘要**

在 `README.md` 的“已验证环境”中只记录：

- git commit；
- Python/PyTorch/UPET/metatrain 版本；
- checkpoint/data SHA；
- n20 counts；
- cache/run IDs；
- best epoch；
- verify 结果。

不得提交 cache、checkpoint、PT/CSV/JSONL outputs。

Commit:

```text
docs: record n20 confidence verification
```

## Task 11: 最终静态检查、代码审查与交付

**Files:**
- Review all files under `Uncertainty_Quantification/ConfidenceHead/`
- Review: `pyproject.toml`
- Review: `tox.ini`

- [ ] **Step 1: 本地运行完整静态检查**

```bash
tox -e lint
git diff --check
```

Expected: exit 0。

- [ ] **Step 2: 验证本地没有动态产物**

```bash
find Uncertainty_Quantification/ConfidenceHead \
  -type f \( -name '*.pt' -o -name '*.ckpt' -o -name '*.png' -o -name '*.pdf' \)
git status --short
```

Expected: find 无输出；工作区只包含当前任务明确修改。

- [ ] **Step 3: 远端重新运行最终动态套件**

```bash
cd /home/bywang/code/UQ/upet_new
git pull --ff-only origin ConfidenceHead
/home/bywang/.conda/envs/upet_new/bin/python -m pytest \
  Uncertainty_Quantification/ConfidenceHead/tests -q
```

Expected: all pass。

- [ ] **Step 4: 远端重新 full verify n20**

```bash
/home/bywang/.conda/envs/upet_new/bin/python \
  Uncertainty_Quantification/ConfidenceHead/scripts/verify.py \
  --run Uncertainty_Quantification/ConfidenceHead/outputs/runs/n20_continuous \
  --full
```

Expected: complete，identity/hash/shape/count 全部通过。

- [ ] **Step 5: 审查最终 diff**

逐项对照设计规格第 22 节的 16 条验收标准。确认：

- 没有旧代码/结果；
- 没有 plotting；
- energy/force readout 严格分离；
- force component-wise；
- energy per-atom；
- fixed-linear 0.5/0.3；
- total-loss EMA；
- exact resume；
- 本地静态、远端动态；
- 没有 GPU/full-data 运行。

- [ ] **Step 6: 最终提交和推送**

```bash
git status --short
git push origin ConfidenceHead
```

不得创建 full-data 或 GPU 作业。
