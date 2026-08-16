"""Transactional orchestration of uncertainty publications for a campaign."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .artifacts import sibling_staging
from .campaign import CampaignConfig, select_campaign_items
from .errors import HardFailure
from .identifiers import validate_artifact_key
from .uq_publication import compute_uncertainty_results, publish_uncertainty_results
from .validation import validate_uq_publication


_UNITS = {
    "energy": "eV",
    "forces": "eV/Angstrom",
    "stress": "eV/Angstrom^3",
}


@dataclass(frozen=True)
class CampaignUQResult:
    """One selected campaign UQ publication outcome."""

    run_label: str
    dataset_label: str
    destination: Path
    skipped: bool


def _validate_campaign_request(campaign: CampaignConfig) -> tuple[str, int]:
    mode = campaign.prediction.mode
    member_count = campaign.prediction.member_count
    if mode != "raw":
        raise HardFailure("campaign uncertainty only supports raw predictions")
    if (
        isinstance(member_count, bool)
        or not isinstance(member_count, int)
        or member_count < 2
    ):
        raise HardFailure("campaign uncertainty member_count must be at least 2")
    return mode, member_count


def compute_campaign_uq(
    campaign: CampaignConfig,
    run_labels: Sequence[str] | None,
    dataset_labels: Sequence[str] | None,
) -> tuple[CampaignUQResult, ...]:
    """Compute selected raw campaign UQ publications transactionally.

    Existing destinations are never overwritten: a complete exact audit is a
    reusable success, and every other existing destination is a hard failure.
    """

    mode, member_count = _validate_campaign_request(campaign)
    runs, datasets = select_campaign_items(campaign, run_labels, dataset_labels)
    outcomes: list[CampaignUQResult] = []
    for run in runs:
        run_root = Path(run.run_root).expanduser().resolve()
        for dataset in datasets:
            storage_key = validate_artifact_key(
                dataset.storage_key, "campaign dataset storage_key"
            )
            prediction_root = run_root / "predictions" / storage_key
            destination = run_root / "uncertainty" / storage_key / mode
            if destination.exists() or destination.is_symlink():
                validate_uq_publication(
                    run_root,
                    split=storage_key,
                    mode=mode,
                    member_count=member_count,
                    units=_UNITS,
                )
                outcomes.append(
                    CampaignUQResult(
                        run_label=run.label,
                        dataset_label=dataset.label,
                        destination=destination,
                        skipped=True,
                    )
                )
                continue
            results = compute_uncertainty_results(
                prediction_root, mode=mode, member_count=member_count
            )
            with sibling_staging(destination) as staging:
                publish_uncertainty_results(
                    staging,
                    results,
                    dataset_key=storage_key,
                    mode=mode,
                    member_count=member_count,
                    units=_UNITS,
                )
                validate_uq_publication(
                    run_root,
                    split=storage_key,
                    mode=mode,
                    member_count=member_count,
                    units=_UNITS,
                    publication_root=staging,
                )
            outcomes.append(
                CampaignUQResult(
                    run_label=run.label,
                    dataset_label=dataset.label,
                    destination=destination,
                    skipped=False,
                )
            )
    return tuple(outcomes)
