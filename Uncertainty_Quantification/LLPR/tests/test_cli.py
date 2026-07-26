from pathlib import Path

import pytest

from Uncertainty_Quantification.LLPR.llpr import cli


@pytest.mark.parametrize(
    ("command", "runner_name"),
    [
        ("build", "run_build"),
        ("calibrate", "run_calibrate"),
        ("evaluate", "run_evaluate"),
        ("import-legacy", "import_legacy"),
        ("verify", "verify_run"),
        ("plot", "run_plot"),
    ],
)
def test_cli_dispatches_once(
    command: str, runner_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []
    monkeypatch.setattr(cli, runner_name, lambda path: calls.append(Path(path)))

    assert cli.main([command, "--config", "config.yaml"]) == 0
    assert calls == [Path("config.yaml")]


def test_run_dispatches_numerical_stages_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Path]] = []
    for name in ("run_build", "run_calibrate", "run_evaluate"):
        monkeypatch.setattr(
            cli,
            name,
            lambda path, stage=name: calls.append((stage, Path(path))),
        )

    assert cli.main(["run", "--config", "config.yaml"]) == 0
    assert calls == [
        ("run_build", Path("config.yaml")),
        ("run_calibrate", Path("config.yaml")),
        ("run_evaluate", Path("config.yaml")),
    ]


def test_run_stops_immediately_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fail(_: Path) -> None:
        calls.append("build")
        raise RuntimeError("expected")

    monkeypatch.setattr(cli, "run_build", fail)
    monkeypatch.setattr(cli, "run_calibrate", lambda _: calls.append("calibrate"))
    monkeypatch.setattr(cli, "run_evaluate", lambda _: calls.append("evaluate"))

    with pytest.raises(RuntimeError, match="expected"):
        cli.main(["run", "--config", "config.yaml"])
    assert calls == ["build"]


def test_help_lists_all_seven_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["--help"])

    assert error.value.code == 0
    output = capsys.readouterr().out
    for command in (
        "build",
        "calibrate",
        "evaluate",
        "run",
        "import-legacy",
        "verify",
        "plot",
    ):
        assert command in output
