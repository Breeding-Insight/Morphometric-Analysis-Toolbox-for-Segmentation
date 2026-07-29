"""CLI argument parsing -- no pipeline execution, so no torch required."""

import pytest

from mats.cli import build_parser, _normalize_argv, _resolve_fetch_only


def test_default_subcommand_inserted():
    # `mats -i in -o out` behaves as `mats run -i in -o out`.
    assert _normalize_argv(["-i", "in", "-o", "out"]) == ["run", "-i", "in", "-o", "out"]


def test_explicit_subcommands_untouched():
    for argv in (["run", "-i", "x"], ["doctor"], ["fetch-weights", "--only", "rf-detr"],
                 ["app", "--server.port", "8502"], ["-h"], ["--help"]):
        assert _normalize_argv(argv) == argv


def test_run_defaults():
    ns = build_parser().parse_args(["run", "-i", "in", "-o", "out"])
    assert ns.command == "run"
    assert ns.input_dir == "in"
    assert ns.output_dir == "out"
    assert ns.output_mode == "masks"
    assert ns.mask_method == "threshold"
    assert ns.threshold_level == "auto"
    assert ns.csv_schema == "full"        # research schema by default
    assert ns.save_axes is False


def test_underscore_and_hyphen_aliases_agree():
    p = build_parser()
    a = p.parse_args(["run", "--input_dir", "x", "--output_dir", "y", "--results_path", "z"])
    b = p.parse_args(["run", "--input-dir", "x", "--output-dir", "y", "--results-path", "z"])
    assert (a.input_dir, a.output_dir, a.results_path) == ("x", "y", "z")
    assert (b.input_dir, b.output_dir, b.results_path) == ("x", "y", "z")


def test_compact_and_axes_opt_in():
    ns = build_parser().parse_args(["run", "-i", "x", "--csv-schema", "compact", "--save-axes"])
    assert ns.csv_schema == "compact"
    assert ns.save_axes is True


def test_invalid_choice_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "-i", "x", "--mask-method", "nonsense"])


def test_fetch_weights_default_targets_rf_detr_only():
    ns = build_parser().parse_args(["fetch-weights"])
    assert _resolve_fetch_only(ns) == "rf-detr"


def test_fetch_weights_all_flag_targets_both():
    ns = build_parser().parse_args(["fetch-weights", "--all"])
    assert _resolve_fetch_only(ns) is None


def test_fetch_weights_explicit_only_wins():
    ns = build_parser().parse_args(["fetch-weights", "--only", "birefnet"])
    assert _resolve_fetch_only(ns) == "birefnet"


def test_fetch_weights_all_and_only_conflict_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["fetch-weights", "--all", "--only", "birefnet"])
