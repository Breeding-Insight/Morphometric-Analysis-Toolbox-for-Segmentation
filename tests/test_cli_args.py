"""CLI argument parsing -- no pipeline execution, so no torch required."""

from types import SimpleNamespace

import pytest

from mats.cli import (
    _normalize_argv,
    _require_local_birefnet_for_run,
    _resolve_fetch_only,
    _resolve_template_dims,
    build_parser,
)
from mats.dimensions import parse_template_dimensions


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
    assert ns.sheet_dimensions is None
    assert ns.template_dimensions is None


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


def test_sheet_and_legacy_dimensions_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["run", "--sheet-dimensions", "12x12in", "-t", "10x9.5in"]
        )


def test_sheet_dimensions_are_converted_to_marker_centre_calibration():
    args = SimpleNamespace(sheet_dimensions="12x12in", template_dimensions=None)
    lm = SimpleNamespace(parse_template_dimensions=parse_template_dimensions)

    assert _resolve_template_dims(args, lm) == (10.0, 9.5, "in")


def test_legacy_template_dimensions_keep_their_historical_meaning():
    args = SimpleNamespace(sheet_dimensions=None, template_dimensions="10.5x9.5in")
    lm = SimpleNamespace(parse_template_dimensions=parse_template_dimensions)

    assert _resolve_template_dims(args, lm) == (10.5, 9.5, "in")


def test_local_birefnet_preflight_skips_otsu(monkeypatch):
    args = build_parser().parse_args(["run", "-i", "x", "--mask-method", "threshold"])
    monkeypatch.setattr("mats.weights.require_local_weight", lambda name: pytest.fail("not needed"))

    assert _require_local_birefnet_for_run(args) is None


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
