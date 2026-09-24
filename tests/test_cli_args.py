"""CLI argument parsing -- no pipeline execution, so no torch required."""

import sys
from types import SimpleNamespace

import pytest

from mats.cli import (
    _cmd_run,
    _measured_methods,
    _normalize_argv,
    _pre_cleanup_methods,
    _print_run_banner,
    _require_local_birefnet_for_run,
    _resolve_fetch_only,
    _resolve_template_dims,
    build_parser,
)
from mats.dimensions import parse_template_dimensions
from mats.mask_settings import CLEAN_MARGIN_DEFAULT, STRAY_GAP_DEFAULT


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
    assert ns.results_unit == "cm"
    assert not ns.measure_pre_cleanup
    assert ns.save_axes is False
    assert ns.sheet_dimensions is None
    assert ns.template_dimensions is None
    assert ns.export == []
    assert ns.pre_cleanup_methods == "selected"
    assert not ns.no_target_boxes and not ns.no_masks and not ns.no_failure_log


def test_repeated_exports_and_both_methods_parse():
    ns = build_parser().parse_args([
        "run", "--export", "pre-cleanup", "--export", "overlay",
        "--pre-cleanup-methods", "both", "--no-target-boxes", "--no-masks",
    ])
    assert ns.export == ["pre-cleanup", "overlay"]
    assert ns.pre_cleanup_methods == "both"
    assert ns.no_target_boxes and ns.no_masks


def test_mask_method_both_measures_and_exports_each_method():
    parse = build_parser().parse_args
    both = parse(["run", "-i", "x", "--mask-method", "both", "--export", "pre-cleanup"])
    assert both.mask_method == "both"
    assert _measured_methods(both) == ("threshold", "birefnet")
    assert _pre_cleanup_methods(both) == ("threshold", "birefnet")

    single = parse(["run", "-i", "x", "--mask-method", "birefnet", "--export", "pre-cleanup"])
    assert _measured_methods(single) == ("birefnet",)
    assert _pre_cleanup_methods(single) == ("birefnet",)
    assert _pre_cleanup_methods(parse(["run", "-i", "x", "--mask-method", "both"])) == ()


def test_unknown_export_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--export", "unknown"])


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


def test_pre_cleanup_measurement_flag_parses_independently_of_export():
    ns = build_parser().parse_args(["run", "--measure-pre-cleanup"])
    assert ns.measure_pre_cleanup
    assert ns.export == []


def test_stray_gap_defaults_and_parses():
    parse = build_parser().parse_args
    assert parse(["run", "-i", "x"]).stray_gap == STRAY_GAP_DEFAULT
    assert parse(["run", "-i", "x", "--measure-pre-cleanup", "--stray-gap", "0"]).stray_gap == 0.0
    assert parse(["run", "-i", "x", "--stray-gap", "1.5"]).stray_gap == 1.5


@pytest.mark.parametrize("value", ["-0.1", "10.5", "nan", "inf", "far"])
def test_stray_gap_rejects_invalid_values(value, capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["run", "-i", "x", "--stray-gap", value])
    assert exc.value.code == 2
    assert "--stray-gap" in capsys.readouterr().err


def test_clean_margin_defaults_and_parses():
    parse = build_parser().parse_args
    assert parse(["run", "-i", "x"]).clean_margin == CLEAN_MARGIN_DEFAULT
    assert parse(["run", "-i", "x", "--clean-margin", "0"]).clean_margin == 0.0
    assert parse(["run", "-i", "x", "--clean-margin", "2.5"]).clean_margin == 2.5


@pytest.mark.parametrize("value", ["-1", "10.5", "nan", "wide"])
def test_clean_margin_rejects_invalid_values(value, capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["run", "-i", "x", "--clean-margin", value])
    assert exc.value.code == 2
    assert "--clean-margin" in capsys.readouterr().err


def test_banner_reports_margin_and_stray_gap_only_for_pre_cleanup(capsys):
    parse = build_parser().parse_args
    _print_run_banner(parse([
        "run", "-i", "x", "--measure-pre-cleanup", "--stray-gap", "0.5", "--clean-margin", "2",
    ]), None)
    out = capsys.readouterr().out
    assert "Clean margin: 2% of the target box's shorter side" in out
    assert "Stray gap: 0.5 x leaf bounding-box diagonal" in out
    _print_run_banner(parse(["run", "-i", "x"]), None)
    out = capsys.readouterr().out
    assert "Stray gap" not in out and "Clean margin" not in out


def _fake_core(monkeypatch, calls):
    """Stand in for mats.core so _cmd_run runs without torch."""
    import mats

    def run_leaf_morpho_batch(**kwargs):
        calls.append(kwargs)
        return {
            "succeeded": 1, "failed": 0, "workers": 1, "worker_reason": "test",
            "methods": ("threshold",), "results_path": kwargs["results_path"],
            "failure_report_path": None,
        }

    fake = SimpleNamespace(
        parse_template_dimensions=parse_template_dimensions,
        method_suffixed_path=lambda path, method: path,
        get_input_images=lambda input_dir: ["leaf.jpg"],
        run_leaf_morpho_batch=run_leaf_morpho_batch,
    )
    monkeypatch.setitem(sys.modules, "mats.core", fake)
    monkeypatch.setattr(mats, "core", fake, raising=False)


def test_stray_gap_reaches_the_pipeline(tmp_path, monkeypatch):
    calls = []
    _fake_core(monkeypatch, calls)
    args = build_parser().parse_args([
        "run", "-i", str(tmp_path), "-o", str(tmp_path / "out"), "-t", "10x10cm",
        "--measure-pre-cleanup", "--stray-gap", "0.6", "--clean-margin", "2",
    ])
    assert _cmd_run(args) == 0
    assert calls[0]["measurement_source"] == "pre-cleanup"
    assert calls[0]["stray_gap"] == 0.6
    assert calls[0]["clean_margin"] == 2.0


@pytest.mark.parametrize("flag", ["--stray-gap", "--clean-margin"])
def test_cleanup_settings_require_pre_cleanup_measurement(flag, tmp_path, monkeypatch, capsys):
    calls = []
    _fake_core(monkeypatch, calls)
    args = build_parser().parse_args([
        "run", "-i", str(tmp_path), "-o", str(tmp_path / "out"), "-t", "10x10cm",
        flag, "0.6",
    ])
    with pytest.raises(SystemExit):
        _cmd_run(args)
    assert f"{flag} requires --measure-pre-cleanup" in capsys.readouterr().err
    assert not calls


def test_results_unit_accepts_supported_choices():
    ns = build_parser().parse_args(["run", "-i", "x", "--results-unit", "in"])
    assert ns.results_unit == "in"


def test_invalid_choice_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "-i", "x", "--mask-method", "nonsense"])


def test_threshold_level_accepts_presets_and_custom_cutoffs():
    parse = build_parser().parse_args
    assert parse(["run", "-i", "x", "--threshold-level", "177"]).threshold_level == 177
    assert parse(["run", "-i", "x", "--threshold-level", "HIGH"]).threshold_level == "high"


@pytest.mark.parametrize("value", ["0", "256", "custom", "12.5"])
def test_threshold_level_rejects_invalid_cutoffs(value, capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["run", "-i", "x", "--threshold-level", value])
    assert exc.value.code == 2
    assert "--threshold-level" in capsys.readouterr().err


@pytest.mark.parametrize("extra", [
    [],
    ["--mask-method", "birefnet", "--export", "pre-cleanup", "--pre-cleanup-methods", "threshold"],
    ["--mask-method", "both"],
])
def test_banner_reports_custom_threshold_whenever_thresholding_runs(extra, capsys):
    args = build_parser().parse_args(["run", "-i", "x", "--threshold-level", "177", *extra])
    _print_run_banner(args, 177)
    assert "Threshold level: custom (177)" in capsys.readouterr().out


def test_banner_omits_threshold_when_birefnet_alone_segments(capsys):
    args = build_parser().parse_args(
        ["run", "-i", "x", "--mask-method", "birefnet", "--threshold-level", "177"]
    )
    _print_run_banner(args, 177)
    assert "Threshold level" not in capsys.readouterr().out


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


@pytest.mark.parametrize("method", ["birefnet", "both"])
def test_local_birefnet_preflight_runs_whenever_birefnet_measures(method, monkeypatch):
    args = build_parser().parse_args(["run", "-i", "x", "--mask-method", method])
    required = []
    monkeypatch.setattr("mats.birefnet_runtime.require_birefnet_dependencies", lambda: None)
    monkeypatch.setattr("mats.weights.require_local_weight", required.append)

    _require_local_birefnet_for_run(args)
    assert required == ["birefnet"]


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
