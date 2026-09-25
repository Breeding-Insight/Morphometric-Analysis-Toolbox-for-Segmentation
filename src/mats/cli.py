"""Command-line interface for MATs.

Subcommands:

    mats run             Batch-measure a folder of images (the default command).
    mats app             Launch the Streamlit GUI.
    mats fetch-weights   Download the model checkpoints.
    mats doctor          Report the environment: weights, devices, decoders.

For muscle-memory compatibility, ``mats -i IN -o OUT ...`` (no subcommand) is
treated as ``mats run -i IN -o OUT ...``.
"""

import argparse
import os
import sys

from .mask_settings import (
    CLEAN_MARGIN_DEFAULT,
    CLEAN_MARGIN_MAX,
    CLEAN_SIZE_DEFAULT,
    CLEAN_SIZE_MAX,
    STRAY_GAP_DEFAULT,
    STRAY_GAP_MAX,
    checked_clean_margin,
    checked_clean_size,
    checked_stray_gap,
)
from .thresholds import parse_threshold_level, threshold_value_for

_RUN_SUBCOMMAND = "run"
_SUBCOMMANDS = {"run", "app", "fetch-weights", "doctor"}


def _fail(message, code=2):
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def _threshold_level_arg(text):
    try:
        return parse_threshold_level(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _stray_gap_arg(text):
    try:
        return checked_stray_gap(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _clean_margin_arg(text):
    try:
        return checked_clean_margin(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _clean_size_arg(text):
    try:
        return checked_clean_size(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def build_parser():
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="mats",
        description="MATs -- field morphometric tools: leaf image -> measurements.",
    )
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser(
        _RUN_SUBCOMMAND,
        help="Batch-measure a folder of images.",
        description="Run the leaf image -> mask -> measurement pipeline over a folder.",
    )
    # Flag spellings preserved from the original CLI (underscore forms), with
    # hyphenated aliases added. Do not remove the underscore forms -- they are
    # referenced in existing scripts and the Open OnDemand launcher.
    run.add_argument('-i', '--input_dir', '--input-dir', dest='input_dir', type=str, default=None,
                     help='Directory of images to analyze.')
    run.add_argument('-o', '--output_dir', '--output-dir', dest='output_dir', default=None,
                     help='Directory for output images (masks / target boxes).')
    run.add_argument('-r', '--results_path', '--results-path', dest='results_path', default=None,
                     help='Path for the measurement CSV (default: ./leaf_morpho_results.csv).')
    run.add_argument('-w', '--workers', default=None, type=int,
                     help='Number of worker processes; overrides the automatic GPU-safe default.')
    dimensions = run.add_mutually_exclusive_group()
    dimensions.add_argument(
        '--sheet-dimensions', dest='sheet_dimensions', type=str, default=None,
        help='Finished Template Creator sheet size as <width>x<height><unit>, e.g. 12x12in or 30x30cm.'
    )
    dimensions.add_argument(
        '-t', '--template_dimensions', '--template-dimensions', dest='template_dimensions',
        type=str, default=None,
        help='Legacy/custom marker-centre calibration area as <width>x<height><unit>.'
    )
    run.add_argument('--output-mode', choices=('masks', 'target-boxes'), default='masks',
                     help='Produce segmentation masks, or only perspective-corrected target boxes.')
    run.add_argument('--mask-method', choices=('threshold', 'birefnet', 'both'), default='threshold',
                     help='Mask method when --output-mode masks. threshold (Otsu) is fast, needs '
                          'no GPU and no extra download; birefnet is more accurate on cluttered '
                          'backgrounds but needs a locally installed ~2.65 GB checkpoint '
                          '(`mats fetch-weights --only birefnet --source lfs`); both measures every '
                          'image with each method and writes one CSV per method, suffixed '
                          '_threshold and _birefnet.')
    run.add_argument('--threshold-level', type=_threshold_level_arg, default='auto',
                     metavar='{auto,low,medium,high,1-255}',
                     help="Cutoff for --mask-method threshold and threshold pre-cleanup exports: "
                          "auto uses Otsu (recommended); low=100, medium=125, high=150; or an "
                          "integer 1-255 for a custom cutoff. Grayscale pixels at or below the "
                          "cutoff are counted as leaf.")
    run.add_argument('--csv-schema', choices=('full', 'compact'), default='full',
                     help='full = area/width/length plus per-axis pixels-per-unit and scale_aspect_ratio '
                          '(research schema); compact = sample_id, area, width, length.')
    run.add_argument('--results-unit', choices=('mm', 'cm', 'in'), default='cm',
                     help='Unit for area, width, length, and pixels-per-unit columns in the CSV '
                          '(default: cm).')
    run.add_argument('--measure-pre-cleanup', action='store_true',
                     help='Derive area, width, and length from the raw binary mask before '
                          'gap closing and hole filling. A band --clean-margin wide along the '
                          'target-box edge (where the printed box outline lands) is cleared, '
                          'the largest object is the leaf, and other pieces are dropped when '
                          'they touch that band or lie farther from the leaf than --stray-gap.')
    run.add_argument('--clean-margin', type=_clean_margin_arg, default=CLEAN_MARGIN_DEFAULT,
                     metavar='PERCENT',
                     help='With --measure-pre-cleanup: width of the band cleared along every '
                          'target-box edge, as a percent of the box\'s shorter side '
                          f'(0-{CLEAN_MARGIN_MAX:g}; 0 clears nothing; '
                          f'default {CLEAN_MARGIN_DEFAULT:g}).')
    run.add_argument('--stray-gap', type=_stray_gap_arg, default=STRAY_GAP_DEFAULT,
                     metavar='FRACTION',
                     help='With --measure-pre-cleanup: drop pieces whose nearest pixel is '
                          'farther from the leaf than this fraction of the leaf\'s '
                          f'bounding-box diagonal (0-{STRAY_GAP_MAX:g}; 0 keeps only the leaf; '
                          f'default {STRAY_GAP_DEFAULT:g}).')
    run.add_argument('--clean-size', type=_clean_size_arg, default=CLEAN_SIZE_DEFAULT,
                     metavar='PX',
                     help='With --measure-pre-cleanup: after the edge margin and stray pieces '
                          'are cleared, remove white specks and fill enclosed holes whose '
                          'inscribed radius is below this many pixels; the leaf is always kept '
                          'and nothing is flash-filled. This is the app\'s Clean size '
                          f'(0-{CLEAN_SIZE_MAX}; 0 turns it off; default {CLEAN_SIZE_DEFAULT}).')
    run.add_argument('--save-axes', action='store_true',
                     help='Also save per-image length/width measurement-axis overlays for QC.')
    run.add_argument('--export', action='append', choices=('pre-cleanup', 'overlay', 'cutout', 'axes'),
                     default=[], help='Additional image export; repeat for multiple kinds.')
    run.add_argument('--pre-cleanup-methods', choices=('selected', 'threshold', 'birefnet', 'both'),
                     default='selected', help='Methods whose binary masks are saved before cleanup; '
                          'selected = every --mask-method method.')
    run.add_argument('--no-target-boxes', action='store_true',
                     help='Do not save new perspective-corrected target-box images.')
    run.add_argument('--no-masks', action='store_true',
                     help='Do not save the cleaned measurement masks.')
    run.add_argument('--no-failure-log', action='store_true',
                     help='Do not write the failures/warnings CSV.')

    app = sub.add_parser('app', help='Launch the Streamlit GUI.')
    app.add_argument('extra', nargs=argparse.REMAINDER,
                     help='Arguments forwarded verbatim to `streamlit run` (e.g. --server.port 8502).')

    fetch = sub.add_parser('fetch-weights', help='Download the model checkpoints.')
    fetch_target = fetch.add_mutually_exclusive_group()
    fetch_target.add_argument('--only', choices=('rf-detr', 'birefnet'), default=None,
                       help='Fetch only one checkpoint (default when no flag is given: rf-detr).')
    fetch_target.add_argument('--all', action='store_true',
                       help='Fetch both checkpoints (RF-DETR + the ~2.65 GB BiRefNet).')
    fetch.add_argument('--force', action='store_true', help='Re-download even if the file already exists.')
    fetch.add_argument('--source', choices=('auto', 'hf', 'lfs'), default='auto',
                       help='Download channel: auto picks Hugging Face if configured, else Git LFS; '
                            'hf/lfs force one channel (lfs requires a Git checkout with Git LFS installed, '
                            'and is the channel to use on networks that block huggingface.co).')

    sub.add_parser('doctor', help='Report weights, devices and QR decoders.')

    return parser


def _normalize_argv(argv):
    """Insert the default 'run' subcommand when none is given.

    So `mats -i in -o out` behaves as `mats run -i in -o out`, while an explicit
    subcommand or a bare `-h/--help` is left untouched.
    """
    if not argv:
        return argv
    first = argv[0]
    if first in _SUBCOMMANDS or first in ('-h', '--help'):
        return argv
    return [_RUN_SUBCOMMAND] + argv


def _measured_methods(args):
    """The methods this run measures with, in output order."""
    return ('threshold', 'birefnet') if args.mask_method == 'both' else (args.mask_method,)


def _pre_cleanup_methods(args):
    """The methods whose pre-cleanup masks this run exports."""
    if 'pre-cleanup' not in args.export:
        return ()
    if args.pre_cleanup_methods == 'selected':
        return _measured_methods(args)
    if args.pre_cleanup_methods == 'both':
        return ('threshold', 'birefnet')
    return (args.pre_cleanup_methods,)


def _uses_threshold(args):
    """Whether this run thresholds for measurements or a pre-cleanup export."""
    return 'threshold' in _measured_methods(args) + _pre_cleanup_methods(args)


def _print_run_banner(args, threshold_value):
    print(f"\nOutput mode: {args.output_mode}")
    print("Scale: independent per-axis pixels-per-cm (anisotropic)")
    if args.output_mode == "masks":
        print(f"Mask method: {args.mask_method}")
        print(f"Measurement source: {'pre-cleanup' if args.measure_pre_cleanup else 'cleaned'}")
        if args.measure_pre_cleanup:
            print(f"Clean margin: {args.clean_margin:g}% of the target box's shorter side")
            print(f"Stray gap: {args.stray_gap:g} x leaf bounding-box diagonal")
            print(f"Clean size: {args.clean_size} px" if args.clean_size else "Clean size: off")
        if _uses_threshold(args):
            if args.threshold_level == "auto":
                print("Threshold level: auto (Otsu's method)")
            elif isinstance(args.threshold_level, int):
                print(f"Threshold level: custom ({threshold_value})")
            else:
                print(f"Threshold level: {args.threshold_level} ({threshold_value})")
    else:
        print("Mask method and threshold level ignored because output mode is target-boxes.")
    print(f"Additional exports: {', '.join(args.export) if args.export else 'none'}")
    if args.output_mode == "target-boxes" and args.export:
        print("Segmentation exports ignored because output mode is target-boxes.")
    elif 'pre-cleanup' in args.export:
        print(f"Pre-cleanup methods: {args.pre_cleanup_methods}")


def _resolve_template_dims(args, lm):
    if args.sheet_dimensions is not None:
        from .template_layout import TemplateLayoutError, build_template_layout

        dims = lm.parse_template_dimensions(args.sheet_dimensions)
        if dims is None:
            _fail('--sheet-dimensions must look like "12x12in" or "30x30cm"')
        try:
            layout = build_template_layout(*dims)
        except TemplateLayoutError as exc:
            _fail(f"invalid --sheet-dimensions: {exc}")
        print(
            "\nPrinted sheet size provided: "
            f"{layout.page_width:g}x{layout.page_length:g}{layout.unit}; "
            "derived marker-centre calibration area: "
            f"{layout.observation_width:g}x{layout.observation_length:g}{layout.unit}"
        )
        return layout.calibration_dimensions

    if args.template_dimensions is None:
        print("\nNo sheet dimensions provided; will read calibration from QR when available.")
        return None
    dims = lm.parse_template_dimensions(args.template_dimensions)
    while dims is None:
        if not sys.stdin.isatty():
            _fail("legacy --template-dimensions must look like 10.5x9.5in or 27x24cm")
        print("\nLegacy calibration dimensions must use <width>x<height><unit> "
              "(e.g., 10.5x9.5in or 27x24cm).")
        args.template_dimensions = input("Please re-enter the legacy calibration area: ")
        dims = lm.parse_template_dimensions(args.template_dimensions)
    w, h, u = dims
    print(
        f"\nLegacy/custom calibration area provided: width={w}{u}, height={h}{u}"
    )
    return dims


def _resolve_input_dir(args):
    input_dir = args.input_dir
    if input_dir is not None:
        return input_dir
    if not sys.stdin.isatty():
        _fail("--input_dir is required in non-interactive mode")
    while True:
        input_dir = input("\nPlease enter the path to the input images to be analyzed: ")
        if os.path.exists(input_dir):
            return input_dir
        print("Invalid path, please try again.")


def _resolve_output_dir(args):
    if args.output_dir is not None:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print("\nOutput images will be written to:", output_dir)
        return output_dir
    if not sys.stdin.isatty():
        _fail("--output_dir is required in non-interactive mode "
              "(pass an empty run with --output-mode target-boxes if you only need the CSV)")
    answer = input("\nOutput directory (leave blank to skip saving images): ").strip()
    if not answer:
        return False
    os.makedirs(answer, exist_ok=True)
    print("Output images will be written to:", answer)
    return answer


def _make_progress_callback():
    """Return a progress callback backed by a tqdm bar (or None if unavailable)."""
    try:
        from tqdm import tqdm
    except Exception:
        return None
    state = {"bar": None}

    def _cb(info):
        bar = state["bar"]
        if bar is None:
            bar = state["bar"] = tqdm(total=info.get("total"), desc="Processing images")
        bar.n = info.get("processed", bar.n)
        bar.refresh()
        if info.get("processed") == info.get("total"):
            bar.close()

    return _cb


def _require_local_birefnet_for_run(args):
    """Stop once, before a batch, when optional BiRefNet is unavailable."""
    if args.output_mode != "masks" or (
        'birefnet' not in _measured_methods(args) + _pre_cleanup_methods(args)
    ):
        return
    from . import weights
    from .birefnet_runtime import require_birefnet_dependencies

    try:
        require_birefnet_dependencies()
        weights.require_local_weight("birefnet")
    except (FileNotFoundError, RuntimeError) as exc:
        _fail(str(exc))


def _cmd_run(args):
    from . import core as lm

    if args.output_mode != 'masks' and args.measure_pre_cleanup:
        _fail('--measure-pre-cleanup requires --output-mode masks')
    if args.pre_cleanup_methods != 'selected' and 'pre-cleanup' not in args.export:
        _fail('--pre-cleanup-methods requires --export pre-cleanup')
    if args.stray_gap != STRAY_GAP_DEFAULT and not args.measure_pre_cleanup:
        _fail('--stray-gap requires --measure-pre-cleanup')
    if args.clean_margin != CLEAN_MARGIN_DEFAULT and not args.measure_pre_cleanup:
        _fail('--clean-margin requires --measure-pre-cleanup')
    if args.clean_size != CLEAN_SIZE_DEFAULT and not args.measure_pre_cleanup:
        _fail('--clean-size requires --measure-pre-cleanup')
    _require_local_birefnet_for_run(args)
    template_dims = _resolve_template_dims(args, lm)
    threshold_value = threshold_value_for(args.threshold_level)
    _print_run_banner(args, threshold_value)

    results_path = args.results_path or os.path.join(os.getcwd(), "leaf_morpho_results.csv")
    if args.output_mode == "masks" and len(_measured_methods(args)) > 1:
        print("\nMeasurement CSVs will be written to:")
        for method in _measured_methods(args):
            print(f"  {method}: {lm.method_suffixed_path(results_path, method)}")
    else:
        print("\nMeasurement CSV will be written to:", results_path)

    input_dir = _resolve_input_dir(args)
    input_images = lm.get_input_images(input_dir)
    if not input_images:
        _fail(f"No images found in {input_dir}")
    print(f"Found {len(input_images)} image(s).")

    output_dir = _resolve_output_dir(args)
    export_options = {
        'target_boxes': not args.no_target_boxes,
        'cleaned_masks': not args.no_masks,
        'pre_cleanup_methods': _pre_cleanup_methods(args),
        'overlay': 'overlay' in args.export,
        'cutout': 'cutout' in args.export,
        'axes': args.save_axes or 'axes' in args.export,
    }

    result = lm.run_leaf_morpho_batch(
        input_images=input_images,
        output_dir=output_dir,
        results_path=results_path,
        template_dimensions=template_dims,
        output_mode=args.output_mode,
        mask_method=args.mask_method,
        threshold_value=threshold_value,
        workers=args.workers,
        write_failures=not args.no_failure_log,
        compact_csv=(args.csv_schema == "compact"),
        results_unit=args.results_unit,
        save_measurement_axes=args.save_axes,
        serialize_model_inference=False,
        progress_callback=_make_progress_callback(),
        export_options=export_options,
        measurement_source='pre-cleanup' if args.measure_pre_cleanup else 'cleaned',
        stray_gap=args.stray_gap,
        clean_margin=args.clean_margin,
        clean_size=args.clean_size,
    )

    print(f"\nDone. {result['succeeded']} succeeded, {result['failed']} failed "
          f"({result['workers']} worker(s): {result['worker_reason']}).")
    if len(result["methods"]) > 1:
        for method in result["methods"]:
            outcome = result["by_method"][method]
            print(f"{method}: {outcome['succeeded']} succeeded, {outcome['failed']} failed.")
            print(f"  Measurement CSV written to: {outcome['results_path']}")
            if outcome["failure_report_path"]:
                print(f"  Failure report written to: {outcome['failure_report_path']}")
        return 0
    print(f"Measurement CSV written to: {result['results_path']}")
    if result.get("failure_report_path"):
        print(f"Failure report written to: {result['failure_report_path']}")
    return 0


def _cmd_app(args):
    from .app.launcher import launch
    return launch(args.extra or [])


def _resolve_fetch_only(args):
    """Translate --only/--all into the target passed to weights.fetch(only=...).

    Bare `mats fetch-weights` fetches RF-DETR only (mandatory, small); BiRefNet
    (optional, ~2.65 GB) needs an explicit `--only birefnet` or `--all`.
    """
    if args.only:
        return args.only
    return None if args.all else "rf-detr"


def _cmd_fetch_weights(args):
    from . import weights
    return weights.fetch(only=_resolve_fetch_only(args), force=args.force, source=args.source)


def _cmd_doctor(_args):
    from . import weights
    return weights.doctor()


_DISPATCH = {
    "run": _cmd_run,
    "app": _cmd_app,
    "fetch-weights": _cmd_fetch_weights,
    "doctor": _cmd_doctor,
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(_normalize_argv(argv))
    if not args.command:
        parser.print_help()
        return 1
    return _DISPATCH[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
