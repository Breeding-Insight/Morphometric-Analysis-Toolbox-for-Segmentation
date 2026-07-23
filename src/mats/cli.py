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

_RUN_SUBCOMMAND = "run"
_SUBCOMMANDS = {"run", "app", "fetch-weights", "doctor"}


def _fail(message, code=2):
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


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
    run.add_argument('-t', '--template_dimensions', '--template-dimensions', dest='template_dimensions',
                     type=str, default=None,
                     help='Physical template size as <width>x<height><unit>, e.g. 10.5x9.5in or 27x24cm.')
    run.add_argument('--output-mode', choices=('masks', 'target-boxes'), default='masks',
                     help='Produce segmentation masks, or only perspective-corrected target boxes.')
    run.add_argument('--mask-method', choices=('birefnet', 'threshold'), default='birefnet',
                     help='Mask method when --output-mode masks. birefnet is accurate but heavy; '
                          'threshold is fast for clean backgrounds.')
    run.add_argument('--threshold-level', choices=('auto', 'low', 'medium', 'high'), default='auto',
                     help="For --mask-method threshold: auto uses Otsu (recommended); "
                          "low=100, medium=125, high=150.")
    run.add_argument('--scale-axis', choices=('average', 'width', 'height'), default='average',
                     help='Deprecated; the CSV always reports mean, width-based and height-based conversions.')
    run.add_argument('--csv-schema', choices=('full', 'compact'), default='full',
                     help='full = all three scale conventions (research schema); '
                          'compact = sample_id, area_cm2, height_cm, length_cm.')
    run.add_argument('--save-axes', action='store_true',
                     help='Also save per-image length/width measurement-axis overlays for QC.')

    app = sub.add_parser('app', help='Launch the Streamlit GUI.')
    app.add_argument('extra', nargs=argparse.REMAINDER,
                     help='Arguments forwarded verbatim to `streamlit run` (e.g. --server.port 8502).')

    fetch = sub.add_parser('fetch-weights', help='Download the model checkpoints.')
    fetch.add_argument('--only', choices=('rf-detr', 'birefnet'), default=None,
                       help='Fetch only one checkpoint (default: both).')
    fetch.add_argument('--force', action='store_true', help='Re-download even if the file already exists.')

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


def _print_run_banner(args, threshold_value):
    print(f"\nOutput mode: {args.output_mode}")
    print("Scale columns: mean, width-based, height-based")
    if args.output_mode == "masks":
        print(f"Mask method: {args.mask_method}")
        if args.mask_method == "threshold":
            if args.threshold_level == "auto":
                print("Threshold level: auto (Otsu's method)")
            else:
                print(f"Threshold level: {args.threshold_level} ({threshold_value})")
    else:
        print("Mask method and threshold level ignored because output mode is target-boxes.")


def _resolve_template_dims(args, lm):
    if args.template_dimensions is None:
        print("\nNo template dimensions provided; will read size/orientation from QR when available.")
        return None
    dims = lm.parse_template_dimensions(args.template_dimensions)
    while dims is None:
        if not sys.stdin.isatty():
            _fail("--template_dimensions must look like 10.5x9.5in or 27x24cm")
        print("\nTemplate dimensions must use the format <width>x<height><unit> "
              "(e.g., 10.5x9.5in or 27x24cm).")
        args.template_dimensions = input("Please re-enter the template dimensions: ")
        dims = lm.parse_template_dimensions(args.template_dimensions)
    w, h, u = dims
    print(f"\nTemplate dimensions provided: width={w}{u}, height={h}{u}")
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


def _cmd_run(args):
    from . import core as lm

    template_dims = _resolve_template_dims(args, lm)
    threshold_value = lm.THRESHOLD_LEVELS[args.threshold_level]
    _print_run_banner(args, threshold_value)

    results_path = args.results_path or os.path.join(os.getcwd(), "leaf_morpho_results.csv")
    print("\nMeasurement CSV will be written to:", results_path)

    input_dir = _resolve_input_dir(args)
    input_images = lm.get_input_images(input_dir)
    if not input_images:
        _fail(f"No images found in {input_dir}")
    print(f"Found {len(input_images)} image(s).")

    output_dir = _resolve_output_dir(args)

    result = lm.run_leaf_morpho_batch(
        input_images=input_images,
        output_dir=output_dir,
        results_path=results_path,
        template_dimensions=template_dims,
        output_mode=args.output_mode,
        mask_method=args.mask_method,
        threshold_value=threshold_value,
        scale_axis=args.scale_axis,
        workers=args.workers,
        write_failures=True,
        compact_csv=(args.csv_schema == "compact"),
        save_measurement_axes=args.save_axes,
        serialize_model_inference=False,
        progress_callback=_make_progress_callback(),
    )

    print(f"\nDone. {result['succeeded']} succeeded, {result['failed']} failed "
          f"({result['workers']} worker(s): {result['worker_reason']}).")
    print(f"Measurement CSV written to: {result['results_path']}")
    if result.get("failure_report_path"):
        print(f"Failure report written to: {result['failure_report_path']}")
    return 0


def _cmd_app(args):
    from .app.launcher import launch
    return launch(args.extra or [])


def _cmd_fetch_weights(args):
    from . import weights
    return weights.fetch(only=args.only, force=args.force)


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
