"""Re-measure one threshold specimen and replace its saved output atomically."""

import csv
import copy
import io
import json
import os
import shutil
import tempfile
from pathlib import Path

import cv2

from mats.mask_cleanup import raw_measurement_mask
from mats.scaling import (
    NA_VALUE,
    compact_measurement_row,
    converted_measurement_row,
    length_conversion_factor,
    result_measurement_fieldnames,
    validate_results_unit,
)
from mats.mask_settings import (
    CLEAN_MARGIN_DEFAULT,
    CLEAN_SIZE_DEFAULT,
    STRAY_GAP_DEFAULT,
    checked_clean_margin,
    checked_clean_size,
    checked_stray_gap,
)


def _results_csv(run):
    path = Path(run["by_method"]["threshold"]["results_path"])
    if not path.is_file():
        raise ValueError("The Classic thresholding results CSV is unavailable.")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    unit = validate_results_unit(run.get("results_unit", "cm"))
    full_area = result_measurement_fieldnames(unit)[0]
    compact_area = result_measurement_fieldnames(unit, compact=True)[0]
    if full_area in fields:
        schema = "full"
    elif compact_area in fields:
        schema = "compact"
    else:
        raise ValueError("The results CSV has no recognized measurement columns.")
    return path, fields, rows, unit, schema


def _sample_row(rows, sample_id):
    matches = [index for index, row in enumerate(rows) if row.get("sample_id") == sample_id]
    if len(matches) != 1:
        raise ValueError("The selected specimen must have exactly one results row.")
    return matches[0]


def _scale_axes(run, sample_id, csv_row, unit):
    saved = run.get("scale_axes_by_sample", {}).get("threshold", {}).get(sample_id)
    if saved is not None:
        axes = tuple(float(value) for value in saved)
    else:
        factor = length_conversion_factor(unit)
        try:
            axes = (
                float(csv_row[f"px_per_{unit}_width"]) * factor,
                float(csv_row[f"px_per_{unit}_height"]) * factor,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Calibration for this specimen is unavailable; rerun analysis before adjusting it."
            ) from exc
    if len(axes) != 2 or not all(0 < value < float("inf") for value in axes):
        raise ValueError("Calibration for this specimen is invalid.")
    return axes


def measure_threshold_adjustment(
    run, pair, cutoff, *, remove_fill=False, clean_margin=None, stray_gap=None,
    clean_size=None,
):
    """Compute the exact saved-mask measurement for one selected specimen.

    ``clean_margin``, ``stray_gap``, and ``clean_size`` default to the run's
    values. The margin and gap apply to pre-cleanup runs and a clean size above
    0, and the margin also to Remove flashfill. A clean size above 0 measures
    Clean image, the mask its preview shows; in a cleaned run it also replaces
    MATS cleanup, and Remove flashfill, in the saved mask.
    """
    from mats import core

    if not isinstance(cutoff, int) or not 0 <= cutoff <= 255:
        raise ValueError("Threshold cutoff must be an integer from 0 to 255.")
    sample_id = str(pair["sample_id"])
    if not sample_id or Path(sample_id).name != sample_id:
        raise ValueError("Invalid specimen ID.")
    source = run.get("measurement_source", "cleaned")
    if source not in {"cleaned", "pre-cleanup"}:
        raise ValueError("Unknown measurement-mask source.")
    target_path = pair.get("target_box")
    target = cv2.imread(str(target_path), cv2.IMREAD_COLOR) if target_path else None
    if target is None:
        raise ValueError("The selected specimen's target-box image is unavailable.")

    results_path, fields, rows, unit, schema = _results_csv(run)
    row_index = _sample_row(rows, sample_id)
    if rows[row_index].get(result_measurement_fieldnames(unit, schema == "compact")[0]) in {
        NA_VALUE, "", None,
    }:
        raise ValueError("Only successfully measured specimens can be adjusted.")
    axes = _scale_axes(run, sample_id, rows[row_index], unit)
    margin = checked_clean_margin(
        run.get("clean_margin", CLEAN_MARGIN_DEFAULT) if clean_margin is None else clean_margin
    )
    gap = checked_stray_gap(
        run.get("stray_gap", STRAY_GAP_DEFAULT) if stray_gap is None else stray_gap
    )
    size = checked_clean_size(
        run.get("clean_size", CLEAN_SIZE_DEFAULT) if clean_size is None else clean_size
    )
    pre_cleanup = source == "pre-cleanup"
    raw = core.threshold_mask(target, cutoff)
    unfilled = raw_measurement_mask(raw, margin, gap, size) if pre_cleanup or size else None
    # A pre-cleanup run keeps its cleaned-mask export and measures the unfilled
    # mask; a cleaned run saves and measures one mask, Clean image when on.
    fill_holes = not remove_fill and (pre_cleanup or not size)
    if size and not pre_cleanup:
        cleaned = unfilled
    elif remove_fill:
        cleaned = core.unfilled_leaf_mask(raw, margin)
    else:
        cleaned = core.clean_leaf_mask(raw.copy())
    measurement_mask = unfilled if pre_cleanup else cleaned
    measurement = core.measurement_row_from_mask(
        sample_id, measurement_mask, *axes, measurement_source=source
    )
    if measurement["leaf_area_cm2"] == NA_VALUE:
        raise ValueError("This threshold produces no measurable leaf mask.")
    converted = (
        compact_measurement_row(measurement, unit)
        if schema == "compact" else converted_measurement_row(measurement, unit)
    )
    return {
        "sample_id": sample_id,
        "results_path": results_path,
        "fields": fields,
        "rows": rows,
        "row_index": row_index,
        "unit": unit,
        "schema": schema,
        "target": target,
        "raw": raw,
        "cleaned": cleaned,
        "measurement_mask": measurement_mask,
        "converted": converted,
        "clean_margin": margin,
        "stray_gap": gap,
        "clean_size": size,
        "fill_holes": fill_holes,
    }


def _encoded_image(path, image):
    suffix = path.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(f"Unsupported saved image format: {suffix}")
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"Could not encode {path.name}")
    return encoded.tobytes()


def _commit_files(changes):
    """Stage every file first; restore earlier files if a later replacement fails."""
    staged = []
    replaced = []
    try:
        for destination, content in changes.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".mats_adjust_", dir=destination.parent)
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
            staged.append((destination, Path(temporary)))
        for destination, temporary in staged:
            backup = None
            if destination.exists():
                fd, backup_name = tempfile.mkstemp(
                    prefix=".mats_adjust_backup_", dir=destination.parent
                )
                os.close(fd)
                backup = Path(backup_name)
                shutil.copy2(destination, backup)
            os.replace(temporary, destination)
            replaced.append((destination, backup))
    except Exception:
        for destination, backup in reversed(replaced):
            if backup is None:
                destination.unlink(missing_ok=True)
            else:
                os.replace(backup, destination)
        raise
    finally:
        for _, temporary in staged:
            temporary.unlink(missing_ok=True)
        for _, backup in replaced:
            if backup is not None:
                backup.unlink(missing_ok=True)


def apply_threshold_adjustment(
    run, pair, cutoff, *, remove_fill=False, clean_margin=None, stray_gap=None,
    clean_size=None,
):
    """Overwrite only this threshold specimen's measurements and dependent images."""
    from mats import core

    prepared = measure_threshold_adjustment(
        run, pair, cutoff, remove_fill=remove_fill,
        clean_margin=clean_margin, stray_gap=stray_gap, clean_size=clean_size,
    )
    sample_id = prepared["sample_id"]
    output_dir = Path(run["output_path"])
    results_path = prepared["results_path"]
    if not results_path.resolve().is_relative_to(output_dir.resolve()):
        raise ValueError("The results CSV is outside this run's output folder.")
    suffix = "_threshold" if len(run["mask_methods"]) > 1 else ""
    artifacts = run.setdefault("artifacts", [])

    def artifact_path(kind):
        return next((
            Path(item["path"]) for item in artifacts
            if item.get("sample_id") == sample_id and item.get("kind") == kind
            and item.get("method") in {None, "threshold"}
        ), None)

    pre_cleanup = run.get("measurement_source", "cleaned") == "pre-cleanup"
    mask_path = artifact_path("mask") or output_dir / f"{sample_id}_mask{suffix}.png"
    raw_path = artifact_path("pre_cleanup")
    if pre_cleanup and raw_path is None:
        raw_path = output_dir / f"{sample_id}_mask_precleanup_threshold.png"
    changes = {mask_path: _encoded_image(mask_path, prepared["cleaned"])}
    if raw_path is not None:
        changes[raw_path] = _encoded_image(raw_path, prepared["raw"])
    if pair.get("raw_mask"):
        preview_raw_path = Path(pair["raw_mask"])
        changes[preview_raw_path] = _encoded_image(preview_raw_path, prepared["raw"])
    # A pre-cleanup run measured its raw mask minus stray pieces (and, at a clean
    # size, small specks and holes); the preview mask holds that, while the
    # pre-cleanup export stays the raw mask.
    measured_path = None
    if pre_cleanup and pair.get("mask") and Path(pair["mask"]) != raw_path:
        measured_path = Path(pair["mask"])
        changes[measured_path] = _encoded_image(measured_path, prepared["measurement_mask"])

    dependent = {
        "overlay": lambda: core.build_overlay_image(prepared["target"], prepared["measurement_mask"]),
        "cutout": lambda: core.build_cutout_image(prepared["target"], prepared["measurement_mask"]),
        "axes": lambda: core.draw_measurement_axes(
            prepared["target"], prepared["measurement_mask"],
            run.get("measurement_source", "cleaned"),
        ),
    }
    for kind, build in dependent.items():
        path = artifact_path(kind)
        if path is not None:
            image = build()
            if image is None:
                raise ValueError(f"Could not regenerate {kind} for this specimen.")
            changes[path] = _encoded_image(path, image)

    converted = prepared["converted"]
    fields_to_update = (
        result_measurement_fieldnames(prepared["unit"], prepared["schema"] == "compact")
    )
    if prepared["schema"] == "full":
        fields_to_update += [
            f"px_per_{prepared['unit']}_width",
            f"px_per_{prepared['unit']}_height",
            "scale_aspect_ratio",
        ]
    prepared["rows"][prepared["row_index"]].update({
        field: str(converted[field]) for field in fields_to_update
    })
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=prepared["fields"])
    writer.writeheader()
    writer.writerows(prepared["rows"])
    changes[results_path] = csv_buffer.getvalue().encode("utf-8")

    metadata_path = Path(f"{results_path}.meta.json")
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Could not read the run's measurement metadata.") from exc
    else:
        metadata = {
            "measurement_source": run.get("measurement_source", "cleaned"),
            "mask_method": "threshold",
            "results_unit": prepared["unit"],
            "csv_schema": prepared["schema"],
        }
        if pre_cleanup:
            metadata["clean_margin"] = run.get("clean_margin", CLEAN_MARGIN_DEFAULT)
            metadata["stray_gap"] = run.get("stray_gap", STRAY_GAP_DEFAULT)
            metadata["clean_size"] = run.get("clean_size", CLEAN_SIZE_DEFAULT)
    # Record the cleanup settings only where they shaped the saved mask.
    clean_on = bool(prepared["clean_size"])
    adjustment = {"cutoff": cutoff, "fill_holes": prepared["fill_holes"]}
    if pre_cleanup or remove_fill or clean_on:
        adjustment["clean_margin"] = prepared["clean_margin"]
    if pre_cleanup or clean_on:
        adjustment["stray_gap"] = prepared["stray_gap"]
        adjustment["clean_size"] = prepared["clean_size"]
    metadata.setdefault("threshold_adjustments", {})[sample_id] = adjustment
    changes[metadata_path] = (json.dumps(metadata, indent=2) + "\n").encode("utf-8")

    _commit_files(changes)
    if artifact_path("mask") is None:
        artifacts.append({
            "path": str(mask_path), "sample_id": sample_id,
            "kind": "mask", "method": "threshold",
        })
    if raw_path is not None and artifact_path("pre_cleanup") is None:
        artifacts.append({
            "path": str(raw_path), "sample_id": sample_id,
            "kind": "pre_cleanup", "method": "threshold",
        })
    if not any(item.get("kind") == "results_metadata" and item.get("method") == "threshold"
               for item in artifacts):
        artifacts.append({
            "path": str(metadata_path), "sample_id": None,
            "kind": "results_metadata", "method": "threshold",
        })
    if pre_cleanup:
        pair["raw_mask"] = str(raw_path)
        pair["mask"] = None if measured_path is None else str(measured_path)
        pair["mask_source"] = "pre-cleanup"
    else:
        pair["mask"] = str(mask_path)
    run.setdefault("threshold_adjustments", {})[sample_id] = dict(adjustment)
    return converted


def apply_threshold_adjustments(
    run, pairs, cutoff, *, remove_fill=False, clean_margin=None, stray_gap=None,
    clean_size=None,
):
    """Apply one set of controls to marked specimens, restoring all on failure."""
    pairs = list(pairs)
    sample_ids = [str(pair["sample_id"]) for pair in pairs]
    if not pairs or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Mark one or more distinct specimens before saving.")
    # Validate all measurements and target images before touching any output.
    for pair in pairs:
        measure_threshold_adjustment(
            run, pair, cutoff, remove_fill=remove_fill,
            clean_margin=clean_margin, stray_gap=stray_gap, clean_size=clean_size,
        )

    output_dir = Path(run["output_path"])
    results_path = Path(run["by_method"]["threshold"]["results_path"])
    suffix = "_threshold" if len(run["mask_methods"]) > 1 else ""
    candidates = {results_path, Path(f"{results_path}.meta.json")}
    for item in run.get("artifacts", []):
        if (item.get("sample_id") in sample_ids
                and item.get("method") in {None, "threshold"}
                and item.get("kind") in {"mask", "pre_cleanup", "overlay", "cutout", "axes"}):
            candidates.add(Path(item["path"]))
    for pair in pairs:
        sample_id = str(pair["sample_id"])
        candidates.add(output_dir / f"{sample_id}_mask{suffix}.png")
        if run.get("measurement_source") == "pre-cleanup":
            candidates.add(output_dir / f"{sample_id}_mask_precleanup_threshold.png")
        for key in ("mask", "raw_mask"):
            if pair.get(key):
                candidates.add(Path(pair[key]))

    original_artifacts = copy.deepcopy(run.get("artifacts", []))
    original_adjustments = copy.deepcopy(run.get("threshold_adjustments", {}))
    original_pairs = [dict(pair) for pair in pairs]
    with tempfile.TemporaryDirectory(prefix="mats_adjust_batch_") as backup_dir:
        backups = {}
        for index, path in enumerate(candidates):
            backup = Path(backup_dir) / str(index)
            if path.exists():
                shutil.copy2(path, backup)
                backups[path] = backup
            else:
                backups[path] = None
        try:
            for pair in pairs:
                apply_threshold_adjustment(
                    run, pair, cutoff, remove_fill=remove_fill,
                    clean_margin=clean_margin, stray_gap=stray_gap, clean_size=clean_size,
                )
        except Exception:
            for path, backup in backups.items():
                if backup is None:
                    path.unlink(missing_ok=True)
                else:
                    shutil.copy2(backup, path)
            run["artifacts"] = original_artifacts
            run["threshold_adjustments"] = original_adjustments
            for pair, original in zip(pairs, original_pairs):
                pair.clear()
                pair.update(original)
            raise
    return sample_ids
