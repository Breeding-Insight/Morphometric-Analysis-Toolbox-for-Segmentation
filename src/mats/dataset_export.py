"""Build training datasets from app or CLI target-box and mask pairs.

Only standard-library modules are imported at module load so this remains safe
for the package's dependency-light CLI and test environments.
"""

from __future__ import annotations

import csv
import io
import json
import random
import zipfile
from pathlib import Path


SPLITS = ("train", "val", "test")
FORMATS = ("png", "yolo-seg", "yolo-detect", "coco")


def pairs_from_manifest(
    artifacts, input_images=(), method=None, measurement_source="cleaned",
    preview_artifacts=(),
):
    """Pair each method's measured mask with its corrected target-box image."""
    if measurement_source not in {"cleaned", "pre-cleanup"}:
        raise ValueError("unknown measurement source")
    by_sample = {}
    raw_kinds = {"preview_raw_mask", "pre_cleanup"}
    mask_kinds = (
        {"preview_mask"} if measurement_source == "pre-cleanup" else {"mask", "preview_mask"}
    )
    for item in (*artifacts, *preview_artifacts):
        sample_id = item.get("sample_id")
        kind = item["kind"]
        if sample_id is None or kind not in {
            "target_box", "preview_target_box", *mask_kinds, *raw_kinds,
        }:
            continue
        if method is not None and kind in {*mask_kinds, *raw_kinds}:
            if item.get("method") != method:
                continue
        by_sample.setdefault(sample_id, {"sample_id": sample_id, "target_box": None, "mask": None})
        if kind in raw_kinds:
            by_sample[sample_id]["raw_mask"] = item["path"]
        elif kind in mask_kinds:
            by_sample[sample_id]["mask"] = item["path"]
        else:
            by_sample[sample_id]["target_box"] = item["path"]
    for path in input_images:
        source = Path(path)
        if source.stem.endswith("_target_box") and source.is_file():
            sample_id = source.stem[:-len("_target_box")]
            if sample_id in by_sample and by_sample[sample_id]["target_box"] is None:
                by_sample[sample_id]["target_box"] = str(source)
    if measurement_source == "pre-cleanup":
        for pair in by_sample.values():
            pair["mask_source"] = measurement_source
    return list(by_sample.values())


def split_counts(total, percentages):
    """Largest-remainder counts; sum is always exactly ``total``."""
    if len(percentages) != 3 or any(not 0 <= value <= 100 for value in percentages):
        raise ValueError("Train, validation, and test must each be between 0 and 100%.")
    if sum(percentages) != 100:
        raise ValueError("Train, validation, and test percentages must total 100%.")
    exact = [total * value / 100 for value in percentages]
    counts = [int(value) for value in exact]
    order = sorted(range(3), key=lambda index: (-(exact[index] - counts[index]), index))
    for index in order[:total - sum(counts)]:
        counts[index] += 1
    return dict(zip(SPLITS, counts))


def read_group_csv(content):
    """Parse optional sample_id,group_id CSV without guessing group boundaries."""
    if not content:
        return {}
    try:
        rows = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
        if not rows.fieldnames or not {"sample_id", "group_id"} <= set(rows.fieldnames):
            raise ValueError("Group CSV needs sample_id and group_id columns.")
        groups = {}
        for row in rows:
            sample_id = (row.get("sample_id") or "").strip()
            group_id = (row.get("group_id") or "").strip()
            if not sample_id or not group_id:
                raise ValueError("Every group CSV row needs a sample_id and group_id.")
            if sample_id in groups and groups[sample_id] != group_id:
                raise ValueError(f"Conflicting group IDs for {sample_id}.")
            groups[sample_id] = group_id
        return groups
    except UnicodeDecodeError as exc:
        raise ValueError("Group CSV must be UTF-8 text.") from exc


def validate_pairs(pairs, mask_source="measured", dataset_format="png"):
    """Read each image once; return usable pairs and explicit exclusions."""
    import cv2
    import numpy as np

    if mask_source not in {"measured", "raw"}:
        raise ValueError("Unknown mask source.")
    seen = set()
    valid, excluded = [], []
    for pair in pairs:
        sample_id = str(pair.get("sample_id", ""))
        image_path = pair.get("target_box")
        mask_path = pair.get("mask" if mask_source == "measured" else "raw_mask")
        reason = None
        if not sample_id or sample_id in seen:
            reason = "missing or duplicate sample ID"
        elif not image_path or not mask_path:
            reason = "image or selected mask is unavailable"
        elif not Path(image_path).is_file() or not Path(mask_path).is_file():
            reason = "image or selected mask file is missing"
        else:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                reason = "image or mask could not be read"
            elif image.shape[:2] != mask.shape[:2]:
                reason = "image and mask dimensions differ"
            elif not np.any(mask):
                reason = "mask is empty"
            elif dataset_format == "yolo-seg" and not _yolo_segments(
                (mask > 0).astype("uint8")
            )[0]:
                reason = "mask has no usable polygon"
        seen.add(sample_id)
        if reason:
            excluded.append({"sample_id": sample_id, "reason": reason})
        else:
            valid.append({
                "sample_id": sample_id, "image": Path(image_path), "mask": Path(mask_path),
                "width": image.shape[1], "height": image.shape[0],
            })
    return valid, excluded


def assign_splits(pairs, percentages=(70, 20, 10), seed=42, groups=None):
    """Deterministically place each group wholly in one split."""
    targets = split_counts(len(pairs), percentages)
    groups = groups or {}
    buckets = {}
    for pair in pairs:
        sample_id = pair["sample_id"]
        group_id = (("group", groups[sample_id]) if sample_id in groups
                    else ("sample", sample_id))
        buckets.setdefault(group_id, []).append(pair)
    items = sorted(buckets.items())
    random.Random(seed).shuffle(items)
    counts = dict.fromkeys(SPLITS, 0)
    assignments = {}
    for _, members in items:
        size = len(members)
        choice = min(SPLITS, key=lambda split: (
            sum(abs(targets[name] - counts[name] - (size if name == split else 0))
                for name in SPLITS),
            -(targets[split] - counts[split]),
            SPLITS.index(split),
        ))
        counts[choice] += size
        for pair in members:
            assignments[pair["sample_id"]] = choice
    return assignments, counts


def _encoded_png(mask):
    import cv2

    ok, data = cv2.imencode(".png", mask)
    if not ok:
        raise ValueError("Could not encode a mask as PNG.")
    return data.tobytes()


def _binary_mask(path):
    import cv2

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {path.name}")
    return (mask > 0).astype("uint8")


def _yolo_segments(mask):
    import cv2

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return [], False
    height, width = mask.shape
    lines = []
    holes = False
    for contour, relation in zip(contours, hierarchy[0]):
        if relation[3] != -1:
            holes = True
            continue
        points = contour.reshape(-1, 2)
        if len(points) < 3 or cv2.contourArea(contour) == 0:
            continue
        coords = " ".join(f"{x / width:.6f} {y / height:.6f}" for x, y in points)
        lines.append(f"0 {coords}")
    return lines, holes


def _bbox(mask):
    import cv2

    points = cv2.findNonZero(mask)
    if points is None:
        raise ValueError("Mask is empty.")
    return cv2.boundingRect(points)


def _coco_rle(mask):
    """Uncompressed COCO RLE in column-major order, preserving mask holes."""
    import numpy as np

    pixels = mask.T.reshape(-1)
    boundaries = np.flatnonzero(pixels[1:] != pixels[:-1]) + 1
    counts = np.diff(np.concatenate(([0], boundaries, [pixels.size]))).tolist()
    if pixels[0]:
        counts.insert(0, 0)
    return {"size": list(mask.shape), "counts": counts}


def write_dataset_zip(
    pairs, dest_path, *, dataset_format="png", percentages=(70, 20, 10), seed=42,
    method="threshold", mask_source="measured", groups=None,
):
    """Validate and stream one method's dataset into a ZIP; return its manifest."""
    if dataset_format not in FORMATS:
        raise ValueError("Unknown dataset format.")
    groups = groups or {}
    valid, excluded = validate_pairs(pairs, mask_source, dataset_format)
    if not valid:
        raise ValueError("No usable image and mask pairs were found for this method.")
    unknown_groups = set(groups) - {pair["sample_id"] for pair in pairs}
    if unknown_groups:
        raise ValueError("Group CSV contains unknown sample IDs: " +
                         ", ".join(sorted(unknown_groups)[:5]))
    assignments, counts = assign_splits(valid, percentages, seed, groups)
    manifest = {
        "format": dataset_format, "label_method": method, "mask_source": mask_source,
        "label_origin": "MATS generated masks; review labels before model training",
        "class_names": ["leaf"], "percentages": dict(zip(SPLITS, percentages)),
        "seed": seed, "counts": counts, "samples": [], "excluded": excluded,
        "conversion_notes": [],
    }
    annotations = {split: {"images": [], "annotations": [], "categories": [
        {"id": 1, "name": "leaf", "supercategory": "plant"}
    ]} for split in SPLITS}
    used_names = set()
    next_annotation_id = 1
    try:
        with zipfile.ZipFile(dest_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for split in SPLITS:
                archive.writestr(f"images/{split}/", b"")
                if dataset_format.startswith("yolo"):
                    archive.writestr(f"labels/{split}/", b"")
                elif dataset_format == "png":
                    archive.writestr(f"masks/{split}/", b"")
            for image_id, pair in enumerate(valid, 1):
                sample_id = pair["sample_id"]
                split = assignments[sample_id]
                # Never put untrusted sample IDs directly into ZIP member paths.
                stem = f"sample_{image_id:06d}"
                image_ext = pair["image"].suffix.lower()
                if image_ext not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
                    raise ValueError(f"Unsupported target-box image type: {image_ext}")
                image_name = stem + image_ext
                image_member = f"images/{split}/{image_name}"
                if image_member in used_names:
                    raise ValueError("Duplicate image name in dataset.")
                used_names.add(image_member)
                archive.write(pair["image"], image_member)
                mask = _binary_mask(pair["mask"])
                if mask.shape != (pair["height"], pair["width"]) or not mask.any():
                    raise ValueError(f"Mask changed during export: {sample_id}")
                entry = {
                    "sample_id": sample_id, "group_id": groups.get(sample_id, sample_id),
                    "split": split, "image": image_member,
                    "width": pair["width"], "height": pair["height"],
                }
                if dataset_format == "png":
                    member = f"masks/{split}/{stem}.png"
                    archive.writestr(member, _encoded_png(mask * 255))
                    entry["label"] = member
                elif dataset_format == "yolo-seg":
                    member = f"labels/{split}/{stem}.txt"
                    lines, has_holes = _yolo_segments(mask)
                    if not lines:
                        raise ValueError(f"Mask has no usable polygon: {sample_id}")
                    archive.writestr(member, "\n".join(lines) + "\n")
                    entry["label"] = member
                    if has_holes:
                        manifest["conversion_notes"].append(
                            f"{sample_id}: YOLO polygons cannot preserve mask holes."
                        )
                    if len(lines) > 1:
                        manifest["conversion_notes"].append(
                            f"{sample_id}: disconnected regions became separate YOLO segments."
                        )
                elif dataset_format == "yolo-detect":
                    member = f"labels/{split}/{stem}.txt"
                    x, y, width, height = _bbox(mask)
                    archive.writestr(member, (
                        f"0 {(x + width / 2) / pair['width']:.6f} "
                        f"{(y + height / 2) / pair['height']:.6f} "
                        f"{width / pair['width']:.6f} {height / pair['height']:.6f}\n"
                    ))
                    entry["label"] = member
                else:
                    x, y, width, height = _bbox(mask)
                    annotations[split]["images"].append({
                        "id": image_id, "file_name": image_name,
                        "width": pair["width"], "height": pair["height"],
                    })
                    annotations[split]["annotations"].append({
                        "id": next_annotation_id, "image_id": image_id, "category_id": 1,
                        "segmentation": _coco_rle(mask), "area": int(mask.sum()),
                        "bbox": [x, y, width, height], "iscrowd": 0,
                    })
                    next_annotation_id += 1
                    entry["label"] = f"annotations/instances_{split}.json"
                manifest["samples"].append(entry)
            if dataset_format.startswith("yolo"):
                archive.writestr("data.yaml", (
                    "train: images/train\nval: images/val\ntest: images/test\n"
                    "names:\n  0: leaf\n"
                ))
            if dataset_format == "coco":
                for split, data in annotations.items():
                    archive.writestr(f"annotations/instances_{split}.json", json.dumps(data))
            archive.writestr("manifest.json", json.dumps(manifest, indent=2))
    except Exception:
        Path(dest_path).unlink(missing_ok=True)
        raise
    return manifest
