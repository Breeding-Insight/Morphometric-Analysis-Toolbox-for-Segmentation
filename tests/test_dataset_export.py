"""Training exports preserve pair alignment and deterministic split assignments."""

import json
import zipfile

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from mats.dataset_export import (
    _coco_rle, assign_splits, read_group_csv, split_counts, write_dataset_zip,
)


def _pair(folder, sample_id, *, hole=False):
    image = np.full((16, 20, 3), 220, dtype=np.uint8)
    mask = np.zeros((16, 20), dtype=np.uint8)
    mask[3:13, 4:15] = 255
    if hole:
        mask[6:9, 8:11] = 0
    image_path = folder / f"{sample_id}.png"
    mask_path = folder / f"{sample_id}_mask.png"
    assert cv2.imwrite(str(image_path), image)
    assert cv2.imwrite(str(mask_path), mask)
    return {"sample_id": sample_id, "target_box": str(image_path), "mask": str(mask_path)}


def test_split_counts_and_grouped_assignments_are_reproducible():
    assert split_counts(10, (70, 20, 10)) == {"train": 7, "val": 2, "test": 1}
    pairs = [{"sample_id": str(index)} for index in range(10)]
    groups = {"0": "plant_a", "1": "plant_a"}
    first, counts = assign_splits(pairs, seed=17, groups=groups)
    second, _ = assign_splits(pairs, seed=17, groups=groups)
    assert first == second
    assert first["0"] == first["1"]
    assert sum(counts.values()) == 10
    with pytest.raises(ValueError, match="total 100"):
        split_counts(10, (70, 20, 9))


def test_png_and_yolo_exports_keep_images_and_labels_aligned(tmp_path):
    pairs = [_pair(tmp_path, f"leaf{index}") for index in range(10)]
    for dataset_format in ("png", "yolo-seg", "yolo-detect"):
        output = tmp_path / f"{dataset_format}.zip"
        manifest = write_dataset_zip(pairs, output, dataset_format=dataset_format)
        assert manifest["counts"] == {"train": 7, "val": 2, "test": 1}
        with zipfile.ZipFile(output) as archive:
            assert json.loads(archive.read("manifest.json"))["samples"] == manifest["samples"]
            for sample in manifest["samples"]:
                assert sample["image"] in archive.namelist()
                assert sample["label"] in archive.namelist()
                if dataset_format == "png":
                    decoded = cv2.imdecode(
                        np.frombuffer(archive.read(sample["label"]), dtype=np.uint8), 0
                    )
                    assert decoded.shape == (16, 20)
                    assert decoded[4, 5] == 255 and decoded[0, 0] == 0
                else:
                    label = archive.read(sample["label"]).decode().strip().split()
                    assert label[0] == "0"
                    assert all(0 <= float(value) <= 1 for value in label[1:])
            if dataset_format.startswith("yolo"):
                assert "data.yaml" in archive.namelist()


def test_coco_rle_preserves_holes_and_invalid_pairs_are_reported(tmp_path):
    good = _pair(tmp_path, "good", hole=True)
    bad = _pair(tmp_path, "bad")
    bad["mask"] = str(tmp_path / "missing.png")
    output = tmp_path / "coco.zip"
    manifest = write_dataset_zip([good, bad], output, dataset_format="coco")
    assert manifest["counts"] == {"train": 1, "val": 0, "test": 0}
    assert manifest["excluded"] == [{
        "sample_id": "bad", "reason": "image or selected mask file is missing",
    }]
    with zipfile.ZipFile(output) as archive:
        annotations = json.loads(archive.read("annotations/instances_train.json"))
    annotation = annotations["annotations"][0]
    assert annotation["area"] == 101
    runs = annotation["segmentation"]["counts"]
    values = []
    for index, length in enumerate(runs):
        values.extend([index % 2] * length)
    restored = np.array(values, dtype=np.uint8).reshape((20, 16)).T
    assert restored[7, 9] == 0 and restored[4, 5] == 1


def test_group_csv_rejects_conflicts():
    assert read_group_csv(b"sample_id,group_id\na,plant1\n") == {"a": "plant1"}
    with pytest.raises(ValueError, match="Conflicting"):
        read_group_csv(b"sample_id,group_id\na,plant1\na,plant2\n")


def test_yolo_seg_excludes_mask_without_a_polygon(tmp_path):
    good = _pair(tmp_path, "good")
    tiny = _pair(tmp_path, "tiny")
    mask = np.zeros((16, 20), dtype=np.uint8)
    mask[5, 6] = 255
    assert cv2.imwrite(tiny["mask"], mask)
    manifest = write_dataset_zip(
        [good, tiny], tmp_path / "small.zip", dataset_format="yolo-seg",
    )
    assert len(manifest["samples"]) == 1
    assert manifest["excluded"] == [{
        "sample_id": "tiny", "reason": "mask has no usable polygon",
    }]
