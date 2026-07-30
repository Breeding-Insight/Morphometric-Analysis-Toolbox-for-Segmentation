from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest
from mats.app.Home import collect_output_pairs, merge_viewer_pairs


HOME_PAGE = Path(__file__).resolve().parents[1] / "src" / "mats" / "app" / "Home.py"


def _write_output_pair(output_dir, sample_id):
    target_box = output_dir / f"{sample_id}_target_box.jpg"
    mask = output_dir / f"{sample_id}_mask.png"
    target_box.write_bytes(b"target")
    mask.write_bytes(b"mask")
    return target_box, mask


def test_collect_output_pairs_ignores_outputs_from_earlier_sessions(tmp_path):
    _write_output_pair(tmp_path, "old")
    new_target, new_mask = _write_output_pair(tmp_path, "new")

    pairs = collect_output_pairs(tmp_path, ["new"])

    assert pairs == [{
        "sample_id": "new",
        "target_box": str(new_target),
        "mask": str(new_mask),
    }]


def test_merge_viewer_pairs_accumulates_current_session_without_duplicates():
    first = {
        "sample_id": "first",
        "target_box": "/outputs/first_target_box.jpg",
        "mask": "/outputs/first_mask.png",
    }
    updated = {
        "sample_id": "first",
        "target_box": "/outputs/recreated_first_target_box.jpg",
        "mask": "/outputs/first_mask.png",
    }
    second = {
        "sample_id": "second",
        "target_box": None,
        "mask": "/outputs/second_mask.png",
    }

    pairs = merge_viewer_pairs([first], [updated, second])

    assert pairs == [updated, second]


def test_home_page_renders_worker_safety_control():
    app = AppTest.from_file(str(HOME_PAGE)).run(timeout=30)

    assert not app.exception
    assert [item.label for item in app.number_input if item.label == "Workers"] == ["Workers"]
