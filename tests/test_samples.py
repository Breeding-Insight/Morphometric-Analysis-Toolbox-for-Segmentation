from mats import samples


def test_every_declared_sample_image_is_present_and_small():
    for sample_set in samples.SAMPLE_SETS:
        paths = samples.sample_paths(sample_set)
        assert len(paths) == len(sample_set["filenames"]), sample_set["key"]
        for path in paths:
            assert path.stat().st_size < 512 * 1024, path


def test_sample_filenames_are_not_mistaken_for_target_boxes():
    # core.is_target_box_image() matches "_target_box" anywhere in the stem,
    # which would make the pipeline skip marker detection entirely.
    for sample_set in samples.SAMPLE_SETS:
        for name in sample_set["filenames"]:
            assert "_target_box" not in name


def test_samples_installed_reports_true_when_images_are_present():
    assert samples.samples_installed()


def test_every_set_hero_is_one_of_its_own_declared_filenames():
    for sample_set in samples.SAMPLE_SETS:
        assert sample_set["hero"] in sample_set["filenames"], sample_set["key"]


def test_sample_metadata_distinguishes_sheet_from_legacy_calibration():
    for sample_set in samples.SAMPLE_SETS:
        assert sample_set["sheet_dimensions"].endswith("in")
        assert sample_set["calibration_dimensions"].endswith("in")


def test_hero_paths_resolve_for_every_set():
    for sample_set in samples.SAMPLE_SETS:
        path = samples.sample_hero_path(sample_set)
        assert path is not None, sample_set["key"]
        assert path.name == sample_set["hero"]


def test_qr_failure_sample_is_declared_in_the_handheld_field_set_and_resolves():
    field_set = next(s for s in samples.SAMPLE_SETS if s["key"] == "handheld_field")
    assert samples.QR_FAILURE_SAMPLE["filename"] in field_set["filenames"]
    assert samples.QR_FAILURE_SAMPLE["directory"] == field_set["directory"]

    path = samples.qr_failure_sample_path()
    assert path is not None
    assert path.name == samples.QR_FAILURE_SAMPLE["filename"]
