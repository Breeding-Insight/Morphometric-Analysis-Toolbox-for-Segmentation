# MATS sample images

Three example photographs of leaves on printed MATS calibration templates,
packaged with `mats-morpho` so the quick start and the in-app **Help** page are
runnable without any download.

## Sets

| Directory | Photos | Legacy calibration (`-t`) | Print sheet | Capture |
|---|---|---|---|---|
| `flat_bench_6x6in/` | 1 | `6x6in` | 8 x 8 in, 0.5 in markers | Indoors, sheet flat on a bench, camera near-perpendicular. Simple leaf. |
| `handheld_field_10.5x9.5in/` | 2 | `10.5x9.5in` | 12 x 12 in, 0.5 in markers | Outdoors, board hand-held at an angle over soil and plastic mulch. Compound leaf. |

These are legacy templates whose margins predate the current Template Creator.
Their compatibility `-t` values are marker-centre calibration areas, not the
outer print sheet. For current Template Creator sheets, use the finished sheet
size with `--sheet-dimensions`; MATS derives the calibration area automatically.

Both "hero" photos (`flat_bench_simple_leaf_01.jpg`,
`handheld_field_compound_leaf_01.jpg`) decode their own QR code cleanly. The
second field photo, `handheld_field_qr_unreadable.jpg`, is **kept
deliberately**: its QR code doesn't decode with any bundled backend (OpenCV,
pyzbar, QReader) because of motion blur. It's a real, reproducible example of
a QR-read failure, used by the Help page to demonstrate the manual fallback —
the run doesn't abort, and that row is written as `NA` until re-run with its
legacy `-t` calibration. Do not treat it as a quality
reject and drop it; its blur is the point.

Run them:

```bash
mats run -i flat_bench_6x6in           -o ~/mats_demo/bench -t 6x6in
mats run -i handheld_field_10.5x9.5in  -o ~/mats_demo/field -t 10.5x9.5in
```

## Provenance

Original photographs by A. J. Ackerman, captured for the MATS manuscript
(target journal: *Plant Phenomics*). The originals are held in the manuscript
repository and are not redistributed here.

## Processing applied

Each image was downscaled and de-identified once, with Pillow:

- EXIF orientation baked into the pixels (`ImageOps.exif_transpose`), then
  **all EXIF, XMP and MPF metadata removed**. The originals carried GPS
  coordinates, device model, and capture timestamps; none of that ships here.
- Resized to a 2000 px long edge with Lanczos resampling.
- Saved as progressive JPEG, quality 85, 4:4:4 chroma (`subsampling=0`) so the
  marker colour the detector was trained on is not degraded.
- The embedded Display P3 ICC profile is retained so colours render correctly
  and the pixel values match the originals the pipeline was validated against.
- Renamed descriptively; all internal breeding-program identifiers removed.
- One hero photo kept per set, plus the deliberate QR-failure photo described
  above; other captures from the same shoots were reviewed and dropped rather
  than padding the package with near-identical images.

No other alteration: no cropping, rotation, colour grading, or retouching.

## License

Released under the same MIT license as the rest of `mats-morpho`
(https://github.com/Breeding-Insight/Morphometric-Analysis-Toolbox-for-Segmentation/blob/main/LICENSE),
copyright A. J. Ackerman. You may reuse them, including for teaching and
benchmarking, with attribution to the MATS project. If you use them in a
publication, please cite the manuscript — see `CITATION.cff`.
