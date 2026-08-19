# Sample images

The example photographs now ship **inside the package** rather than in this
folder, so `pip install mats-morpho` gives you runnable samples with no
checkout, and the repository does not carry two copies of the same binaries.

Find them:

```bash
python -c "from mats import samples; print(samples.SAMPLES_DIR)"
```

Or open the **Help** page in the app (`mats app`), which displays them with
annotations and offers a ZIP download.

Two legacy-template sets, each with its own marker-centre calibration area:

```bash
mats run -i <samples>/flat_bench_6x6in          -o /tmp/mats_demo -t 6x6in
mats run -i <samples>/handheld_field_10.5x9.5in -o /tmp/mats_demo -t 10.5x9.5in
```

Source, processing, and license: `src/mats/app/assets/samples/README.md`.
