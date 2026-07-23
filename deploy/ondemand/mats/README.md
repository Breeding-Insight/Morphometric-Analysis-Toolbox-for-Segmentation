# MATs — Open OnDemand app

A Batch Connect interactive app that serves the MATs Streamlit UI on an HPC
compute node and exposes it through the Open OnDemand reverse proxy.

## What you need on the cluster (one-time)

1. **A conda env with MATs installed.** From a clone of the repo:
   ```bash
   conda env create -f environment.yml     # creates env "mats" (includes zbar)
   conda activate mats
   pip install -e ".[app]"
   ```
   Then set `CONDA_ENV` in `template/script.sh.erb` to that env name (`mats`).
2. **The model checkpoints.** Fetch them once, ideally to a shared location:
   ```bash
   export MATS_WEIGHTS_DIR=/shared/models/mats
   mats fetch-weights
   mats doctor            # confirm they resolve
   ```
   Point the same `MATS_WEIGHTS_DIR` at that path in `template/script.sh.erb`.
3. On Linux, `pyzbar` needs the system lib `zbar`. The conda env above installs
   it; otherwise `conda install -c conda-forge zbar` or the distro package
   (`libzbar0`).

Note: `mats app` runs from any directory, so — unlike the old launcher — there
is no repo path to configure. Only the conda env and (optionally) the weights
location need setting.

## Install the OOD app

Copy this folder into your OOD sandbox apps dir on the cluster:

```bash
mkdir -p ~/ondemand/dev
cp -r deploy/ondemand/mats ~/ondemand/dev/
```

It then shows up under **My Sandbox Apps (Development)** in the OOD dashboard.
To publish for everyone, an OOD admin copies it to the system apps directory
(typically `/var/www/ood/apps/sys/`).

## Customize before first launch

- `form.yml`: set `cluster` and the GPU/CPU `partition` option values.
- `submit.yml.erb`: adjust SLURM flags if your scheduler differs (PBS/LSF).
- `template/script.sh.erb`: set `CONDA_SH`, `CONDA_ENV`, and optionally
  `MATS_WEIGHTS_DIR`.

## Proxy note (the usual gotcha)

Streamlit serves assets from absolute paths, so it must know the proxy prefix.
This app assumes OOD's **relative-node** proxy (`/rnode/HOST/PORT/...`, which
forwards the prefix unchanged) and sets Streamlit's `--server.baseUrlPath`
accordingly. If your site uses the **absolute-node** proxy (`/node/HOST/PORT/...`,
which strips the prefix), remove `--server.baseUrlPath` from `script.sh.erb` and
change `rnode` to `node` in `view.html.erb`.

If the page loads blank or the app "keeps connecting," it's almost always this
proxy/baseUrlPath mismatch.
