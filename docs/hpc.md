# Running on HPC

MATs runs fine as a plain CLI batch job, and there is also an Open OnDemand app
for the GUI.

## As a batch job

Install once into a conda env on the cluster:

```bash
conda env create -f environment.yml
conda activate mats
pip install -e ".[app]"
export MATS_WEIGHTS_DIR=/project/<your_project>/mats_weights   # shared, readable
mkdir -p "$MATS_WEIGHTS_DIR"
# Run from a Git checkout with Git LFS installed. It materializes both files
# in that checkout's weights/ directory; copy the verified files to /project.
mats fetch-weights --all
cp weights/rf_detr_marker.pth "$MATS_WEIGHTS_DIR/"
cp weights/birefnet_leaf.pth "$MATS_WEIGHTS_DIR/"
mats doctor
```

On USDA **SCINet** (Ceres/Atlas), a `/project` directory is a mounted filesystem
shared across the project, so every job reads the weights in place — no per-user
copy. Materialize the weights once in a Git checkout, copy the verified files to
that path, and point `MATS_WEIGHTS_DIR` at it for all users. External collaborators
without SCINet accounts can pull the same directory via a **Globus guest collection**
(they need a free Globus login).

Set `MATS_NO_AUTO_FETCH=1` in your jobs so a misconfigured path fails fast with a
clear error instead of triggering a 2.65 GB download on a login or compute node
when the weights should already be staged.

Then a SLURM job is just:

```bash
#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

module load miniconda3 && source activate mats   # or your conda activate
export MATS_WEIGHTS_DIR=/project/<your_project>/mats_weights
export MATS_NO_AUTO_FETCH=1                        # weights are pre-staged; don't download

mats run \
  -i "$SCRATCH/leaf_images" \
  -o "$SCRATCH/leaf_out" \
  -r "$SCRATCH/leaf_out/results.csv" \
  --sheet-dimensions 12x12in \
  --mask-method birefnet
```

Always pass `-i` and `-o` in a job: with no terminal attached, `mats run` will
not prompt — it exits with an error instead of hanging.

A GPU makes BiRefNet much faster; CPU works but is slow. For clean backgrounds,
`--mask-method threshold` avoids the GPU entirely.

## As an Open OnDemand app (GUI on a compute node)

The Batch Connect app is in [../deploy/ondemand/mats/](../deploy/ondemand/mats/).
Its README covers installation, the form fields, and the reverse-proxy setup.
The short version:

```bash
mkdir -p ~/ondemand/dev
cp -r deploy/ondemand/mats ~/ondemand/dev/
```

then edit `form.yml` (cluster + partitions) and `template/script.sh.erb`
(`CONDA_ENV`, optional `MATS_WEIGHTS_DIR`). Because `mats app` runs from any
directory, there is no repo path to configure.
