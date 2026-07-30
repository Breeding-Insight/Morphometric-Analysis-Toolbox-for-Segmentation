"""Install and diagnose the optional BiRefNet leaf-segmentation model."""

from pathlib import Path

import streamlit as st

from mats import weights
from mats.app import branding
from mats.devices import birefnet_device_report


def _human_bytes(value):
    value = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024


def _show_status(level, text):
    getattr(st, {"success": "success", "warning": "warning", "error": "error"}[level])(text)


st.set_page_config(
    page_title="MATS — BiRefNet setup",
    page_icon=branding.page_icon(),
    layout="wide",
)
branding.apply_logo()

st.title("BiRefNet setup")
st.caption(
    "Install the optional leaf-segmentation model once. It's committed to this repository "
    "via Git LFS, but excluded from a plain `git clone` to keep clones small -- install it "
    "here from Hugging Face or straight from this repository."
)

device = birefnet_device_report()
with st.container(border=True):
    st.subheader("Compute availability")
    _show_status(device.severity, device.detail)
    st.caption(
        "A green result confirms that PyTorch can use an accelerator. A yellow CPU result is supported, "
        "but BiRefNet processing will be slower."
    )
    if device.torch_version:
        st.caption(f"PyTorch {device.torch_version} · selected device: {device.device}")

status = weights.get_weight_status("birefnet")
with st.container(border=True):
    st.subheader("Model checkpoint")
    if status.state == "ready":
        st.success(f"BiRefNet is installed at `{status.path}`.")
    elif status.state == "invalid":
        st.error(f"BiRefNet checkpoint needs attention: {status.detail}")
    else:
        st.warning(
            f"BiRefNet is not installed. Expected download: {_human_bytes(status.expected_size_bytes)}."
        )
    st.caption(f"Install location: `{status.path}`")

    destination = Path(status.path).parent
    try:
        free = weights.free_bytes(destination if destination.exists() else destination.parent)
        st.caption(f"Free disk space near the install location: {_human_bytes(free)}")
    except OSError:
        pass

    by_id = {source.id: source for source in status.sources}
    available = [source for source in status.sources if source.available]

    if not available:
        st.info(
            "Neither download source is available in this build. Set `MATS_WEIGHTS_HF_REPO` "
            "to a Hugging Face repository, or run this page from a Git clone with Git LFS "
            "installed. See the manual options below."
        )
        choice = None
    else:
        captions = {
            "hf": "Public download. Requires internet access to huggingface.co.",
            "lfs": "Works on networks that block Hugging Face, including USDA. Requires this "
                   "page to be running from a `git clone` with Git LFS installed.",
        }
        options = [source.id for source in status.sources]
        choice = st.radio(
            "Download source",
            options=options,
            format_func=lambda source_id: by_id[source_id].label,
            captions=[
                captions[source.id] if source.available else f"Unavailable: {source.reason}"
                for source in status.sources
            ],
            index=next(i for i, source in enumerate(status.sources) if source.available),
            key="birefnet_source",
        )
        if not by_id[choice].available:
            st.warning(f"{by_id[choice].label} is unavailable: {by_id[choice].reason}")

    if st.button(
        "Download BiRefNet",
        type="primary",
        icon=":material/download:",
        disabled=choice is None or not by_id[choice].available or status.state == "ready",
        width="content",
        key="download_birefnet",
    ):
        progress = st.progress(0, text="Preparing download…")
        with st.status("Installing BiRefNet…", expanded=True) as install_status:
            def update(phase, completed, total):
                fraction = min(1.0, completed / max(total, 1))
                labels = {
                    "preparing": "Preparing download…",
                    "downloading": f"Downloading {_human_bytes(completed)} of {_human_bytes(total)}…",
                    "verifying": "Verifying checkpoint…",
                    "complete": "Installation complete.",
                }
                progress.progress(fraction, text=labels.get(phase, "Installing BiRefNet…"))

            try:
                installed = weights.install_weight("birefnet", source=choice, progress_callback=update)
            except weights.WeightDownloadError as exc:
                install_status.update(label="BiRefNet installation failed", state="error", expanded=True)
                st.error(str(exc))
            else:
                install_status.update(label="BiRefNet installed", state="complete", expanded=False)
                progress.progress(1.0, text="Installation complete.")
                st.success(f"Installed and verified at `{installed.path}`.")
                st.rerun()

with st.expander("Manual and HPC installation"):
    st.markdown(
        "- **Shared filesystem (recommended for an institution, e.g. SCINet):** set "
        "`MATS_WEIGHTS_DIR` to a shared directory containing `birefnet_leaf.pth` -- read in "
        "place, no download for anyone who can mount it.\n"
        "- Or set `BIREFNET_CHECKPOINT` to an explicit checkpoint path.\n"
        "- Or, from a terminal in a Git checkout: "
        "`git lfs pull --include=\"weights/birefnet_leaf.pth\"`.\n"
        "- On air-gapped systems, pre-stage the checkpoint and verify its SHA-256 before "
        "launching MATS."
    )
