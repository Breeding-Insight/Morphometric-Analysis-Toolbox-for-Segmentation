"""Install and diagnose the optional BiRefNet leaf-segmentation model."""

from pathlib import Path

import streamlit as st

from mats import weights
from mats.app import branding
from mats.app.runtime_paths import display_path
from mats.birefnet_runtime import birefnet_runtime_status
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
    "BiRefNet is optional. MATS uses it only when its checkpoint is already installed locally; "
    "choosing Otsu never downloads or loads it."
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

runtime = birefnet_runtime_status()
with st.container(border=True):
    st.subheader("Local runtime")
    _show_status("success" if runtime.ready else "error", runtime.detail)

status = weights.get_weight_status("birefnet")
with st.container(border=True):
    st.subheader("Model checkpoint")
    if status.state == "ready":
        st.success(f"BiRefNet is installed at `{display_path(status.path)}`.")
    elif status.state == "invalid":
        st.error(f"BiRefNet checkpoint needs attention: {status.detail}")
    else:
        st.warning(
            f"BiRefNet is not installed. Expected download: {_human_bytes(status.expected_size_bytes)}."
        )
    st.caption(
        f"Install location on the machine running MATS: `{display_path(status.path)}`"
    )

    destination = Path(status.path).parent
    try:
        free = weights.free_bytes(destination if destination.exists() else destination.parent)
        st.caption(f"Free disk space near the install location: {_human_bytes(free)}")
    except OSError:
        pass

    lfs = next(source for source in status.sources if source.id == "lfs")
    if not lfs.available:
        st.info(
            "Automatic model downloads are disabled. To fetch the optional checkpoint here, "
            "run this page from a Git checkout with Git LFS installed, or use one of the "
            "manual local-placement options below."
        )
        can_fetch = False
    else:
        st.caption(
            "Fetch is explicit: the button below pulls the checkpoint from this repository via Git LFS."
        )
        can_fetch = True

    if st.button(
        "Fetch BiRefNet from this repository",
        type="primary",
        icon=":material/download:",
        disabled=not can_fetch or status.state == "ready",
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
                installed = weights.install_weight("birefnet", source="lfs", progress_callback=update)
            except weights.WeightDownloadError as exc:
                install_status.update(label="BiRefNet installation failed", state="error", expanded=True)
                st.error(str(exc))
            else:
                install_status.update(label="BiRefNet installed", state="complete", expanded=False)
                progress.progress(1.0, text="Installation complete.")
                st.success(f"Installed and verified at `{display_path(installed.path)}`.")
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
