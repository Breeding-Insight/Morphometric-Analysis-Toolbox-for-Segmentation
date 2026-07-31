"""Explain and diagnose optional QR fallbacks without changing the environment."""

import streamlit as st

from mats.app import branding
from mats.app.runtime_paths import (
    current_python,
    display_path,
    mats_command,
    python_command,
    shell_language,
    source_install_root,
)
from mats.qr_runtime import qr_runtime_status


def _show_backend_status(backend):
    message = f"**{backend.name}** — {backend.detail}"
    if backend.available:
        st.success(message, icon=":material/check_circle:")
    else:
        st.warning(message, icon=":material/warning:")


st.set_page_config(
    page_title="MATS — Robust QR setup",
    page_icon=branding.page_icon(),
    layout="wide",
)
branding.apply_logo()

st.title("Robust QR setup")
st.caption(
    "Optional fallbacks for difficult template QR codes. This page only explains "
    "and checks the setup; it never changes your Python or Conda environment."
)
st.caption(
    "Paths and commands below are resolved for the machine currently running MATS."
)

status = qr_runtime_status()
with st.container(border=True):
    st.subheader("QR decoder status", anchor=False)
    _show_backend_status(status.opencv)
    _show_backend_status(status.pyzbar)
    _show_backend_status(status.qreader)
    if status.full_fallback_available:
        st.success(
            "Full robust QR fallback coverage is ready: OpenCV, pyzbar, and QReader "
            "will be tried in that order.",
            icon=":material/verified:",
        )
    elif status.enhanced_available:
        st.info(
            "One robust QR fallback is ready. Install zbar as described below only "
            "if you want to enable pyzbar as well.",
            icon=":material/info:",
        )
    else:
        st.warning(
            "Only OpenCV is available. Clear QR codes may work, but glare, blur, "
            "skew, or poor contrast can leave a measurement without a scale and "
            "produce `NA` values.",
            icon=":material/qr_code_scanner:",
        )

with st.container(border=True):
    st.subheader("Do you need robust QR?", anchor=False)
    st.markdown(
        "- **No extra setup:** enter the finished printed-sheet size when every "
        "image uses the same current Template Creator sheet.\n"
        "- **OpenCV only:** use QR mode with clear, well-lit codes.\n"
        "- **Robust QR:** add fallbacks when field images commonly have glare, blur, "
        "skew, or low contrast."
    )
    st.caption(
        "QReader is useful without Conda. Its detector model is downloaded only "
        "when MATS first needs that fallback."
    )

with st.container(border=True):
    st.subheader("Install the optional Python fallbacks", anchor=False)
    st.markdown(
        "Install the `qr` extra in the same Python environment that runs MATS. "
        "It adds both QReader and pyzbar."
    )
    st.code(
        python_command("-m", "pip", "install", "mats-morpho[qr]"),
        language=shell_language(),
    )
    source_root = source_install_root()
    if source_root is not None:
        with st.expander("If MATS is installed from this source checkout"):
            st.caption(f"Source checkout: `{display_path(source_root)}`")
            st.code(
                python_command("-m", "pip", "install", "-e", f"{source_root}[app,qr]"),
                language=shell_language(),
            )

with st.container(border=True):
    st.subheader("Enable the full pyzbar fallback", anchor=False)
    st.markdown(
        "pyzbar also needs the native **zbar** library. Conda is optional: use it "
        "only when MATS already runs in a Conda environment. Otherwise use your "
        "operating system's package manager."
    )
    st.code("brew install zbar", language=shell_language())
    st.caption("macOS with Homebrew")
    st.code("sudo apt install libzbar0", language=shell_language())
    st.caption("Debian or Ubuntu")
    st.code("conda install -c conda-forge zbar", language=shell_language())
    st.caption("Existing Conda environment only")
    st.info(
        "On shared or managed systems, ask your administrator to provide zbar. "
        "QReader can still be used without it.",
        icon=":material/admin_panel_settings:",
    )

with st.container(border=True):
    st.subheader("Restart and verify", anchor=False)
    st.warning(
        "Restart MATS after installing anything. Decoder modules are loaded when "
        "the app starts, so a browser refresh alone is not enough.",
        icon=":material/restart_alt:",
    )
    st.code(mats_command("app"), language=shell_language())
    st.caption(f"MATS is currently running with: `{current_python()}`")
    st.caption("After restart, return here or run `mats doctor` to confirm the available decoders.")
