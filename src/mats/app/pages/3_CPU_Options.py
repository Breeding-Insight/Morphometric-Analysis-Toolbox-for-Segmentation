"""Explain and configure method-aware CPU/GPU execution policies."""

import streamlit as st

from mats.app import branding
from mats.app.compute import (
    CPU_WORKERS_KEY,
    HYBRID_FANOUT_DISABLED_KEY,
    activate_cpu,
    birefnet_parallel_unlocked,
    break_glass_unlocked,
    confirm_high_risk_workers,
    confirm_parallel_birefnet,
    get_compute_settings,
    hybrid_fanout_disabled,
    reengage_gpu,
    resolve_execution_plan,
    restore_automatic_compute,
    selected_mask_method,
    set_cpu_workers,
)


WORKER_WIDGET_KEY = "compute_options_worker_input"


@st.cache_resource
def load_pipeline_module():
    """Load the shared pipeline module without initializing models twice."""
    from mats import core
    return core


def _worker_changed() -> None:
    set_cpu_workers(st.session_state[WORKER_WIDGET_KEY])


st.set_page_config(
    page_title="MATS — Compute options",
    page_icon=branding.page_icon(),
    layout="wide",
)
branding.apply_logo()

st.title("Compute options")
st.caption(
    "MATS automatically chooses the fastest safe pipeline for the selected segmentation method. "
    "Advanced overrides are available for diagnostics and exceptional hardware constraints."
)

try:
    lm = load_pipeline_module()
except ModuleNotFoundError as exc:
    st.error(f"Failed to import the MATS pipeline: missing dependency `{exc.name or exc}`.")
    st.stop()
except Exception as exc:
    st.error(f"Failed to import the MATS pipeline: {exc}")
    st.stop()

mask_method = selected_mask_method()
method_label = "BiRefNet" if mask_method == "birefnet" else "Classic thresholding (Otsu)"
settings = get_compute_settings(lm)
available_workers = settings.available_workers
birefnet_unlocked = birefnet_parallel_unlocked(available_workers)
plan = resolve_execution_plan(
    settings,
    mask_method,
    birefnet_parallel_allowed=birefnet_unlocked,
)
safe_worker_max = lm.worker_risk_report(1, available_workers).normal_limit
standard_unlocked = break_glass_unlocked(available_workers)

if mask_method == "birefnet" and not birefnet_unlocked:
    max_workers = 1
    worker_value = 1
else:
    max_workers = available_workers if standard_unlocked else safe_worker_max
    worker_value = min(settings.cpu_workers, max_workers)
    if settings.cpu_workers != worker_value:
        st.session_state[CPU_WORKERS_KEY] = worker_value
        settings = get_compute_settings(lm)

if st.session_state.get(WORKER_WIDGET_KEY) != worker_value:
    st.session_state[WORKER_WIDGET_KEY] = worker_value

with st.container(border=True):
    st.subheader("Current execution")
    status_color = "green" if plan.uses_gpu else "orange"
    status_icon = ":material/check_circle:" if plan.uses_gpu else ":material/memory:"
    st.badge(plan.label, icon=status_icon, color=status_color)
    st.caption(plan.detail)
    st.caption(f"Segmentation method: {method_label}")
    if mask_method == "birefnet" and not plan.uses_gpu:
        gpu_notice = (
            "A usable GPU is available but disabled by an advanced CPU-only override."
            if settings.accelerator_available
            else "No usable CUDA or Apple MPS accelerator was detected."
        )
        st.warning(
            f"BiRefNet is running on CPU. {gpu_notice} This may be substantially slower and "
            "use significant system memory.",
            icon=":material/warning:",
        )
    if settings.accelerator_available and settings.cpu_active:
        if st.button("Re-engage GPU", icon=":material/restart_alt:", width="content"):
            reengage_gpu()
            st.rerun()

with st.container(border=True):
    st.subheader("CPU worker allocation")
    st.caption(
        f"{available_workers} CPU worker(s) are available to this app. The allocation is based "
        "on machine capacity, not the number of images in the batch."
    )
    if mask_method == "birefnet" and not birefnet_unlocked:
        st.info(
            "BiRefNet is capped at one effective worker by default. CPU-parallel BiRefNet needs "
            "a separate break-glass acknowledgement.",
            icon=":material/info:",
        )
    selected_workers = int(st.number_input(
        "CPU workers",
        min_value=1,
        max_value=max_workers,
        step=1,
        key=WORKER_WIDGET_KEY,
        on_change=_worker_changed,
        help=(
            "Otsu uses these workers automatically with GPU RF-DETR. BiRefNet is restricted "
            "to one worker until its separate break-glass acknowledgement is completed."
        ),
    ))
    worker_risk = lm.worker_risk_report(selected_workers, available_workers)
    st.badge(
        f"{worker_risk.label} · {selected_workers}/{available_workers} workers · "
        f"{worker_risk.utilization:.0%}",
        icon=(":material/check_circle:" if worker_risk.level == "low" else ":material/warning:"),
        color=worker_risk.color,
        help=(
            "Green: 25% or less. Yellow: 26–50%. Red: 51–75%. "
            "Counts above 75% require a one-run break-glass acknowledgement."
        ),
    )

    if mask_method == "birefnet" and settings.cpu_active and not birefnet_unlocked:
        if st.button(
            "Unlock CPU-parallel BiRefNet",
            icon=":material/warning:",
            width="content",
        ):
            confirm_parallel_birefnet(available_workers)
    if worker_risk.requires_break_glass:
        st.error(
            "Break-glass mode is active. This worker count may make the computer "
            "unresponsive or crash; save work before running."
        )
    elif available_workers > safe_worker_max and not standard_unlocked and not (
        mask_method == "birefnet" and not birefnet_unlocked
    ):
        if st.button(
            "Unlock high-risk worker counts",
            icon=":material/warning:",
            width="content",
        ):
            confirm_high_risk_workers(available_workers)
    elif standard_unlocked:
        st.caption("CPU load unlock is valid for one high-risk run.")

with st.expander("How MATS uses your hardware", expanded=True, icon=":material/help:"):
    st.markdown(
        "### Otsu thresholding → GPU/CPU fan-out\n\n"
        "Otsu thresholding (`cv2.THRESH_OTSU`) is a per-image OpenCV histogram operation. "
        "It has no GPU path in MATS. With a GPU available, RF-DETR runs through one controlled "
        "GPU lane while independent Otsu, morphology, measurement, and output stages fan out "
        "across the selected CPU workers.\n\n"
        "### BiRefNet → one GPU inference lane\n\n"
        "BiRefNet and RF-DETR use one GPU lane by default. GPU inference already parallelizes "
        "within each call; concurrent model copies would multiply VRAM use and can cause "
        "contention or out-of-memory failures. If BiRefNet falls back to CPU, MATS warns you "
        "and keeps one effective worker until you explicitly break the glass.\n\n"
        "### Worker limits\n\n"
        "More CPU workers can improve throughput until CPU, memory bandwidth, disk I/O, or the "
        "GPU lane is saturated. Workers are a concurrency ceiling, so unused workers remain idle "
        "for small batches."
    )

with st.expander("Advanced execution overrides", icon=":material/tune:"):
    if mask_method == "threshold" and settings.accelerator_available and not settings.cpu_active:
        st.toggle(
            "Disable GPU/CPU fan-out",
            key=HYBRID_FANOUT_DISABLED_KEY,
            persist_state="session",
            help="Runs GPU RF-DETR and Otsu processing serially. This usually reduces throughput.",
        )
    else:
        st.caption("GPU/CPU fan-out is available only for Otsu when automatic GPU execution is active.")

    if not settings.cpu_active:
        if st.button("Force CPU-only pipeline", icon=":material/memory:", width="content"):
            activate_cpu()
            st.rerun()
    if st.button("Restore automatic defaults", icon=":material/refresh:", width="content"):
        restore_automatic_compute()
        st.rerun()
