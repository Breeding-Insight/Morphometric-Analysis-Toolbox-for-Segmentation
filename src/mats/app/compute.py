"""Shared compute-mode state and controls for the Streamlit application."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from mats.devices import birefnet_device_report


COMPUTE_MODE_KEY = "compute_execution_mode"
CPU_WORKERS_KEY = "compute_cpu_workers"
BREAK_GLASS_KEY = "break_glass_available_workers"
HYBRID_FANOUT_DISABLED_KEY = "compute_hybrid_fanout_disabled"
BIREFNET_PARALLEL_KEY = "birefnet_parallel_available_workers"


@dataclass(frozen=True)
class ComputeSettings:
    """The effective compute settings for the current browser session."""

    mode: str
    cpu_workers: int
    available_workers: int
    accelerator_available: bool
    accelerator_label: str
    accelerator_detail: str

    @property
    def cpu_active(self) -> bool:
        return self.mode == "cpu"

    @property
    def execution_device(self) -> str:
        return "cpu" if self.cpu_active else "auto"

    @property
    def effective_workers(self) -> int:
        return self.cpu_workers if self.cpu_active else 1


@dataclass(frozen=True)
class ExecutionPlan:
    """Resolved, method-aware execution policy for one MATS run."""

    policy: str
    execution_device: str
    workers: int
    label: str
    detail: str
    uses_gpu: bool
    uses_cpu_fanout: bool


def _accelerator_label(device: str) -> str:
    if device.startswith("cuda"):
        return "CUDA GPU"
    if device == "mps":
        return "Apple MPS GPU"
    return "GPU"


def get_compute_settings(pipeline) -> ComputeSettings:
    """Initialize and return shared compute state.

    A usable accelerator is the default. CPU workers are deliberately based on
    machine capacity only, never the current input batch size.
    """
    report = birefnet_device_report()
    available_workers = pipeline.available_cpu_workers()
    default_cpu_workers = max(1, available_workers // 2)

    st.session_state.setdefault(CPU_WORKERS_KEY, default_cpu_workers)
    st.session_state[CPU_WORKERS_KEY] = min(
        available_workers,
        max(1, int(st.session_state[CPU_WORKERS_KEY])),
    )
    if COMPUTE_MODE_KEY not in st.session_state:
        st.session_state[COMPUTE_MODE_KEY] = "auto" if report.accelerated else "cpu"
    if st.session_state[COMPUTE_MODE_KEY] not in {"auto", "cpu"}:
        st.session_state[COMPUTE_MODE_KEY] = "auto" if report.accelerated else "cpu"
    if not report.accelerated:
        st.session_state[COMPUTE_MODE_KEY] = "cpu"

    return ComputeSettings(
        mode=st.session_state[COMPUTE_MODE_KEY],
        cpu_workers=st.session_state[CPU_WORKERS_KEY],
        available_workers=available_workers,
        accelerator_available=report.accelerated,
        accelerator_label=_accelerator_label(report.device),
        accelerator_detail=report.detail,
    )


def activate_cpu() -> None:
    """Make the selected CPU worker count effective."""
    st.session_state[COMPUTE_MODE_KEY] = "cpu"


def set_cpu_workers(workers: int) -> None:
    """Store a CPU worker preference without changing the active compute mode."""
    st.session_state[CPU_WORKERS_KEY] = max(1, int(workers))


def reengage_gpu() -> None:
    """Return to automatic accelerator selection and revoke high-risk access."""
    st.session_state[COMPUTE_MODE_KEY] = "auto"
    st.session_state.pop(BREAK_GLASS_KEY, None)
    st.session_state.pop(BIREFNET_PARALLEL_KEY, None)


def restore_automatic_compute() -> None:
    """Clear advanced execution overrides and return to automatic mode."""
    reengage_gpu()
    st.session_state.pop(HYBRID_FANOUT_DISABLED_KEY, None)


def hybrid_fanout_disabled() -> bool:
    return bool(st.session_state.get(HYBRID_FANOUT_DISABLED_KEY, False))


def selected_mask_method() -> str:
    """Return the segmentation method selected on Home, defaulting to Otsu."""
    return (
        "birefnet"
        if st.session_state.get("segmentation_method") == "BiRefNet"
        else "threshold"
    )


def break_glass_unlocked(available_workers: int) -> bool:
    """Return whether this exact allocation has a one-run high-risk override."""
    if st.session_state.get(BREAK_GLASS_KEY) != available_workers:
        st.session_state.pop(BREAK_GLASS_KEY, None)
        return False
    return True


def birefnet_parallel_unlocked(available_workers: int) -> bool:
    """Return whether multi-worker CPU BiRefNet has one-run authorization."""
    if st.session_state.get(BIREFNET_PARALLEL_KEY) != available_workers:
        st.session_state.pop(BIREFNET_PARALLEL_KEY, None)
        return False
    return True


def resolve_execution_plan(
    settings: ComputeSettings,
    mask_method: str,
    *,
    birefnet_parallel_allowed: bool = False,
) -> ExecutionPlan:
    """Resolve automatic defaults and explicit overrides into a runnable plan."""
    if mask_method == "birefnet":
        if settings.cpu_active:
            workers = settings.cpu_workers if birefnet_parallel_allowed else 1
            return ExecutionPlan(
                "cpu_parallel_birefnet" if workers > 1 else "cpu_serial_birefnet",
                "cpu",
                workers,
                f"GPU INACTIVE · BIREFNET ON CPU · {workers} WORKER{'S' if workers != 1 else ''}",
                "BiRefNet is on CPU. Multi-worker mode requires a separate break-glass acknowledgement.",
                False,
                workers > 1,
            )
        return ExecutionPlan(
            "gpu_serial_birefnet",
            "auto",
            1,
            f"GPU ACTIVE · RF-DETR + BIREFNET · {settings.accelerator_label}",
            "Neural inference uses one GPU lane to avoid duplicate model memory and VRAM contention.",
            True,
            False,
        )

    if settings.cpu_active:
        return ExecutionPlan(
            "cpu_parallel_otsu",
            "cpu",
            settings.cpu_workers,
            f"GPU INACTIVE · OTSU CPU · {settings.cpu_workers} WORKER{'S' if settings.cpu_workers != 1 else ''}",
            "RF-DETR, Otsu, and measurement stages are running on CPU workers.",
            False,
            settings.cpu_workers > 1,
        )
    if hybrid_fanout_disabled():
        return ExecutionPlan(
            "gpu_serial_otsu",
            "auto",
            1,
            f"GPU ACTIVE · RF-DETR · {settings.accelerator_label}",
            "GPU/CPU fan-out is disabled by an advanced override; Otsu processing is serial.",
            True,
            False,
        )
    return ExecutionPlan(
        "hybrid_otsu",
        "hybrid",
        settings.cpu_workers,
        (
            f"HYBRID ACTIVE · RF-DETR GPU · OTSU CPU · "
            f"{settings.cpu_workers} WORKER{'S' if settings.cpu_workers != 1 else ''}"
        ),
        "RF-DETR uses one GPU inference lane while CPU workers run Otsu and measurements.",
        True,
        settings.cpu_workers > 1,
    )


@st.dialog(
    "Break glass: high-risk CPU allocation",
    width="medium",
    icon=":material/warning:",
    dismissible=False,
)
def confirm_high_risk_workers(available_workers: int) -> None:
    """Require an explicit, single-run acknowledgement for >75% CPU use."""
    dialog_nonce = st.session_state.get("break_glass_dialog_nonce", 0)
    st.error(
        f"You are unlocking up to {available_workers} workers. This can use more than "
        "75% of the CPU allocation and may freeze or crash the computer."
    )
    st.markdown(
        "- Save important work and close unnecessary applications first.\n"
        "- Model-backed runs can consume substantial memory and increase heat/fan activity.\n"
        "- Other users of this shared server may be affected.\n"
        "- CUDA/MPS remains disabled while CPU mode is active."
    )
    saved_work = st.checkbox(
        "I have saved important work and closed unnecessary applications.",
        key=f"break_glass_saved_work_{dialog_nonce}",
    )
    understands_risk = st.checkbox(
        "I understand this run may freeze or crash the computer.",
        key=f"break_glass_understands_risk_{dialog_nonce}",
    )
    if st.button("Keep the safe worker limit", width="stretch"):
        st.session_state["break_glass_dialog_nonce"] = dialog_nonce + 1
        st.rerun()
    if st.button(
        "Break the glass for one run",
        type="primary",
        icon=":material/warning:",
        width="stretch",
        disabled=not (saved_work and understands_risk),
    ):
        st.session_state[BREAK_GLASS_KEY] = available_workers
        st.session_state["break_glass_dialog_nonce"] = dialog_nonce + 1
        st.rerun()


@st.dialog(
    "Break glass: CPU-parallel BiRefNet",
    width="medium",
    icon=":material/warning:",
    dismissible=False,
)
def confirm_parallel_birefnet(available_workers: int) -> None:
    """Require a distinct acknowledgement before running BiRefNet on CPU workers."""
    dialog_nonce = st.session_state.get("birefnet_parallel_dialog_nonce", 0)
    st.error(
        "BiRefNet normally uses one GPU worker. CPU-parallel BiRefNet can consume "
        "substantial RAM, create model contention, and make the computer unresponsive."
    )
    st.markdown(
        "- A usable GPU is faster for BiRefNet in the normal case.\n"
        "- Save important work and close unnecessary applications first.\n"
        "- This unlock applies to one run only; the normal CPU load limit still applies."
    )
    saved_work = st.checkbox(
        "I have saved important work and understand the memory risk.",
        key=f"birefnet_parallel_saved_work_{dialog_nonce}",
    )
    understands_risk = st.checkbox(
        "I understand this may be slower or make the computer unstable.",
        key=f"birefnet_parallel_understands_risk_{dialog_nonce}",
    )
    if st.button("Keep one BiRefNet worker", width="stretch"):
        st.session_state["birefnet_parallel_dialog_nonce"] = dialog_nonce + 1
        st.rerun()
    if st.button(
        "Unlock CPU-parallel BiRefNet for one run",
        type="primary",
        icon=":material/warning:",
        width="stretch",
        disabled=not (saved_work and understands_risk),
    ):
        st.session_state[BIREFNET_PARALLEL_KEY] = available_workers
        st.session_state["birefnet_parallel_dialog_nonce"] = dialog_nonce + 1
        st.rerun()
