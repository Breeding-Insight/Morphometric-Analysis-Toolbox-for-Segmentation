"""Compute status and the latest Home preflight report."""

import streamlit as st

from mats.app import branding
from mats.app.Home import (
    build_preflight_checks, load_pipeline_module, render_diagnostics,
)
from mats.app.compute import (
    birefnet_parallel_unlocked, break_glass_unlocked, get_compute_settings,
    resolve_execution_plan,
)


st.set_page_config(page_title="MATS — Diagnostics", page_icon=branding.page_icon(), layout="wide")
branding.apply_logo()

config = st.session_state.get("diagnostics_context")
inputs = st.session_state.get("diagnostics_inputs")
if config is None or inputs is None:
    st.title("Diagnostics")
    st.info("Open Home to check the current analysis settings and input images.")
    try:
        st.page_link("Home.py", label="Open Home", icon=":material/home:")
    except (KeyError, ValueError):
        st.caption("Open Home from the sidebar.")
else:
    lm = load_pipeline_module()
    compute = get_compute_settings(lm)
    workers = compute.available_workers
    plan = resolve_execution_plan(
        compute,
        "birefnet" if config["needs_birefnet"] else "threshold",
        birefnet_parallel_allowed=birefnet_parallel_unlocked(workers),
    )
    risk = lm.worker_risk_report(plan.workers, workers)
    checks, _, _ = build_preflight_checks(
        lm, config, *inputs, compute, plan, risk, break_glass_unlocked(workers),
    )
    render_diagnostics(compute, plan, config, checks)
