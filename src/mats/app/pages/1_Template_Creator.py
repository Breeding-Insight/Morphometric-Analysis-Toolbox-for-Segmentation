from pathlib import Path

import streamlit as st

from mats.template_exports import render_template_idml, render_template_pdf
from mats.template_layout import (
    MAX_ASPECT_RATIO,
    TemplateLayoutError,
    build_template_layout,
    format_measurement,
    maximum_template_edge,
    minimum_template_edge,
    round_to_increment,
)


APP_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = APP_DIR / "assets"


def _clear_generated_template():
    st.session_state.pop("generated_template", None)


def _normalize_dimensions():
    """Keep both dimension widgets on the promised half-unit grid."""
    for key in ("template_width", "template_length"):
        st.session_state[key] = round_to_increment(st.session_state[key])
    _clear_generated_template()


def _convert_dimension_unit():
    previous_unit = st.session_state.template_previous_unit
    current_unit = st.session_state.template_unit
    if previous_unit == current_unit:
        return

    factor = 2.54 if previous_unit == "in" and current_unit == "cm" else 1.0 / 2.54
    minimum = minimum_template_edge(current_unit)
    maximum = maximum_template_edge(current_unit)
    for key in ("template_width", "template_length"):
        converted = round_to_increment(st.session_state[key] * factor)
        st.session_state[key] = min(max(converted, minimum), maximum)
    st.session_state.template_previous_unit = current_unit
    _clear_generated_template()


def _layout_fingerprint(layout):
    return (
        layout.unit,
        round(layout.page_width_in, 8),
        round(layout.page_length_in, 8),
        layout.qr_payload,
        layout.marker_diameter_in,
        tuple(round(value, 8) for value in layout.qr_bounds_in),
    )


def _file_stem(layout):
    width = format_measurement(layout.page_width)
    length = format_measurement(layout.page_length)
    return f"{width}x{length}{layout.unit}_template"


mark_icon = ASSETS_DIR / "mats_mark_circles_color.svg"
st.set_page_config(
    page_title="MATs — Template creator",
    page_icon=str(mark_icon) if mark_icon.is_file() else "🌿",
    layout="wide",
)

logo_wide = ASSETS_DIR / "mats_logo_horizontal_circles_color.svg"
if logo_wide.is_file():
    st.logo(
        str(logo_wide),
        size="large",
        icon_image=str(mark_icon) if mark_icon.is_file() else None,
    )

st.session_state.setdefault("template_unit", "in")
st.session_state.setdefault("template_previous_unit", "in")
st.session_state.setdefault("template_width", 12.0)
st.session_state.setdefault("template_length", 12.0)

st.title("Template creator")
st.caption(
    "Choose the finished print-sheet size. Marker size and calibrated area are "
    "calculated automatically."
)

with st.expander("How this works"):
    st.markdown(
        "- Enter only the finished sheet's **width** and **length**.\n"
        f"- Templates may be no more elongated than **{MAX_ASPECT_RATIO:g}:1** "
        "so the calibrated area remains usable.\n"
        "- Marker diameter scales automatically from the sheet's longest edge: "
        "0.5 in below 12 in, 0.75 in from 12 to under 18 in, 1 in from 18 "
        "to under 24 in, and 1.5 in at 24 in and above.\n"
        "- Metric marker tiers are 1.5 cm below 30.5 cm, 2 cm to under 45.5 cm, "
        "2.5 cm to under 61 cm, and 4 cm at 61 cm and above.\n"
        "- Dimensions snap to 0.5 in or 0.5 cm. The observation box uses fixed "
        "margins: 1.5 in top / 1 in elsewhere, or 3.5 cm top / 2.5 cm elsewhere.\n"
        "- The QR code is centered in the top header.\n"
        "- The QR code stores the derived marker-to-marker observation area, "
        "which MATs uses for measurement scale.\n"
        "- Use PDF for printing. IDML is editable in Adobe InDesign; changing "
        "marker positions or scaling the artwork changes the calibration."
    )

unit = st.segmented_control(
    "Unit",
    options=["in", "cm"],
    format_func={"in": "Inches", "cm": "Centimeters"}.get,
    required=True,
    key="template_unit",
    on_change=_convert_dimension_unit,
    persist_state="page",
)

minimum = minimum_template_edge(unit)
maximum = maximum_template_edge(unit)
step = 0.5

width_column, length_column = st.columns(2)
with width_column:
    width = st.number_input(
        "Template width",
        min_value=minimum,
        max_value=maximum,
        step=step,
        format="%.1f",
        key="template_width",
        on_change=_normalize_dimensions,
        help="Horizontal size of the finished print sheet.",
        icon=":material/width:",
        persist_state="page",
    )
with length_column:
    length = st.number_input(
        "Template length",
        min_value=minimum,
        max_value=maximum,
        step=step,
        format="%.1f",
        key="template_length",
        on_change=_normalize_dimensions,
        help="Vertical size of the finished print sheet.",
        icon=":material/height:",
        persist_state="page",
    )

layout = None
layout_error = None
try:
    layout = build_template_layout(width, length, unit)
except TemplateLayoutError as exc:
    layout_error = str(exc)
    st.error(layout_error, icon=":material/error:")

if layout is not None:
    with st.container(border=True):
        st.subheader("Calculated layout")
        summary_columns = st.columns(3)
        summary_columns[0].metric(
            "Print sheet",
            f"{format_measurement(layout.page_width)} × "
            f"{format_measurement(layout.page_length)} {layout.unit}",
        )
        summary_columns[1].metric(
            "Observation area",
            f"{format_measurement(layout.observation_width)} × "
            f"{format_measurement(layout.observation_length)} {layout.unit}",
            help="Distance between marker centers; this is encoded in the QR code.",
        )
        summary_columns[2].metric(
            "Marker diameter",
            f"{format_measurement(layout.marker_diameter)} {layout.unit}",
        )
        st.caption(
            f"Aspect ratio: {layout.aspect_ratio:.2f}:1 "
            f"(maximum {MAX_ASPECT_RATIO:g}:1) · QR payload: "
            f"`{layout.qr_payload}` · Margins: "
            f"{format_measurement(layout.in_unit(layout.top_inset_in))}{layout.unit} top, "
            f"{format_measurement(layout.in_unit(layout.side_inset_in))}{layout.unit} elsewhere"
        )

prepare = st.button(
    "Prepare downloads",
    type="primary",
    icon=":material/file_export:",
    disabled=layout_error is not None,
)

if prepare and layout is not None:
    try:
        pdf_bytes = render_template_pdf(layout)
        idml_bytes = render_template_idml(layout)
    except (ImportError, OSError, ValueError) as exc:
        st.error(f"Template export failed: {exc}", icon=":material/error:")
    else:
        st.session_state.generated_template = {
            "fingerprint": _layout_fingerprint(layout),
            "pdf": pdf_bytes,
            "idml": idml_bytes,
            "file_stem": _file_stem(layout),
        }

generated = st.session_state.get("generated_template")
if (
    generated is not None
    and layout is not None
    and generated["fingerprint"] == _layout_fingerprint(layout)
):
    st.success("Your print and editable files are ready.")
    with st.container(horizontal=True):
        st.download_button(
            "Download print PDF",
            data=generated["pdf"],
            file_name=f"{generated['file_stem']}.pdf",
            mime="application/pdf",
            type="primary",
            icon=":material/picture_as_pdf:",
            on_click="ignore",
        )
        st.download_button(
            "Download editable IDML",
            data=generated["idml"],
            file_name=f"{generated['file_stem']}.idml",
            mime="application/vnd.adobe.indesign-idml-package",
            icon=":material/edit_document:",
            on_click="ignore",
        )
    st.caption(
        "Print the PDF at 100% scale. If you edit the IDML, do not move the "
        "markers independently or scale the page artwork."
    )
