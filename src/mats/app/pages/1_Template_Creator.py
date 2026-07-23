import io
from pathlib import Path

import streamlit as st

# Single source of truth for the dimension format the pipeline expects. The QR
# payload generated below must round-trip through this parser, so importing it
# (rather than re-copying the regex) keeps templates and the pipeline in sync.
from mats.dimensions import parse_template_dimensions

APP_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = APP_DIR / "assets"


# Extracted from the CMYK fill operator in the InDesign-authored template
# PDFs' content streams -- reused so generated markers match the color the
# RF-DETR marker detector was trained against.
MARKER_CMYK = (0.15, 1.0, 1.0, 0.0)

TOP_CLEARANCE_IN = 1.4
BOTTOM_MARGIN_IN = 0.3
QR_SIZE_IN = 1.0
QR_TOP_MARGIN_IN = 0.3
QR_LEFT_MARGIN_IN = 2.0
LABEL_LEFT_MARGIN_IN = 0.2
LABEL_TOP_MARGIN_IN = 0.2


def render_template_pdf(canvas_w, canvas_h, box_w, box_h, unit, marker_diameter):
    from reportlab.lib.colors import CMYKColor, black
    from reportlab.lib.units import cm, inch
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas as pdfcanvas
    import qrcode

    unit_pt = inch if unit == "in" else cm

    page_w = canvas_w * unit_pt
    page_h = canvas_h * unit_pt
    box_w_pt = box_w * unit_pt
    box_h_pt = box_h * unit_pt
    marker_r = (marker_diameter * unit_pt) / 2

    top_clearance_pt = TOP_CLEARANCE_IN * inch
    bottom_margin_pt = BOTTOM_MARGIN_IN * inch

    if box_w_pt > page_w or (box_h_pt + top_clearance_pt + bottom_margin_pt) > page_h:
        raise ValueError(
            "Observation box does not fit within the template canvas at the "
            "required clearances. Increase the canvas size or reduce the box."
        )

    box_x = (page_w - box_w_pt) / 2
    box_y_top = page_h - top_clearance_pt
    box_y_bottom = box_y_top - box_h_pt

    # QR encodes the observation-box size (not the canvas) -- this is what the
    # pipeline auto-detects when --template_dimensions isn't supplied manually.
    payload = f"{box_w:g}x{box_h:g}{unit}"
    if parse_template_dimensions(payload) != (float(box_w), float(box_h), unit):
        raise ValueError(f"Generated QR payload '{payload}' failed to round-trip; aborting.")

    buf = io.BytesIO()
    c = pdfcanvas.Canvas(buf, pagesize=(page_w, page_h))

    c.setStrokeColor(black)
    c.setLineWidth(1)
    c.rect(box_x, box_y_bottom, box_w_pt, box_h_pt, stroke=1, fill=0)

    marker_color = CMYKColor(*MARKER_CMYK)
    c.setFillColor(marker_color)
    c.setStrokeColor(marker_color)
    for cx, cy in (
        (box_x, box_y_top),
        (box_x + box_w_pt, box_y_top),
        (box_x, box_y_bottom),
        (box_x + box_w_pt, box_y_bottom),
    ):
        c.circle(cx, cy, marker_r, stroke=0, fill=1)

    qr_img = qrcode.make(payload)
    qr_buf = io.BytesIO()
    qr_img.save(qr_buf, format="PNG")
    qr_buf.seek(0)
    qr_pt = QR_SIZE_IN * inch
    qr_x = QR_LEFT_MARGIN_IN * inch
    qr_y = page_h - QR_TOP_MARGIN_IN * inch - qr_pt
    c.drawImage(ImageReader(qr_buf), qr_x, qr_y, width=qr_pt, height=qr_pt)

    c.setFillColor(black)
    c.setFont("Helvetica", 10)
    label_x = LABEL_LEFT_MARGIN_IN * inch
    label_y = page_h - LABEL_TOP_MARGIN_IN * inch - 10
    lines = [
        f"{canvas_w:g}{unit} x {canvas_h:g}{unit} template",
        f"{box_w:g}{unit} x {box_h:g}{unit} observation box",
        f"{marker_diameter:g}{unit} marker diameter",
    ]
    for i, line in enumerate(lines):
        c.drawString(label_x, label_y - i * 14, line)

    c.showPage()
    c.save()
    return buf.getvalue()


def main():
    mark_icon = ASSETS_DIR / "mats_mark.svg"
    st.set_page_config(
        page_title="MATs — Template Creator",
        page_icon=str(mark_icon) if mark_icon.is_file() else "🌿",
        layout="wide",
    )

    logo_wide = ASSETS_DIR / "mats_logo_horizontal.svg"
    if logo_wide.is_file():
        st.logo(
            str(logo_wide),
            icon_image=str(mark_icon) if mark_icon.is_file() else None,
        )

    st.title("Template Creator")
    st.caption("Generate a print-ready calibration template at any size.")

    with st.expander("How this works"):
        st.markdown(
            "- The **observation box** is the area the four markers bound; it "
            "does not have to match the canvas size or aspect ratio (e.g. a "
            "square canvas can hold a non-square box).\n"
            "- The QR code encodes the observation-box size in the exact "
            "`<width>x<height><unit>` format the pipeline's `--template_dimensions` "
            "flag expects, so the app can auto-detect scale from a photo "
            "without the box size being typed in manually.\n"
            "- Marker color matches the CMYK value used in the original "
            "InDesign-authored templates."
        )

    unit = st.selectbox("Unit", ["in", "cm"], index=0)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Canvas (print sheet)")
        canvas_w = st.number_input("Canvas width", min_value=1.0, value=12.0, step=0.5)
        canvas_h = st.number_input("Canvas height", min_value=1.0, value=12.0, step=0.5)
    with col2:
        st.subheader("Observation box")
        box_w = st.number_input("Box width", min_value=0.5, value=10.5, step=0.5)
        box_h = st.number_input("Box height", min_value=0.5, value=9.5, step=0.5)

    default_marker = min(1.0, max(0.25, round(min(box_w, box_h) / 16, 2)))
    marker_diameter = st.number_input(
        "Marker diameter",
        min_value=0.1,
        value=default_marker,
        step=0.05,
        help="Corner fiducial marker diameter. Larger canvases typically use a larger marker.",
    )

    if st.button("Generate template PDF"):
        try:
            pdf_bytes = render_template_pdf(
                canvas_w, canvas_h, box_w, box_h, unit, marker_diameter
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            file_name = f"{canvas_w:g}x{canvas_h:g}{unit}_template.pdf"
            st.success("Template generated.")
            st.download_button(
                "Download template PDF",
                data=pdf_bytes,
                file_name=file_name,
                mime="application/pdf",
            )


if __name__ == "__main__":
    main()
