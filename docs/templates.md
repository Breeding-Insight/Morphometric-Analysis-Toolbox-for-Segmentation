# Calibration templates

MATs measures in real-world units by referencing four printed corner markers of
known spacing. The **Template Creator** generates the printable template; you can
also use your own as long as it follows the same conventions.

## Generating a template

In the app, open **Template Creator** in the sidebar:

1. Choose the **unit** (`in` or `cm`).
2. Set the **canvas** (the print sheet) and the **observation box** (the region
   the four markers bound). They need not match — a square sheet can hold a
   non-square box.
3. Set the **marker diameter**.
4. Click **Generate template PDF** and download it.

The generated PDF contains:

- a rectangular **observation box** outline,
- four **corner markers** in the exact color the detector was trained on
  (CMYK 0.15, 1.0, 1.0, 0.0),
- a **QR code** encoding the box size as `<w>x<h><unit>`, and
- a text label of the sizes.

## Printing and photographing

- Print at **100% scale** — turn off "fit to page" / "shrink to fit." Scaling
  breaks the real-world calibration.
- Lay leaves flat inside the observation box, not overlapping the markers.
- Photograph the whole sheet, as flat and square-on as practical, with all four
  markers in frame. MATs corrects moderate perspective, but keep the markers
  crisp and unobstructed.

## How the QR fits in

The QR encodes the observation-box dimensions. When you don't pass
`--template_dimensions` (CLI) or leave the field blank (GUI), MATs reads the QR
to recover the scale automatically. The Template Creator verifies each payload
round-trips through the pipeline's parser before writing the PDF, so a generated
template is always readable.

## Marker color

The corner markers use a specific CMYK value so they match what the RF-DETR
detector learned. If you author templates in other software, reproduce that
color (CMYK 0.15, 1.0, 1.0, 0.0) for best detection.
