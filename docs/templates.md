# Calibration templates

MATs measures in real-world units by referencing four printed corner markers of
known spacing. The **Template Creator** generates the printable template; you can
also use your own as long as it follows the same conventions.

## Generating a template

In the app, open **Template Creator** in the sidebar:

1. Choose the **unit** (`in` or `cm`).
2. Enter the finished print sheet's **width** and **length**.
3. Review the automatically calculated observation area and marker diameter.
4. Click **Prepare downloads**, then download the print PDF or editable IDML.

Dimensions use 0.5-unit increments in either unit. If you type another value,
the app rounds it to the nearest 0.5; switching units converts each dimension
and applies the same rounding. The width-to-length ratio cannot exceed
**1.5:1**. For example, 12 × 8 in is allowed, while 10 × 5 in is not. The app
reports the minimum acceptable shorter edge when a sheet is too elongated.

The marker-center observation box uses fixed unit-native margins:

| Unit | Top margin | Other margins |
| --- | ---: | ---: |
| Inches | 1.5 in | 1 in |
| Centimeters | 3.5 cm | 2.5 cm |

Therefore, the observation width is the sheet width minus 2 in or 5 cm, and
the observation length is the sheet length minus 2.5 in or 6 cm. These margins
do not change when a larger marker tier is selected.

Marker diameter is selected from the sheet's longest edge:

| Longest edge (inches) | Marker diameter |
| --- | --- |
| Under 12 in | 0.5 in |
| 12 in to under 18 in | 0.75 in |
| 18 in to under 24 in | 1 in |
| 24 in and above | 1.5 in |

Centimeter templates use matching clean metric tiers:

| Longest edge (centimeters) | Marker diameter |
| --- | --- |
| Under 30.5 cm | 1.5 cm |
| 30.5 cm to under 45.5 cm | 2 cm |
| 45.5 cm to under 61 cm | 2.5 cm |
| 61 cm and above | 4 cm |

The generated PDF contains:

- a rectangular **observation box** outline,
- four **corner markers** in the exact color the detector was trained on
  (CMYK 0.15, 1.0, 1.0, 0.0),
- a **QR code**, centered in the top header, encoding the box size as
  `<w>x<h><unit>`, and
- a text label of the sizes.

The **PDF is the recommended print file**. The IDML contains editable native
InDesign objects on separate **Calibration geometry** and **QR and labels**
layers. Editing the annotation text is safe, but moving markers, resizing the
observation box, or scaling the artwork changes the calibration. If marker
spacing is changed, the QR payload must be updated to exactly match it.

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
round-trips through the pipeline's parser before writing either export, so a
generated template is always readable. The entered width and length describe
the outer print sheet; the calculated observation area describes the
marker-center spacing and is the value stored in the QR.

## Marker color

The corner markers use a specific CMYK value so they match what the RF-DETR
detector learned. If you author templates in other software, reproduce that
color (CMYK 0.15, 1.0, 1.0, 0.0) for best detection.
