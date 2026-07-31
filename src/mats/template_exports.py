"""Document renderers for calibration templates."""

import io
from xml.sax.saxutils import escape
import zipfile

from .dimensions import parse_template_dimensions
from .template_layout import (
    POINTS_PER_INCH,
    TemplateLayout,
)


# Extracted from the CMYK fill operator in the InDesign-authored template PDFs.
MARKER_CMYK = (0.15, 1.0, 1.0, 0.0)
IDML_MIME_TYPE = "application/vnd.adobe.indesign-idml-package"
IDML_PACKAGE_NAMESPACE = "http://ns.adobe.com/AdobeInDesign/idml/1.0/packaging"
LABEL_FONT_SIZE = 8
LABEL_LEADING = 10


def _assert_qr_payload(layout):
    parsed = parse_template_dimensions(layout.qr_payload)
    expected = (
        float(f"{layout.observation_width:.4f}"),
        float(f"{layout.observation_length:.4f}"),
        layout.unit,
    )
    if parsed != expected:
        raise ValueError(
            f"Generated QR payload '{layout.qr_payload}' failed to round-trip; "
            "aborting."
        )


def render_template_pdf(layout):
    """Return a print-ready PDF for ``layout``."""
    from reportlab.lib.colors import CMYKColor, black
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas as pdfcanvas
    import qrcode

    if not isinstance(layout, TemplateLayout):
        raise TypeError("layout must be a TemplateLayout")
    _assert_qr_payload(layout)

    page_w = layout.page_width_in * POINTS_PER_INCH
    page_h = layout.page_length_in * POINTS_PER_INCH
    marker_r = layout.marker_diameter_in * POINTS_PER_INCH / 2.0
    top, left, bottom, right = (
        value * POINTS_PER_INCH for value in layout.observation_bounds_in
    )

    buf = io.BytesIO()
    document = pdfcanvas.Canvas(buf, pagesize=(page_w, page_h))

    document.setStrokeColor(black)
    document.setLineWidth(1)
    document.rect(
        left,
        page_h - bottom,
        right - left,
        bottom - top,
        stroke=1,
        fill=0,
    )

    marker_color = CMYKColor(*MARKER_CMYK)
    document.setFillColor(marker_color)
    document.setStrokeColor(marker_color)
    for center_x, center_y in layout.marker_centers_in:
        document.circle(
            center_x * POINTS_PER_INCH,
            page_h - center_y * POINTS_PER_INCH,
            marker_r,
            stroke=0,
            fill=1,
        )

    qr_img = qrcode.make(layout.qr_payload)
    qr_buf = io.BytesIO()
    qr_img.save(qr_buf, format="PNG")
    qr_buf.seek(0)
    qr_top, qr_left, qr_bottom, qr_right = layout.qr_bounds_in
    qr_width = (qr_right - qr_left) * POINTS_PER_INCH
    qr_height = (qr_bottom - qr_top) * POINTS_PER_INCH
    document.drawImage(
        ImageReader(qr_buf),
        qr_left * POINTS_PER_INCH,
        page_h - qr_bottom * POINTS_PER_INCH,
        width=qr_width,
        height=qr_height,
    )

    document.setFillColor(black)
    document.setFont("Helvetica", LABEL_FONT_SIZE)
    label_top, label_left, _, _ = layout.label_bounds_in
    label_x = label_left * POINTS_PER_INCH
    label_y = page_h - label_top * POINTS_PER_INCH - LABEL_FONT_SIZE
    for index, line in enumerate(layout.artwork_labels):
        document.drawString(label_x, label_y - index * LABEL_LEADING, line)

    document.showPage()
    document.save()
    return buf.getvalue()


def _xml_header(kind):
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<?aid style="50" type="{kind}" readerVersion="6.0" '
        'featureSet="257"?>\n'
    )


def _idml_number(value):
    value = 0.0 if abs(float(value)) < 1e-9 else float(value)
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _path_point(x, y, left_x=None, left_y=None, right_x=None, right_y=None):
    left_x = x if left_x is None else left_x
    left_y = y if left_y is None else left_y
    right_x = x if right_x is None else right_x
    right_y = y if right_y is None else right_y
    return (
        '<PathPoint Anchor="{anchor}" LeftDirection="{left}" '
        'RightDirection="{right}"/>'
    ).format(
        anchor=f"{_idml_number(x)} {_idml_number(y)}",
        left=f"{_idml_number(left_x)} {_idml_number(left_y)}",
        right=f"{_idml_number(right_x)} {_idml_number(right_y)}",
    )


def _rect_path(top, left, bottom, right):
    points = (
        _path_point(left, top),
        _path_point(right, top),
        _path_point(right, bottom),
        _path_point(left, bottom),
    )
    return (
        '<GeometryPath PathOpen="false"><PathPointArray>'
        + "".join(points)
        + "</PathPointArray></GeometryPath>"
    )


def _oval_path(center_x, center_y, radius):
    # Cubic Bezier control distance for a circle.
    control = radius * 0.5522847498307936
    points = (
        _path_point(
            center_x,
            center_y - radius,
            center_x - control,
            center_y - radius,
            center_x + control,
            center_y - radius,
        ),
        _path_point(
            center_x + radius,
            center_y,
            center_x + radius,
            center_y - control,
            center_x + radius,
            center_y + control,
        ),
        _path_point(
            center_x,
            center_y + radius,
            center_x + control,
            center_y + radius,
            center_x - control,
            center_y + radius,
        ),
        _path_point(
            center_x - radius,
            center_y,
            center_x - radius,
            center_y + control,
            center_x - radius,
            center_y - control,
        ),
    )
    return (
        '<GeometryPath PathOpen="false"><PathPointArray>'
        + "".join(points)
        + "</PathPointArray></GeometryPath>"
    )


def _page_item(
    tag,
    self_id,
    path_geometry,
    fill,
    stroke,
    stroke_weight,
    *,
    name,
    layer="layer_geometry",
):
    return (
        f'<{tag} Self="{self_id}" ContentType="Unassigned" '
        f'Name="{escape(name)}" ItemLayer="{layer}" ParentPage="page_mats" '
        'ItemTransform="1 0 0 1 0 0" '
        f'FillColor="{fill}" StrokeColor="{stroke}" '
        f'StrokeWeight="{_idml_number(stroke_weight)}">'
        f"<Properties><PathGeometry>{path_geometry}</PathGeometry></Properties>"
        f"</{tag}>"
    )


def _qr_matrix(payload):
    import qrcode
    from qrcode.constants import ERROR_CORRECT_M

    qr = qrcode.QRCode(
        version=None,
        error_correction=ERROR_CORRECT_M,
        box_size=1,
        border=4,
    )
    qr.add_data(payload)
    qr.make(fit=True)
    return qr.get_matrix()


def _spread_xml(layout):
    page_w = layout.page_width_in * POINTS_PER_INCH
    page_h = layout.page_length_in * POINTS_PER_INCH
    page_left = -page_w / 2.0
    page_top = -page_h / 2.0

    observation_top, observation_left, observation_bottom, observation_right = (
        value * POINTS_PER_INCH for value in layout.observation_bounds_in
    )
    observation = _page_item(
        "Rectangle",
        "observation_box",
        _rect_path(
            page_top + observation_top,
            page_left + observation_left,
            page_top + observation_bottom,
            page_left + observation_right,
        ),
        "Swatch/None",
        "Color/Black",
        1,
        name="Observation box",
    )

    marker_radius = layout.marker_diameter_in * POINTS_PER_INCH / 2.0
    markers = []
    for index, (center_x, center_y) in enumerate(layout.marker_centers_in, start=1):
        markers.append(
            _page_item(
                "Oval",
                f"marker_{index}",
                _oval_path(
                    page_left + center_x * POINTS_PER_INCH,
                    page_top + center_y * POINTS_PER_INCH,
                    marker_radius,
                ),
                "Color/MATS Marker",
                "Swatch/None",
                0,
                name=f"Calibration marker {index}",
            )
        )

    matrix = _qr_matrix(layout.qr_payload)
    qr_top, qr_left, qr_bottom, qr_right = layout.qr_bounds_in
    qr_module_width = (qr_right - qr_left) * POINTS_PER_INCH / len(matrix)
    qr_module_height = (qr_bottom - qr_top) * POINTS_PER_INCH / len(matrix)
    qr_paths = []
    for row_index, row in enumerate(matrix):
        for column_index, is_dark in enumerate(row):
            if not is_dark:
                continue
            top = page_top + (qr_top * POINTS_PER_INCH) + row_index * qr_module_height
            left = page_left + (qr_left * POINTS_PER_INCH) + column_index * qr_module_width
            qr_paths.append(
                _rect_path(
                    top,
                    left,
                    top + qr_module_height,
                    left + qr_module_width,
                )
            )
    qr_code = _page_item(
        "Polygon",
        "qr_code",
        "".join(qr_paths),
        "Color/Black",
        "Swatch/None",
        0,
        name=f"QR code ({layout.qr_payload})",
        layer="layer_annotations",
    )

    label_top, label_left, label_bottom, label_right = (
        value * POINTS_PER_INCH for value in layout.label_bounds_in
    )
    label_path = _rect_path(
        page_top + label_top,
        page_left + label_left,
        page_top + label_bottom,
        page_left + label_right,
    )
    label_frame = (
        '<TextFrame Self="label_frame" Name="Sheet and calibration information" '
        'ParentStory="story_labels" '
        'PreviousTextFrame="n" NextTextFrame="n" ContentType="TextType" '
        'ItemLayer="layer_annotations" ParentPage="page_mats" '
        'ItemTransform="1 0 0 1 0 0" FillColor="Swatch/None" '
        'StrokeColor="Swatch/None" StrokeWeight="0">'
        f"<Properties><PathGeometry>{label_path}</PathGeometry></Properties>"
        '<TextFramePreference TextColumnCount="1" TextColumnGutter="12" '
        'FirstBaselineOffset="AscentOffset" VerticalJustification="TopAlign">'
        '<Properties><InsetSpacing type="list">'
        '<ListItem type="unit">0</ListItem><ListItem type="unit">0</ListItem>'
        '<ListItem type="unit">0</ListItem><ListItem type="unit">0</ListItem>'
        "</InsetSpacing></Properties></TextFramePreference>"
        "</TextFrame>"
    )

    return (
        _xml_header("document")
        + f'<idPkg:Spread xmlns:idPkg="{IDML_PACKAGE_NAMESPACE}" DOMVersion="6.0">'
        '<Spread Self="spread_mats" PageCount="1" '
        'ItemTransform="1 0 0 1 0 0" ShowMasterItems="true">'
        f'<Page Self="page_mats" GeometricBounds="0 0 {_idml_number(page_h)} '
        f'{_idml_number(page_w)}" ItemTransform="1 0 0 1 '
        f'{_idml_number(page_left)} {_idml_number(page_top)}" '
        'Name="1" AppliedMaster="n"/>'
        + observation
        + "".join(markers)
        + qr_code
        + label_frame
        + "</Spread></idPkg:Spread>"
    )


def _story_xml(layout):
    lines = layout.artwork_labels
    contents = "<Br/>".join(f"<Content>{escape(line)}</Content>" for line in lines)
    return (
        _xml_header("document")
        + f'<idPkg:Story xmlns:idPkg="{IDML_PACKAGE_NAMESPACE}" DOMVersion="6.0">'
        '<Story Self="story_labels" UserText="true" TrackChanges="false" '
        'StoryTitle="$ID/">'
        '<StoryPreference OpticalMarginAlignment="false" OpticalMarginSize="12" '
        'FrameType="TextFrameType" StoryOrientation="Horizontal" '
        'StoryDirection="LeftToRightDirection"/>'
        '<ParagraphStyleRange '
        'AppliedParagraphStyle="ParagraphStyle/$ID/[No paragraph style]">'
        '<CharacterStyleRange '
        'AppliedCharacterStyle="CharacterStyle/$ID/[No character style]" '
        f'AppliedFont="Arial" FontStyle="Regular" PointSize="{LABEL_FONT_SIZE}" '
        f'Leading="{LABEL_LEADING}" '
        'FillColor="Color/Black">'
        + contents
        + "</CharacterStyleRange></ParagraphStyleRange></Story></idPkg:Story>"
    )


def _designmap_xml():
    return (
        _xml_header("document")
        + f'<Document xmlns:idPkg="{IDML_PACKAGE_NAMESPACE}" DOMVersion="6.0" '
        'Self="d" StoryList="story_labels" ActiveLayer="layer_annotations" '
        'ZeroPoint="0 0">'
        '<idPkg:Graphic src="Resources/Graphic.xml"/>'
        '<idPkg:Styles src="Resources/Styles.xml"/>'
        '<idPkg:Fonts src="Resources/Fonts.xml"/>'
        '<idPkg:Preferences src="Resources/Preferences.xml"/>'
        '<Layer Self="layer_geometry" Name="Calibration geometry" Visible="true" '
        'Locked="false" Printable="true"/>'
        '<Layer Self="layer_annotations" Name="QR and labels" Visible="true" '
        'Locked="false" Printable="true"/>'
        '<idPkg:Spread src="Spreads/Spread_mats.xml"/>'
        '<idPkg:Story src="Stories/Story_labels.xml"/>'
        "</Document>"
    )


def _graphic_xml():
    return (
        _xml_header("document")
        + f'<idPkg:Graphic xmlns:idPkg="{IDML_PACKAGE_NAMESPACE}" DOMVersion="6.0">'
        '<Color Self="Color/Paper" Model="Process" Space="CMYK" '
        'ColorValue="0 0 0 0" Name="Paper" ColorEditable="true" '
        'ColorRemovable="false" Visible="true"/>'
        '<Color Self="Color/Black" Model="Process" Space="CMYK" '
        'ColorValue="0 0 0 100" Name="Black" ColorEditable="false" '
        'ColorRemovable="false" Visible="true"/>'
        '<Color Self="Color/MATS Marker" Model="Process" Space="CMYK" '
        'ColorValue="15 100 100 0" Name="MATS Marker" ColorEditable="true" '
        'ColorRemovable="true" Visible="true"/>'
        '<Swatch Self="Swatch/None" Name="None" ColorEditable="false" '
        'ColorRemovable="false" Visible="true"/>'
        '<StrokeStyle Self="StrokeStyle/$ID/Solid" Name="$ID/Solid"/>'
        "</idPkg:Graphic>"
    )


def _styles_xml():
    return (
        _xml_header("document")
        + f'<idPkg:Styles xmlns:idPkg="{IDML_PACKAGE_NAMESPACE}" DOMVersion="6.0">'
        '<RootCharacterStyleGroup Self="character_style_root">'
        '<CharacterStyle Self="CharacterStyle/$ID/[No character style]" '
        'Name="$ID/[No character style]"/></RootCharacterStyleGroup>'
        '<RootParagraphStyleGroup Self="paragraph_style_root">'
        '<ParagraphStyle Self="ParagraphStyle/$ID/[No paragraph style]" '
        'Name="$ID/[No paragraph style]" FillColor="Color/Black" '
        'PointSize="10" FontStyle="Regular">'
        '<Properties><AppliedFont type="string">Arial</AppliedFont>'
        "</Properties></ParagraphStyle></RootParagraphStyleGroup>"
        '<RootObjectStyleGroup Self="object_style_root">'
        '<ObjectStyle Self="ObjectStyle/$ID/[None]" Name="$ID/[None]" '
        'FillColor="Swatch/None" StrokeColor="Swatch/None" StrokeWeight="0"/>'
        "</RootObjectStyleGroup></idPkg:Styles>"
    )


def _fonts_xml():
    return (
        _xml_header("document")
        + f'<idPkg:Fonts xmlns:idPkg="{IDML_PACKAGE_NAMESPACE}" DOMVersion="6.0">'
        '<FontFamily Self="font_family_arial" Name="Arial">'
        '<Font Self="font_arial_regular" FontFamily="Arial" '
        'Name="Arial\tRegular" PostScriptName="ArialMT" Status="Installed" '
        'FontStyleName="Regular" FontType="TrueType" FullName="Arial"/>'
        "</FontFamily></idPkg:Fonts>"
    )


def _preferences_xml(layout):
    page_w = layout.page_width_in * POINTS_PER_INCH
    page_h = layout.page_length_in * POINTS_PER_INCH
    return (
        _xml_header("document")
        + f'<idPkg:Preferences xmlns:idPkg="{IDML_PACKAGE_NAMESPACE}" '
        'DOMVersion="6.0">'
        f'<DocumentPreference PageHeight="{_idml_number(page_h)}" '
        f'PageWidth="{_idml_number(page_w)}" PagesPerDocument="1" '
        'FacingPages="false" AllowPageShuffle="true" '
        'PageOrientation="Portrait"/>'
        "</idPkg:Preferences>"
    )


def _container_xml():
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<container version="1.0" '
        'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        '<rootfiles><rootfile full-path="designmap.xml" '
        f'media-type="{IDML_MIME_TYPE}"/></rootfiles></container>'
    )


def _write_idml_part(package, name, content, compression=zipfile.ZIP_DEFLATED):
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = compression
    package.writestr(info, content.encode("utf-8"))


def render_template_idml(layout):
    """Return an editable IDML package for ``layout``."""
    if not isinstance(layout, TemplateLayout):
        raise TypeError("layout must be a TemplateLayout")
    _assert_qr_payload(layout)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as package:
        # IDML follows the UCF convention: mimetype is first and uncompressed.
        _write_idml_part(
            package,
            "mimetype",
            IDML_MIME_TYPE,
            compression=zipfile.ZIP_STORED,
        )
        _write_idml_part(package, "META-INF/container.xml", _container_xml())
        _write_idml_part(package, "designmap.xml", _designmap_xml())
        _write_idml_part(package, "Resources/Graphic.xml", _graphic_xml())
        _write_idml_part(package, "Resources/Styles.xml", _styles_xml())
        _write_idml_part(package, "Resources/Fonts.xml", _fonts_xml())
        _write_idml_part(
            package,
            "Resources/Preferences.xml",
            _preferences_xml(layout),
        )
        _write_idml_part(package, "Spreads/Spread_mats.xml", _spread_xml(layout))
        _write_idml_part(package, "Stories/Story_labels.xml", _story_xml(layout))

    return buf.getvalue()
