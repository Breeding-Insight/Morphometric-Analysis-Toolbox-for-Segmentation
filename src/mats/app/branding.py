"""Shared MATS branding for the Streamlit pages."""

import base64
from datetime import date
from functools import lru_cache
from html import escape
from pathlib import Path

import streamlit as st

from mats import __version__

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_WIDE = ASSETS_DIR / "mats_logo_horizontal_circles_color.svg"
LOGO_MARK = ASSETS_DIR / "mats_mark_circles_color.svg"
LOGOS_DIR = ASSETS_DIR / "logos"

FUNDING_STATEMENT = (
    "Breeding Insight is funded by the U.S. Department of Agriculture (USDA) "
    "Agricultural Research Service (ARS) through University of Florida/IFAS. "
    "Formerly at Cornell University."
)

# White logos for the dark footer band, as on Breeding Insight's other app
# landing pages (e.g. AlloMate). (file, alt text naming the organization,
# display height in px)
FUNDER_LOGOS = (
    ("breeding-insight-white.png", "Breeding Insight", 56),
    ("usda-ars-white.png", "USDA Agricultural Research Service", 44),
    ("uf-ifas-white.png", "University of Florida Institute of Food and Agricultural Sciences (UF/IFAS)", 44),
    ("cornell-white.png", "Cornell University", 44),
)

FOOTER_LINKS = (
    ("About Breeding Insight", "https://breedinginsight.org/about-bi/"),
    ("Contact us", "https://breedinginsight.org/contact-us/"),
    ("Source code", "https://github.com/Breeding-Insight/Morphometric-Analysis-Toolbox-for-Segmentation"),
)

# Footer band #12483B (a dark green already used in Home.py). White text is
# 10.4:1 on it and the muted #D4E4DA is 7.9:1.
_FUNDER_FOOTER_CSS = """
<style>
.mats-funder-footer {
    background: #12483B;
    border-radius: 0.75rem;
    color: #FFFFFF;
    margin-top: 3rem;
    padding: 1.75rem 1.5rem 1.25rem;
}
.mats-funder-logos {
    align-items: center;
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem 3rem;
    justify-content: center;
    list-style: none;
    margin: 0 0 1.25rem !important;
    padding: 0 !important;
}
.mats-funder-logos li { margin: 0 !important; padding: 0 !important; }
.mats-funder-logos li::marker { content: none; }
.mats-funder-logos img { display: block; max-width: 100%; width: auto; }
.mats-funder-footer .mats-funding {
    color: #FFFFFF;
    font-size: 1rem;
    font-weight: 500;
    line-height: 1.6;
    margin: 0 auto 1rem;
    max-width: none;  /* one or two lines at desktop widths */
    text-align: center;
}
.mats-footer-bottom {
    border-top: 1px solid rgba(212, 228, 218, 0.35);
    color: #D4E4DA;
    display: flex;
    flex-wrap: wrap;
    font-size: 0.9rem;
    gap: 0.75rem 1.5rem;
    justify-content: space-between;
    padding-top: 0.9rem;
}
.mats-footer-bottom p { color: #D4E4DA; margin: 0; }
.mats-footer-bottom ul {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem 1.25rem;
    list-style: none;
    margin: 0 !important;
    padding: 0 !important;
}
.mats-footer-bottom li { margin: 0 !important; padding: 0 !important; }
.mats-footer-bottom li::marker { content: none; }
.mats-funder-footer a { color: #FFFFFF !important; text-decoration: underline; text-underline-offset: 3px; }
.mats-funder-footer a:hover { color: #D4E4DA !important; }
.mats-funder-footer a:focus-visible { outline: 2px solid #FFFFFF !important; outline-offset: 3px !important; }
.mats-visually-hidden {
    border: 0; clip: rect(0 0 0 0); height: 1px; margin: -1px; overflow: hidden;
    padding: 0; position: absolute; white-space: nowrap; width: 1px;
}
@media (max-width: 640px) {
    .mats-funder-footer { padding: 2rem 1rem 1.25rem; }
    .mats-funder-logos { gap: 1.25rem 2rem; }
    .mats-funder-logos img { max-height: 36px; }
    .mats-funder-logos li:first-child img { max-height: 46px; }
    .mats-footer-bottom { justify-content: center; text-align: center; }
}
</style>
"""

# st.logo's biggest built-in size ("large") renders at 2rem; override it to
# stay legible at header scale.
_LOGO_HEIGHT = "8rem"
_SIDEBAR_LOGO_TOP_OFFSET = "2rem"
_SIDEBAR_NAV_TOP_GAP = "0.5rem"

# The collapsed-state header icon (stHeaderLogo) lives in a fixed 3.75rem
# header bar, so it gets its own modest size -- not the sidebar's 8rem -- plus
# a nudge down/right off the header's top-left corner where it defaults to
# sitting flush (default margin-left is explicitly 0), which otherwise reads
# as clipped/crowded next to the sidebar-expand control.
_HEADER_LOGO_HEIGHT = "2.5rem"

# Scoped to stSidebarLogo only: the wide lockup shown when the sidebar is
# expanded. st.logo also renders the separate stHeaderLogo above, which
# shares the same .stLogo class -- enlarging it the same way overflows the
# header bar and clips at the top.
_LOGO_CSS = f"""
<style>
[data-testid="stSidebarLogo"] {{
    height: {_LOGO_HEIGHT} !important;
    max-width: 100%;
    object-fit: contain;
    margin-top: {_SIDEBAR_LOGO_TOP_OFFSET} !important;
}}
[data-testid="stLogoSpacer"] {{
    height: calc({_LOGO_HEIGHT} + {_SIDEBAR_LOGO_TOP_OFFSET}) !important;
}}
[data-testid="stSidebarNav"] {{
    padding-top: {_SIDEBAR_NAV_TOP_GAP} !important;
}}
[data-testid="stHeaderLogo"] {{
    height: {_HEADER_LOGO_HEIGHT} !important;
    margin-top: 0.75rem !important;
    margin-left: 0.5rem !important;
}}
</style>
"""


# WCAG 2.x A/AA fixes shared by every page (apply_logo runs on all of them).
# Colors come from the theme in .streamlit/config.toml: #1F6F5B (primaryColor)
# and #5F6C66 (grayColor). Contrast ratios are against the theme backgrounds
# #FBFCF8 / #F4F7F1 (sidebar) / #F3F7F0 (inline code).
_A11Y_CSS = """
<style>
/* 1.4.3: Streamlit dims captions with opacity: 0.6 (about 3.8:1). Render
   them opaque in the theme gray instead, 5.1-5.3:1 on every background
   used. Links keep their color. */
[data-testid="stCaptionContainer"] {
    opacity: 1 !important;
}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] *:not(a) {
    color: #5F6C66 !important;
}
/* 1.4.3: icons Streamlit draws at 60% text opacity (input icons, sidebar
   arrows) measure about 3.7:1; Siteimprove treats icon-font glyphs as text. */
[data-testid="stTextInputIcon"] [data-testid="stIconMaterial"],
[data-testid="stNumberInputIcon"] [data-testid="stIconMaterial"],
[data-baseweb="input"] [data-testid="stIconMaterial"],
[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"],
[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"] {
    color: #5F6C66 !important;
}
/* Disabled checkbox/radio labels (e.g. Export types not saved in this run)
   fade to 2.3:1. WCAG exempts inactive controls, but checkers such as
   Siteimprove still flag them; the theme gray stays visibly lighter than
   active labels (#20302A) while reading at 5.3:1. */
[data-testid="stCheckbox"] label:has(input:disabled) [data-testid="stMarkdownContainer"],
[data-testid="stCheckbox"] label:has(input:disabled) [data-testid="stMarkdownContainer"] *,
[data-testid="stRadio"] label:has(input:disabled) [data-testid="stMarkdownContainer"],
[data-testid="stRadio"] label:has(input:disabled) [data-testid="stMarkdownContainer"] * {
    color: #5F6C66 !important;
    opacity: 1 !important;
}
/* 2.1.1: st.code renders <pre> with overflow:auto, a scroll region that
   keyboard users can't reach (Siteimprove: "Scrollable element is not
   keyboard accessible"). Wrap every line, breaking long paths anywhere, so
   the block never needs to scroll. Covers blocks without wrap_lines too. */
[data-testid="stCode"] pre {
    overflow: visible !important;
}
[data-testid="stCode"] pre,
[data-testid="stCode"] pre code {
    white-space: pre-wrap !important;
    overflow-wrap: anywhere !important;
}
/* 1.4.3: syntax-highlight colors in st.code are 2.4-3.5:1 on the code
   background #EDF3EA. Darker steps of the theme's own hues, all >= 4.8:1. */
[data-testid="stCode"] .token.comment,
[data-testid="stCode"] .token.prolog,
[data-testid="stCode"] .token.doctype,
[data-testid="stCode"] .token.cdata { color: #5F6C66 !important; }
[data-testid="stCode"] .token.parameter,
[data-testid="stCode"] .token.variable,
[data-testid="stCode"] .token.attr-name,
[data-testid="stCode"] .token.property,
[data-testid="stCode"] .token.tag { color: #1C6B80 !important; }
[data-testid="stCode"] .token.function,
[data-testid="stCode"] .token.builtin,
[data-testid="stCode"] .token.class-name,
[data-testid="stCode"] .token.selector { color: #1D5F8A !important; }
[data-testid="stCode"] .token.string,
[data-testid="stCode"] .token.char,
[data-testid="stCode"] .token.attr-value,
[data-testid="stCode"] .token.inserted,
[data-testid="stCode"] .token.url { color: #1A6B4A !important; }
[data-testid="stCode"] .token.number,
[data-testid="stCode"] .token.boolean,
[data-testid="stCode"] .token.constant,
[data-testid="stCode"] .token.symbol { color: #1F6F5B !important; }
[data-testid="stCode"] .token.keyword,
[data-testid="stCode"] .token.operator,
[data-testid="stCode"] .token.important,
[data-testid="stCode"] .token.deleted,
[data-testid="stCode"] .token.regex { color: #9C3C30 !important; }
[data-testid="stCode"] .token.punctuation { color: #20302A !important; }
/* 2.4.7 / 1.4.11: a visible focus ring for keyboard users, in the brand
   green (5.2-5.9:1). Streamlit's default is a faint tint only. */
a:focus-visible,
button:focus-visible,
summary:focus-visible,
[role="tab"]:focus-visible,
[role="button"]:focus-visible,
[tabindex]:not([tabindex="-1"]):focus-visible,
[data-baseweb="input"]:focus-within,
[data-baseweb="textarea"]:focus-within,
[data-baseweb="select"] > div:focus-within,
label:has(> input[type="checkbox"]:focus-visible),
label:has(> input[type="radio"]:focus-visible) {
    outline: 2px solid #1F6F5B !important;
    outline-offset: 2px !important;
}
/* 1.4.11: text fields, number fields and selects need a 3:1 edge. The theme
   border (#D4DED3, 1.3:1) stays on cards and dividers; form controls get
   a darker step of the same sage (3.1-3.3:1). */
[data-baseweb="input"],
[data-baseweb="textarea"],
[data-baseweb="select"] > div {
    border-color: #7F8F86 !important;
}
</style>
"""


@lru_cache(maxsize=None)
def _data_uri(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def funder_footer_html():
    """HTML for the funder footer band: logos, funding statement, links."""
    items = []
    for filename, alt, height in FUNDER_LOGOS:
        path = LOGOS_DIR / filename
        if path.is_file():
            items.append(
                f'<li><img src="{_data_uri(str(path))}" alt="{escape(alt)}" '
                f'style="height:{height}px"></li>'
            )
    logos = (
        f'<ul class="mats-funder-logos" aria-label="Breeding Insight and its funders">{"".join(items)}</ul>'
        if items
        else ""
    )
    links = "".join(
        f'<li><a href="{escape(url)}" target="_blank" rel="noopener noreferrer">{escape(label)}'
        '<span class="mats-visually-hidden"> (opens in a new tab)</span></a></li>'
        for label, url in FOOTER_LINKS
    )
    bottom = (
        '<div class="mats-footer-bottom">'
        f"<p>&copy; {date.today().year} Breeding Insight &middot; MATS {escape(__version__)}</p>"
        f'<ul aria-label="Breeding Insight links">{links}</ul>'
        "</div>"
    )
    return (
        f'{_FUNDER_FOOTER_CSS}<footer class="mats-funder-footer">'
        f'{logos}<p class="mats-funding">{escape(FUNDING_STATEMENT)}</p>{bottom}</footer>'
    )


def render_funder_footer():
    """Render the funder logos and funding statement at the bottom of a page.

    Lives in the main area rather than the sidebar, which is collapsed on
    phones, so it is visible at every width.
    """
    st.html(funder_footer_html())


def page_icon():
    """Value for st.set_page_config(page_icon=...)."""
    return str(LOGO_MARK) if LOGO_MARK.is_file() else "🌿"


def apply_logo():
    """Render the header/sidebar lockup and the shared accessibility styles."""
    st.markdown(_A11Y_CSS, unsafe_allow_html=True)
    if not LOGO_WIDE.is_file():
        return
    st.logo(
        str(LOGO_WIDE),
        size="large",
        icon_image=str(LOGO_MARK) if LOGO_MARK.is_file() else None,
    )
    st.markdown(_LOGO_CSS, unsafe_allow_html=True)
