"""Shared MATS branding for the Streamlit pages."""

from pathlib import Path

import streamlit as st

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_WIDE = ASSETS_DIR / "mats_logo_horizontal_circles_color.svg"
LOGO_MARK = ASSETS_DIR / "mats_mark_circles_color.svg"

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
