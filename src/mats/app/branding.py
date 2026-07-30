"""Shared MATS branding for the Streamlit pages."""

from pathlib import Path

import streamlit as st

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_WIDE = ASSETS_DIR / "mats_logo_horizontal_circles_color.svg"
LOGO_MARK = ASSETS_DIR / "mats_mark_circles_color.svg"

# st.logo's biggest built-in size ("large") renders at 2rem; override it to
# stay legible at header scale.
_LOGO_HEIGHT = "8rem"

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
}}
[data-testid="stLogoSpacer"] {{
    height: {_LOGO_HEIGHT} !important;
}}
[data-testid="stHeaderLogo"] {{
    height: {_HEADER_LOGO_HEIGHT} !important;
    margin-top: 0.75rem !important;
    margin-left: 0.5rem !important;
}}
</style>
"""


def page_icon():
    """Value for st.set_page_config(page_icon=...)."""
    return str(LOGO_MARK) if LOGO_MARK.is_file() else "🌿"


def apply_logo():
    """Render the header/sidebar lockup at the enlarged MATS size."""
    if not LOGO_WIDE.is_file():
        return
    st.logo(
        str(LOGO_WIDE),
        size="large",
        icon_image=str(LOGO_MARK) if LOGO_MARK.is_file() else None,
    )
    st.markdown(_LOGO_CSS, unsafe_allow_html=True)
