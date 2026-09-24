"""Interactive, sample-only threshold and clean-image preview for the Analyze workbench."""

import base64
from pathlib import Path

import cv2
import streamlit as st

from mats.mask_cleanup import (
    CLEAN_RADIUS_DEFAULT, CLEAN_RADIUS_MAX, clean_levels, clean_raw_mask,
)
from mats.mask_settings import CLEAN_MARGIN_DEFAULT, STRAY_GAP_DEFAULT

_PREVIEW_HTML = """
<div class="mats-threshold-preview">
  <div class="mats-threshold-control" hidden>
    <label for="mats-threshold-range">Threshold preview: <output class="mats-threshold-value"></output></label>
    <input id="mats-threshold-range" class="mats-threshold-range" type="range" min="0" max="255" step="1" />
    <div class="mats-threshold-presets" aria-hidden="true">
      <span style="left:39.2%">low · 100</span>
      <span style="left:49.0%">med · 125</span>
      <span style="left:58.8%">high · 150</span>
    </div>
  </div>
  <div class="mats-clean-control" hidden>
    <div class="mats-clean-label">
      <label for="mats-clean-range">Clean size: <output class="mats-clean-value"></output> px</label>
      <span class="mats-help" tabindex="0" aria-label="About clean size">
        <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M11 18h2v-2h-2v2zm1-16C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm0-14c-2.21 0-4 1.79-4 4h2c0-1.1.9-2 2-2s2 .9 2 2c0 2-3 1.75-3 5h2c0-2.25 3-2.5 3-5 0-2.21-1.79-4-4-4z"/></svg>
        <span id="mats-clean-help" class="mats-help-tip" role="tooltip"></span>
      </span>
    </div>
    <input id="mats-clean-range" class="mats-clean-range" type="range" min="0" step="1"
      aria-describedby="mats-clean-help" />
  </div>
  <div class="mats-fill-control" hidden>
    <label class="mats-fill-label"
      title="Keep enclosed holes unfilled while retaining the other mask cleanup.">
      <input class="mats-remove-fill" type="checkbox" />
      Remove flashfill from this preview
    </label>
    <p class="mats-fill-note" hidden></p>
  </div>
  <div class="mats-threshold-panels">
    <figure>
      <canvas class="mats-threshold-mask" aria-label="Mask preview"></canvas>
      <figcaption>Mask</figcaption>
    </figure>
    <figure class="mats-threshold-leaf-panel">
      <canvas class="mats-threshold-leaf" aria-label="Masked leaf preview in color"></canvas>
      <figcaption>Masked leaf</figcaption>
    </figure>
  </div>
  <p class="mats-threshold-status" aria-live="polite"></p>
</div>
"""

_PREVIEW_CSS = """
.mats-threshold-preview { width: 100%; }
.mats-threshold-preview label { display: block; margin-bottom: .35rem; font-weight: 600; }
.mats-threshold-preview input { width: 100%; accent-color: var(--st-primary-color); }
.mats-threshold-presets { position: relative; height: 1.9rem; margin: -.15rem .45rem 0; }
.mats-threshold-presets span {
  position: absolute; top: 0; transform: translateX(-50%); white-space: nowrap;
  color: var(--st-text-color); font-size: .7rem; text-align: center;
}
/* A tick from the bar down to each preset label. */
.mats-threshold-presets span::before {
  content: ""; display: block; width: 2px; height: .55rem; margin: 0 auto .1rem;
  border-radius: 1px; background: var(--st-text-color); opacity: .6;
}
.mats-threshold-panels {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(12rem, 1fr));
  gap: .75rem; margin-top: .65rem;
}
.mats-threshold-panels figure { margin: 0; min-width: 0; }
.mats-threshold-preview [hidden] { display: none; }
.mats-clean-control { margin-top: .5rem; }
.mats-clean-label { display: flex; align-items: center; gap: .35rem; margin-bottom: .35rem; }
.mats-threshold-preview .mats-clean-label label { margin-bottom: 0; }
.mats-fill-control { margin-top: .5rem; }
.mats-threshold-preview label.mats-fill-label {
  display: inline-flex; align-items: center; gap: .45rem; margin: 0;
  font-weight: 400; cursor: pointer;
}
.mats-threshold-preview input.mats-remove-fill { width: auto; margin: 0; }
.mats-threshold-preview label.mats-fill-label:has(input:disabled) { cursor: not-allowed; opacity: .6; }
.mats-fill-note { margin: .25rem 0 0; color: var(--st-text-color); font-size: .8rem; opacity: .75; }
.mats-help {
  position: relative; display: inline-flex; color: var(--st-text-color);
  opacity: .6; cursor: help; border-radius: 50%;
}
.mats-help:hover, .mats-help:focus-visible { opacity: 1; }
.mats-help-tip {
  position: absolute; left: 0; bottom: calc(100% + .4rem); z-index: 10;
  width: max-content; max-width: 20rem; padding: .5rem .65rem;
  border-radius: var(--st-base-radius); background: var(--st-text-color);
  color: var(--st-background-color); font-size: .8rem; font-weight: 400;
  line-height: 1.4; visibility: hidden; opacity: 0; pointer-events: none;
  transition: opacity .12s;
}
.mats-help:hover .mats-help-tip, .mats-help:focus-visible .mats-help-tip {
  visibility: visible; opacity: 1;
}
.mats-threshold-panels canvas {
  display: block; width: 100%; height: auto; max-height: 32rem;
  object-fit: contain; background: var(--st-secondary-background-color);
}
.mats-threshold-panels figcaption {
  margin-top: .25rem; color: var(--st-text-color); font-size: .8rem;
}
.mats-threshold-status { margin: .4rem 0 0; color: var(--st-text-color); font-size: .85rem; }
"""

_PREVIEW_JS = """
export default function(component) {
  const { data, parentElement, setTriggerValue } = component;
  const find = (selector) => parentElement.querySelector(selector);
  const thresholdControl = find('.mats-threshold-control');
  const slider = find('input.mats-threshold-range');
  const valueLabel = find('output.mats-threshold-value');
  const cleanControl = find('.mats-clean-control');
  const cleanSlider = find('input.mats-clean-range');
  const cleanLabel = find('output.mats-clean-value');
  const cleanHelp = find('.mats-clean-control .mats-help-tip');
  const fillControl = find('.mats-fill-control');
  const fillBox = find('input.mats-remove-fill');
  const fillNote = find('.mats-fill-note');
  const maskCanvas = find('canvas.mats-threshold-mask');
  const leafCanvas = find('canvas.mats-threshold-leaf');
  const leafPanel = find('.mats-threshold-leaf-panel');
  const status = find('.mats-threshold-status');
  if (![thresholdControl, slider, valueLabel, cleanControl, cleanSlider, cleanLabel,
        cleanHelp, fillControl, fillBox, fillNote, maskCanvas, leafCanvas, leafPanel,
        status].every(Boolean)) return;

  const maskContext = maskCanvas.getContext('2d');
  const leafContext = leafCanvas.getContext('2d');
  // A grayscale image means a live threshold; otherwise the mask is fixed.
  const thresholded = Boolean(data.grayscale_image);
  // Levels make the clean-size slider live; a clean size of 0 turns it off.
  const cleaning = Boolean(data.clean_levels_image);
  const cleanActive = () => cleaning && Number(cleanSlider.value) > 0;
  let pixels = null;
  let baseMask = null;
  let levels = null;
  let colors = null;
  let frame = null;
  let disposed = false;
  let dragging = false;
  thresholdControl.hidden = !thresholded;
  cleanControl.hidden = !cleaning;
  leafPanel.hidden = !data.color_image;
  if (thresholded) {
    slider.value = String(Number(data.cutoff));
    valueLabel.textContent = slider.value;
  }
  cleanSlider.max = String(Number(data.clean_max));
  cleanSlider.value = String(Number(data.clean_radius));
  cleanLabel.textContent = cleanSlider.value;
  cleanHelp.textContent = data.clean_help || '';
  fillControl.hidden = !data.fill_toggle;
  fillBox.checked = Boolean(data.remove_fill);
  fillBox.disabled = Boolean(data.fill_note);
  fillNote.textContent = data.fill_note || '';
  fillNote.hidden = !data.fill_note;

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = reject;
      image.src = src;
    });
  }

  // Nearest-neighbour sampling keeps the grayscale, color, mask, and level
  // pixels aligned one-to-one on the preview canvases.
  function readPixels(image) {
    const offscreen = document.createElement('canvas');
    offscreen.width = maskCanvas.width;
    offscreen.height = maskCanvas.height;
    const offscreenContext = offscreen.getContext('2d', { willReadFrequently: true });
    offscreenContext.imageSmoothingEnabled = false;
    offscreenContext.drawImage(image, 0, 0, offscreen.width, offscreen.height);
    return offscreenContext.getImageData(0, 0, offscreen.width, offscreen.height).data;
  }

  // Paint the binary mask, and beside it the sample's own colors wherever the
  // mask is foreground; excluded pixels stay transparent.
  function paint(isForeground) {
    const mask = maskContext.createImageData(maskCanvas.width, maskCanvas.height);
    const leaf = colors
      ? leafContext.createImageData(leafCanvas.width, leafCanvas.height)
      : null;
    for (let i = 0; i < mask.data.length; i += 4) {
      mask.data[i + 3] = 255;
      if (!isForeground(i)) continue;
      mask.data[i] = 255;
      mask.data[i + 1] = 255;
      mask.data[i + 2] = 255;
      if (leaf) {
        leaf.data[i] = colors[i];
        leaf.data[i + 1] = colors[i + 1];
        leaf.data[i + 2] = colors[i + 2];
        leaf.data[i + 3] = 255;
      }
    }
    maskContext.putImageData(mask, 0, 0);
    if (leaf) leafContext.putImageData(leaf, 0, 0);
  }

  // The edge band that mats.mask_cleanup.clear_margin sets to background:
  // live_margin percent of the shorter side, in canvas pixels.
  function marginBand() {
    const shorter = Math.min(maskCanvas.width, maskCanvas.height);
    return Math.round(shorter * Number(data.live_margin || 0) / 100);
  }

  function drawRaw() {
    frame = null;
    if (!pixels || disposed) return;
    const cutoff = Number(slider.value);
    const band = marginBand();
    const width = maskCanvas.width;
    const height = maskCanvas.height;
    paint((i) => {
      if (pixels[i] > cutoff) return false;
      if (!band) return true;
      const x = (i / 4) % width;
      const y = Math.floor(i / 4 / width);
      return x >= band && y >= band && x < width - band && y < height - band;
    });
    const edge = band ? ' with the edge margin cleared' : '';
    if (cleanActive()) {
      status.textContent = `Live mask${edge} before cleaning; release to see the clean result`;
    } else if (data.measurement_source === 'pre-cleanup') {
      status.textContent = `Live raw mask${edge}; release to see the measured mask`;
    } else {
      status.textContent = `Live mask${edge} before cleanup; release to see the cleaned result`;
    }
  }

  // Levels hold each pixel's keep and fill radii for the mask they were
  // measured on (mats.mask_cleanup); with a threshold, that is one cutoff.
  function levelsCurrent() {
    return Boolean(levels) &&
      (!thresholded || Number(slider.value) === Number(data.clean_cutoff));
  }

  function drawClean() {
    frame = null;
    if (disposed || !levelsCurrent()) return;
    const radius = Number(cleanSlider.value);
    paint((i) => radius < levels[i] || (levels[i + 1] <= radius && radius <= levels[i + 2]));
    status.textContent = radius < 2
      ? 'Clean preview: edge margin and stray pieces removed; no specks or holes change at this size'
      : `Clean preview: edge margin and stray pieces removed; specks removed and holes filled below ${radius} px`;
  }

  function showSettled() {
    if (disposed || dragging) return;
    if (cleanActive()) {
      if (levelsCurrent()) drawClean(); else drawRaw();
      return;
    }
    if (!thresholded) {
      if (!baseMask) return;
      paint((i) => baseMask[i] >= 128);
      status.textContent = data.mask_status || 'Saved measurement mask';
      return;
    }
    if (!pixels) return;
    const cutoff = Number(slider.value);
    if (cutoff !== Number(data.cleaned_cutoff) || !data.cleaned_image) {
      drawRaw();
      return;
    }
    loadImage(data.cleaned_image).then((cleaned) => {
      if (disposed || dragging || Number(slider.value) !== cutoff) return;
      const settled = readPixels(cleaned);
      paint((i) => settled[i] >= 128);
      if (data.measurement_source === 'pre-cleanup') {
        status.textContent =
          'Pre-cleanup measurement mask: edge margin cleared and stray pieces dropped';
      } else {
        status.textContent = data.remove_fill
          ? 'Preview with hole filling removed and the edge margin cleared'
          : 'Cleaned preview from the MATS pipeline';
      }
    }, () => {});
  }

  slider.oninput = () => {
    dragging = true;
    valueLabel.textContent = slider.value;
    if (frame !== null) cancelAnimationFrame(frame);
    frame = requestAnimationFrame(drawRaw);
  };
  slider.onchange = () => {
    dragging = false;
    if (frame !== null) cancelAnimationFrame(frame);
    drawRaw();
    if (cleanActive()) {
      status.textContent = 'Measuring speck and hole sizes for this cutoff…';
    } else if (data.measurement_source === 'pre-cleanup') {
      status.textContent = 'Dropping stray pieces for this cutoff…';
    } else {
      status.textContent = 'Applying MATS cleanup to this sample…';
    }
    setTriggerValue('cutoff', Number(slider.value));
  };
  cleanSlider.oninput = () => {
    cleanLabel.textContent = cleanSlider.value;
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
    if (!cleanActive()) {
      showSettled();
      return;
    }
    if (levelsCurrent()) frame = requestAnimationFrame(drawClean);
  };
  cleanSlider.onchange = () => {
    setTriggerValue('clean_radius', Number(cleanSlider.value));
  };
  fillBox.onchange = () => {
    // The server recomputes the mask; hold the box until it answers.
    fillBox.disabled = true;
    status.textContent = fillBox.checked
      ? 'Removing flashfill from this preview…'
      : 'Restoring flashfill to this preview…';
    setTriggerValue('remove_fill', fillBox.checked);
  };

  const optional = (src) => (src ? loadImage(src) : Promise.resolve(null));
  Promise.all([
    optional(data.grayscale_image),
    optional(data.mask_image),
    optional(data.clean_levels_image),
    optional(data.color_image).catch(() => null),
  ]).then(([gray, mask, levelsImage, colorImage]) => {
    if (disposed) return;
    const source = gray || mask || levelsImage;
    if (!source) {
      status.textContent = 'Could not load this sample for preview.';
      return;
    }
    const scale = Math.min(1, 1400 / Math.max(source.naturalWidth, source.naturalHeight));
    const width = Math.max(1, Math.round(source.naturalWidth * scale));
    const height = Math.max(1, Math.round(source.naturalHeight * scale));
    maskCanvas.width = leafCanvas.width = width;
    maskCanvas.height = leafCanvas.height = height;
    pixels = gray ? readPixels(gray) : null;
    baseMask = mask ? readPixels(mask) : null;
    levels = levelsImage ? readPixels(levelsImage) : null;
    colors = colorImage ? readPixels(colorImage) : null;
    leafPanel.hidden = !colors;
    showSettled();
  }, () => { status.textContent = 'Could not load this sample for preview.'; });

  return () => {
    disposed = true;
    if (frame !== null) cancelAnimationFrame(frame);
    slider.oninput = null;
    slider.onchange = null;
    cleanSlider.oninput = null;
    cleanSlider.onchange = null;
    fillBox.onchange = null;
  };
}
"""


def _png_data_url(image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("could not encode threshold preview")
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


@st.cache_data(max_entries=8, show_spinner=False)
def grayscale_sample(path):
    """Return exact OpenCV grayscale pixels and the sample's Otsu cutoff."""
    target = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if target is None:
        raise ValueError(f"Could not read sample target box: {Path(path).name}")
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    otsu_cutoff, _ = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    return _png_data_url(gray), int(otsu_cutoff)


@st.cache_data(max_entries=8, show_spinner=False)
def color_sample(path):
    """Return the sample's color target box for the masked-leaf preview panel."""
    target = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if target is None:
        raise ValueError(f"Could not read sample target box: {Path(path).name}")
    # Display only: JPEG keeps a full-resolution color payload small, and the
    # mask itself always comes from the lossless grayscale image.
    ok, encoded = cv2.imencode(".jpg", target, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("could not encode color preview")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


@st.cache_data(max_entries=8, show_spinner=False)
def cleaned_sample(path, cutoff, *, fill_holes=True, clean_margin=CLEAN_MARGIN_DEFAULT):
    """Run the production mask cleanup on just the selected threshold sample.

    ``fill_holes=False`` is Remove flashfill, which clears the edge margin first.
    """
    from mats import core

    target = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if target is None:
        raise ValueError(f"Could not read sample target box: {Path(path).name}")
    raw = core.threshold_mask(target, cutoff)
    if fill_holes:
        return _png_data_url(core.clean_leaf_mask(raw.copy()))
    return _png_data_url(core.unfilled_leaf_mask(raw, clean_margin))


@st.cache_data(max_entries=8, show_spinner=False)
def unfilled_sample(raw_mask_path, clean_margin=CLEAN_MARGIN_DEFAULT):
    """Keep the largest raw component, past the edge margin, without filling holes."""
    from mats import core

    raw = cv2.imread(str(raw_mask_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError(f"Could not read raw mask: {Path(raw_mask_path).name}")
    return _png_data_url(core.unfilled_leaf_mask(raw, clean_margin))


@st.cache_data(max_entries=8, show_spinner=False)
def pre_cleanup_sample(path, cutoff, clean_margin=CLEAN_MARGIN_DEFAULT,
                       stray_gap=STRAY_GAP_DEFAULT):
    """The pre-cleanup measurement mask for one cutoff (``clean_raw_mask``)."""
    from mats import core

    target = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if target is None:
        raise ValueError(f"Could not read sample target box: {Path(path).name}")
    raw = core.threshold_mask(target, cutoff)
    return _png_data_url(clean_raw_mask(raw, clean_margin, stray_gap))


def _levels_data_url(levels):
    # OpenCV writes BGR; reverse so the browser reads the channels in order.
    return _png_data_url(cv2.cvtColor(levels, cv2.COLOR_RGB2BGR))


@st.cache_data(max_entries=8, show_spinner=False)
def clean_levels_for_threshold(path, cutoff, stray_gap=STRAY_GAP_DEFAULT,
                               clean_margin=CLEAN_MARGIN_DEFAULT):
    """Speck and hole sizes of the raw threshold mask, for the clean slider."""
    from mats import core

    target = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if target is None:
        raise ValueError(f"Could not read sample target box: {Path(path).name}")
    raw = core.threshold_mask(target, cutoff)
    return _levels_data_url(clean_levels(raw, stray_gap, clean_margin))


@st.cache_data(max_entries=8, show_spinner=False)
def clean_levels_for_mask(path, mtime_ns, stray_gap=STRAY_GAP_DEFAULT,
                          clean_margin=CLEAN_MARGIN_DEFAULT):
    """Speck and hole sizes of a saved raw mask; ``mtime_ns`` keys the cache."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read raw mask: {Path(path).name}")
    return _levels_data_url(clean_levels(mask, stray_gap, clean_margin))


@st.cache_data(max_entries=8, show_spinner=False)
def pre_cleanup_mask_sample(path, mtime_ns, clean_margin=CLEAN_MARGIN_DEFAULT,
                            stray_gap=STRAY_GAP_DEFAULT):
    """A saved raw mask's pre-cleanup measurement mask; ``mtime_ns`` keys the cache."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read raw mask: {Path(path).name}")
    return _png_data_url(clean_raw_mask(mask, clean_margin, stray_gap))


@st.cache_data(max_entries=8, show_spinner=False)
def mask_sample(path, mtime_ns):
    """A saved mask for the fixed-mask preview; ``mtime_ns`` keys the cache."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {Path(path).name}")
    return _png_data_url(mask)


def show_threshold_preview(
    *, key, grayscale_image=None, cutoff=None, measurement_source="cleaned",
    mask_image=None, mask_status=None, color_image=None, cleaned_image=None,
    cleaned_cutoff=None, remove_fill=False, clean_levels_image=None, clean_cutoff=None,
    clean_radius=CLEAN_RADIUS_DEFAULT, clean_help=None, fill_toggle=False, fill_note=None,
    live_margin=0, on_cutoff_change=None, on_clean_radius_change=None,
    on_remove_fill_change=None,
):
    """Mount the local drag preview; emit values only when a drag ends.

    With ``grayscale_image`` the threshold is live; otherwise ``mask_image`` is
    shown as is, labelled ``mask_status``. ``clean_levels_image`` switches on the
    clean-size slider; above 0 it previews Clean image, and ``clean_help`` fills
    the help icon beside it. ``fill_toggle`` shows the Remove flashfill checkbox
    under it, checked when ``remove_fill``; ``fill_note`` disables it and says why.
    ``cleaned_image`` is the server's settled mask for ``cleaned_cutoff``, and
    ``live_margin`` is the edge-margin percent cleared while dragging.
    """
    # Register in the active Streamlit runtime. AppTest and the running app use
    # separate registries, so a registration made at import time can be stale.
    component = st.components.v2.component(
        "mats_threshold_preview", html=_PREVIEW_HTML, css=_PREVIEW_CSS, js=_PREVIEW_JS,
    )
    return component(
        key=key,
        data={
            "grayscale_image": grayscale_image,
            "mask_image": mask_image,
            "mask_status": mask_status,
            "color_image": color_image,
            "cutoff": cutoff,
            "measurement_source": measurement_source,
            "cleaned_image": cleaned_image,
            "cleaned_cutoff": cleaned_cutoff,
            "remove_fill": remove_fill,
            "clean_levels_image": clean_levels_image,
            "clean_cutoff": clean_cutoff,
            "clean_radius": clean_radius,
            "clean_help": clean_help,
            "clean_max": CLEAN_RADIUS_MAX,
            "fill_toggle": fill_toggle,
            "fill_note": fill_note,
            "live_margin": live_margin,
        },
        on_cutoff_change=on_cutoff_change,
        on_clean_radius_change=on_clean_radius_change,
        on_remove_fill_change=on_remove_fill_change,
    )
