"""Adjust's specimen table: choose the specimen to view and mark specimens for adjustment."""

import streamlit as st

_TABLE_HTML = """
<div class="mats-specimen-table">
  <div class="mats-table-scroll">
    <table>
      <thead><tr></tr></thead>
      <tbody></tbody>
    </table>
  </div>
</div>
"""

_TABLE_CSS = """
.mats-specimen-table {
  color: var(--st-text-color); font-family: var(--st-font); font-size: .875rem;
}
.mats-table-scroll {
  position: relative; overflow: auto;
  border: 1px solid var(--st-dataframe-border-color, var(--st-border-color));
  border-radius: var(--st-base-radius);
}
.mats-specimen-table table { width: 100%; border-collapse: separate; border-spacing: 0; }
.mats-specimen-table th {
  position: sticky; top: 0; z-index: 1; padding: 0; text-align: left; font-weight: 400;
  vertical-align: bottom;
  color: color-mix(in srgb, var(--st-text-color) 65%, transparent);
  background: var(--st-dataframe-header-background-color, var(--st-secondary-background-color));
  border-bottom: 1px solid var(--st-dataframe-border-color, var(--st-border-color));
}
.mats-specimen-table th button {
  all: unset; box-sizing: border-box; width: 100%; padding: .45rem .6rem; cursor: pointer;
}
.mats-specimen-table th button:focus-visible { outline: 2px solid var(--st-primary-color); }
.mats-specimen-table td {
  padding: .4rem .6rem; white-space: nowrap;
  border-bottom: 1px solid var(--st-dataframe-border-color, var(--st-border-color));
}
.mats-specimen-table tbody tr:last-child td { border-bottom: 0; }
.mats-specimen-table .mats-num { text-align: right; font-variant-numeric: tabular-nums; }
.mats-specimen-table th.mats-num button { text-align: right; }
.mats-specimen-table .mats-col-view,
.mats-specimen-table .mats-col-mark { width: 1%; text-align: center; }
.mats-specimen-table th.mats-col-view,
.mats-specimen-table th.mats-col-mark { padding: .45rem .6rem; }
/* Wrap the long header rather than push the table wider than Adjust. */
.mats-specimen-table th.mats-col-mark { min-width: 5.5rem; }
.mats-specimen-table tbody tr { cursor: pointer; }
.mats-specimen-table tbody tr:hover td {
  background: color-mix(in srgb, var(--st-primary-color) 6%, transparent);
}
.mats-specimen-table tbody tr.mats-viewed td {
  background: color-mix(in srgb, var(--st-primary-color) 12%, transparent);
}
.mats-specimen-table input { margin: 0; cursor: pointer; accent-color: var(--st-primary-color); }
/* The whole Marked cell toggles its checkbox. */
.mats-specimen-table .mats-mark-hit {
  display: flex; align-items: center; justify-content: center;
  margin: -.4rem -.6rem; padding: .4rem .6rem; cursor: pointer;
}
.mats-specimen-table .mats-sort { margin-left: .3rem; font-size: .7em; }
"""

_TABLE_JS = """
export default function(component) {
  const { data, parentElement, setTriggerValue } = component;
  const scroller = parentElement.querySelector('.mats-table-scroll');
  const head = parentElement.querySelector('thead tr');
  const body = parentElement.querySelector('tbody');
  if (!scroller || !head || !body) return;

  // Sort order and scroll survive data updates because the table DOM does.
  const state = parentElement.__matsTable || (parentElement.__matsTable = {
    sort: null, scrolled: false, group: `mats-view-${Math.random().toString(36).slice(2)}`,
  });
  const columns = data.columns || [];
  const markable = Boolean(data.markable);
  const marked = new Set(data.marked || []);
  let view = data.view;
  scroller.style.maxHeight = `${Number(data.height) || 300}px`;

  const numeric = (column) => column.decimals !== null && column.decimals !== undefined;
  const format = (value, column) => {
    if (value === null || value === undefined || value === '') return '';
    return numeric(column) ? Number(value).toFixed(column.decimals) : String(value);
  };

  function sortedRows() {
    const rows = (data.rows || []).slice();
    const sort = state.sort;
    const column = sort && columns.find((item) => item.key === sort.key);
    if (!column) return rows;
    rows.sort((a, b) => {
      const x = a[column.key];
      const y = b[column.key];
      if (x === null || x === undefined) return 1;   // blanks last either way
      if (y === null || y === undefined) return -1;
      const order = numeric(column)
        ? x - y
        : String(x).localeCompare(String(y), undefined, { numeric: true });
      return sort.descending ? -order : order;
    });
    return rows;
  }

  function headerCell(text, className) {
    const th = document.createElement('th');
    th.scope = 'col';
    if (className) th.className = className;
    th.textContent = text;
    return th;
  }

  function renderHead() {
    const cells = [headerCell('View', 'mats-col-view')];
    for (const column of columns) {
      const active = Boolean(state.sort) && state.sort.key === column.key;
      const th = headerCell('', numeric(column) ? 'mats-num' : '');
      th.setAttribute('aria-sort', active
        ? (state.sort.descending ? 'descending' : 'ascending') : 'none');
      const button = document.createElement('button');
      button.type = 'button';
      button.dataset.sort = column.key;
      button.textContent = column.label;
      const arrow = document.createElement('span');
      arrow.className = 'mats-sort';
      arrow.textContent = active ? (state.sort.descending ? '▼' : '▲') : '';
      button.append(arrow);
      th.append(button);
      cells.push(th);
    }
    if (markable) cells.push(headerCell('Marked for Adjustment', 'mats-col-mark'));
    head.replaceChildren(...cells);
  }

  function renderBody() {
    const fragment = document.createDocumentFragment();
    for (const row of sortedRows()) {
      const id = String(row.sample_id);
      const tr = document.createElement('tr');
      tr.dataset.id = id;
      tr.classList.toggle('mats-viewed', id === view);
      const viewCell = document.createElement('td');
      viewCell.className = 'mats-col-view';
      const radio = document.createElement('input');
      radio.type = 'radio';
      radio.name = state.group;
      radio.checked = id === view;
      radio.setAttribute('aria-label', `View ${id}`);
      viewCell.append(radio);
      tr.append(viewCell);
      for (const column of columns) {
        const td = document.createElement('td');
        if (numeric(column)) td.className = 'mats-num';
        td.textContent = format(row[column.key], column);
        tr.append(td);
      }
      if (markable) {
        const td = document.createElement('td');
        td.className = 'mats-col-mark';
        const label = document.createElement('label');
        label.className = 'mats-mark-hit';
        const box = document.createElement('input');
        box.type = 'checkbox';
        box.checked = marked.has(id);
        box.setAttribute('aria-label', `Mark ${id} for adjustment`);
        label.append(box);
        td.append(label);
        tr.append(td);
      }
      fragment.append(tr);
    }
    body.replaceChildren(fragment);
  }

  function render() {
    renderHead();
    renderBody();
  }

  function choose(id) {
    view = id;
    for (const tr of body.rows) {
      const on = tr.dataset.id === id;
      tr.classList.toggle('mats-viewed', on);
      tr.querySelector('input[type=radio]').checked = on;
    }
    setTriggerValue('view', id);
  }

  head.onclick = (event) => {
    const button = event.target.closest('button[data-sort]');
    if (!button) return;
    const key = button.dataset.sort;
    const again = Boolean(state.sort) && state.sort.key === key && !state.sort.descending;
    state.sort = { key, descending: again };
    render();
  };
  body.onclick = (event) => {
    if (event.target.closest('.mats-col-mark')) return;   // marking never changes the view
    const tr = event.target.closest('tr');
    if (tr && tr.dataset.id !== view) choose(tr.dataset.id);
  };
  body.onchange = (event) => {
    const box = event.target;
    if (box.type !== 'checkbox') return;
    const id = box.closest('tr').dataset.id;
    if (box.checked) marked.add(id); else marked.delete(id);
    setTriggerValue('mark', { id, marked: box.checked });
  };

  render();
  // On first mount, bring the viewed specimen into sight.
  if (!state.scrolled) {
    state.scrolled = true;
    const viewed = body.querySelector('tr.mats-viewed');
    if (viewed && viewed.offsetTop + viewed.offsetHeight > scroller.clientHeight) {
      scroller.scrollTop = viewed.offsetTop - head.offsetHeight;
    }
  }

  return () => {
    head.onclick = null;
    body.onclick = null;
    body.onchange = null;
  };
}
"""


def show_specimen_table(
    *, key, columns, rows, view, marked=(), markable=False, height=300,
    on_view_change=None, on_mark_change=None,
):
    """Mount the specimen table.

    ``columns`` are ``{"key", "label", "decimals"}`` dicts (``decimals`` None for
    text) and ``rows`` carry those keys, including ``sample_id``. Clicking a row
    emits ``view`` (its sample id); with ``markable``, ticking **Marked for
    Adjustment** emits ``mark`` as ``{"id", "marked"}``.
    """
    # Register in the active Streamlit runtime, as threshold_preview does.
    component = st.components.v2.component(
        "mats_specimen_table", html=_TABLE_HTML, css=_TABLE_CSS, js=_TABLE_JS,
    )
    return component(
        key=key,
        data={
            "columns": list(columns),
            "rows": list(rows),
            "view": view,
            "marked": list(marked),
            "markable": markable,
            "height": height,
        },
        on_view_change=on_view_change,
        on_mark_change=on_mark_change,
    )
