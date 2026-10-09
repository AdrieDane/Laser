"""Compressed, paginated payloads for large self-contained LASER reports.

The report remains a single offline HTML file.  Detailed feature rows are
stored as a gzip-compressed JSON payload and only the requested page is added
to the browser DOM.  This prevents thousands of SVG traces and spectra from
being parsed when a report is opened or when another tab is being reviewed.
"""

from __future__ import annotations

import base64
import gzip
import html
import json
import re
from typing import Iterable, Mapping


LAZY_REPORT_FORMAT_VERSION = 2
DEFAULT_REPORT_PAGE_SIZE = 50


def decode_lazy_feature_payloads(document: str) -> list[dict[str, object]]:
    """Decode every embedded lazy-row payload in an HTML document.

    This is intentionally a Python-side companion to the browser decoder.  It
    lets report validation, tests and future migration tooling inspect the
    user-visible feature evidence without depending on its compressed storage
    representation.
    """
    payloads: list[dict[str, object]] = []
    for encoded in re.findall(
        r"data-laser-lazy-payload>([^<]+)</textarea>", document
    ):
        raw = gzip.decompress(base64.b64decode(encoded)).decode("utf-8")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Lazy report payload must be a JSON object")
        payloads.append(value)
    return payloads


def searchable_report_text(document: str) -> str:
    """Return the static HTML plus decoded row/section markup for validation."""
    parts = [document]
    for payload in decode_lazy_feature_payloads(document):
        sections = payload.get("sections", {})
        fragments = payload.get("fragments", {})
        rows = payload.get("rows", [])
        if isinstance(sections, dict):
            parts.extend(str(value) for value in sections.values())
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                markup = str(row.get("html", ""))
                if isinstance(fragments, dict):
                    markup = re.sub(
                        r"@@LASER_FRAGMENT:([0-9]+)@@",
                        lambda match: str(fragments.get(match.group(1), "")),
                        markup,
                    )
                parts.append(markup)
    return "\n".join(parts)


LAZY_REPORT_STYLE = """
.lazy-report-controls{position:sticky;top:43px;z-index:4;display:flex;
 flex-wrap:wrap;align-items:center;gap:7px;padding:8px;margin:0 0 7px;
 background:rgba(247,247,245,.96);border:1px solid #d9d9d5;border-radius:7px}
.lazy-report-controls input{min-width:240px;padding:6px 8px;border:1px solid #c9c9c5;
 border-radius:5px;font:inherit}.lazy-report-controls button{padding:5px 10px;
 border:1px solid #c9c9c5;border-radius:5px;background:#fff;font:inherit;cursor:pointer}
.lazy-report-controls button:disabled{opacity:.45;cursor:default}
.lazy-report-controls select{padding:5px;border:1px solid #c9c9c5;border-radius:5px;
 background:#fff;font:inherit}.lazy-report-status{color:#5d5d58;font-size:12px}
.lazy-report-error{padding:12px;border:1px solid #e2aaa5;background:#fff0ef;
 color:#8e2822;border-radius:6px}.lazy-row-heading{color:#666;font-size:12px;
 margin:4px 0 8px}
"""


LAZY_REPORT_SCRIPT = r"""
(function(){
  if(window.LASERLazyReports){return;}
  const state = new WeakMap();
  function bytesFromBase64(value){
    const raw = atob(value.replace(/\s+/g, ''));
    const bytes = new Uint8Array(raw.length);
    for(let i=0;i<raw.length;i++){bytes[i]=raw.charCodeAt(i);}
    return bytes;
  }
  async function decodePayload(holder){
    if(!('DecompressionStream' in window)){
      throw new Error('This report needs a current Chrome, Edge, Firefox or Safari browser.');
    }
    const bytes = bytesFromBase64(holder.value || holder.textContent || '');
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    const text = await new Response(stream).text();
    return JSON.parse(text);
  }
  function plainText(markup){
    const box = document.createElement('div'); box.innerHTML = markup;
    return (box.textContent || '').toLocaleLowerCase();
  }
  function render(root){
    const current = state.get(root); if(!current){return;}
    const query = current.search.toLocaleLowerCase().trim();
    const matching = [];
    current.payload.rows.forEach((row,index)=>{
      if(!query || current.searchText[index].includes(query)){matching.push(index);}
    });
    const pageSize = current.pageSize;
    const pages = Math.max(1, Math.ceil(matching.length/pageSize));
    current.page = Math.max(0, Math.min(current.page,pages-1));
    const shown = matching.slice(current.page*pageSize,(current.page+1)*pageSize);
    let lastSection = null; const pieces = [];
    shown.forEach(index=>{
      const row=current.payload.rows[index];
      if(row.section !== lastSection){
        pieces.push(current.payload.sections[row.section] || '');
        lastSection=row.section;
      }
      const fragments=current.payload.fragments || {};
      pieces.push(row.html.replace(
        /@@LASER_FRAGMENT:([0-9]+)@@/g,
        (_,key)=>fragments[key] || ''
      ));
    });
    current.target.innerHTML = pieces.join('') || "<p class='empty-message'>No matching feature rows.</p>";
    current.previous.disabled=current.page===0;
    current.next.disabled=current.page>=pages-1;
    const first=matching.length ? current.page*pageSize+1 : 0;
    const last=Math.min((current.page+1)*pageSize,matching.length);
    current.status.textContent=`${first.toLocaleString()}–${last.toLocaleString()} of ${matching.length.toLocaleString()} rows · page ${current.page+1} of ${pages}`;
  }
  async function hydrate(root){
    if(!root || state.has(root)){return;}
    const holder=root.querySelector('[data-laser-lazy-payload]');
    const target=root.querySelector('[data-laser-lazy-target]');
    if(!holder || !target){return;}
    target.innerHTML="<p class='empty-message'>Loading detailed feature evidence…</p>";
    try{
      const payload=await decodePayload(holder);
      const current={payload,target,page:0,pageSize:Number(payload.page_size)||50,
        search:'',searchText:payload.rows.map(row=>plainText(row.html)),
        previous:root.querySelector('[data-lazy-previous]'),
        next:root.querySelector('[data-lazy-next]'),
        status:root.querySelector('[data-lazy-status]')};
      state.set(root,current); holder.remove();
      const search=root.querySelector('[data-lazy-search]');
      const size=root.querySelector('[data-lazy-page-size]');
      search.addEventListener('input',()=>{current.search=search.value;current.page=0;render(root);});
      size.value=String(current.pageSize);
      size.addEventListener('change',()=>{current.pageSize=Number(size.value)||50;current.page=0;render(root);});
      current.previous.addEventListener('click',()=>{current.page--;render(root);root.scrollIntoView({block:'start'});});
      current.next.addEventListener('click',()=>{current.page++;render(root);root.scrollIntoView({block:'start'});});
      render(root);
    }catch(error){target.innerHTML=`<div class='lazy-report-error'>Could not open detailed rows: ${String(error.message||error)}</div>`;}
  }
  function hydrateWithin(container){
    (container || document).querySelectorAll('[data-laser-lazy-report]').forEach(hydrate);
  }
  window.LASERLazyReports={hydrateWithin};
  document.addEventListener('DOMContentLoaded',()=>{
    if(window.LASER_LAZY_DEFER_HIDDEN){
      const openPanel=()=>{
        const selected=document.querySelector('input.tab:checked');
        if(!selected){return;}
        const key=selected.id.replace(/^tab-/, '');
        hydrateWithin(document.querySelector(`.panel-${CSS.escape(key)}`));
      };
      document.querySelectorAll('input.tab').forEach(input=>input.addEventListener('change',openPanel));
      openPanel();
    }else{hydrateWithin(document);}
  });
})();
"""


def lazy_feature_rows_markup(
    rows: Iterable[Mapping[str, str]],
    sections: Mapping[str, str],
    *,
    fragments: Mapping[str, str] | None = None,
    page_size: int = DEFAULT_REPORT_PAGE_SIZE,
    heading: str = "Detailed feature rows",
) -> str:
    """Return an offline, compressed and browser-paginated feature list."""
    payload = {
        "format_version": LAZY_REPORT_FORMAT_VERSION,
        "page_size": max(1, int(page_size)),
        "sections": dict(sections),
        "fragments": dict(fragments or {}),
        "rows": [dict(row) for row in rows],
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded = base64.b64encode(gzip.compress(raw, compresslevel=9)).decode("ascii")
    safe_heading = html.escape(heading)
    return (
        "<div class='lazy-feature-report' data-laser-lazy-report>"
        f"<div class='lazy-row-heading'>{safe_heading}. Rows are loaded in pages "
        "to keep this report responsive.</div>"
        "<div class='lazy-report-controls'>"
        "<input type='search' data-lazy-search placeholder='Filter feature rows…' "
        "aria-label='Filter feature rows'>"
        "<button type='button' data-lazy-previous>Previous</button>"
        "<button type='button' data-lazy-next>Next</button>"
        "<label>Rows <select data-lazy-page-size>"
        "<option>25</option><option selected>50</option><option>100</option>"
        "</select></label><span class='lazy-report-status' data-lazy-status></span>"
        "</div><div data-laser-lazy-target></div>"
        f"<textarea hidden data-laser-lazy-payload>{encoded}</textarea></div>"
    )
