"""Mounted, offline Plotly charts with local linked selection.

Uses the LASER-4D be478d6 approach: retain the graph across component updates,
paint selection in the browser immediately, and update Python in a fragment.
"""
from functools import lru_cache

import plotly.io as pio
from plotly.offline import get_plotlyjs
import streamlit as st

from presentation.viewer_theme import current_theme, themed_viewer_figure


@lru_cache(maxsize=1)
def _runtime():
    return ("const Plotly = (() => { const module={exports:{}}; const exports=module.exports;\n"
            + get_plotlyjs() + "\nreturn module.exports;})();\n" + r'''
window.__laserStandardPlotly = Plotly;
const styles = new CSSStyleSheet();
styles.replaceSync(Array.from(document.querySelectorAll('style[id^="plotly.js-style"]'))
  .flatMap(style => Array.from(style.sheet?.cssRules ?? [], rule => rule.cssText)).join("\n"));
window.__laserStandardPlotStyles = styles;
document.dispatchEvent(new Event('laser-standard-plot-ready'));
export default function() {}
''')


_JS = r'''
export default function(component) {
  const {data, parentElement, setTriggerValue} = component;
  const host = parentElement.host ?? parentElement;
  let state = parentElement.__standardPlot;
  if (!state) {
    const graph = document.createElement('div');
    graph.style.width = '100%';
    graph.style.position = 'relative';
    parentElement.appendChild(graph);
    state = {graph, spec:null, channel:null, modifiers:{}, hovered:null, busy:false, disposed:false};
    parentElement.__standardPlot = state;
    state.observer = new ResizeObserver(() => {
      if (state.graph._fullLayout) window.__laserStandardPlotly?.Plots.resize(state.graph);
    });
    state.observer.observe(host);
  }
  state.disposed = false;
  state.data = data;
  state.emit = setTriggerValue;
  state.graph.style.height = data.height + 'px';
  if (state.cleanup) cancelAnimationFrame(state.cleanup);
  window.__laserStandardSelections ??= new Map();
  const selection = () => {
    if (!window.__laserStandardSelections.has(state.data.channel))
      window.__laserStandardSelections.set(state.data.channel, {keys:state.data.selected ?? [], anchor:null, activeKey:state.data.activeKey});
    return window.__laserStandardSelections.get(state.data.channel);
  };
  const paint = () => {
    const graph = state.graph, layout = graph._fullLayout;
    if (!layout) return;
    const keys = new Set(selection().keys);
    let layer = graph.querySelector(':scope > .standard-selection-layer');
    if (!layer) {
      layer = document.createElement('div'); layer.className = 'standard-selection-layer';
      Object.assign(layer.style, {position:'absolute', inset:'0', pointerEvents:'none', zIndex:'5'});
      graph.appendChild(layer);
    }
    const boxes = [];
    if (state.data.role === 'diagnostic') {
      const axis = layout.yaxis;
      const height = Math.abs(axis.c2p(1)-axis.c2p(0))*.9;
      state.data.orderedKeys.forEach((key,index) => {
        if (!keys.has(key)) return;
        const row = document.createElement('div');
        Object.assign(row.style, {position:'absolute', left:layout.xaxis._offset+'px',
          width:layout._size.w+'px', top:(axis._offset+axis.c2p(index)-height/2)+'px',
          height:height+'px', border:'1px solid #d62728', background:'rgba(214,39,40,.05)'});
        boxes.push(row);
      });
      for (const tick of graph.querySelectorAll('.ytick')) {
        const value = tick.__data__?.x;
        const index = typeof value === 'number' ? value : state.data.rowIds.indexOf(value);
        const text = tick.querySelector('text');
        if (text) text.style.fontWeight = keys.has(state.data.orderedKeys[index]) ? '700' : 'normal';
      }
    } else {
      for (const trace of graph._fullData ?? []) {
        if (!String(trace.mode ?? '').includes('markers') || !trace.customdata) continue;
        for (let i=0;i<trace.customdata.length;i++) {
          if (!keys.has(trace.customdata[i]?.at(-1))) continue;
          const x = layout.xaxis?._offset+layout.xaxis?.d2p(trace.x[i]);
          const y = layout.yaxis?._offset+layout.yaxis?.d2p(trace.y[i]);
          if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
          const marker = document.createElement('div');
          Object.assign(marker.style, {position:'absolute', left:(x-7)+'px', top:(y-7)+'px',
            width:'14px',height:'14px',border:'2px solid #d62728',borderRadius:'50%',background:'rgba(214,39,40,.18)'});
          boxes.push(marker);
        }
      }
    }
    layer.replaceChildren(...boxes);
    for (const point of graph.querySelectorAll('.scatterlayer .point')) {
      const trace=point.closest('.trace')?.__data__?.[0]?.trace;
      const index=point.__data__?.i, key=trace?.customdata?.[index]?.at(-1);
      const base=Array.isArray(trace?.marker?.color) ? trace.marker.color[index] : trace?.marker?.color;
      if (base) point.style.fill=keys.has(key) ? '#d62728' : base;
    }
  };
  const centerSelectedRow = () => {
    if (state.data.role !== 'diagnostic' || !state.graph._fullLayout) return;
    const current = selection();
    const active = current.keys.includes(current.activeKey) ? current.activeKey : current.keys.at(-1);
    const index = state.data.orderedKeys.indexOf(active);
    if (index < 0) return;
    const signature = state.data.channel + '::' + active;
    if (state.centeredSelection === signature) return;
    // Cross the component's shadow boundary, stopping at the inner scroll panel.
    // Never move the document or the main grid viewer.
    let viewport = host.parentElement;
    while (viewport && viewport !== document.body && viewport !== document.documentElement) {
      const overflow = getComputedStyle(viewport).overflowY;
      if (/^(auto|scroll)$/.test(overflow) && viewport.scrollHeight > viewport.clientHeight) {
        const axis = state.graph._fullLayout.yaxis;
        const rowY = state.graph.getBoundingClientRect().top + axis._offset + axis.c2p(index);
        const viewportY = viewport.getBoundingClientRect().top + viewport.clientTop;
        const target = viewport.scrollTop + rowY - viewportY - viewport.clientHeight / 2;
        viewport.scrollTop = Math.max(0, Math.min(viewport.scrollHeight - viewport.clientHeight, target));
        state.centeredSelection = signature;
        return;
      }
      viewport = viewport.parentElement ?? viewport.getRootNode()?.host;
    }
  };
  const choose = (key, modifiers, boxKeys=null) => {
    if (!key && !boxKeys) return;
    const current=selection(), ordered=state.data.orderedKeys;
    const toggle=modifiers.ctrlKey || modifiers.metaKey;
    if (boxKeys) current.keys=toggle ? [...new Set([...current.keys,...boxKeys])] : boxKeys;
    else if (modifiers.shiftKey && ordered.includes(current.anchor) && ordered.includes(key)) {
      const a=ordered.indexOf(current.anchor), b=ordered.indexOf(key);
      const range=ordered.slice(Math.min(a,b),Math.max(a,b)+1);
      current.keys=toggle ? [...new Set([...current.keys,...range])] : range;
    } else if (toggle) {
      current.keys=current.keys.includes(key) ? current.keys.filter(k=>k!==key) : [...current.keys,key];
      current.anchor=key;
    } else {current.keys=[key];current.anchor=key;}
    current.activeKey=key;
    document.dispatchEvent(new CustomEvent('laser-standard-selection',{detail:{channel:state.data.channel}}));
    state.emit('point',{customdata:['','',key], selectedKeys:current.keys, activeKey:key,
      ctrlKey:!!modifiers.ctrlKey,metaKey:!!modifiers.metaKey,shiftKey:!!modifiers.shiftKey,
      view:{xaxis:state.graph._fullLayout?.xaxis?.range,yaxis:state.graph._fullLayout?.yaxis?.range},timestamp:Date.now()});
  };
  state.paint=paint;
  const receive = event => {if (event.detail?.channel===state.data.channel) {state.paint();centerSelectedRow();}};
  if (state.receive) document.removeEventListener('laser-standard-selection',state.receive);
  state.receive=receive;document.addEventListener('laser-standard-selection',receive);
  const ready = window.__laserStandardPlotly ? Promise.resolve() : new Promise(resolve =>
    document.addEventListener('laser-standard-plot-ready',resolve,{once:true}));
  ready.then(async()=>{
    if (state.disposed) return;
    if (parentElement instanceof ShadowRoot && !parentElement.adoptedStyleSheets.includes(window.__laserStandardPlotStyles))
      parentElement.adoptedStyleSheets=[...parentElement.adoptedStyleSheets,window.__laserStandardPlotStyles];
    if (state.spec !== data.spec) {
      const spec=JSON.parse(data.spec);
      await window.__laserStandardPlotly.react(state.graph,spec.data,spec.layout,data.config);
      state.spec=data.spec;
    }
    if (!state.bound) {
      state.graph.addEventListener('pointerdown',event=>{state.contextHandled=false;state.modifiers={ctrlKey:event.ctrlKey,metaKey:event.metaKey,shiftKey:event.shiftKey};},true);
      state.graph.on('plotly_hover',event=>{state.hovered=event.points?.find(p=>p.customdata?.length>=3) ?? null;});
      state.graph.on('plotly_unhover',()=>{state.hovered=null;});
      state.graph.on('plotly_click',event=>{
        if (state.contextHandled) {state.contextHandled=false;return;}
        const key=event.points?.map(point=>point.customdata?.at(-1)).find(key=>state.data.validKeys.includes(key));
        if (key) choose(key,{...state.modifiers,...event.event});
      });
      state.graph.addEventListener('contextmenu',event=>{
        if (!event.ctrlKey) return;
        const key=state.hovered?.customdata?.at(-1);
        if (state.data.validKeys.includes(key)) {event.preventDefault();state.contextHandled=true;choose(key,event);}
      });
      state.graph.on('plotly_selected',event=>{
        const keys=[...new Set((event?.points??[]).map(p=>p.customdata?.at(-1)).filter(k=>state.data.validKeys.includes(k)))];
        if (keys.length) choose(keys.at(-1),state.modifiers,keys);
      });
      state.graph.on('plotly_relayout',()=>state.paint());
      state.bound=true;
    }
    paint();
    centerSelectedRow();
  }).catch(error=>console.error('LASER reviewer plot update failed',error));
  return ()=>{state.cleanup=requestAnimationFrame(()=>{
    if (host.isConnected) return;
    state.disposed=true;state.observer.disconnect();
    document.removeEventListener('laser-standard-selection',state.receive);
    window.__laserStandardPlotly?.purge(state.graph);
  });};
}
'''


def persistent_review_plot(figure, *, key, channel, ordered_keys, selected_keys,
                           on_point_change, role="main", row_ids=(), config=None, valid_keys=None):
    st.components.v2.component("standard_plotly_runtime", js=_runtime())(key="standard-plot-runtime-" + key, height=0)
    renderer = st.components.v2.component("standard_persistent_review_plot", js=_JS,
        css=":host { all: initial; display: block; width: 100%; contain: layout style; }")
    figure = themed_viewer_figure(figure, current_theme())
    return renderer(key=key, height=int(figure.layout.height or 450), width="stretch",
        data={"spec": pio.to_json(figure, validate=False), "height": int(figure.layout.height or 450),
              "channel": str(st.session_state.get("standard_review_run_token", "")) + "::" + channel, "role": role, "orderedKeys": list(ordered_keys),
              "rowIds": list(row_ids), "selected": list(selected_keys), "validKeys": list(valid_keys if valid_keys is not None else ordered_keys),
              "activeKey": st.session_state.get("standard_active_grid_keys", {}).get(channel),
              "config": config or {}}, on_point_change=on_point_change, default={})
