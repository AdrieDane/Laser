"""Display projections, group colours and repeated cached reviewer updates."""
from itertools import product
from pathlib import Path
import shutil
import subprocess
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from presentation.standard_grid import annotation_diagnostics as diagnostics
from presentation.standard_grid.dimensions import (
    DIMENSIONS, dimension_values, kendrick_mass_defect, projected_grid_line,
)
from standard_lipidomics.grid_browser import _sample_intensity_figure
from presentation.standard_grid.persistent_plot import _JS


@pytest.mark.parametrize('x,y', list(product(DIMENSIONS, repeat=2)))
def test_mesh_projects_to_both_selected_dimensions(x, y):
    rt = np.array([4., 5.])
    mz = np.array([700., 700. + 14.015650064])
    xs, ys = projected_grid_line((rt, mz), x, y)
    np.testing.assert_allclose(xs, dimension_values(rt, mz, x))
    np.testing.assert_allclose(ys, dimension_values(rt, mz, y))


def test_kendrick_homologues_align_and_nan_stays_missing():
    values = kendrick_mass_defect([700., 714.015650064, np.nan])
    assert values[0] == pytest.approx(values[1])
    assert np.isnan(values[2])


def test_sample_bars_hide_labels_and_preserve_metadata_groups_in_hover():
    row = pd.Series({'name': 'PC(34:1)', 'adduct': '[M+H]+', 'feat_a': 10, 'feat_b': 20, 'feat_c': 30})
    figure = _sample_intensity_figure(row, ['a', 'b', 'c'], {'a': 'Control', 'b': 'Case', 'c': 'Control'})
    traces = {trace.name: trace for trace in figure.data}
    assert set(traces) == {'Control', 'Case'}
    assert traces['Control'].marker.color != traces['Case'].marker.color
    assert list(traces['Control'].customdata) == ['a', 'c']
    assert list(traces['Control'].x) == [0, 2]
    assert figure.layout.xaxis.showticklabels is False


def test_diagnostic_rows_survive_cached_selection_updates(monkeypatch):
    frame = pd.DataFrame({'carbon': [34], 'db': [1], 'name': ['PC(34:1)'],
                          'adduct': ['[M+H]+'], 'absresid': [.1], 'key': ['key1']})
    class Session(dict):
        __setattr__ = dict.__setitem__
    session = Session()
    monkeypatch.setattr(diagnostics.st, 'session_state', session)
    build = Mock(return_value=('header', 'body'))
    render = Mock()
    monkeypatch.setattr(diagnostics, '_diagnostic_figures', build)
    monkeypatch.setattr(diagnostics, '_render_figures', render)
    for active in (set(), {'key1'}):
        diagnostics._render_annotation_diagnostics(frame, ['sample'], chart_key='PC',
            active_keys=active, services=Mock())
    assert build.call_count == 1
    assert render.call_count == 2
    for call in render.call_args_list:
        assert call.args[1].row_id.tolist() == ['0000|PC(34:1)|[M+H]+']


def test_linked_selection_centers_only_diagnostic_scroll_panel():
    node = shutil.which('node')
    bundled = Path('/Applications/ChatGPT.app/Contents/Resources/cua_node/bin/node')
    node = node or (str(bundled) if bundled.is_file() else None)
    if not node:
        pytest.skip('Node is required for browser-local interaction checks')
    script = r'''
import assert from 'node:assert/strict';
const listeners = new Map();
const layer = {style:{}, replaceChildren(){}};
const viewport = {scrollTop:0, scrollHeight:2230, clientHeight:470, clientTop:0,
  getBoundingClientRect:()=>({top:100}), overflow:'auto'};
const wrapper = {parentElement:viewport, overflow:'visible'};
const host = {parentElement:wrapper, isConnected:true};
const graph = {style:{}, querySelector:()=>layer, querySelectorAll:()=>[],
  appendChild(){}, addEventListener(){}, on(){},
  getBoundingClientRect:()=>({top:100-viewport.scrollTop})};
globalThis.document = {body:{}, documentElement:{}, createElement:()=>graph,
  addEventListener:(name,fn)=>listeners.set(name,fn), removeEventListener(){}};
globalThis.getComputedStyle = element=>({overflowY:element.overflow});
globalThis.ResizeObserver = class {observe(){} disconnect(){}};
globalThis.ShadowRoot = class {};
globalThis.window = {__laserStandardPlotly:{react:async()=>{graph._fullLayout={
  yaxis:{_offset:10,c2p:index=>(99-index)*22},xaxis:{_offset:100},_size:{w:600}};}}};
const parentElement = {host, appendChild(){}};
const component = {parentElement,setTriggerValue:()=>assert.fail('Scroll must stay local'),
  data:{channel:'PC',role:'diagnostic',height:2230,spec:'{}',config:{},
    selected:[],orderedKeys:Array.from({length:100},(_,i)=>'key'+i),rowIds:[]}};
'''
    script += '\nconst render = (' + _JS.replace('export default function', 'function', 1) + ');\n'
    script += r'''
render(component);
await new Promise(resolve=>setImmediate(resolve));
const select = (keys,activeKey)=>{
  window.__laserStandardSelections.set('PC',{keys,activeKey});
  listeners.get('laser-standard-selection')({detail:{channel:'PC'}});
};
select(['key50'],'key50');
assert.equal(viewport.scrollTop,10+49*22-235);
// A repaint or server component update must preserve manual scrolling.
viewport.scrollTop=300;
listeners.get('laser-standard-selection')({detail:{channel:'PC'}});
render(component);
await new Promise(resolve=>setImmediate(resolve));
assert.equal(viewport.scrollTop,300);
// Multi-selection follows the clicked active row, not the first selected row.
select(['key50','key70'],'key70');
assert.equal(viewport.scrollTop,10+29*22-235);
select(['key0'],'key0');
assert.equal(viewport.scrollTop,2230-470);
select(['key99'],'key99');
assert.equal(viewport.scrollTop,0);
// Other charts and absent rows never scroll this panel or the document.
viewport.scrollTop=90;
select(['absent'],'absent');
assert.equal(viewport.scrollTop,90);
component.data.role='main';
render(component);
select(['key80'],'key80');
assert.equal(viewport.scrollTop,90);
'''
    completed = subprocess.run([node, '--input-type=module'], input=script,
                               capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stderr
