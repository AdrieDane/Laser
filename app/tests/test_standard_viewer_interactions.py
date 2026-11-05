"""Display projections, group colours and repeated cached reviewer updates."""
from itertools import product
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from presentation.standard_grid import annotation_diagnostics as diagnostics
from presentation.standard_grid.dimensions import (
    DIMENSIONS, dimension_values, kendrick_mass_defect, projected_grid_line,
)
from standard_lipidomics.grid_browser import _sample_intensity_figure


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
