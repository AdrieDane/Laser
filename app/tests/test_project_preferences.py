from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from standard_lipidomics.preferences import default_project, save_default_project

APP=Path(__file__).resolve().parents[1]/'app.py'


def test_saved_project_is_the_default_for_every_module(tmp_path,monkeypatch):
    monkeypatch.setenv('LASER_STANDARD_SETTINGS',str(tmp_path/'settings.json'))
    monkeypatch.delenv('LASER_STANDARD_PROJECT',raising=False)
    project=tmp_path/'project'
    project.mkdir()
    save_default_project(project)
    assert default_project()==str(project)
    for module in ('calibration','annotation','untargeted','open_annotation','open_untargeted'):
        app=AppTest.from_file(str(APP),default_timeout=60).run()
        app.button(key=f'standard-module-{module}').click().run()
        assert not app.exception
        assert next(widget for widget in app.text_input if widget.label=='Project folder').value==str(project)


def test_environment_can_override_a_saved_default(tmp_path,monkeypatch):
    monkeypatch.setenv('LASER_STANDARD_SETTINGS',str(tmp_path/'settings.json'))
    save_default_project(tmp_path)
    monkeypatch.setenv('LASER_STANDARD_PROJECT','/explicit/project')
    assert default_project()=='/explicit/project'
    monkeypatch.setenv('LASER_STANDARD_PROJECT','')
    assert default_project()==''


def test_missing_or_invalid_local_settings_are_safe(tmp_path,monkeypatch):
    monkeypatch.setenv('LASER_STANDARD_SETTINGS',str(tmp_path/'settings.json'))
    monkeypatch.delenv('LASER_STANDARD_PROJECT',raising=False)
    assert default_project()==''
    (tmp_path/'settings.json').write_text('{broken')
    assert default_project()==''
    with pytest.raises(ValueError):
        save_default_project(tmp_path/'missing')
