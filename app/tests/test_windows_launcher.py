"""Regression checks for the shared Windows server launcher."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_windows_launcher_uses_the_hub_server_contract():
    launcher = (ROOT / "Launch LASER Standard.bat").read_text(encoding="utf-8")

    assert 'set "LASER_STANDARD_HOST=0.0.0.0"' in launcher
    assert 'set "LASER_STANDARD_PORT=8085"' in launcher
    assert "-m pip install -e ." in launcher
    assert '--server.address "%LASER_STANDARD_HOST%"' in launcher
    assert '--server.port "%LASER_STANDARD_PORT%"' in launcher
    assert "--server.headless true" in launcher
    assert "--browser.gatherUsageStats false" in launcher


def test_python_314_support_is_declared_consistently():
    launcher = (ROOT / "Launch LASER Standard.bat").read_text(encoding="utf-8")
    setup = (ROOT / "Setup (Windows).bat").read_text(encoding="utf-8")
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version_module = (ROOT / "app" / "version.py").read_text(encoding="utf-8")

    assert "py -3.14 -m venv .venv" in launcher
    assert "py -3.14 -m venv .venv" in setup
    assert "Python 3.10-3.14" in setup
    assert 'requires-python = ">=3.10,<3.15"' in project
    assert 'version = "0.1.1"' in project
    assert '__version__ = "0.1.1"' in version_module
