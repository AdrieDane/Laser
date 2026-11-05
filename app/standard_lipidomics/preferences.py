"""Per-user defaults shared by every LASER module."""
import json
import os
from pathlib import Path
import sys

from output_io import atomic_write_json


def settings_path():
    if os.environ.get('LASER_STANDARD_SETTINGS'):
        return Path(os.environ['LASER_STANDARD_SETTINGS']).expanduser()
    if sys.platform == 'darwin':
        parent = Path.home() / 'Library/Application Support'
    elif os.name == 'nt':
        parent = Path(os.environ.get('APPDATA', str(Path.home() / 'AppData/Roaming')))
    else:
        parent = Path(os.environ.get('XDG_CONFIG_HOME', str(Path.home() / '.config')))
    return parent / 'LASER Standard Lipidomics' / 'settings.json'


def default_project():
    explicit = os.environ.get('LASER_STANDARD_PROJECT')
    if explicit is not None:
        return explicit
    try:
        settings = json.loads(settings_path().read_text(encoding='utf-8'))
        return str(settings.get('project_folder', ''))
    except (OSError, ValueError, TypeError):
        return ''


def save_default_project(project):
    path = Path(project).expanduser().resolve()
    if not path.is_dir():
        raise ValueError('Choose an existing project folder.')
    target = settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, {'project_folder': str(path)})
    return target
