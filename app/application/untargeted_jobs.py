"""Independent untargeted workers with frozen inputs and reopenable results."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from uuid import uuid4

import pandas as pd

from application.annotation_jobs import launch_worker, read_annotation_job, _now
from output_io import atomic_write_json
from standard_lipidomics.project_layout import xcms_project_root
from standard_lipidomics.acquisition_method import load_acquisition_method, normalize_acquisition_method
from standard_lipidomics.project_store import (
    load_annotation_workflow, load_untargeted_workflow, save_untargeted_workflow,
)
from standard_lipidomics.untargeted import (
    UntargetedComparisonCriteria, analyse_standard_untargeted, attach_excel_sample_metadata,
)
from standard_lipidomics.workflow import discover_standard_streams, load_standard_feature_list


def load_untargeted_inputs(project):
    """Use completed annotation when available, preserving accepted exclusions."""
    root = xcms_project_root(project)
    if (root / 'LASER' / 'LASER_PROJECT.json').is_file() or (root / 'LASER' / 'Annotation' / 'LASER_PROJECT.json').is_file():
        result, saved = load_annotation_workflow(project)
        provenance = str(saved['root'])
    else:
        sources = discover_standard_streams(project)
        if not sources:
            raise ValueError('No XCMS features or completed annotation were found.')
        result = {'engine': 'untargeted', 'streams': {
            name: {'features': load_standard_feature_list(path)} for name, path in sources.items()},
            'automatic_annotations': pd.DataFrame(), 'xcms_input_dir': str(root)}
        provenance = 'raw XCMS features; no completed annotation'
    result['untargeted_sample_metadata_audit'] = attach_excel_sample_metadata(result, root)
    result['acquisition_method'] = load_acquisition_method(root)
    return result, provenance


def active_untargeted_job(project):
    pointer = xcms_project_root(project) / '.laser/jobs/untargeted/current.json'
    if not pointer.is_file():
        return None
    directory = Path(json.loads(pointer.read_text())['directory'])
    return directory if read_annotation_job(directory)['status'] in {'queued', 'running'} else None


def start_untargeted_job(project, criteria, *, result=None, provenance='', existing='archive'):
    root = xcms_project_root(project)
    jobs = root / '.laser/jobs/untargeted'
    jobs.mkdir(parents=True, exist_ok=True)
    lock = jobs / 'launch.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError('An untargeted analysis is already being started.') from exc
    os.close(descriptor)
    directory = None
    try:
        active = active_untargeted_job(root)
        if active:
            return active
        if result is None:
            result, provenance = load_untargeted_inputs(root)
        result = {**result, 'acquisition_method': normalize_acquisition_method(
            result.get('acquisition_method') or load_acquisition_method(root))}
        directory = jobs / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid4().hex[:10])
        directory.mkdir()
        # Portable project tables freeze coordinates, sample levels, group maps,
        # and accepted annotations without serializing executable Python objects.
        save_untargeted_workflow(directory / 'input_snapshot', result,
            {'features': pd.DataFrame(), 'significant_features': pd.DataFrame(), 'criteria': {}},
            inputs={'project': str(root), 'annotation_source': provenance})
        atomic_write_json(directory / 'request.json', {'project': str(root), 'criteria': vars(criteria),
            'existing': existing, 'annotation_source': provenance, 'created_at': _now()})
        atomic_write_json(directory / 'status.json', {'status': 'queued', 'progress': 0.,
            'message': 'Starting untargeted analysis', 'updated_at': _now()})
        process = launch_worker(directory, 'application.untargeted_jobs', 'untargeted.log')
        atomic_write_json(directory / 'process.json', {'pid': process.pid})
        atomic_write_json(jobs / 'current.json', {'directory': str(directory)})
        return directory
    except Exception:
        if directory is not None and not (directory / 'process.json').is_file():
            shutil.rmtree(directory)
        raise
    finally:
        lock.unlink(missing_ok=True)


def run_untargeted_job(directory):
    directory = Path(directory).resolve()
    request = json.loads((directory / 'request.json').read_text())

    def progress(fraction, message):
        print(f"[{_now()}] {message}", flush=True)
        atomic_write_json(directory / 'status.json', {'status': 'running', 'progress': fraction,
            'message': message, 'updated_at': _now()})

    try:
        progress(.02, 'Loading frozen inputs')
        result, _ = load_untargeted_workflow(directory / 'input_snapshot')
        criteria = UntargetedComparisonCriteria(**request['criteria'])
        analysis = analyse_standard_untargeted(result, criteria, progress=progress)
        progress(.96, 'Saving untargeted analysis')
        paths, archive = save_untargeted_workflow(request['project'], result, analysis,
            inputs={'project': request['project'], 'annotation_source': request['annotation_source'],
                    'input_snapshot': str(directory / 'input_snapshot')}, existing=request['existing'])
        atomic_write_json(directory / 'status.json', {'status': 'complete', 'progress': 1.,
            'message': 'Untargeted analysis complete', 'project_manifest': str(paths['project_manifest']),
            'archive': str(archive) if archive else None, 'updated_at': _now()})
        print('Untargeted analysis complete', flush=True)
        return 0
    except Exception as exc:
        traceback.print_exc()
        atomic_write_json(directory / 'status.json', {'status': 'failed', 'progress': 0.,
            'message': str(exc), 'updated_at': _now()})
        return 1


if __name__ == '__main__':
    raise SystemExit(run_untargeted_job(sys.argv[1]))
