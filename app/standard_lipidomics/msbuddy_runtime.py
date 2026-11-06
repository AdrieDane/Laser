"""Optional pinned native engine in a separate interpreter; no GUI downloads."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading

from standard_lipidomics.preferences import settings_path

MODELS = {
    'common_db_v0.2.4.joblib': 'cd3509150091a911ef8da85874bf71015fe7d0969b32dd97ccabefd9193dd8ec',
    'formula_db_v0.2.4.joblib': '9bac35f339f3d4c98eea558ffa95af37ba432d4d599148a98b27f8888d135997',
    'ml_v0.3.0.joblib': 'd24aa86b2ec0eea956993c8f9559c3e52e1d9ad31589513668a30268bfe4bee8',
}


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def configuration_path():
    return Path(os.environ.get('LASER_MSBUDDY_CONFIG', settings_path().with_name('msbuddy_runtime.json'))).expanduser()


def native_environment(python):
    script = '''import sys,json,hashlib,importlib.metadata as m
from pathlib import Path
assert (3,9)<=sys.version_info[:2]<(3,13), 'msbuddy requires an isolated supported Python 3.9-3.12 environment'
assert m.version('msbuddy')=='1.0.0', 'msbuddy release differs from the pinned version'
folder=Path(m.distribution('msbuddy').locate_file('msbuddy'))
sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.py')}
print(json.dumps({'python_version':sys.version,'packages':{d.metadata['Name']:d.version for d in m.distributions()},'msbuddy_sources':sources},sort_keys=True))'''
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    value = subprocess.run([str(python), '-c', script], capture_output=True, text=True, env=environment, timeout=30)
    if value.returncode:
        raise ValueError('msbuddy interpreter/package preflight failed: '+value.stderr[-1000:])
    return json.loads(value.stdout)


def freeze_runtime(config=None):
    """Resolve settings once at launch; changed binaries/models cannot be reused."""
    try:
        if config is None:
            config = json.loads(configuration_path().read_text())
        if config.get('schema_version') != 1 or set(config)-{'schema_version', 'python', 'data', 'libomp'}:
            raise ValueError('Unsupported msbuddy runtime configuration.')
        # Preserve the venv interpreter path: resolving its symlink would run
        # the base interpreter without the isolated packages.
        python, data = Path(config['python']).expanduser().absolute(), Path(config['data']).expanduser().resolve()
        if not python.is_file() or not os.access(python, os.X_OK):
            raise ValueError('msbuddy interpreter is missing or not executable.')
        for name, sha in MODELS.items():
            if digest(data/name) != sha:
                raise ValueError('msbuddy database/model differs from the pinned release: '+name)
        libomp = Path(config['libomp']).expanduser().resolve() if config.get('libomp') else None
        worker = Path(__file__).with_name('msbuddy_worker.py')
        return {'available': True, 'version': '1.0.0', 'python': str(python), 'data': str(data),
                'python_sha256': digest(python), 'model_hashes': dict(MODELS),
                'environment': native_environment(python),
                'worker_sha256': digest(worker), 'libomp': str(libomp) if libomp else '',
                'libomp_sha256': digest(libomp) if libomp else None}
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        return {'available': False, 'version': '1.0.0', 'reason': str(exc)}


def verify_runtime(runtime):
    if not runtime.get('available'):
        return False
    current = freeze_runtime({'schema_version': 1, **{k: runtime[k] for k in ('python', 'data', 'libomp')}})
    return current == runtime


def run_native(inputs, runtime, ppm, progress=None):
    """All hypothesis cases retained; failures never silently become predictions."""
    records, receipt = {}, {'runtime': runtime, 'requested_cases': len(inputs), 'status': 'unavailable'}
    if not inputs:
        receipt['status'] = 'no queries'
        return records, receipt
    if not runtime.get('available'):
        receipt['reason'] = runtime.get('reason', 'msbuddy not configured')
        return records, receipt
    if not verify_runtime(runtime):
        receipt['reason'] = 'Frozen msbuddy runtime changed before execution'
        return records, receipt
    try:
        with tempfile.TemporaryDirectory(prefix='laser-msbuddy-') as folder:
            folder = Path(folder)
            source, output = folder/'queries.json', folder/'results'
            source.write_text(json.dumps(inputs, allow_nan=False))
            environment = dict(os.environ, NUMBA_NUM_THREADS='2', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            if runtime['libomp']:
                environment['DYLD_LIBRARY_PATH'] = str(Path(runtime['libomp']).parent)
            command = [runtime['python'], '-u', str(Path(__file__).with_name('msbuddy_worker.py')),
                       '--inputs', str(source), '--data', runtime['data'], '--output', str(output), '--ppm', str(ppm)]
            # Python passes some loader variables through special *_PYTHON fields
            # on macOS; remove inherited overrides belonging to the GUI runtime.
            environment.pop('PYTHONPATH', None)
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, text=True, env=environment)
            watchdog = threading.Timer(1200, process.kill)
            watchdog.start()
            try:
                for line in process.stdout:
                    print(line.rstrip(), flush=True)
                    match = re.match(r'msbuddy (\d+) / (\d+)', line)
                    if match and progress:
                        count, total = map(int, match.groups())
                        progress(count/max(1, total), f'msbuddy: {count:,}/{total:,} ion hypotheses')
                code = process.wait()
            finally:
                watchdog.cancel()
                process.stdout.close()
            predictions = output/'predictions.jsonl'
            if predictions.is_file():
                for line in predictions.read_text().splitlines():
                    record = json.loads(line)
                    if record['query_id'] in records:
                        raise ValueError('Duplicate native query ID')
                    records[record['query_id']] = record
            expected = {q['query_id'] for q in inputs}
            if set(records)-expected:
                raise ValueError('Unexpected native query IDs')
            native_receipt = output/'frozen_receipt.json'
            receipt.update(status='complete' if code == 0 else 'partial native failure', exit_code=code,
                           input_sha256=digest(source), returned_cases=len(records),
                           native=json.loads(native_receipt.read_text()) if native_receipt.is_file() else {})
            if not verify_runtime(runtime):
                raise ValueError('Frozen msbuddy runtime changed during execution')
    except (OSError, ValueError, KeyError) as exc:
        records = {}
        receipt.update(status='unavailable', reason=str(exc))
    return records, receipt
